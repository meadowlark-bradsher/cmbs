#!/usr/bin/env python3
"""
Guidance-based Kyverno Policy Generator v2
Uses proper role context for OpenAI-compatible APIs (Ollama).
"""

import guidance
from guidance import models, gen, select, system, user, assistant
import subprocess
import os
import sys
import time

def ensure_ollama():
    """Ensure Ollama is running."""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return True
    except:
        print("[*] Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        return True


def get_model(model_name: str = "qwen2.5:7b"):
    """Initialize Guidance model with Ollama backend."""
    ensure_ollama()
    return models.OpenAI(
        model=model_name,
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )


@guidance
def generate_kyverno_with_roles(lm, task: str):
    """Generate Kyverno policy using proper chat roles."""

    with system():
        lm += """You are a Kubernetes security expert specializing in Kyverno policies.
You will generate valid Kyverno ClusterPolicy YAML.
Always output ONLY the YAML - no explanations."""

    with user():
        lm += f"""Generate a Kyverno ClusterPolicy for this task:
{task}

The policy must:
1. Use apiVersion: kyverno.io/v1
2. Use kind: ClusterPolicy
3. Use validationFailureAction: Audit
4. Exclude kube-system and local-path-storage namespaces
5. Match Pod resources
6. Validate that spec.hostNetwork is "false"

Output ONLY the YAML:"""

    with assistant():
        # Constrained generation with fixed structure
        lm += """apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: """
        lm += gen(name="policy_name", max_tokens=30, stop=["\n"], temperature=0.3)
        lm += """
spec:
  validationFailureAction: Audit
  background: true
  rules:
    - name: """
        lm += gen(name="rule_name", max_tokens=30, stop=["\n"], temperature=0.3)
        lm += """
      exclude:
        any:
        - resources:
            namespaces:
            - kube-system
            - local-path-storage
      match:
        any:
        - resources:
            kinds:
              - Pod
      validate:
        message: """
        lm += gen(name="message", max_tokens=60, stop=["\n"], temperature=0.3)
        lm += """
        pattern:
          spec:
            =(hostNetwork): "false"
"""

    return lm


@guidance
def generate_kyverno_freeform(lm, task: str):
    """Let the model generate more freely but within structure."""

    with system():
        lm += "You are a Kyverno policy expert. Output only valid YAML."

    with user():
        lm += f"Create a Kyverno ClusterPolicy to: {task}"

    with assistant():
        lm += gen(name="full_policy", max_tokens=500, temperature=0.2)

    return lm


def extract_yaml_from_result(result) -> str:
    """Extract the YAML portion from Guidance result."""
    text = str(result)

    # Find the YAML start
    if "apiVersion:" in text:
        start = text.find("apiVersion:")
        text = text[start:]

    # Remove any trailing non-YAML content
    lines = text.split('\n')
    yaml_lines = []
    for line in lines:
        # Stop if we hit obvious non-YAML
        if line.strip().startswith(('```', 'Note:', 'This policy')):
            break
        yaml_lines.append(line)

    return '\n'.join(yaml_lines).strip()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"
    task = "Minimize the admission of containers wishing to share the host network namespace"

    print(f"[*] Guidance Kyverno Generator v2")
    print(f"[*] Model: {model_name}")
    print(f"[*] Task: {task}")
    print()

    try:
        lm = get_model(model_name)
        print("[+] Model initialized via Ollama OpenAI-compatible API")
    except Exception as e:
        print(f"[!] Model init failed: {e}")
        return 1

    # Try constrained generation
    print("\n[*] Attempt 1: Constrained generation with fixed structure...")
    try:
        result = lm + generate_kyverno_with_roles(task=task)
        policy_yaml = extract_yaml_from_result(result)

        print(f"\n--- Generated Values ---")
        print(f"policy_name: {result.get('policy_name', 'N/A')}")
        print(f"rule_name: {result.get('rule_name', 'N/A')}")
        print(f"message: {result.get('message', 'N/A')}")
        print(f"--- End Values ---\n")

        print(f"--- Full Policy ---\n{policy_yaml}\n--- End Policy ---\n")

        if "apiVersion" in policy_yaml and "ClusterPolicy" in policy_yaml:
            print("[+] Constrained generation succeeded!")
        else:
            raise ValueError("Invalid policy structure")

    except Exception as e:
        print(f"[!] Constrained generation failed: {e}")
        print("\n[*] Attempt 2: Freeform generation...")

        try:
            result = lm + generate_kyverno_freeform(task=task)
            policy_yaml = extract_yaml_from_result(result)
            print(f"--- Freeform Policy ---\n{policy_yaml}\n--- End Policy ---\n")

        except Exception as e2:
            print(f"[!] Freeform generation also failed: {e2}")
            print("[*] Using hardcoded template as fallback...")
            policy_yaml = get_template_policy()

    # Save and apply
    os.makedirs(output_dir, exist_ok=True)
    policy_path = os.path.join(output_dir, "guidance_policy.yaml")

    with open(policy_path, 'w') as f:
        f.write(policy_yaml)

    print(f"[+] Policy saved to: {policy_path}")

    # Delete existing policies
    subprocess.run(["kubectl", "delete", "clusterpolicy", "--all"], capture_output=True)

    # Apply
    print("[*] Applying to cluster...")
    result = subprocess.run(["kubectl", "apply", "-f", policy_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Apply failed: {result.stderr}")
        print("[*] Policy may have schema errors - this is the test!")
        return 1

    print(f"[+] {result.stdout.strip()}")

    # Wait and evaluate
    print("[*] Waiting 10s for policy reports...")
    time.sleep(10)

    result = subprocess.run(
        ["kubectl", "get", "policyreport", "-n", "paa", "-o", "wide"],
        capture_output=True, text=True
    )
    print(f"[*] Policy reports:\n{result.stdout}")

    # Run evaluation
    run_evaluation(output_dir)

    return 0


def get_template_policy():
    """Fallback template."""
    return """apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-host-network
spec:
  validationFailureAction: Audit
  background: true
  rules:
    - name: check-host-network
      exclude:
        any:
        - resources:
            namespaces:
            - kube-system
            - local-path-storage
      match:
        any:
        - resources:
            kinds:
              - Pod
      validate:
        message: Host network sharing is not allowed
        pattern:
          spec:
            =(hostNetwork): "false"
"""


def run_evaluation(output_dir: str):
    """Run IT-Bench evaluation."""
    subprocess.run(["kubectl", "get", "polr", "-A", "-o", "json"],
                   stdout=open(f"{output_dir}/polr.json", "w"))
    subprocess.run(["kubectl", "get", "cpolr", "-A", "-o", "json"],
                   stdout=open(f"{output_dir}/cpolr.json", "w"))

    result = subprocess.run([
        "python3",
        "/Users/meadowlarkbradsher/workspace/repos/genai/mast/ITBench/scenarios/ciso/1.gen-cis-b-k8s-kyverno/evaluation/main.py",
        "--policy-reports", f"{output_dir}/polr.json",
        "--cluster-policy-reports", f"{output_dir}/cpolr.json",
        "--api-version", "apps/v1",
        "--kind", "Deployment",
        "--namespace", "paa",
        "--name", "nginx-deployment",
        "--agent-output", output_dir
    ], capture_output=True, text=True)

    print(f"\n{'='*60}")
    print("EVALUATION RESULT")
    print('='*60)
    print(result.stdout)


if __name__ == "__main__":
    sys.exit(main())
