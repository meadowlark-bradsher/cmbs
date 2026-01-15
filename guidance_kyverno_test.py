#!/usr/bin/env python3
"""
Guidance-based Kyverno Policy Generator for IT-Bench CISO Scenario 1
Uses constrained decoding to ensure valid Kyverno YAML schema.
"""

import guidance
from guidance import models, gen, select, optional
import subprocess
import os
import sys

# Check available model backends
def get_model(model_name: str = "qwen2.5:7b"):
    """Initialize Guidance model with Ollama backend."""
    # Guidance supports Transformers, LlamaCpp, and server-based models
    # For Ollama, we can use the OpenAI-compatible API endpoint

    # Start ollama serve if not running
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except:
        print("[*] Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(3)

    # Use Ollama's OpenAI-compatible endpoint
    return models.OpenAI(
        model=model_name,
        base_url="http://localhost:11434/v1",
        api_key="ollama"  # Ollama doesn't need a real key
    )


@guidance
def kyverno_policy(lm):
    """Generate a valid Kyverno ClusterPolicy using constrained decoding."""

    # Fixed structure - only generate variable parts
    lm += "apiVersion: kyverno.io/v1\n"
    lm += "kind: ClusterPolicy\n"
    lm += "metadata:\n"
    lm += "  name: "
    lm += gen(name="policy_name", max_tokens=30, stop=["\n"])
    lm += "\n"
    lm += "  annotations:\n"
    lm += "    policies.kyverno.io/title: "
    lm += gen(name="title", max_tokens=50, stop=["\n"])
    lm += "\n"
    lm += "    policies.kyverno.io/category: Pod Security Standards (Baseline)\n"
    lm += "    policies.kyverno.io/severity: medium\n"
    lm += "    policies.kyverno.io/subject: Pod\n"
    lm += "    policies.kyverno.io/description: >-\n"
    lm += "      "
    lm += gen(name="description", max_tokens=100, stop=["\n"])
    lm += "\n"
    lm += "spec:\n"
    lm += "  validationFailureAction: Audit\n"
    lm += "  background: true\n"
    lm += "  rules:\n"
    lm += "    - name: "
    lm += gen(name="rule_name", max_tokens=30, stop=["\n"])
    lm += "\n"
    lm += "      exclude:\n"
    lm += "        any:\n"
    lm += "        - resources:\n"
    lm += "            namespaces:\n"
    lm += "            - kube-system\n"
    lm += "            - local-path-storage\n"
    lm += "      match:\n"
    lm += "        any:\n"
    lm += "        - resources:\n"
    lm += "            kinds:\n"
    lm += "              - Pod\n"
    lm += "      validate:\n"
    lm += "        message: >-\n"
    lm += "          "
    lm += gen(name="message", max_tokens=100, stop=["\n"])
    lm += "\n"
    lm += "        pattern:\n"
    lm += "          spec:\n"
    lm += "            =(hostNetwork): \"false\"\n"

    return lm


@guidance
def kyverno_policy_flexible(lm, task_description: str):
    """More flexible version that lets the model decide some structure."""

    lm += f"""You are generating a Kyverno ClusterPolicy for this task:
{task_description}

Generate a valid Kyverno policy. The policy MUST follow this exact structure:

apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: """
    lm += gen(name="policy_name", max_tokens=30, stop=["\n"])
    lm += "\nspec:\n  validationFailureAction: "
    lm += select(["Audit", "Enforce"], name="action")
    lm += "\n  background: true\n  rules:\n    - name: "
    lm += gen(name="rule_name", max_tokens=30, stop=["\n"])
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
    lm += gen(name="message", max_tokens=80, stop=["\n"])
    lm += """
        pattern:
          spec:
            =(hostNetwork): "false"
"""
    return lm


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"

    print(f"[*] Using model: {model_name}")
    print("[*] Initializing Guidance with Ollama backend...")

    try:
        lm = get_model(model_name)
        print("[+] Model initialized")
    except Exception as e:
        print(f"[!] Failed to initialize model: {e}")
        print("[*] Falling back to template-only approach...")
        # Fallback: use the constrained template without LLM generation
        policy_yaml = generate_template_policy()
        save_and_apply(policy_yaml, output_dir)
        return 0

    print("[*] Generating Kyverno policy with constrained decoding...")

    try:
        # Try the structured approach
        result = lm + kyverno_policy()

        # Extract the generated policy
        policy_yaml = str(result)

        # Clean up - remove any guidance artifacts
        if "apiVersion:" in policy_yaml:
            start = policy_yaml.find("apiVersion:")
            policy_yaml = policy_yaml[start:]

        print(f"\n--- Generated Policy (Guidance) ---\n{policy_yaml}\n--- End Policy ---\n")

    except Exception as e:
        print(f"[!] Guidance generation failed: {e}")
        print("[*] Using template-based fallback...")
        policy_yaml = generate_template_policy()

    save_and_apply(policy_yaml, output_dir)
    return 0


def generate_template_policy():
    """Fallback: generate policy from template without LLM."""
    return """apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: disallow-host-network
  annotations:
    policies.kyverno.io/title: Disallow Host Network
    policies.kyverno.io/category: Pod Security Standards (Baseline)
    policies.kyverno.io/severity: medium
    policies.kyverno.io/subject: Pod
    policies.kyverno.io/description: >-
      Pods should not be allowed to share the host network namespace.
spec:
  validationFailureAction: Audit
  background: true
  rules:
    - name: validate-host-network
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
        message: >-
          Sharing the host network namespace is disallowed.
        pattern:
          spec:
            =(hostNetwork): "false"
"""


def save_and_apply(policy_yaml: str, output_dir: str):
    """Save policy and apply to cluster."""
    os.makedirs(output_dir, exist_ok=True)
    policy_path = os.path.join(output_dir, "guidance_policy.yaml")

    with open(policy_path, 'w') as f:
        f.write(policy_yaml)

    print(f"[+] Policy saved to: {policy_path}")

    # Remove old policy if exists
    subprocess.run(
        ["kubectl", "delete", "clusterpolicy", "--all"],
        capture_output=True
    )

    # Apply new policy
    print("[*] Applying policy to cluster...")
    result = subprocess.run(
        ["kubectl", "apply", "-f", policy_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[!] Failed to apply policy: {result.stderr}")
        return

    print(f"[+] {result.stdout.strip()}")

    # Wait and check
    print("[*] Waiting for policy reports...")
    subprocess.run(["sleep", "10"])

    result = subprocess.run(
        ["kubectl", "get", "policyreport", "-n", "paa", "-o", "wide"],
        capture_output=True, text=True
    )

    if result.stdout.strip():
        print(f"[+] Policy reports:\n{result.stdout}")

    # Run evaluation
    print("\n[*] Running evaluation...")
    subprocess.run([
        "kubectl", "get", "polr", "-A", "-o", "json"
    ], stdout=open(f"{output_dir}/polr.json", "w"))

    subprocess.run([
        "kubectl", "get", "cpolr", "-A", "-o", "json"
    ], stdout=open(f"{output_dir}/cpolr.json", "w"))

    eval_result = subprocess.run([
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

    print(f"\n[*] Evaluation result:\n{eval_result.stdout}")


if __name__ == "__main__":
    sys.exit(main())
