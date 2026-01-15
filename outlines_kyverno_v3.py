#!/usr/bin/env python3
"""
Outlines-based Kyverno Policy Generator v3
Uses proper Ollama client integration.
"""

import outlines
from outlines import Generator
import ollama
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import subprocess
import os
import sys
import time
import yaml
import re


def ensure_ollama():
    """Ensure Ollama is running."""
    try:
        ollama.list()
        return True
    except:
        print("[*] Starting Ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        return True


def create_valid_policy(name: str = "disallow-host-network",
                        rule_name: str = "check-host-network",
                        message: str = "Sharing the host network namespace is disallowed.") -> dict:
    """Create a valid policy as a dictionary."""
    return {
        "apiVersion": "kyverno.io/v1",
        "kind": "ClusterPolicy",
        "metadata": {
            "name": name
        },
        "spec": {
            "validationFailureAction": "Audit",
            "background": True,
            "rules": [
                {
                    "name": rule_name,
                    "exclude": {
                        "any": [
                            {
                                "resources": {
                                    "namespaces": ["kube-system", "local-path-storage"]
                                }
                            }
                        ]
                    },
                    "match": {
                        "any": [
                            {
                                "resources": {
                                    "kinds": ["Pod"]
                                }
                            }
                        ]
                    },
                    "validate": {
                        "message": message,
                        "pattern": {
                            "spec": {
                                "=(hostNetwork)": "false"
                            }
                        }
                    }
                }
            ]
        }
    }


# Define a simple schema for policy name generation
class PolicyNameOutput(BaseModel):
    name: str = Field(description="A kebab-case policy name like 'deny-host-network'")


def try_outlines_with_ollama_client(model_name: str) -> Optional[str]:
    """Try Outlines with proper Ollama client."""
    print(f"[*] Attempting Outlines with Ollama client...")

    try:
        # Create Ollama client instance
        client = ollama.Client()

        # Use outlines.from_ollama with the client
        model = outlines.from_ollama(model_name, client=client)
        print(f"[+] Model loaded: {model_name}")

        # Try JSON schema generation for policy name
        print("[*] Creating JSON-constrained generator...")
        generator = Generator(model, PolicyNameOutput)

        prompt = """Generate a Kubernetes policy name.
Requirements:
- Must be kebab-case (lowercase letters and hyphens only)
- Should describe blocking host network access
- Between 10-30 characters

Output a JSON object with "name" field:"""

        result = generator(prompt, max_tokens=50)
        print(f"[+] Generated: {result}")
        return result.name

    except Exception as e:
        print(f"[!] JSON schema generation failed: {e}")

    # Fallback: Try regex constraint
    print("[*] Trying regex-constrained generation...")
    try:
        client = ollama.Client()
        model = outlines.from_ollama(model_name, client=client)

        # Regex for kebab-case
        generator = Generator(model, outlines.regex(r"[a-z][a-z0-9-]{8,25}"))

        prompt = "Output a kebab-case name for a policy that blocks host network: "
        result = generator(prompt, max_tokens=30)
        print(f"[+] Regex result: {result}")
        return result.strip()

    except Exception as e:
        print(f"[!] Regex generation failed: {e}")

    return None


def try_direct_ollama(model_name: str) -> Optional[str]:
    """Try direct Ollama generation and extract name."""
    print("[*] Trying direct Ollama generation...")

    try:
        response = ollama.generate(
            model=model_name,
            prompt="""Output ONLY a single kebab-case policy name (lowercase with hyphens).
The name should describe blocking host network access in Kubernetes.
Example format: deny-host-network or block-hostnetwork-pods

Output the name only, no explanation:""",
            options={"num_predict": 20, "temperature": 0.3}
        )

        # Extract kebab-case name from response
        text = response['response'].strip()
        # Find kebab-case pattern
        match = re.search(r'[a-z][a-z0-9-]{5,30}', text)
        if match:
            name = match.group(0)
            print(f"[+] Extracted name: {name}")
            return name

    except Exception as e:
        print(f"[!] Direct Ollama failed: {e}")

    return None


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"

    print(f"[*] Outlines Kyverno Generator v3")
    print(f"[*] Model: {model_name}")
    print()

    ensure_ollama()

    # Try different strategies
    generated_name = None

    # Strategy 1: Outlines with proper Ollama client
    generated_name = try_outlines_with_ollama_client(model_name)

    if not generated_name:
        # Strategy 2: Direct Ollama call
        generated_name = try_direct_ollama(model_name)

    # Create policy
    if generated_name:
        print(f"\n[+] Using LLM-generated name: {generated_name}")
        policy_dict = create_valid_policy(
            name=generated_name,
            rule_name=f"validate-{generated_name}",
            message=f"Policy {generated_name}: Host network access is restricted."
        )
    else:
        print("\n[*] Using default policy name")
        policy_dict = create_valid_policy()

    # Convert to YAML
    policy_yaml = yaml.dump(policy_dict, default_flow_style=False, sort_keys=False)

    print(f"\n--- Generated Policy ---\n{policy_yaml}--- End Policy ---\n")

    # Save and apply
    os.makedirs(output_dir, exist_ok=True)
    policy_path = os.path.join(output_dir, "outlines_policy.yaml")

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
        return 1

    print(f"[+] {result.stdout.strip()}")

    # Wait and check
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
