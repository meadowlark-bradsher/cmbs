#!/usr/bin/env python3
"""
Naive Ollama Agent Test for IT-Bench CISO Scenario 1
Tests a simple LLM call to generate a Kyverno policy.
"""

import subprocess
import json
import sys
import os

GOAL = """I would like to check if the following condition is satisfied, given a Kubernetes cluster with `kubeconfig.yaml`:
    Minimize the admission of containers wishing to share the host network namespace.

To check the condition, do the following steps:
- deploy a Kyverno policy to the cluster
- check if the policy is correctly deployed.

If deploying the policy failed and if you can fix the issue, you will do it and try deploying again.
Once you get a final answer, you can quit the work.

The cluster's kubeconfig is at `/tmp/itbench-agent/kubeconfig.yaml`.
"""

SYSTEM_PROMPT = """You are a Kubernetes security expert. Generate a Kyverno ClusterPolicy that detects pods using hostNetwork.

Requirements:
1. Output ONLY valid YAML - no explanations before or after
2. Use apiVersion: kyverno.io/v1
3. Use kind: ClusterPolicy
4. The policy should audit (not block) pods with hostNetwork: true
5. Exclude kube-system and local-path-storage namespaces

Output the complete YAML policy now:"""


def call_ollama(model: str, prompt: str) -> str:
    """Call Ollama and return the response."""
    cmd = ["ollama", "run", model, prompt]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.stdout.strip()


def extract_yaml(text: str) -> str:
    """Extract YAML from response, handling markdown code blocks."""
    lines = text.split('\n')
    in_yaml = False
    yaml_lines = []

    for line in lines:
        if line.strip().startswith('```yaml') or line.strip().startswith('```'):
            if in_yaml:
                break  # End of code block
            in_yaml = True
            continue
        if in_yaml:
            yaml_lines.append(line)
        elif line.strip().startswith('apiVersion:'):
            # Raw YAML without code blocks
            yaml_lines = lines[lines.index(line):]
            break

    return '\n'.join(yaml_lines).strip()


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"

    print(f"[*] Using model: {model}")
    print(f"[*] Goal: Minimize admission of containers sharing host network namespace")
    print(f"[*] Generating Kyverno policy...")

    # Generate policy
    response = call_ollama(model, SYSTEM_PROMPT)
    print(f"\n[*] Raw response length: {len(response)} chars")

    # Extract YAML
    policy_yaml = extract_yaml(response)

    if not policy_yaml or 'apiVersion' not in policy_yaml:
        print(f"[!] Failed to generate valid YAML")
        print(f"[!] Response:\n{response[:500]}...")
        return 1

    # Save policy
    policy_path = os.path.join(output_dir, "policy.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(policy_path, 'w') as f:
        f.write(policy_yaml)

    print(f"[+] Policy saved to: {policy_path}")
    print(f"\n--- Generated Policy ---\n{policy_yaml}\n--- End Policy ---\n")

    # Apply policy
    print("[*] Applying policy to cluster...")
    result = subprocess.run(
        ["kubectl", "apply", "-f", policy_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[!] Failed to apply policy: {result.stderr}")
        return 1

    print(f"[+] {result.stdout.strip()}")

    # Wait for policy to be ready
    print("[*] Waiting for policy to be ready...")
    subprocess.run(["sleep", "5"])

    # Check for policy reports
    print("[*] Checking for policy reports in 'paa' namespace...")
    result = subprocess.run(
        ["kubectl", "get", "policyreport", "-n", "paa", "-o", "wide"],
        capture_output=True, text=True
    )

    if result.stdout.strip():
        print(f"[+] Policy reports found:\n{result.stdout}")
    else:
        print("[!] No policy reports yet (may take a moment)")

    # Summary
    print("\n" + "="*60)
    print("NAIVE TEST COMPLETE")
    print("="*60)
    print(f"Model: {model}")
    print(f"Policy generated: {os.path.exists(policy_path)}")
    print(f"Policy applied: {result.returncode == 0}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
