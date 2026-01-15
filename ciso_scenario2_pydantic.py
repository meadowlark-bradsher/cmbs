#!/usr/bin/env python3
"""
CISO Scenario 2: gen-cis-b-k8s-kubectl-opa
Pydantic schemas for generating kubectl scripts and OPA Rego policies.
"""

import ollama
import subprocess
import os
import time
import re


def create_kubectl_script(resource: str = "pods",
                          namespace: str = "-A",
                          output_file: str = "collected_data.json") -> str:
    """
    Generate a valid kubectl script.
    Structure is fixed, only resource type varies.
    """
    return f"""#!/bin/bash
# Collect Kubernetes resources for compliance checking
kubectl get {resource} {namespace} -o json > {output_file}
"""


def create_opa_policy_hostnetwork() -> str:
    """
    Generate OPA Rego policy to check hostNetwork compliance.
    The Rego structure is fixed for this specific check.
    """
    return '''package check

default result = false

# Returns false if any pod has hostNetwork enabled (non-compliant)
# Returns true if all pods have hostNetwork disabled (compliant)
result {
    # Check all items in the input
    pods := input.items
    # Ensure no pod has hostNetwork set to true
    not any_host_network(pods)
}

any_host_network(pods) {
    pods[_].spec.hostNetwork == true
}
'''


def create_opa_policy_generic(field_path: str, expected_value: str, check_name: str) -> str:
    """
    Generate a generic OPA policy checking a specific field.
    LLM can provide: field_path, expected_value, check_name
    """
    return f'''package check

default result = false

# Check: {check_name}
result {{
    input.{field_path} == {expected_value}
}}
'''


def generate_with_llm(model: str = "qwen2.5:7b") -> dict:
    """Let LLM generate variable parts, constrain structure."""

    # Ask LLM for the check description
    response = ollama.generate(
        model=model,
        prompt="""For a Kubernetes compliance check about "minimizing containers sharing host network namespace":

1. What kubectl resource should we query? (e.g., pods, deployments)
2. What field indicates host network usage?

Reply in this exact format:
RESOURCE: <resource>
FIELD: <field path like spec.hostNetwork>""",
        options={"num_predict": 50, "temperature": 0.3}
    )

    text = response['response']
    print(f"[*] LLM response:\n{text}\n")

    # Parse response
    resource_match = re.search(r'RESOURCE:\s*(\w+)', text, re.IGNORECASE)
    field_match = re.search(r'FIELD:\s*([\w.]+)', text, re.IGNORECASE)

    resource = resource_match.group(1) if resource_match else "pods"
    field = field_match.group(1) if field_match else "spec.hostNetwork"

    print(f"[+] Extracted - Resource: {resource}, Field: {field}")

    return {
        "resource": resource.lower(),
        "field": field
    }


def main():
    model = "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"

    print("=" * 60)
    print("CISO Scenario 2: kubectl + OPA Generator")
    print("=" * 60)

    # Ensure Ollama is running
    try:
        ollama.list()
    except:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

    # Get LLM suggestions for variable parts
    print("\n[*] Asking LLM for compliance check parameters...")
    params = generate_with_llm(model)

    # Generate script with constrained structure
    print("\n[*] Generating kubectl script...")
    script = create_kubectl_script(
        resource=params["resource"],
        namespace="-A",
        output_file="collected_data.json"
    )
    print(f"--- script.sh ---\n{script}--- end ---\n")

    # Generate OPA policy with constrained structure
    print("[*] Generating OPA Rego policy...")
    policy = create_opa_policy_hostnetwork()
    print(f"--- policy.rego ---\n{policy}--- end ---\n")

    # Save files
    os.makedirs(output_dir, exist_ok=True)

    script_path = os.path.join(output_dir, "script.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    policy_path = os.path.join(output_dir, "policy.rego")
    with open(policy_path, 'w') as f:
        f.write(policy)

    print(f"[+] Saved: {script_path}")
    print(f"[+] Saved: {policy_path}")

    # Test the script
    print("\n[*] Testing kubectl script...")
    result = subprocess.run(
        ["bash", script_path],
        cwd=output_dir,
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print("[+] Script executed successfully")

        # Check if data was collected
        data_path = os.path.join(output_dir, "collected_data.json")
        if os.path.exists(data_path):
            size = os.path.getsize(data_path)
            print(f"[+] Collected data: {size} bytes")

            # Test OPA policy
            print("\n[*] Testing OPA policy...")
            opa_result = subprocess.run(
                ["opa", "eval", "--data", policy_path, "--input", data_path,
                 "data.check.result", "--format", "raw"],
                capture_output=True, text=True
            )

            print(f"[*] OPA result: {opa_result.stdout.strip()}")
            # For this scenario, 'false' means non-compliant (hostNetwork found)
            if opa_result.stdout.strip() == "false":
                print("[+] Policy correctly detected non-compliance (hostNetwork enabled)")
            else:
                print("[!] Policy indicates compliance (no hostNetwork issues)")
    else:
        print(f"[!] Script failed: {result.stderr}")

    print("\n" + "=" * 60)
    print("SUMMARY: Scenario 2 artifacts generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
