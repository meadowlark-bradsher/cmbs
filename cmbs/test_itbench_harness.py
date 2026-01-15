#!/usr/bin/env python3
"""
Test script for ITBench harness adapter.

This script simulates the ITBench harness by:
1. Creating scenario_data.json in ITBench format
2. Running the CMBS ITBench adapter
3. Checking the output

Usage:
    python -m cmbs.test_itbench_harness

Prerequisites:
    - Kind cluster running with Kyverno installed
    - Fault injected (pod with hostNetwork=true)
    - Ollama running with qwen2.5:7b model
"""

import json
import os
import subprocess
import sys
import tempfile


def get_kubeconfig() -> str:
    """Get current kubeconfig content."""
    kubeconfig_path = os.environ.get('KUBECONFIG', os.path.expanduser('~/.kube/config'))
    with open(kubeconfig_path, 'r') as f:
        return f.read()


def create_scenario_data(output_path: str) -> None:
    """Create ITBench-format scenario_data.json."""
    scenario_data = {
        "metadata": {
            "name": "cis-b.5.2.5-gen",
            "description": "CIS Benchmark for K8S 5.2.5 (Kyverno)",
            "scenario_type": "Gen-CIS-b-K8s-Kyverno"
        },
        "goal_template": """I would like to check if the following condition is satisfied, given a Kubernetes cluster with `kubeconfig.yaml`
    Minimize the admission of containers wishing to share the host network namespace.

To check the condition, do the following steps.
- deploy a Kyverno policy to the cluster
- check if the policy is correctly deployed.

If deploying the policy failed and if you can fix the issue, you will do it and try deploying again.
Once you get a final answer, you can quit the work.

The cluster's kubeconfig is at `{{ kubeconfig }}`.
""",
        "vars": {
            "kubeconfig": get_kubeconfig()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(scenario_data, f, indent=2)

    print(f"[+] Created scenario_data.json at: {output_path}")


def check_prerequisites() -> bool:
    """Check that prerequisites are met."""
    print("[*] Checking prerequisites...")

    # Check kubectl
    result = subprocess.run(
        ["kubectl", "cluster-info"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[!] kubectl not connected to cluster")
        return False
    print("[+] kubectl connected")

    # Check Kyverno
    result = subprocess.run(
        ["kubectl", "get", "deployment", "-n", "kyverno", "kyverno-admission-controller"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[!] Kyverno not installed")
        return False
    print("[+] Kyverno installed")

    # Check fault injection
    result = subprocess.run(
        ["kubectl", "get", "pod", "-n", "paa", "-o", "jsonpath={.items[*].spec.hostNetwork}"],
        capture_output=True, text=True
    )
    if "true" not in result.stdout:
        print("[!] No fault injected (no pod with hostNetwork=true)")
        print("    Setting up fault...")
        setup_fault()
    print("[+] Fault injected (hostNetwork=true)")

    # Check Ollama
    try:
        import ollama
        ollama.list()
        print("[+] Ollama running")
    except Exception as e:
        print(f"[!] Ollama not available: {e}")
        return False

    return True


def setup_fault():
    """Set up the fault if not present."""
    subprocess.run(["kubectl", "create", "namespace", "paa"], capture_output=True)
    subprocess.run([
        "kubectl", "run", "nginx-violation",
        "-n", "paa",
        "--image=nginx:latest",
        "--overrides", '{"spec":{"hostNetwork":true}}'
    ], capture_output=True)


def cleanup():
    """Clean up test artifacts."""
    subprocess.run(["kubectl", "delete", "clusterpolicy", "--all"], capture_output=True)


def main():
    print("=" * 60)
    print("CMBS ITBench Harness Adapter Test")
    print("=" * 60)

    if not check_prerequisites():
        print("[!] Prerequisites not met")
        return 1

    # Create temp directories
    test_dir = tempfile.mkdtemp(prefix="cmbs-itbench-test-")
    scenario_data_path = os.path.join(test_dir, "scenario_data.json")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Test directory: {test_dir}")

    # Create scenario data
    create_scenario_data(scenario_data_path)

    # Clean up any existing policies
    print("[*] Cleaning up existing policies...")
    cleanup()

    # Run the adapter
    print("\n[*] Running CMBS ITBench adapter...")
    print("=" * 60)

    from .itbench_harness import main as harness_main
    sys.argv = [
        "itbench_harness",
        "--scenario-data", scenario_data_path,
        "--output-dir", output_dir,
        "--model", "qwen2.5:7b",
        "--max-steps", "30",
        "--timeout", "180",
        "-v",
    ]

    try:
        result_code = harness_main()
    except SystemExit as e:
        result_code = e.code if isinstance(e.code, int) else 1

    print("=" * 60)

    # Check outputs
    print("\n[*] Checking outputs...")
    output_files = os.listdir(output_dir)
    print(f"[*] Output files: {output_files}")

    # Check for policy file
    policy_found = any(f.endswith('.yaml') or f.endswith('.yml') for f in output_files)
    print(f"[*] Policy file found: {policy_found}")

    # Check result file
    result_file = os.path.join(output_dir, "agent-result.json")
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)
        print(f"[*] Agent result: {json.dumps(result, indent=2)}")

    # Check cluster for policy
    print("\n[*] Checking cluster state...")
    subprocess.run(["kubectl", "get", "clusterpolicy"], capture_output=False)

    # Check policy reports
    print("\n[*] Checking policy reports...")
    subprocess.run(["kubectl", "get", "policyreport", "-n", "paa"], capture_output=False)

    print("\n" + "=" * 60)
    print(f"Test directory preserved at: {test_dir}")
    print("=" * 60)

    return result_code


if __name__ == "__main__":
    sys.exit(main())
