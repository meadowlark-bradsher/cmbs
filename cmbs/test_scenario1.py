#!/usr/bin/env python3
"""
CMBS Test: Scenario 1 (Kyverno)

This script demonstrates the CMBS supervisor running on IT-Bench CISO Scenario 1.
It shows how the supervisor:
1. Allows artifact generation
2. Allows execution
3. Blocks premature posture declarations
4. Allows termination only when epistemic requirements are met

Usage:
    python -m cmbs.test_scenario1

Prerequisites:
    - Kind cluster running with Kyverno installed
    - Fault injected (pod with hostNetwork=true in namespace paa)
    - Ollama running with qwen2.5:7b model
"""

import sys
import os
import subprocess
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmbs.runner import CMBSRunner
from cmbs.agents.ollama_agent import OllamaAgent
from cmbs.document_oracle import load_document


# Goal from IT-Bench CISO Scenario 1
SCENARIO_1_GOAL = """
I would like to check if the following condition is satisfied, given a Kubernetes cluster with `kubeconfig.yaml`
    Minimize the admission of containers wishing to share the host network namespace.

To check the condition, do the following steps.
- deploy a Kyverno policy to the cluster
- check if the policy is correctly deployed.

If deploying the policy failed and if you can fix the issue, you will do it and try deploying again.
Once you get a final answer, you can quit the work.
"""


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
        print("    Run: kubectl create ns paa && kubectl run nginx --image=nginx -n paa --overrides='{\"spec\":{\"hostNetwork\":true}}'")
        return False
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
    print("[*] Setting up fault injection...")

    # Create namespace
    subprocess.run(
        ["kubectl", "create", "namespace", "paa"],
        capture_output=True
    )

    # Create pod with hostNetwork=true
    pod_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: nginx-violation
  namespace: paa
spec:
  hostNetwork: true
  containers:
  - name: nginx
    image: nginx:latest
"""
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=pod_yaml,
        capture_output=True, text=True
    )
    print("[+] Fault injected")


def cleanup():
    """Clean up test artifacts."""
    print("[*] Cleaning up...")
    subprocess.run(
        ["kubectl", "delete", "clusterpolicy", "--all"],
        capture_output=True
    )
    subprocess.run(
        ["kubectl", "delete", "ns", "paa"],
        capture_output=True
    )


def main():
    print("=" * 60)
    print("CMBS Test: Scenario 1 (Kyverno)")
    print("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        print("\n[!] Prerequisites not met. Setting up...")
        setup_fault()
        time.sleep(3)
        if not check_prerequisites():
            print("[!] Setup failed. Please check your environment.")
            return 1

    # Load Kyverno documentation for DSRO
    doc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent-docs", "kyverno_cheat_sheet.txt"
    )
    if os.path.exists(doc_path):
        print(f"\n[*] Loading Kyverno documentation from: {doc_path}")
        document_oracle = load_document(doc_path)
        print(f"[+] Loaded {len(document_oracle.get_available_sections())} sections, "
              f"{len(document_oracle.get_available_keywords())} keywords")
    else:
        print(f"\n[!] Warning: Kyverno documentation not found at {doc_path}")
        document_oracle = None

    # Create agent and runner
    print("\n[*] Creating CMBS runner with Ollama agent...")
    agent = OllamaAgent(model="qwen2.5:7b", temperature=0.3)
    runner = CMBSRunner(
        agent=agent,
        work_dir="/tmp/cmbs-scenario1",
        max_steps=40,
        timeout_seconds=300.0,  # 5 minutes for DSRO exploration
        verbose=True,
        document_oracle=document_oracle,
    )

    # Run the scenario
    print("\n[*] Starting CMBS-supervised run...")
    result = runner.run(SCENARIO_1_GOAL)

    # Save trace
    trace_path = "/tmp/cmbs-scenario1/trace.json"
    runner.save_trace(trace_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Final Posture: {result.final_posture}")
    print(f"Steps Taken: {result.steps_taken}")
    print(f"Elapsed Time: {result.elapsed_time:.1f}s")
    print(f"Termination: {result.termination_reason}")
    print(f"Trace: {trace_path}")

    # Run official evaluation
    print("\n[*] Running official evaluation...")
    work_dir = "/tmp/cmbs-scenario1"

    # Collect cluster state
    subprocess.run(
        ["kubectl", "get", "clusterpolicy", "-o", "yaml"],
        stdout=open(f"{work_dir}/clusterpolicies.yaml", "w"),
        stderr=subprocess.DEVNULL
    )
    subprocess.run(
        ["kubectl", "get", "policyreport", "-A", "-o", "yaml"],
        stdout=open(f"{work_dir}/policyreports.yaml", "w"),
        stderr=subprocess.DEVNULL
    )

    print("\n" + "=" * 60)
    if result.final_posture == "non_compliant":
        print("SCENARIO 1: EXPECTED OUTCOME (non_compliant)")
        print("The supervisor allowed the agent to:")
        print("  1. Generate a policy")
        print("  2. Deploy the policy")
        print("  3. Check for violations")
        print("  4. Declare non-compliance")
        print("  5. Terminate")
    else:
        print(f"SCENARIO 1: UNEXPECTED OUTCOME ({result.final_posture})")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
