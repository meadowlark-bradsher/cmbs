#!/usr/bin/env python3
"""
CMBS Test: Scenario 2 with DSRO (OPA/Rego)

This script tests CMBS with Document Search as Repair Obligation (DSRO).
When the agent gets stuck generating valid Rego, it can consult the
Rego cheat sheet documentation.

Usage:
    python -m cmbs.test_scenario2_dsro

Prerequisites:
    - Kind cluster running
    - OPA installed (brew install opa)
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
from cmbs.ccil import CCILConfig
from cmbs.document_oracle import load_document


# Goal for Scenario 2: OPA/Rego based compliance check
SCENARIO_2_GOAL = """
I would like to check if the following condition is satisfied, given a Kubernetes cluster:
    Minimize the admission of containers wishing to share the host network namespace.

To check the condition, do the following steps:
1. Use kubectl to collect pod data from all namespaces as JSON
2. Write an OPA Rego policy that checks if any pod has hostNetwork=true
3. Run opa eval to check the collected data against the policy
4. Report whether the cluster is compliant or non-compliant

If your Rego policy has syntax errors, fix them and try again.
If you get stuck, you may search the Rego documentation for help.

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

    # Check OPA
    result = subprocess.run(
        ["opa", "version"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[!] OPA not installed")
        print("    Run: brew install opa")
        return False
    print("[+] OPA installed")

    # Check fault injection
    result = subprocess.run(
        ["kubectl", "get", "pod", "-n", "paa", "-o", "jsonpath={.items[*].spec.hostNetwork}"],
        capture_output=True, text=True
    )
    if "true" not in result.stdout:
        print("[!] No fault injected (no pod with hostNetwork=true)")
        print("    Setting up fault...")
        setup_fault()
        time.sleep(3)
    print("[+] Fault injected (hostNetwork=true)")

    # Check Ollama
    try:
        import ollama
        ollama.list()
        print("[+] Ollama running")
    except Exception as e:
        print(f"[!] Ollama not available: {e}")
        return False

    # Check Rego cheat sheet exists
    doc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent-docs", "rego_cheat_sheet.txt"
    )
    if not os.path.exists(doc_path):
        print(f"[!] Rego cheat sheet not found: {doc_path}")
        return False
    print(f"[+] Rego documentation available")

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


def main():
    print("=" * 60)
    print("CMBS Test: Scenario 2 with DSRO (OPA/Rego)")
    print("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        print("\n[!] Prerequisites not met.")
        return 1

    # Load document oracle for DSRO
    doc_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agent-docs", "rego_cheat_sheet.txt"
    )
    print(f"\n[*] Loading Rego documentation from: {doc_path}")
    document_oracle = load_document(doc_path)
    print(f"[+] Loaded {len(document_oracle.get_available_sections())} sections, "
          f"{len(document_oracle.get_available_keywords())} keywords")

    # Create agent and runner with DSRO enabled
    print("\n[*] Creating CMBS runner with DSRO enabled...")
    agent = OllamaAgent(model="qwen2.5:7b", temperature=0.3)

    ccil_config = CCILConfig(
        enabled=True,
        log_level="full",  # Full logging for debugging
    )

    runner = CMBSRunner(
        agent=agent,
        work_dir="/tmp/cmbs-scenario2",
        max_steps=40,
        timeout_seconds=300.0,
        verbose=True,
        ccil_enabled=True,
        ccil_config=ccil_config,
        document_oracle=document_oracle,
    )

    # Run the scenario
    print("\n[*] Starting CMBS-supervised run with DSRO...")
    print("[*] Agent can use DOCUMENT_SEARCH when stuck in repair loop")
    result = runner.run(SCENARIO_2_GOAL)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Final Posture: {result.final_posture}")
    print(f"Steps Taken: {result.steps_taken}")
    print(f"Elapsed Time: {result.elapsed_time:.1f}s")
    print(f"Termination: {result.termination_reason}")
    print(f"Log Dir: {result.log_dir}")

    # Check if DSO was used
    dso_mask = runner.supervisor.masks.dso
    if dso_mask.probe_history:
        print(f"\n[*] DSRO Activity:")
        print(f"    Probes made: {len(dso_mask.probe_history)}")
        for kind, target in dso_mask.probe_history:
            print(f"    - {kind}: {target}")
        print(f"    Exit reason: {dso_mask.exit_reason}")
    else:
        print("\n[*] DSRO was not used (agent didn't enter document search)")

    # Check work directory for artifacts
    work_dir = "/tmp/cmbs-scenario2"
    print(f"\n[*] Artifacts in {work_dir}:")
    for f in os.listdir(work_dir):
        fpath = os.path.join(work_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    {f}: {size} bytes")

    print("\n" + "=" * 60)
    if result.final_posture == "non_compliant":
        print("SCENARIO 2: EXPECTED OUTCOME (non_compliant)")
        print("The agent successfully:")
        print("  1. Collected pod data with kubectl")
        print("  2. Generated a valid Rego policy")
        print("  3. Ran OPA eval")
        print("  4. Detected the hostNetwork violation")
    else:
        print(f"SCENARIO 2: OUTCOME ({result.final_posture})")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
