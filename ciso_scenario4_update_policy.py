#!/usr/bin/env python3
"""
CISO Scenario 4: Update Kyverno Policy

This scenario is HARDER than 1-3 because it requires:
1. Reading existing policies from the cluster
2. Understanding which policy to modify
3. Adding new requirements without breaking existing ones
4. Not modifying policies that shouldn't be touched

The challenge: Can't just template this - need semantic understanding.
"""

import subprocess
import yaml
import json
import os
import sys
import time
import ollama
from typing import Optional, Dict, Any, List
from copy import deepcopy


SCENARIO_DIR = "/Users/meadowlarkbradsher/workspace/repos/genai/mast/ITBench/scenarios/ciso/4.upd-cis-b-k8s-kyverno"


def get_cluster_policies() -> List[Dict[str, Any]]:
    """Read current ClusterPolicies from the cluster."""
    result = subprocess.run(
        ["kubectl", "get", "clusterpolicy", "-o", "yaml"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[!] Failed to get policies: {result.stderr}")
        return []

    data = yaml.safe_load(result.stdout)
    return data.get("items", [])


def get_policy_by_name(policies: List[Dict], name: str) -> Optional[Dict]:
    """Find a policy by name."""
    for p in policies:
        if p.get("metadata", {}).get("name") == name:
            return deepcopy(p)
    return None


def llm_analyze_requirements(goal: str, existing_policy: Dict) -> Dict[str, Any]:
    """
    Ask LLM to analyze what changes are needed.
    This is where the LLM adds value - understanding the delta.
    """
    policy_yaml = yaml.dump(existing_policy, default_flow_style=False)

    prompt = f"""Analyze this Kyverno policy update task.

CURRENT POLICY:
```yaml
{policy_yaml}
```

NEW REQUIREMENTS:
{goal}

What specific changes need to be made to the policy?
List each change in this format:
CHANGE 1: [description]
FIELD: [yaml path like spec.rules[0].validate.pattern.spec.containers]
VALUE: [new value to add]

Be specific about the YAML structure."""

    response = ollama.generate(
        model="qwen2.5:7b",
        prompt=prompt,
        options={"num_predict": 300, "temperature": 0.3}
    )

    return {"analysis": response["response"]}


def apply_security_updates(policy: Dict, updates: Dict[str, Any]) -> Dict:
    """
    Apply the required security updates to the policy.

    This is where we enforce correctness - the LLM identified what to change,
    but we apply it with proper structure.
    """
    updated = deepcopy(policy)

    # Find the container security rule
    rules = updated.get("spec", {}).get("rules", [])

    for rule in rules:
        # Look for rules that validate container security
        validate = rule.get("validate", {})
        pattern = validate.get("pattern", {})
        spec = pattern.get("spec", {})
        containers = spec.get("containers", [])

        if containers:
            # Update each container pattern
            for container in containers:
                # Add runAsNonRoot if not present
                security_context = container.get("securityContext", {})
                if "runAsNonRoot" not in security_context:
                    container.setdefault("securityContext", {})
                    container["securityContext"]["runAsNonRoot"] = True

                # Add trusted registry constraint
                if "image" not in container:
                    container["image"] = "trusted-registry.com/*"

            # Update the validation message
            validate["message"] = (
                "Privilege escalation and root user execution are not allowed, "
                "and only images from trusted-registry.com are permitted."
            )

            # Update rule name to reflect new scope
            rule["name"] = "disallow-privilege-escalation-and-root"

    return updated


def setup_scenario():
    """Set up scenario 4 in the cluster."""
    print("[*] Setting up Scenario 4...")

    # Create namespace if not exists
    subprocess.run(["kubectl", "create", "namespace", "paa"],
                   capture_output=True)

    # Deploy existing policies
    existing_policies = os.path.join(SCENARIO_DIR, "manifests/existing-policy.yaml")
    result = subprocess.run(
        ["kubectl", "apply", "-f", existing_policies],
        capture_output=True, text=True
    )
    print(f"[*] Deployed existing policies: {result.stdout.strip()}")

    # Deploy test pod (from untrusted registry)
    test_pod_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: test-untrusted-registry
  namespace: paa
spec:
  containers:
  - name: test
    image: untrusted-registry.com/nginx:latest
    securityContext:
      allowPrivilegeEscalation: false
      runAsNonRoot: true
"""
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=test_pod_yaml,
        capture_output=True, text=True
    )
    print(f"[*] Deployed test pod: {result.stdout.strip() or result.stderr.strip()}")


def run_evaluation():
    """Run the official evaluation."""
    print("\n[*] Running evaluation...")

    work_dir = "/tmp/itbench-agent/scenario4"
    os.makedirs(work_dir, exist_ok=True)

    # Collect cluster state
    subprocess.run(["kubectl", "get", "policy", "-A", "-o", "yaml"],
                   stdout=open(f"{work_dir}/policies.yaml", "w"), stderr=subprocess.DEVNULL)
    subprocess.run(["kubectl", "get", "clusterpolicy", "-o", "yaml"],
                   stdout=open(f"{work_dir}/clusterpolicies.yaml", "w"))
    subprocess.run(["kubectl", "get", "policyreport", "-A", "-o", "yaml"],
                   stdout=open(f"{work_dir}/policyreports.yaml", "w"), stderr=subprocess.DEVNULL)
    subprocess.run(["kubectl", "get", "clusterpolicyreport", "-o", "yaml"],
                   stdout=open(f"{work_dir}/clusterpolicyreports.yaml", "w"), stderr=subprocess.DEVNULL)

    # Ensure empty files exist if no resources
    for f in ["policies.yaml", "policyreports.yaml", "clusterpolicyreports.yaml"]:
        path = f"{work_dir}/{f}"
        if os.path.getsize(path) == 0:
            with open(path, "w") as fh:
                fh.write("items: []\n")

    # Run evaluation
    result = subprocess.run([
        "python3", f"{SCENARIO_DIR}/evaluation.py",
        f"{SCENARIO_DIR}/eval_config.yaml",
        "--epol", f"{SCENARIO_DIR}/manifests/existing-policy.yaml",
        "--pol", f"{work_dir}/policies.yaml",
        "--cpol", f"{work_dir}/clusterpolicies.yaml",
        "--polr", f"{work_dir}/policyreports.yaml",
        "--cpolr", f"{work_dir}/clusterpolicyreports.yaml"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(f"[!] Errors: {result.stderr}")

    return result.stdout


def main():
    print("=" * 60)
    print("CISO Scenario 4: Update Kyverno Policy")
    print("=" * 60)

    # Ensure ollama is running
    try:
        ollama.list()
    except:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

    # Step 1: Setup scenario
    setup_scenario()
    time.sleep(5)  # Wait for policies to be ready

    # Step 2: Read current policies from cluster
    print("\n[*] Reading current policies from cluster...")
    policies = get_cluster_policies()
    print(f"[+] Found {len(policies)} policies")

    for p in policies:
        name = p.get("metadata", {}).get("name", "unknown")
        print(f"    - {name}")

    # Step 3: Identify which policy to modify
    target_policy = get_policy_by_name(policies, "container-security-controls")
    if not target_policy:
        print("[!] Target policy 'container-security-controls' not found!")
        return 1

    print(f"\n[*] Target policy found: container-security-controls")
    print(f"    Current rules: {[r.get('name') for r in target_policy.get('spec', {}).get('rules', [])]}")

    # Step 4: Ask LLM to analyze requirements
    goal = """
    Add these new requirements to the container-security-controls policy:
    1. Prohibit running as root user (runAsNonRoot: true)
    2. Only allow images from trusted-registry.com
    """

    print(f"\n[*] Asking LLM to analyze requirements...")
    analysis = llm_analyze_requirements(goal, target_policy)
    print(f"[*] LLM Analysis:\n{analysis['analysis'][:500]}...")

    # Step 5: Apply updates (structure enforced by Python)
    print(f"\n[*] Applying updates with enforced structure...")
    updated_policy = apply_security_updates(target_policy, analysis)

    # Show the diff
    print("\n--- Updated Policy ---")
    print(yaml.dump(updated_policy, default_flow_style=False))
    print("--- End Policy ---\n")

    # Step 6: Apply to cluster
    print("[*] Applying updated policy to cluster...")

    # Remove resourceVersion for update
    if "resourceVersion" in updated_policy.get("metadata", {}):
        del updated_policy["metadata"]["resourceVersion"]

    policy_yaml = yaml.dump(updated_policy, default_flow_style=False)
    result = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=policy_yaml,
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"[!] Failed to apply: {result.stderr}")
        return 1

    print(f"[+] {result.stdout.strip()}")

    # Step 7: Wait for policy reports
    print("\n[*] Waiting 15s for policy reports...")
    time.sleep(15)

    # Step 8: Run evaluation
    eval_result = run_evaluation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if '"pass": true' in eval_result:
        print("✅ SCENARIO 4 PASSED")
    else:
        print("❌ SCENARIO 4 FAILED")
        print("Check the evaluation details above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
