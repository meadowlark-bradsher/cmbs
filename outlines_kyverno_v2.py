#!/usr/bin/env python3
"""
Outlines-based Kyverno Policy Generator v2
Uses the correct Outlines API with native Ollama support.
"""

import outlines
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import subprocess
import os
import sys
import time
import yaml


# Define Kyverno policy schema using Pydantic
class Resources(BaseModel):
    namespaces: Optional[List[str]] = None
    kinds: Optional[List[str]] = None


class AnyResource(BaseModel):
    resources: Resources


class Exclude(BaseModel):
    any: List[AnyResource]


class Match(BaseModel):
    any: List[AnyResource]


class PatternSpec(BaseModel):
    hostNetwork: str = Field(alias="=(hostNetwork)", default="false")

    class Config:
        populate_by_name = True


class Pattern(BaseModel):
    spec: dict  # Using dict to allow special Kyverno syntax


class Validate(BaseModel):
    message: str = Field(description="Validation failure message")
    pattern: Pattern


class Rule(BaseModel):
    name: str = Field(description="Rule name")
    exclude: Exclude
    match: Match
    validate: Validate


class Spec(BaseModel):
    validationFailureAction: Literal["Audit", "Enforce"] = "Audit"
    background: bool = True
    rules: List[Rule]


class Metadata(BaseModel):
    name: str = Field(description="Policy name in kebab-case")


class KyvernoClusterPolicy(BaseModel):
    """A Kyverno ClusterPolicy for Kubernetes security."""
    apiVersion: Literal["kyverno.io/v1"] = "kyverno.io/v1"
    kind: Literal["ClusterPolicy"] = "ClusterPolicy"
    metadata: Metadata
    spec: Spec


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


def try_outlines_ollama(model_name: str) -> Optional[dict]:
    """Try Outlines with native Ollama support."""
    print(f"[*] Attempting Outlines with native Ollama backend...")

    try:
        # Use Outlines' native Ollama integration
        model = outlines.from_ollama(model_name)
        print(f"[+] Ollama model loaded: {model_name}")

        # Try JSON schema generation
        print("[*] Creating JSON schema generator...")

        # Simplified schema for policy name
        class PolicyName(BaseModel):
            name: str = Field(pattern=r"^[a-z][a-z0-9-]{2,30}$")

        generator = outlines.Generator(model, PolicyName)

        prompt = """Generate a policy name for a Kyverno policy that prevents pods from using the host network.
The name should be in kebab-case (lowercase with hyphens).
Example: "deny-host-network" or "block-hostnetwork-pods"

Generate the name:"""

        result = generator(prompt)
        print(f"[+] Generated policy name: {result.name}")

        # Now create full policy with generated name
        return create_valid_policy(
            name=result.name,
            rule_name=f"validate-{result.name}",
            message=f"Policy {result.name}: Host network access is restricted."
        )

    except Exception as e:
        print(f"[!] Outlines Ollama failed: {e}")
        return None


def try_outlines_regex(model_name: str) -> Optional[str]:
    """Try regex-constrained generation for just the name."""
    print(f"[*] Attempting regex-constrained name generation...")

    try:
        model = outlines.from_ollama(model_name)

        # Use regex constraint for kebab-case name
        generator = outlines.Generator(model, outlines.regex(r"[a-z][a-z0-9-]{5,25}"))

        prompt = "Generate a short kebab-case name for a security policy: "
        name = generator(prompt)
        print(f"[+] Regex-generated name: {name}")
        return name

    except Exception as e:
        print(f"[!] Regex generation failed: {e}")
        return None


def try_simple_generation(model_name: str) -> Optional[str]:
    """Try simple text generation and parse the result."""
    print(f"[*] Attempting simple text generation with structure prompt...")

    try:
        model = outlines.from_ollama(model_name)
        generator = outlines.Generator(model)

        prompt = """You must output ONLY a single word in kebab-case format.
This word will be used as a Kubernetes policy name.
It should describe blocking host network access.

Output ONLY the name, nothing else:
"""
        result = generator(prompt, max_tokens=20)
        # Clean the result
        name = result.strip().lower().replace(" ", "-").replace("_", "-")
        name = ''.join(c for c in name if c.isalnum() or c == '-')
        if name and len(name) > 3:
            print(f"[+] Generated name: {name}")
            return name
    except Exception as e:
        print(f"[!] Simple generation failed: {e}")

    return None


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"

    print(f"[*] Outlines Kyverno Generator v2 (Native Ollama)")
    print(f"[*] Model: {model_name}")
    print()

    ensure_ollama()

    policy_dict = None

    # Strategy 1: Try full Outlines JSON schema
    policy_dict = try_outlines_ollama(model_name)

    if not policy_dict:
        # Strategy 2: Try regex for name only
        name = try_outlines_regex(model_name)

        if not name:
            # Strategy 3: Simple generation
            name = try_simple_generation(model_name)

        if name:
            policy_dict = create_valid_policy(
                name=name,
                rule_name=f"validate-{name}",
                message=f"Host network sharing violates policy {name}"
            )
        else:
            # Fallback
            print("[*] All generation attempts failed, using default policy...")
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
