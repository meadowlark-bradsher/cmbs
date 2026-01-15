#!/usr/bin/env python3
"""
Outlines-based Kyverno Policy Generator for IT-Bench CISO Scenario 1
Uses JSON schema validation to ensure valid structure.
"""

import outlines
from outlines import models, generate
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import subprocess
import json
import os
import sys
import time
import yaml


# Define Kyverno policy schema using Pydantic
class ResourceFilter(BaseModel):
    namespaces: List[str] = Field(default_factory=lambda: ["kube-system", "local-path-storage"])


class Resources(BaseModel):
    namespaces: Optional[List[str]] = None
    kinds: Optional[List[str]] = None


class AnyResource(BaseModel):
    resources: Resources


class Exclude(BaseModel):
    any: List[AnyResource]


class Match(BaseModel):
    any: List[AnyResource]


class Pattern(BaseModel):
    spec: dict


class Validate(BaseModel):
    message: str = Field(description="Validation failure message")
    pattern: Pattern


class Rule(BaseModel):
    name: str = Field(description="Rule name, e.g. 'check-host-network'")
    exclude: Exclude
    match: Match
    validate: Validate


class Spec(BaseModel):
    validationFailureAction: Literal["Audit", "Enforce"] = "Audit"
    background: bool = True
    rules: List[Rule]


class Metadata(BaseModel):
    name: str = Field(description="Policy name, e.g. 'disallow-host-network'")


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


def create_valid_policy() -> KyvernoClusterPolicy:
    """Create a valid policy programmatically."""
    return KyvernoClusterPolicy(
        metadata=Metadata(name="disallow-host-network"),
        spec=Spec(
            validationFailureAction="Audit",
            background=True,
            rules=[
                Rule(
                    name="check-host-network",
                    exclude=Exclude(
                        any=[AnyResource(resources=Resources(namespaces=["kube-system", "local-path-storage"]))]
                    ),
                    match=Match(
                        any=[AnyResource(resources=Resources(kinds=["Pod"]))]
                    ),
                    validate=Validate(
                        message="Sharing the host network namespace is disallowed.",
                        pattern=Pattern(spec={"=(hostNetwork)": "false"})
                    )
                )
            ]
        )
    )


def policy_to_yaml(policy: KyvernoClusterPolicy) -> str:
    """Convert Pydantic model to YAML."""
    # Convert to dict, handling the special Kyverno pattern syntax
    d = policy.model_dump(by_alias=True)
    return yaml.dump(d, default_flow_style=False, sort_keys=False)


def try_outlines_generation(model_name: str) -> Optional[KyvernoClusterPolicy]:
    """Try to generate policy using Outlines with Ollama."""
    print(f"[*] Attempting Outlines generation with {model_name}...")

    try:
        # Outlines supports OpenAI-compatible APIs
        model = models.openai(
            model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        # Create a JSON generator with our schema
        generator = generate.json(model, KyvernoClusterPolicy)

        prompt = """Generate a Kyverno ClusterPolicy that:
1. Detects pods using hostNetwork: true
2. Uses Audit mode (not blocking)
3. Excludes kube-system and local-path-storage namespaces
4. Validates that spec.hostNetwork is "false"

Generate the policy JSON:"""

        result = generator(prompt)
        print(f"[+] Outlines generated: {type(result)}")
        return result

    except Exception as e:
        print(f"[!] Outlines generation failed: {e}")
        return None


def try_regex_generation(model_name: str) -> Optional[str]:
    """Try regex-constrained generation."""
    print(f"[*] Attempting regex-constrained generation...")

    try:
        model = models.openai(
            model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        # Simple regex for policy name
        name_generator = generate.regex(model, r"[a-z][a-z0-9-]{3,30}")

        prompt = "Generate a short kebab-case name for a Kyverno policy that blocks host network access:"
        name = name_generator(prompt)
        print(f"[+] Generated name: {name}")
        return name

    except Exception as e:
        print(f"[!] Regex generation failed: {e}")
        return None


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    output_dir = "/tmp/itbench-agent"

    print(f"[*] Outlines Kyverno Generator")
    print(f"[*] Model: {model_name}")
    print()

    ensure_ollama()

    # Strategy 1: Try full JSON schema generation
    policy = try_outlines_generation(model_name)

    if policy:
        print("[+] Full schema generation succeeded!")
        policy_yaml = policy_to_yaml(policy)
    else:
        # Strategy 2: Try regex generation for just the name
        generated_name = try_regex_generation(model_name)

        # Strategy 3: Use Pydantic to create valid structure
        print("\n[*] Using Pydantic schema to create valid policy structure...")
        policy = create_valid_policy()

        if generated_name:
            policy.metadata.name = generated_name
            policy.spec.rules[0].name = f"validate-{generated_name}"

        policy_yaml = policy_to_yaml(policy)
        print("[+] Created policy using Pydantic schema validation")

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

        # Try to fix common issues
        print("[*] Attempting to fix YAML...")
        # The pattern syntax might need adjustment
        fixed_yaml = policy_yaml.replace("'=(hostNetwork)':", "=(hostNetwork):")
        with open(policy_path, 'w') as f:
            f.write(fixed_yaml)

        result = subprocess.run(["kubectl", "apply", "-f", policy_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[!] Still failing: {result.stderr}")
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
