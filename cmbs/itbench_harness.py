#!/usr/bin/env python3
"""
ITBench Harness Adapter for CMBS

This module bridges the CMBS runner to the ITBench harness interface.

ITBench expects:
- Input: scenario_data.json with goal_template and vars (kubeconfig, etc.)
- Output: Agent artifacts in a specified workdir
- Artifacts are tarred and uploaded by the harness

This adapter:
1. Reads scenario_data.json from ITBench-provided path
2. Extracts goal and writes kubeconfig to temp location
3. Runs the CMBS supervised agent loop
4. Copies generated artifacts to the ITBench output directory
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from datetime import datetime

from .runner import CMBSRunner
from .agents.ollama_agent import OllamaAgent


def load_scenario_data(scenario_path: str) -> dict:
    """Load scenario data from ITBench-provided JSON file."""
    with open(scenario_path, 'r') as f:
        return json.load(f)


def setup_kubeconfig(kubeconfig_content: str, workdir: str) -> str:
    """Write kubeconfig content to a file and return the path."""
    kubeconfig_path = os.path.join(workdir, "kubeconfig.yaml")
    with open(kubeconfig_path, 'w') as f:
        f.write(kubeconfig_content)
    os.environ['KUBECONFIG'] = kubeconfig_path
    return kubeconfig_path


def substitute_goal_template(goal_template: str, vars: dict, workdir: str) -> str:
    """
    Substitute template variables in the goal.

    ITBench uses {{ kubeconfig }} and {{ path_to_inventory }} placeholders.
    """
    goal = goal_template

    # Handle kubeconfig substitution
    if 'kubeconfig' in vars and '{{ kubeconfig }}' in goal:
        kubeconfig_path = os.path.join(workdir, "kubeconfig.yaml")
        goal = goal.replace('{{ kubeconfig }}', kubeconfig_path)

    # Handle ansible inventory substitution
    if 'ansible_ini' in vars and '{{ path_to_inventory }}' in goal:
        inventory_path = os.path.join(workdir, "inventory.ansible.ini")
        with open(inventory_path, 'w') as f:
            f.write(vars['ansible_ini'])
        goal = goal.replace('{{ path_to_inventory }}', inventory_path)

    return goal


def copy_artifacts(src_dir: str, dst_dir: str) -> list:
    """
    Copy generated artifacts from CMBS workdir to ITBench output dir.

    Returns list of copied files.
    """
    copied = []
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    # Ensure destination exists
    dst_path.mkdir(parents=True, exist_ok=True)

    # Copy all files (not directories)
    for item in src_path.iterdir():
        if item.is_file():
            dst_file = dst_path / item.name
            shutil.copy2(item, dst_file)
            copied.append(str(dst_file))

    return copied


def run_cmbs_agent(
    goal: str,
    workdir: str,
    output_dir: str,
    model: str = "qwen2.5:7b",
    max_steps: int = 30,
    timeout_seconds: float = 180.0,
    verbose: bool = True,
) -> dict:
    """
    Run the CMBS-supervised agent and return results.

    Args:
        goal: The scenario goal (with substituted variables)
        workdir: Working directory for agent artifacts
        output_dir: ITBench output directory for final artifacts
        model: Ollama model to use
        max_steps: Maximum steps before forced termination
        timeout_seconds: Timeout before forced termination
        verbose: Print progress to stdout

    Returns:
        Dictionary with run results
    """
    # Create agent and runner
    agent = OllamaAgent(model=model, temperature=0.3)
    runner = CMBSRunner(
        agent=agent,
        work_dir=workdir,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        verbose=verbose,
    )

    # Run the scenario
    result = runner.run(goal)

    # Copy artifacts to ITBench output directory
    copied_files = copy_artifacts(workdir, output_dir)

    # Build result summary
    return {
        "success": result.success,
        "final_posture": result.final_posture,
        "steps_taken": result.steps_taken,
        "elapsed_time": result.elapsed_time,
        "termination_reason": result.termination_reason,
        "run_id": result.run_id,
        "artifacts": copied_files,
    }


def main():
    """
    Main entry point for ITBench harness integration.

    Usage:
        python -m cmbs.itbench_harness \\
            --scenario-data /tmp/agent/scenario_data.json \\
            --output-dir /tmp/agent \\
            --model qwen2.5:7b
    """
    parser = argparse.ArgumentParser(
        description="CMBS ITBench Harness Adapter"
    )
    parser.add_argument(
        "--scenario-data",
        type=str,
        required=True,
        help="Path to scenario_data.json from ITBench",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write agent output artifacts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model to use (default: qwen2.5:7b)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum steps before termination (default: 30)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        default=None,
        help="Path to write result JSON (default: <output-dir>/agent-result.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CMBS ITBench Harness Adapter")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load scenario data
    print(f"\n[*] Loading scenario data from: {args.scenario_data}")
    try:
        scenario_data = load_scenario_data(args.scenario_data)
    except Exception as e:
        print(f"[!] Failed to load scenario data: {e}")
        sys.exit(1)

    # Extract metadata
    metadata = scenario_data.get("metadata", {})
    print(f"[*] Scenario: {metadata.get('name', 'unknown')}")
    print(f"[*] Type: {metadata.get('scenario_type', 'unknown')}")
    print(f"[*] Description: {metadata.get('description', 'unknown')}")

    # Setup working directories
    workdir = tempfile.mkdtemp(prefix="cmbs-itbench-")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"[*] Working directory: {workdir}")
    print(f"[*] Output directory: {output_dir}")

    # Setup kubeconfig if present
    vars_data = scenario_data.get("vars", {})
    if "kubeconfig" in vars_data:
        kubeconfig_path = setup_kubeconfig(vars_data["kubeconfig"], workdir)
        print(f"[*] Kubeconfig written to: {kubeconfig_path}")

    # Substitute goal template
    goal_template = scenario_data.get("goal_template", "")
    goal = substitute_goal_template(goal_template, vars_data, workdir)

    print(f"\n[*] Goal (first 200 chars):")
    print(f"    {goal[:200]}...")

    # Run the CMBS agent
    print(f"\n[*] Starting CMBS agent with model: {args.model}")
    print(f"[*] Max steps: {args.max_steps}, Timeout: {args.timeout}s")
    print()

    result = run_cmbs_agent(
        goal=goal,
        workdir=workdir,
        output_dir=output_dir,
        model=args.model,
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
        verbose=args.verbose,
    )

    # Write result file
    result_file = args.result_file or os.path.join(output_dir, "agent-result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("CMBS Run Complete")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Final Posture: {result['final_posture']}")
    print(f"Steps Taken: {result['steps_taken']}")
    print(f"Elapsed Time: {result['elapsed_time']:.1f}s")
    print(f"Termination: {result['termination_reason']}")
    print(f"Artifacts copied: {len(result['artifacts'])}")
    for artifact in result['artifacts']:
        print(f"  - {artifact}")
    print(f"Result file: {result_file}")

    # Cleanup temp workdir
    shutil.rmtree(workdir, ignore_errors=True)

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())
