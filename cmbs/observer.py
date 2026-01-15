"""
CMBS Observer

Observes execution outcomes and updates masks accordingly.
This is the ONLY place where masks are modified.

The observer:
- Watches command execution results
- Checks for artifact creation
- Detects violations in output
- Updates masks based on OBSERVED REALITY only

The observer does NOT:
- Parse Kyverno/Rego/Ansible schemas
- Evaluate policy correctness
- Infer CIS semantics
"""

import subprocess
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .masks import Masks, AffordanceState, EvidenceState


@dataclass
class Observation:
    """
    The result of observing an execution.

    This is what the supervisor uses to update masks.
    """
    command: Optional[str] = None
    success: bool = False
    return_code: int = -1
    stdout: str = ""
    stderr: str = ""
    artifact_path: Optional[str] = None
    artifact_exists: bool = False

    def has_violation_signal(self) -> bool:
        """
        Detect if output indicates policy violations.

        This is intentionally simple pattern matching.
        We look for common signals without understanding schema.
        """
        combined = (self.stdout + self.stderr).lower()

        # Kyverno policy report signals
        if "fail" in combined and ("policyreport" in combined or "policy" in combined):
            return True

        # OPA result signals
        if '"result": false' in combined or "result: false" in combined:
            return True
        if re.search(r'\bresult\b.*\bfalse\b', combined):
            return True

        return False

    def has_clean_signal(self) -> bool:
        """
        Detect if output indicates no violations.

        Again, simple pattern matching without schema knowledge.
        """
        combined = (self.stdout + self.stderr).lower()

        # Kyverno - no fails in policy report
        if "policyreport" in combined and "fail" not in combined and "pass" in combined:
            return True

        # OPA - result true
        if '"result": true' in combined or "result: true" in combined:
            return True

        return False


class Observer:
    """
    Observes execution and updates masks.

    This is a passive observer - it only watches and records.
    It never executes commands itself (that's the runner's job).
    """

    def __init__(self, work_dir: str = "/tmp/cmbs-agent"):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.last_observation: Optional[Observation] = None

    def observe_command(
        self,
        command: str,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> Observation:
        """
        Create an observation from a command execution.

        This is called AFTER a command has been run.
        """
        obs = Observation(
            command=command,
            success=(return_code == 0),
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )
        self.last_observation = obs
        return obs

    def observe_artifact(self, path: str) -> Observation:
        """
        Create an observation about an artifact.

        Checks if file exists and is non-empty.
        """
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        obs = Observation(
            artifact_path=path,
            artifact_exists=exists,
        )
        self.last_observation = obs
        return obs

    def update_masks(self, masks: Masks, observation: Observation) -> None:
        """
        Update masks based on observation.

        This is purely observational - no inference beyond what we see.
        """
        # Update affordances based on command type
        if observation.command:
            cmd = observation.command.lower()

            if "kubectl" in cmd:
                masks.affordance.k8s_policy = (
                    AffordanceState.AVAILABLE if observation.success
                    else AffordanceState.UNAVAILABLE
                    if "not found" in observation.stderr.lower()
                    or "connection refused" in observation.stderr.lower()
                    else masks.affordance.k8s_policy
                )

            if "opa" in cmd:
                masks.affordance.opa_eval = (
                    AffordanceState.AVAILABLE if observation.success
                    else AffordanceState.UNAVAILABLE
                    if "not found" in observation.stderr.lower()
                    else masks.affordance.opa_eval
                )

            if "ansible" in cmd:
                masks.affordance.ansible_exec = (
                    AffordanceState.AVAILABLE if observation.success
                    else AffordanceState.UNAVAILABLE
                    if "not found" in observation.stderr.lower()
                    else masks.affordance.ansible_exec
                )

        # Update evidence based on success
        if observation.artifact_exists:
            if masks.evidence.state == EvidenceState.NONE:
                masks.evidence.state = EvidenceState.ATTEMPTED

        # Update repair_required and evidence based on execution outcome
        if observation.command:
            cmd = observation.command.lower()
            is_execution_cmd = any(x in cmd for x in [
                "kubectl apply", "opa eval", "ansible-playbook"
            ])

            if is_execution_cmd:
                if observation.success:
                    # Success: clear repair obligation, advance evidence
                    masks.evidence.state = EvidenceState.SUCCESSFUL
                    masks.repair_required = False
                else:
                    # Failure: set repair obligation
                    # Evidence stays at attempted (not successful)
                    if masks.evidence.state == EvidenceState.NONE:
                        masks.evidence.state = EvidenceState.ATTEMPTED
                    masks.repair_required = True

        # Update posture based on violation signals
        if observation.has_violation_signal():
            # Violations observed - cannot be compliant
            masks.posture.compliant = False

        if observation.has_clean_signal():
            # Clean signal - cannot be non_compliant
            masks.posture.non_compliant = False

    def execute_and_observe(
        self,
        command: str,
        masks: Masks,
        timeout: int = 60,
    ) -> Observation:
        """
        Execute a command and observe the result.

        This is a convenience method that runs a command and updates masks.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir,
            )
            obs = self.observe_command(
                command=command,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            obs = self.observe_command(
                command=command,
                return_code=-1,
                stdout="",
                stderr="Command timed out",
            )
        except Exception as e:
            obs = self.observe_command(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
            )

        self.update_masks(masks, obs)
        return obs

    def write_artifact_and_observe(
        self,
        path: str,
        content: str,
        masks: Masks,
    ) -> Observation:
        """
        Write an artifact and observe the result.

        This is a convenience method that writes a file and updates masks.
        """
        full_path = os.path.join(self.work_dir, path) if not path.startswith("/") else path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, "w") as f:
                f.write(content)
        except Exception as e:
            print(f"[Observer] Failed to write artifact: {e}")

        obs = self.observe_artifact(full_path)
        self.update_masks(masks, obs)
        return obs
