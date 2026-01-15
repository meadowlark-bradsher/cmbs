"""
CMBS Supervisor

The permissive, stateful gatekeeper that observes execution outcomes
and blocks only epistemically invalid transitions.

Design checksum: "Could this supervisor pass a Turing test as a dumb traffic light?"

The supervisor is:
- boring
- stubborn
- annoying
- but fair

The supervisor MUST:
- Track observable execution outcomes
- Maintain belief masks derived from those outcomes
- Decide whether a proposed action is allowed, blocked, or must continue
- Decide whether termination is earned

The supervisor MUST NOT:
- Generate content
- Fix YAML / Rego / Ansible
- Suggest commands
- Infer CIS semantics
- Encode schemas
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time

from .masks import Masks, AffordanceState, EvidenceState
from .agent_protocol import (
    AgentStep,
    AgentAction,
    ActionType,
    SUPERVISOR_MESSAGES,
)
from .observer import Observer


class Verdict(str, Enum):
    """Supervisor decision on an agent step."""
    ALLOW = "allow"       # Action is permitted, execute it
    BLOCK = "block"       # Action is not permitted, must continue
    CONTINUE = "continue" # Step was invalid, re-emit
    TERMINATE = "terminate"  # Agent may exit


@dataclass
class SupervisorResponse:
    """Response from supervisor to agent."""
    verdict: Verdict
    message: str
    masks_snapshot: dict  # Current mask state for agent visibility

    def __str__(self) -> str:
        return f"[{self.verdict.value.upper()}] {self.message}"


class Supervisor:
    """
    The CMBS supervisor.

    Controls the agent loop by:
    1. Validating step schema
    2. Checking belief consistency with masks
    3. Checking action legality
    4. Deciding verdict

    Never helps. Never hints. Only gatekeeps.
    """

    def __init__(
        self,
        observer: Observer,
        max_steps: int = 50,
        timeout_seconds: float = 200.0,
        degeneracy_threshold: int = 3,  # Block after N identical failures
    ):
        self.observer = observer
        self.masks = Masks()
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.step_count = 0
        self.start_time = time.time()

        # Execution degeneracy guard: track identical failed artifacts
        # This is byte-level, schema-agnostic progress detection
        self.degeneracy_threshold = degeneracy_threshold
        self.last_failed_checksum: dict[str, str] = {}  # path -> checksum
        self.repeat_fail_count: dict[str, int] = {}     # path -> count

        # Revision degeneracy guard: track identical revisions
        # "A revision that produces identical bytes is not a revision"
        self.revision_threshold = 2  # Lower than execution (revise should change something)
        self.last_revision_checksum: dict[str, str] = {}  # path -> checksum
        self.revision_repeat_count: dict[str, int] = {}   # path -> count

    def elapsed_time(self) -> float:
        """Time elapsed since supervisor started."""
        return time.time() - self.start_time

    def time_remaining(self) -> float:
        """Time remaining before timeout."""
        return max(0, self.timeout_seconds - self.elapsed_time())

    def is_timed_out(self) -> bool:
        """Check if we've exceeded timeout."""
        return self.elapsed_time() >= self.timeout_seconds

    def is_step_limit_reached(self) -> bool:
        """Check if we've exceeded step limit."""
        return self.step_count >= self.max_steps

    def _checksum(self, path: str) -> Optional[str]:
        """
        Compute byte-level checksum of an artifact.

        This is schema-agnostic - we hash raw bytes, not parsed content.
        Returns None if file doesn't exist.
        """
        import hashlib
        import os
        full_path = os.path.join(self.observer.work_dir, path) if not path.startswith("/") else path
        if not os.path.exists(full_path):
            return None
        with open(full_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def record_failed_execution(self, artifact_path: str) -> None:
        """
        Record a failed execution for degeneracy detection.

        Called by runner after execution fails.
        """
        cs = self._checksum(artifact_path)
        if cs is None:
            return

        if cs == self.last_failed_checksum.get(artifact_path):
            # Same artifact failed again
            self.repeat_fail_count[artifact_path] = self.repeat_fail_count.get(artifact_path, 0) + 1
        else:
            # Different artifact - reset count
            self.last_failed_checksum[artifact_path] = cs
            self.repeat_fail_count[artifact_path] = 1

    def clear_failed_checksum(self, artifact_path: str) -> None:
        """Clear checksum tracking on successful execution."""
        self.last_failed_checksum.pop(artifact_path, None)
        self.repeat_fail_count.pop(artifact_path, None)

    def is_degenerate_execution(self, artifact_path: str) -> bool:
        """
        Check if executing this artifact would be degenerate.

        Returns True if same artifact has failed >= threshold times.
        """
        return self.repeat_fail_count.get(artifact_path, 0) >= self.degeneracy_threshold

    def record_revision(self, artifact_path: str) -> None:
        """
        Record a revision for degeneracy detection.

        Called by runner after REVISE_ARTIFACT action completes.
        Tracks whether revision produced different content.
        """
        cs = self._checksum(artifact_path)
        if cs is None:
            return

        if cs == self.last_revision_checksum.get(artifact_path):
            # Identical content - not a real revision
            self.revision_repeat_count[artifact_path] = self.revision_repeat_count.get(artifact_path, 0) + 1
        else:
            # Different content - genuine revision
            self.last_revision_checksum[artifact_path] = cs
            self.revision_repeat_count[artifact_path] = 1
            # A genuine revision also resets execution degeneracy
            self.clear_failed_checksum(artifact_path)

    def is_degenerate_revision(self, artifact_path: str) -> bool:
        """
        Check if revising this artifact would be degenerate.

        Returns True if same content has been "revised" >= threshold times.
        """
        return self.revision_repeat_count.get(artifact_path, 0) >= self.revision_threshold

    def clear_revision_tracking(self, artifact_path: str) -> None:
        """Clear revision tracking (e.g., on successful execution)."""
        self.last_revision_checksum.pop(artifact_path, None)
        self.revision_repeat_count.pop(artifact_path, None)

    def _response(self, verdict: Verdict, message: str) -> SupervisorResponse:
        """Create a supervisor response."""
        return SupervisorResponse(
            verdict=verdict,
            message=message,
            masks_snapshot=self.masks.to_dict(),
        )

    def _check_belief_consistency(self, step: AgentStep) -> Optional[str]:
        """
        Check if agent's claimed belief is consistent with observed reality.

        Returns error message if inconsistent, None if OK.
        """
        belief = step.belief

        # Check evidence claim
        claimed_evidence = belief.evidence
        actual_evidence = self.masks.evidence.state.value

        # Agent can't claim more evidence than observed
        evidence_order = {"none": 0, "attempted": 1, "successful": 2}
        if evidence_order.get(claimed_evidence, 0) > evidence_order.get(actual_evidence, 0):
            return f"Claimed evidence '{claimed_evidence}' exceeds observed '{actual_evidence}'"

        # Check posture claim against admissibility
        claimed_posture = belief.posture
        if claimed_posture == "compliant" and not self.masks.posture.compliant:
            return "Claimed 'compliant' but violations have been observed"
        if claimed_posture == "non_compliant" and not self.masks.posture.non_compliant:
            return "Claimed 'non_compliant' but clean results have been observed"

        return None

    def _check_action_legality(self, action: AgentAction) -> tuple[Verdict, str]:
        """
        Check if the proposed action is legal given current masks.

        Returns (verdict, message).
        """
        # Rule 0: Block posture/termination during repair obligation
        if self.masks.repair_required:
            if action.type in (ActionType.DECLARE_POSTURE, ActionType.TERMINATE):
                return Verdict.BLOCK, "Repair required; fix failed execution before claiming posture or terminating."

        # Rule 0.5: Block degenerate re-execution of same failed artifact
        if self.masks.repair_required:
            if action.type in (ActionType.EXECUTE_KUBECTL, ActionType.EXECUTE_OPA, ActionType.EXECUTE_ANSIBLE):
                # Derive artifact path from context (policy.yaml is primary)
                artifact_path = "policy.yaml"
                if self.is_degenerate_execution(artifact_path):
                    return Verdict.BLOCK, (
                        f"Identical artifact re-executed {self.degeneracy_threshold} times. "
                        "Revise artifact or change approach."
                    )

        # Rule 0.6: Block degenerate revision (identical content repeated)
        if action.type == ActionType.REVISE_ARTIFACT:
            artifact_path = action.payload.get("path", "policy.yaml")
            if self.is_degenerate_revision(artifact_path):
                return Verdict.BLOCK, (
                    f"Revision produced identical content {self.revision_threshold} times. "
                    "Change content or approach."
                )

        # Rule 0.7: Enforce action preconditions (capability checks)
        # Execution actions require corresponding generative actions to have occurred
        if action.type in (ActionType.EXECUTE_KUBECTL, ActionType.EXECUTE_OPA):
            if not self.masks.capability.policy_written:
                return Verdict.BLOCK, (
                    "Attempted to execute before any policy was written. "
                    "Generate a policy first."
                )

        if action.type == ActionType.EXECUTE_SCRIPT:
            if not self.masks.capability.script_written:
                return Verdict.BLOCK, (
                    "Attempted to execute script before any script was written. "
                    "Generate a script first."
                )

        if action.type == ActionType.EXECUTE_ANSIBLE:
            if not self.masks.capability.playbook_written:
                return Verdict.BLOCK, (
                    "Attempted to run Ansible before any playbook was written. "
                    "Generate a playbook first."
                )

        # Rule 0.8: Derivation obligation for edit actions
        # EDIT_POLICY is epistemically meaningful only as a transformation
        # of previously observed state - you can't edit what you haven't seen
        if action.type == ActionType.EDIT_POLICY:
            if not self.masks.capability.policy_observed:
                return Verdict.BLOCK, (
                    "Attempted to edit policy before observing existing state. "
                    "Use read_resource to observe the current policy first."
                )

        # Rule 1: Posture declaration requires successful evidence
        if action.type == ActionType.DECLARE_POSTURE:
            if self.masks.evidence.state != EvidenceState.SUCCESSFUL:
                return Verdict.BLOCK, SUPERVISOR_MESSAGES["posture_requires_evidence"]

            # Check that declared posture is admissible
            posture = action.payload.get("posture")
            if posture == "compliant" and not self.masks.posture.compliant:
                return Verdict.BLOCK, "Cannot declare compliant; violations observed."
            if posture == "non_compliant" and not self.masks.posture.non_compliant:
                return Verdict.BLOCK, "Cannot declare non_compliant; clean results observed."

            # Record posture for stability tracking
            self.masks.stability.record_posture(posture)
            return Verdict.ALLOW, "Posture declaration accepted."

        # Rule 2: Termination requires epistemic completion
        if action.type == ActionType.TERMINATE:
            if not self.masks.termination_allowed():
                reasons = []
                if self.masks.evidence.state != EvidenceState.SUCCESSFUL:
                    reasons.append("evidence not successful")
                if not self.masks.posture.exactly_one_admissible():
                    reasons.append("posture not determined")
                if not self.masks.stability.is_stable():
                    reasons.append("posture not stable")
                if self.masks.repair_required:
                    reasons.append("repair obligation pending")
                detail = "; ".join(reasons) if reasons else "requirements not met"
                return Verdict.BLOCK, f"Termination blocked; {detail}. Continue."

            return Verdict.TERMINATE, "Termination allowed."

        # Rule 3: Affordance gating (weak)
        required_affordance = action.requires_affordance()
        if required_affordance:
            affordance_state = getattr(
                self.masks.affordance,
                required_affordance,
                AffordanceState.UNKNOWN
            )
            if affordance_state == AffordanceState.UNAVAILABLE:
                return Verdict.BLOCK, SUPERVISOR_MESSAGES["affordance_unavailable"]

        # Rule 4: Everything else is allowed
        return Verdict.ALLOW, "Action allowed."

    def evaluate_step(self, step: AgentStep) -> SupervisorResponse:
        """
        Evaluate an agent step and return a verdict.

        This is the main entry point for the supervisor.
        """
        self.step_count += 1

        # Check time/step limits
        if self.is_timed_out():
            return self._response(
                Verdict.TERMINATE,
                f"Timeout reached ({self.timeout_seconds}s). Forced termination."
            )

        if self.is_step_limit_reached():
            return self._response(
                Verdict.TERMINATE,
                f"Step limit reached ({self.max_steps}). Forced termination."
            )

        # 1. Validate schema
        is_valid, error = step.validate_schema()
        if not is_valid:
            return self._response(Verdict.CONTINUE, f"Invalid format: {error}")

        # 2. Check belief consistency
        inconsistency = self._check_belief_consistency(step)
        if inconsistency:
            return self._response(Verdict.CONTINUE, f"Belief inconsistent: {inconsistency}")

        # 3. Check action legality
        verdict, message = self._check_action_legality(step.action)
        return self._response(verdict, message)

    def get_status(self) -> dict:
        """Get current supervisor status for debugging/logging."""
        return {
            "step_count": self.step_count,
            "elapsed_time": self.elapsed_time(),
            "time_remaining": self.time_remaining(),
            "masks": self.masks.to_dict(),
        }

    def __str__(self) -> str:
        return (
            f"Supervisor(step={self.step_count}, "
            f"elapsed={self.elapsed_time():.1f}s, "
            f"evidence={self.masks.evidence.state.value}, "
            f"term_allowed={self.masks.termination_allowed()})"
        )
