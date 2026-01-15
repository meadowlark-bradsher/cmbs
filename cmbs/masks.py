"""
CMBS Belief Masks

The six masks that track observable execution state:
- Affordance: what execution paths are possible
- Evidence: how far execution has progressed
- Posture: which outcomes are still admissible
- Stability: whether beliefs are oscillating
- Repair: whether agent is obligated to fix a failed execution
- Termination: derived eligibility to terminate

These masks are updated ONLY from observed execution outcomes.
They never encode domain knowledge or schemas.

Repair obligation: When an execution attempt fails, the agent is
required to remain in a repair loop until execution succeeds or
time expires. This is epistemic discipline, not competence.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Literal


class AffordanceState(str, Enum):
    """What the environment appears to support."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class EvidenceState(str, Enum):
    """How far execution has actually progressed."""
    NONE = "none"
    ATTEMPTED = "attempted"
    SUCCESSFUL = "successful"


@dataclass
class AffordanceMask:
    """
    Mask A - Tracks what execution paths are currently possible.

    Updated by:
    - Presence of kubeconfig
    - Success/failure of kubectl
    - Success/failure of opa eval
    - Success/failure of ansible-playbook

    Used to gate:
    - Policy generation attempts
    - Execution commands
    """
    k8s_policy: AffordanceState = AffordanceState.UNKNOWN
    opa_eval: AffordanceState = AffordanceState.UNKNOWN
    ansible_exec: AffordanceState = AffordanceState.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "k8s_policy": self.k8s_policy.value,
            "opa_eval": self.opa_eval.value,
            "ansible_exec": self.ansible_exec.value,
        }


@dataclass
class EvidenceMask:
    """
    Mask B - Tracks whether anything real has happened.

    Updated by:
    - Artifact creation → attempted
    - Successful execution → successful

    Used to gate:
    - Posture declaration
    - Termination
    """
    state: EvidenceState = EvidenceState.NONE

    def to_dict(self) -> dict:
        return {"state": self.state.value}


@dataclass
class PostureMask:
    """
    Mask C - Tracks which postures are still logically possible.

    Updated by:
    - Observed violations → compliant becomes False
    - Explicit negative results → non_compliant becomes False

    Rules:
    - Absence of evidence does NOT eliminate non_compliance
    - Execution failure does NOT collapse posture

    Used to gate:
    - Declared posture
    """
    compliant: bool = True
    non_compliant: bool = True

    def to_dict(self) -> dict:
        return {
            "compliant": self.compliant,
            "non_compliant": self.non_compliant,
        }

    def exactly_one_admissible(self) -> bool:
        """Returns True if exactly one posture is still admissible."""
        return self.compliant != self.non_compliant


@dataclass
class CapabilityMask:
    """
    Mask G - Tracks what generative and observational actions have occurred.

    This enforces action preconditions: execution actions are permitted
    only if the corresponding generative actions have occurred earlier
    in the episode.

    This prevents "phantom action" failures where agents believe they
    wrote artifacts that were never created.

    Updated by:
    - GENERATE_POLICY, REVISE_ARTIFACT (policy), EDIT_POLICY → policy_written = True
    - GENERATE_SCRIPT → script_written = True
    - GENERATE_PLAYBOOK → playbook_written = True
    - READ_RESOURCE (for policies) → policy_observed = True

    Used to gate:
    - EXECUTE_KUBECTL, EXECUTE_OPA → requires policy_written
    - EXECUTE_SCRIPT → requires script_written
    - EXECUTE_ANSIBLE → requires playbook_written
    - EDIT_POLICY → requires policy_observed (derivation obligation)
    """
    policy_written: bool = False
    script_written: bool = False
    playbook_written: bool = False
    # Derivation source: observed existing cluster state
    policy_observed: bool = False

    def to_dict(self) -> dict:
        return {
            "policy_written": self.policy_written,
            "script_written": self.script_written,
            "playbook_written": self.playbook_written,
            "policy_observed": self.policy_observed,
        }


@dataclass
class StabilityMask:
    """
    Mask D - Ensures beliefs do not oscillate.

    Updated by:
    - Comparing posture across last N steps

    Used to gate:
    - Termination
    """
    posture_history: list = field(default_factory=list)
    window_size: int = 3

    def record_posture(self, posture: str) -> None:
        """Record a posture claim."""
        self.posture_history.append(posture)
        # Keep only last N entries
        if len(self.posture_history) > self.window_size * 2:
            self.posture_history = self.posture_history[-self.window_size:]

    def is_stable(self) -> bool:
        """
        Returns True if posture has been stable over the window.
        Requires at least window_size consistent claims.
        """
        if len(self.posture_history) < self.window_size:
            return False
        recent = self.posture_history[-self.window_size:]
        return len(set(recent)) == 1

    def to_dict(self) -> dict:
        return {
            "posture_stable": self.is_stable(),
            "history": self.posture_history[-self.window_size:],
        }


@dataclass
class Masks:
    """
    Complete mask state for the CMBS supervisor.

    This is the ONLY state the supervisor maintains.
    All masks are derived from observed execution outcomes.
    """
    affordance: AffordanceMask = field(default_factory=AffordanceMask)
    evidence: EvidenceMask = field(default_factory=EvidenceMask)
    posture: PostureMask = field(default_factory=PostureMask)
    stability: StabilityMask = field(default_factory=StabilityMask)
    capability: CapabilityMask = field(default_factory=CapabilityMask)

    # Mask F - Repair/troubleshooting obligation
    # When True, agent must continue repair loop until success
    # Set on failed execution, cleared only on successful execution
    repair_required: bool = False

    def termination_allowed(self) -> bool:
        """
        Derived termination eligibility.

        Termination requires:
        - evidence == successful
        - exactly one posture admissible
        - posture stable
        - no repair obligation pending
        """
        return (
            self.evidence.state == EvidenceState.SUCCESSFUL
            and self.posture.exactly_one_admissible()
            and self.stability.is_stable()
            and not self.repair_required
        )

    def to_dict(self) -> dict:
        """Export masks as dictionary for agent visibility."""
        return {
            "affordances": self.affordance.to_dict(),
            "evidence": self.evidence.to_dict(),
            "posture": self.posture.to_dict(),
            "stability": self.stability.to_dict(),
            "capability": self.capability.to_dict(),
            "repair_required": self.repair_required,
            "termination_allowed": self.termination_allowed(),
        }

    def __str__(self) -> str:
        return (
            f"Masks(\n"
            f"  affordance: k8s={self.affordance.k8s_policy.value}, "
            f"opa={self.affordance.opa_eval.value}, "
            f"ansible={self.affordance.ansible_exec.value}\n"
            f"  evidence: {self.evidence.state.value}\n"
            f"  posture: compliant={self.posture.compliant}, "
            f"non_compliant={self.posture.non_compliant}\n"
            f"  stability: stable={self.stability.is_stable()}\n"
            f"  repair_required: {self.repair_required}\n"
            f"  termination_allowed: {self.termination_allowed()}\n"
            f")"
        )
