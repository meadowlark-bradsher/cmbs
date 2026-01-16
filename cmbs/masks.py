"""
CMBS Belief Masks

The seven masks that track observable execution state:
- Affordance: what execution paths are possible
- Evidence: how far execution has progressed
- Posture: which outcomes are still admissible
- Stability: whether beliefs are oscillating
- Repair: whether agent is obligated to fix a failed execution
- Capability: what generative actions have occurred
- DSO: Document Search Obligation state (DSRO feature)
- Termination: derived eligibility to terminate

These masks are updated ONLY from observed execution outcomes.
They never encode domain knowledge or schemas.

Repair obligation: When an execution attempt fails, the agent is
required to remain in a repair loop until execution succeeds or
time expires. This is epistemic discipline, not competence.

Document Search Obligation (DSRO): When the agent chooses document
search as a repair action, they enter DSO and must demonstrate
measurable learning (via CCIL metrics) before exiting.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Literal, List, Tuple, Optional


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
class DSOMask:
    """
    Mask H - Document Search Obligation state (DSRO feature).

    Tracks whether the agent is in the Document Search Obligation state
    and enforces the non-repetition rule.

    Entry: Agent selects DOCUMENT_SEARCH as repair action
    Exit: CCIL metrics show measurable belief change OR all probes exhausted

    Design invariant:
    "Choosing document search creates an obligation to learn.
     Learning is measured by belief movement, not by claims."
    """
    # Whether DSO is currently active
    active: bool = False

    # Probe history for non-repetition rule: [(kind, target), ...]
    probe_history: List[Tuple[str, str]] = field(default_factory=list)

    # CCIL metrics at DSO entry (for measuring change)
    entry_entropy_posture: Optional[float] = None
    entry_capability_opa: Optional[float] = None
    entry_repair_pressure: Optional[float] = None

    # Exit thresholds (belief change required to exit)
    delta_threshold: float = 0.05  # Minimum change in any metric

    # Exit reason (set when DSO ends)
    exit_reason: Optional[str] = None  # "belief_change" | "exhausted" | "timeout"

    def enter(
        self,
        entropy_posture: float,
        capability_opa: float,
        repair_pressure: float,
    ) -> None:
        """Enter DSO state, recording entry metrics."""
        self.active = True
        self.probe_history = []
        self.entry_entropy_posture = entropy_posture
        self.entry_capability_opa = capability_opa
        self.entry_repair_pressure = repair_pressure
        self.exit_reason = None

    def exit(self, reason: str) -> None:
        """Exit DSO state with given reason."""
        self.active = False
        self.exit_reason = reason

    def record_probe(self, kind: str, target: str) -> None:
        """Record a probe in history."""
        self.probe_history.append((kind, target))

    def is_probe_repeated(self, kind: str, target: str) -> bool:
        """Check if this probe has been made before."""
        return (kind, target) in self.probe_history

    def get_probe_count(self) -> int:
        """Return number of probes made."""
        return len(self.probe_history)

    def can_exit_with_belief_change(
        self,
        current_entropy_posture: float,
        current_capability_opa: float,
        current_repair_pressure: float,
    ) -> bool:
        """
        Check if CCIL metrics show sufficient change to exit DSO.

        Exit is allowed if ANY of:
        - entropy_posture decreased >= delta_threshold
        - capability_opa increased >= delta_threshold
        - repair_pressure decreased >= delta_threshold
        """
        if not self.active:
            return True  # Not in DSO, can always exit

        if self.entry_entropy_posture is None:
            return False  # No entry metrics recorded

        # Check each exit condition
        entropy_decrease = self.entry_entropy_posture - current_entropy_posture
        capability_increase = current_capability_opa - self.entry_capability_opa
        pressure_decrease = self.entry_repair_pressure - current_repair_pressure

        return (
            entropy_decrease >= self.delta_threshold or
            capability_increase >= self.delta_threshold or
            pressure_decrease >= self.delta_threshold
        )

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "probe_count": len(self.probe_history),
            "probe_history": self.probe_history,
            "entry_metrics": {
                "entropy_posture": self.entry_entropy_posture,
                "capability_opa": self.entry_capability_opa,
                "repair_pressure": self.entry_repair_pressure,
            } if self.active else None,
            "exit_reason": self.exit_reason,
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
    dso: DSOMask = field(default_factory=DSOMask)

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
            "dso": self.dso.to_dict(),
            "repair_required": self.repair_required,
            "termination_allowed": self.termination_allowed(),
        }

    def __str__(self) -> str:
        dso_status = f"active={self.dso.active}, probes={self.dso.get_probe_count()}" if self.dso.active else "inactive"
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
            f"  dso: {dso_status}\n"
            f"  termination_allowed: {self.termination_allowed()}\n"
            f")"
        )
