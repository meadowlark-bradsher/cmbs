"""
CMBS Agent Protocol

Defines the structured communication between agent and supervisor.
The agent must emit:
- belief: its current understanding of the world
- action: what it wants to do next
- free_text: optional explanation (only accepted on termination)

The supervisor returns:
- verdict: ALLOW, BLOCK, CONTINUE, or TERMINATE
- message: neutral procedural feedback (never hints or suggestions)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any


class ActionType(str, Enum):
    """Actions the agent can propose."""
    # Artifact generation (new content, ex nihilo)
    GENERATE_POLICY = "generate_policy"
    GENERATE_SCRIPT = "generate_script"
    GENERATE_PLAYBOOK = "generate_playbook"

    # Artifact revision (modify existing local content after seeing errors)
    REVISE_ARTIFACT = "revise_artifact"

    # Derived artifact (edit requires prior observation of source)
    # Use this when modifying an existing cluster resource
    EDIT_POLICY = "edit_policy"

    # Observation (witness existing state)
    READ_RESOURCE = "read_resource"  # Observe existing cluster resource
    READ_OUTPUT = "read_output"
    CHECK_STATUS = "check_status"

    # Execution
    EXECUTE_KUBECTL = "execute_kubectl"
    EXECUTE_OPA = "execute_opa"
    EXECUTE_ANSIBLE = "execute_ansible"
    EXECUTE_SCRIPT = "execute_script"

    # State transitions
    DECLARE_POSTURE = "declare_posture"
    TERMINATE = "terminate"

    # Meta
    RETRY = "retry"
    CONTINUE = "continue"


@dataclass
class AgentBelief:
    """
    The agent's claimed belief state.

    This is what the agent CLAIMS to believe.
    The supervisor validates it against observed reality (masks).
    """
    affordances: dict = field(default_factory=lambda: {
        "k8s_policy": "unknown",
        "opa_eval": "unknown",
        "ansible_exec": "unknown",
    })
    posture: str = "unknown"  # unknown | compliant | non_compliant
    evidence: str = "none"    # none | attempted | successful

    def to_dict(self) -> dict:
        return {
            "affordances": self.affordances,
            "posture": self.posture,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentBelief":
        return cls(
            affordances=d.get("affordances", {
                "k8s_policy": "unknown",
                "opa_eval": "unknown",
                "ansible_exec": "unknown",
            }),
            posture=d.get("posture", "unknown"),
            evidence=d.get("evidence", "none"),
        )


@dataclass
class AgentAction:
    """
    An action the agent wants to perform.

    The supervisor checks if this action is legal given current masks.
    """
    type: ActionType
    payload: dict = field(default_factory=dict)
    # For execute actions: {"command": "kubectl apply -f ..."}
    # For generate actions: {"content": "apiVersion: ..."}
    # For declare_posture: {"posture": "compliant" | "non_compliant"}
    # For terminate: {}

    def requires_affordance(self) -> Optional[str]:
        """Returns the affordance this action requires, if any."""
        mapping = {
            ActionType.GENERATE_POLICY: "k8s_policy",
            ActionType.EXECUTE_KUBECTL: "k8s_policy",
            ActionType.EXECUTE_OPA: "opa_eval",
            ActionType.EXECUTE_ANSIBLE: "ansible_exec",
            ActionType.GENERATE_PLAYBOOK: "ansible_exec",
        }
        return mapping.get(self.type)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentAction":
        return cls(
            type=ActionType(d["type"]),
            payload=d.get("payload", {}),
        )


@dataclass
class AgentStep:
    """
    A complete agent output for one step.

    Contains:
    - belief: what the agent claims to believe
    - action: what the agent wants to do
    - free_text: optional explanation (discarded unless terminating)
    """
    belief: AgentBelief
    action: AgentAction
    free_text: str = ""

    def to_dict(self) -> dict:
        return {
            "belief": self.belief.to_dict(),
            "action": self.action.to_dict(),
            "free_text": self.free_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentStep":
        return cls(
            belief=AgentBelief.from_dict(d.get("belief", {})),
            action=AgentAction.from_dict(d["action"]),
            free_text=d.get("free_text", ""),
        )

    def validate_schema(self) -> tuple[bool, str]:
        """
        Validates that the step has required fields.
        Returns (is_valid, error_message).
        """
        if not self.action:
            return False, "Missing action"

        if self.action.type == ActionType.DECLARE_POSTURE:
            if "posture" not in self.action.payload:
                return False, "declare_posture requires posture in payload"
            if self.action.payload["posture"] not in ("compliant", "non_compliant"):
                return False, "posture must be 'compliant' or 'non_compliant'"

        if self.action.type in (
            ActionType.EXECUTE_KUBECTL,
            ActionType.EXECUTE_OPA,
            ActionType.EXECUTE_ANSIBLE,
            ActionType.EXECUTE_SCRIPT,
        ):
            if "command" not in self.action.payload:
                return False, f"{self.action.type.value} requires command in payload"

        if self.action.type in (
            ActionType.GENERATE_POLICY,
            ActionType.GENERATE_SCRIPT,
            ActionType.GENERATE_PLAYBOOK,
            ActionType.EDIT_POLICY,
        ):
            if "content" not in self.action.payload:
                return False, f"{self.action.type.value} requires content in payload"

        if self.action.type == ActionType.READ_RESOURCE:
            if "command" not in self.action.payload:
                return False, "read_resource requires command in payload"

        return True, ""


# Standard responses from supervisor to agent
SUPERVISOR_MESSAGES = {
    "invalid_schema": "Invalid step format. Re-emit with correct structure.",
    "belief_inconsistent": "Belief inconsistent with observed outcomes. Update belief.",
    "posture_requires_evidence": "Posture declaration not permitted without successful execution. Continue.",
    "termination_blocked": "Termination blocked; epistemic requirements not met. Continue.",
    "affordance_unavailable": "Action blocked; required capability unavailable. Continue.",
    "action_blocked": "Action not permitted. Continue.",
    "continue": "Continue.",
}
