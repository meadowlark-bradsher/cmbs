"""
CMBS - Constraint-Mask Belief System for IT-Bench CISO

A runtime supervisor that enforces epistemic discipline without supplying
domain knowledge. The supervisor observes execution outcomes and blocks
only epistemically invalid transitions (lies and premature exits).

Design principles:
- Permissive by default
- No domain knowledge (Kyverno/Rego/Ansible schemas)
- No hints or suggestions
- Only blocks: false claims, premature termination

See: CMBS-for-ITBench.md, supervisor.md, mask-inventory.md
"""

from .masks import Masks, AffordanceState, EvidenceState
from .supervisor import Supervisor, Verdict
from .agent_protocol import AgentStep, AgentBelief, AgentAction, ActionType
from .observer import Observer
from .runner import CMBSRunner

__all__ = [
    "Masks",
    "AffordanceState",
    "EvidenceState",
    "Supervisor",
    "Verdict",
    "AgentStep",
    "AgentBelief",
    "AgentAction",
    "ActionType",
    "Observer",
    "CMBSRunner",
]
