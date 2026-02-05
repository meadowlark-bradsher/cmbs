"""
CMBS v0 Core.

CMBS is a belief-state accounting system. It tracks hypotheses, eliminations,
entropy, and obligation discipline. It does not select probes, interpret
observables, or manage workflows. Semantics live in adapters.
"""

from .core import (
    CMBSCore,
    EliminationEvent,
    ObligationExitResult,
    ProbeResult,
    TerminationResult,
)
from .belief_server import (
    AuditEntry,
    BeliefServer,
    BeliefSnapshot,
    OntologyBundle,
)
from .adapters.legacy import (
    LegacyEliminationEvent,
    LegacyReplayAdapter,
    submit_legacy_elimination,
)

__all__ = [
    "CMBSCore",
    "EliminationEvent",
    "ObligationExitResult",
    "ProbeResult",
    "TerminationResult",
    "AuditEntry",
    "BeliefServer",
    "BeliefSnapshot",
    "OntologyBundle",
    "LegacyEliminationEvent",
    "LegacyReplayAdapter",
    "submit_legacy_elimination",
]
