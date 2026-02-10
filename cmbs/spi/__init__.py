"""SPI surface for CMBS adapters."""

from .adapter import discover_providers
from .hypothesis_provider import HypothesisProvider
from .elimination_store import (
    EliminationProvenance,
    EliminationResult,
    EliminationStore,
    RecoveredState,
)

__all__ = [
    "HypothesisProvider",
    "discover_providers",
    "EliminationProvenance",
    "EliminationResult",
    "EliminationStore",
    "RecoveredState",
]
