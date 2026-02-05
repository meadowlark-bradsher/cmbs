"""CMBS adapters (core boundary implementations)."""

from .legacy import LegacyEliminationEvent, LegacyReplayAdapter, submit_legacy_elimination
from .types import Action, AdapterActionContext, BeliefAdapter, BeliefMessage, EliminateMessage
from .twenty_questions import TwentyQAdapter, TwentyQKit, TwentyQOracle
from .itbench import ITBenchAdapter, ITBenchKit, ITBenchOracle

__all__ = [
    "Action",
    "AdapterActionContext",
    "BeliefAdapter",
    "BeliefMessage",
    "EliminateMessage",
    "LegacyEliminationEvent",
    "LegacyReplayAdapter",
    "submit_legacy_elimination",
    "TwentyQAdapter",
    "TwentyQKit",
    "TwentyQOracle",
    "ITBenchAdapter",
    "ITBenchKit",
    "ITBenchOracle",
]
