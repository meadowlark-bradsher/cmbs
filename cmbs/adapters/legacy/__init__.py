"""Legacy replay adapter (transitional)."""

from .replay import LegacyEliminationEvent, LegacyReplayAdapter, submit_legacy_elimination

__all__ = [
    "LegacyEliminationEvent",
    "LegacyReplayAdapter",
    "submit_legacy_elimination",
]
