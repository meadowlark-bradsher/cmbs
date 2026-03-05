"""Models for CMBS v2 operator-log sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OperationSpec:
    """Client-submitted operation payload."""

    op_id: Optional[str]
    op_type: str
    payload: Dict[str, Any]
    source_id: str
    preconditions: List[str] = field(default_factory=list)
    commutativity_key: Optional[str] = None
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class OperationEnvelope:
    """Append-only operation envelope persisted in a branch log."""

    op_id: str
    seq: int
    branch_id: str
    origin_seq: int
    op_type: str
    payload: Dict[str, Any]
    source_id: str
    preconditions: List[str]
    commutativity_key: Optional[str]
    idempotency_key: Optional[str]
    accepted: bool
    rejected_reason: Optional[str]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "seq": self.seq,
            "branch_id": self.branch_id,
            "origin_seq": self.origin_seq,
            "type": self.op_type,
            "payload": self.payload,
            "source_id": self.source_id,
            "preconditions": list(self.preconditions),
            "commutativity_key": self.commutativity_key,
            "idempotency_key": self.idempotency_key,
            "accepted": self.accepted,
            "rejected_reason": self.rejected_reason,
            "created_at": self.created_at,
        }


@dataclass
class BranchRecord:
    """Mutable branch state for a session."""

    branch_id: str
    from_branch: Optional[str]
    from_seq: int
    note: Optional[str]
    created_at: float
    op_log: List[OperationEnvelope] = field(default_factory=list)
    idempotency_index: Dict[str, OperationEnvelope] = field(default_factory=dict)

    @property
    def head_seq(self) -> int:
        return len(self.op_log)


@dataclass
class SessionRecord:
    """Top-level v2 session state."""

    sid: str
    ontology: Dict[str, Any]
    initial_hypotheses: List[str]
    default_reducer: str
    metadata: Dict[str, Any]
    created_at: float
    branches: Dict[str, BranchRecord] = field(default_factory=dict)
