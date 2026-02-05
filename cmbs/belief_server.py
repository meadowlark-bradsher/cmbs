"""
Belief Server v1 (in-memory).

Wraps the CMBSCore kernel with session management, audit logging, and
policy checks specified in BELIEF_SERVER_SPEC.md.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .core import CMBSCore
from .belief_state import BeliefState
from .spi.hypothesis_provider import HypothesisProvider


@dataclass(frozen=True)
class OntologyBundle:
    hypothesis_space_id: str
    hypothesis_version: str
    causal_graph_ref: str
    causal_graph_version: str


@dataclass
class BeliefSnapshot:
    session_id: str
    ontology: OntologyBundle
    survivors: List[str]
    terminated: bool
    active_obligation_id: Optional[str]
    audit_head_event_id: Optional[str]

    @property
    def n_survivors(self) -> int:
        return len(self.survivors)

    @property
    def entropy_proxy(self) -> float:
        n = self.n_survivors
        return 0.0 if n <= 1 else math.log2(n)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ontology": {
                "hypothesis_space_id": self.ontology.hypothesis_space_id,
                "hypothesis_version": self.ontology.hypothesis_version,
                "causal_graph_ref": self.ontology.causal_graph_ref,
                "causal_graph_version": self.ontology.causal_graph_version,
            },
            "survivors": list(self.survivors),
            "n_survivors": self.n_survivors,
            "entropy_proxy": self.entropy_proxy,
            "terminated": self.terminated,
            "active_obligation_id": self.active_obligation_id,
            "audit_head_event_id": self.audit_head_event_id,
        }


@dataclass
class AuditEntry:
    event_id: str
    ts: float
    verb: str
    payload: Dict[str, Any]
    survivors_before_hash: str
    survivors_after_hash: str
    delta: Dict[str, List[str]]
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "event_id": self.event_id,
            "ts": self.ts,
            "verb": self.verb,
            "payload": self.payload,
            "survivors_before_hash": self.survivors_before_hash,
            "survivors_after_hash": self.survivors_after_hash,
            "delta": self.delta,
        }
        if self.notes is not None:
            data["notes"] = self.notes
        return data


@dataclass
class ObservationRecord:
    applied_eliminated: List[str]
    ignored_eliminated: List[str]
    audit_event_id: str


@dataclass
class SessionState:
    session_id: str
    ontology: OntologyBundle
    kernel: Optional[CMBSCore]
    belief_state: Optional[BeliefState]
    hypothesis_ids: Set[str]
    terminated: bool = False
    active_obligation_id: Optional[str] = None
    obligation_min_elims: Optional[int] = None
    obligation_elim_count: int = 0
    audit: List[AuditEntry] = field(default_factory=list)
    audit_head_event_id: Optional[str] = None
    audit_head_hash: str = ""
    observation_index: Dict[str, ObservationRecord] = field(default_factory=dict)


class BeliefServerError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class BeliefServer:
    def __init__(
        self,
        validate_hypotheses: bool = False,
        belief_provider: Optional[HypothesisProvider] = None,
    ) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._validate_hypotheses = validate_hypotheses
        self._belief_provider = belief_provider

    def declare_session(
        self,
        ontology: OntologyBundle,
        hypotheses: Optional[List[str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, BeliefSnapshot]:
        session_id = str(uuid.uuid4())
        if self._belief_provider is not None:
            provider = self._belief_provider
            hypothesis_ids = list(provider.hypothesis_ids())
            belief_state = BeliefState(provider)
            state = SessionState(
                session_id=session_id,
                ontology=ontology,
                kernel=None,
                belief_state=belief_state,
                hypothesis_ids=set(hypothesis_ids),
            )
        else:
            if hypotheses is None:
                raise BeliefServerError(
                    code="INVALID_REQUEST",
                    message="Hypotheses are required when no belief provider is configured.",
                )
            kernel = CMBSCore(set(hypotheses))
            state = SessionState(
                session_id=session_id,
                ontology=ontology,
                kernel=kernel,
                belief_state=None,
                hypothesis_ids=set(hypotheses),
            )
        self._sessions[session_id] = state
        declared_hypotheses = sorted(state.hypothesis_ids)
        payload = {
            "ontology": {
                "hypothesis_space_id": ontology.hypothesis_space_id,
                "hypothesis_version": ontology.hypothesis_version,
                "causal_graph_ref": ontology.causal_graph_ref,
                "causal_graph_version": ontology.causal_graph_version,
            },
            "hypotheses": declared_hypotheses,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        self._append_audit(
            state,
            "DECLARE_SESSION",
            payload,
            eliminated=[],
            survivors_before=set(),
            survivors_after=set(self._survivors(state)),
        )
        return session_id, self._snapshot(state)

    def eliminate(
        self,
        session_id: str,
        source_id: str,
        observation_id: str,
        eliminated: List[str],
        justification: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str], BeliefSnapshot, str]:
        state = self._get(session_id)
        self._ensure_not_terminated(state)
        if state.belief_state is not None:
            raise BeliefServerError(
                code="CONFLICT",
                message="SPI sessions must use apply_probe instead of eliminate.",
            )

        observation_key = f"{source_id}::{observation_id}"
        if observation_key in state.observation_index:
            record = state.observation_index[observation_key]
            return (
                list(record.applied_eliminated),
                list(record.ignored_eliminated),
                self._snapshot(state),
                record.audit_event_id,
            )

        before = set(self._survivors(state))
        if self._validate_hypotheses:
            invalid = sorted(set(eliminated) - state.hypothesis_ids)
            if invalid:
                raise BeliefServerError(
                    code="INVALID_HYPOTHESIS_ID",
                    message="Elimination contains unknown hypothesis ids.",
                    details={"invalid": invalid},
                )

        applied = sorted(set(eliminated) & before)
        ignored = sorted(set(eliminated) - set(applied))

        probe_id = observation_key
        observable_id = source_id
        if state.kernel is None:
            raise BeliefServerError(
                code="CONFLICT",
                message="Belief kernel unavailable for eliminate.",
            )
        state.kernel.submit_probe_result(
            probe_id=probe_id,
            observable_id=observable_id,
            eliminated=set(eliminated),
        )
        after = set(self._survivors(state))

        payload = {
            "source_id": source_id,
            "observation_id": observation_id,
            "eliminated": list(eliminated),
            "justification": justification or {},
        }
        event_id = self._append_audit(
            state,
            "ELIMINATE",
            payload,
            eliminated=applied,
            survivors_before=before,
            survivors_after=after,
        )
        state.observation_index[observation_key] = ObservationRecord(
            applied_eliminated=applied,
            ignored_eliminated=ignored,
            audit_event_id=event_id,
        )
        return applied, ignored, self._snapshot(state), event_id

    def apply_probe(
        self,
        session_id: str,
        probe_id: str,
        response: Any,
        source_id: str = "provider",
    ) -> Tuple[List[str], List[str], BeliefSnapshot, str]:
        state = self._get(session_id)
        self._ensure_not_terminated(state)
        if state.belief_state is None:
            raise BeliefServerError(
                code="CONFLICT",
                message="apply_probe is only available for SPI sessions.",
            )

        observation_id = f"{probe_id}:{response}"
        observation_key = f"{source_id}::{observation_id}"
        if observation_key in state.observation_index:
            record = state.observation_index[observation_key]
            return (
                list(record.applied_eliminated),
                list(record.ignored_eliminated),
                self._snapshot(state),
                record.audit_event_id,
            )

        before = set(self._survivors(state))
        eliminated_set = state.belief_state.apply_probe(probe_id, response)
        after = set(self._survivors(state))
        applied = sorted(eliminated_set)
        ignored: List[str] = []

        if state.active_obligation_id is not None:
            state.obligation_elim_count += len(applied)

        payload = {
            "source_id": source_id,
            "probe_id": probe_id,
            "response": response,
        }
        event_id = self._append_audit(
            state,
            "APPLY_PROBE",
            payload,
            eliminated=applied,
            survivors_before=before,
            survivors_after=after,
        )
        state.observation_index[observation_key] = ObservationRecord(
            applied_eliminated=applied,
            ignored_eliminated=ignored,
            audit_event_id=event_id,
        )
        return applied, ignored, self._snapshot(state), event_id

    def query_belief(self, session_id: str) -> BeliefSnapshot:
        state = self._get(session_id)
        return self._snapshot(state)

    def audit_trace(self, session_id: str, since_event_id: Optional[str] = None) -> List[AuditEntry]:
        state = self._get(session_id)
        if since_event_id is None:
            return list(state.audit)
        for idx, entry in enumerate(state.audit):
            if entry.event_id == since_event_id:
                return list(state.audit[idx + 1 :])
        return list(state.audit)

    def enter_obligation(
        self,
        session_id: str,
        obligation_id: str,
        min_total_eliminations: int,
    ) -> Tuple[BeliefSnapshot, str]:
        state = self._get(session_id)
        self._ensure_not_terminated(state)
        if state.active_obligation_id is not None:
            raise BeliefServerError(
                code="CONFLICT",
                message="An obligation is already active.",
                details={"active_obligation_id": state.active_obligation_id},
            )
        if state.belief_state is not None:
            state.active_obligation_id = obligation_id
            state.obligation_min_elims = min_total_eliminations
            state.obligation_elim_count = 0
        else:
            if state.kernel is None:
                raise BeliefServerError(
                    code="CONFLICT",
                    message="Belief kernel unavailable for enter_obligation.",
                )
            state.kernel.enter_obligation(
                obligation_id=obligation_id,
                min_eliminations=min_total_eliminations,
            )
            state.active_obligation_id = obligation_id
        payload = {
            "obligation_id": obligation_id,
            "min_total_eliminations": min_total_eliminations,
        }
        event_id = self._append_audit(state, "ENTER_OBLIGATION", payload, eliminated=[])
        return self._snapshot(state), event_id

    def request_exit(
        self,
        session_id: str,
        obligation_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, BeliefSnapshot, str]:
        state = self._get(session_id)
        self._ensure_not_terminated(state)
        if state.active_obligation_id != obligation_id:
            raise BeliefServerError(
                code="OBLIGATION_NOT_FOUND",
                message="Obligation not active.",
                details={"obligation_id": obligation_id},
            )
        if state.belief_state is not None:
            min_elims = state.obligation_min_elims or 0
            approved = state.obligation_elim_count >= min_elims
            reason = "approved" if approved else "minimum eliminations not met"
            if approved:
                state.active_obligation_id = None
                state.obligation_min_elims = None
                state.obligation_elim_count = 0
        else:
            if state.kernel is None:
                raise BeliefServerError(
                    code="CONFLICT",
                    message="Belief kernel unavailable for request_exit.",
                )
            result = state.kernel.request_obligation_exit(obligation_id=obligation_id)
            approved = result.permitted
            reason = result.error or ("approved" if approved else "minimum eliminations not met")
            if approved:
                state.active_obligation_id = None
        payload = {
            "obligation_id": obligation_id,
            "context": context or {},
            "approved": approved,
        }
        event_id = self._append_audit(state, "REQUEST_EXIT", payload, eliminated=[])
        return approved, reason, self._snapshot(state), event_id

    def declare_conclusion(
        self,
        session_id: str,
        conclusion_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, BeliefSnapshot, str]:
        state = self._get(session_id)
        self._ensure_not_terminated(state)
        accepted = state.active_obligation_id is None
        reason = "accepted" if accepted else "active obligation blocks conclusion"
        if accepted and state.belief_state is None:
            if state.kernel is None:
                raise BeliefServerError(
                    code="CONFLICT",
                    message="Belief kernel unavailable for declare_conclusion.",
                )
            state.kernel.declare_conclusion(conclusion_id=conclusion_id)
        payload = {
            "conclusion_id": conclusion_id,
            "context": context or {},
            "accepted": accepted,
        }
        event_id = self._append_audit(state, "DECLARE_CONCLUSION", payload, eliminated=[])
        return accepted, reason, self._snapshot(state), event_id

    def request_termination(
        self,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, BeliefSnapshot, str]:
        state = self._get(session_id)
        self._ensure_not_terminated(state)
        if state.active_obligation_id is not None:
            approved = False
            reason = "active obligation blocks termination"
        else:
            force = bool((context or {}).get("force"))
            approved = (len(self._survivors(state)) == 1) or force
            reason = "approved" if approved else "termination requires singleton survivor"

        if approved:
            state.terminated = True

        payload = {
            "context": context or {},
            "approved": approved,
        }
        event_id = self._append_audit(state, "REQUEST_TERMINATION", payload, eliminated=[])
        return approved, reason, self._snapshot(state), event_id

    def _get(self, session_id: str) -> SessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise BeliefServerError(code="SESSION_NOT_FOUND", message="Session not found.")
        return state

    def _ensure_not_terminated(self, state: SessionState) -> None:
        if state.terminated:
            raise BeliefServerError(code="SESSION_TERMINATED", message="Session is terminated.")

    def _snapshot(self, state: SessionState) -> BeliefSnapshot:
        survivors = sorted(self._survivors(state))
        return BeliefSnapshot(
            session_id=state.session_id,
            ontology=state.ontology,
            survivors=survivors,
            terminated=state.terminated,
            active_obligation_id=state.active_obligation_id,
            audit_head_event_id=state.audit_head_event_id,
        )

    def _append_audit(
        self,
        state: SessionState,
        verb: str,
        payload: Dict[str, Any],
        eliminated: List[str],
        notes: Optional[str] = None,
        survivors_before: Optional[Set[str]] = None,
        survivors_after: Optional[Set[str]] = None,
    ) -> str:
        if survivors_before is None:
            survivors_before = set(self._survivors(state))
        if survivors_after is None:
            survivors_after = set(self._survivors(state))
        before_hash = _hash_survivors(
            survivors_before, state.session_id, state.audit_head_hash
        )
        after_hash = _hash_survivors(
            survivors_after, state.session_id, state.audit_head_hash
        )
        event_id = str(uuid.uuid4())
        entry = AuditEntry(
            event_id=event_id,
            ts=time.time(),
            verb=verb,
            payload=payload,
            survivors_before_hash=before_hash,
            survivors_after_hash=after_hash,
            delta={"eliminated": list(eliminated)},
            notes=notes,
        )
        state.audit.append(entry)
        state.audit_head_event_id = event_id
        state.audit_head_hash = after_hash
        return event_id

    def _survivors(self, state: SessionState) -> Set[str]:
        if state.belief_state is not None:
            return set(state.belief_state.survivors)
        if state.kernel is None:
            raise BeliefServerError(
                code="CONFLICT",
                message="Belief kernel unavailable for survivors.",
            )
        return set(state.kernel.survivors)


def _hash_survivors(survivors: Set[str], session_id: str, prev_event_hash: str) -> str:
    serialized = "\n".join(sorted(survivors))
    payload = f"{serialized}|{session_id}|{prev_event_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
