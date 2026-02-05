"""
FastAPI transport for Belief Server v1 (in-memory).
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .belief_server import BeliefServer, BeliefServerError, OntologyBundle

app = FastAPI(title="Belief Server v1")
server = BeliefServer()


class OntologyBundleModel(BaseModel):
    hypothesis_space_id: str
    hypothesis_version: str
    causal_graph_ref: str
    causal_graph_version: str


class DeclareSessionRequest(BaseModel):
    ontology: OntologyBundleModel
    hypotheses: List[str]
    metadata: Optional[Dict[str, Any]] = None


class EliminateRequest(BaseModel):
    source_id: str
    observation_id: str
    eliminated: List[str]
    justification: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class EnterObligationRequest(BaseModel):
    obligation_id: str
    min_total_eliminations: int


class RequestExitRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None


class DeclareConclusionRequest(BaseModel):
    conclusion_id: str
    context: Optional[Dict[str, Any]] = None


class RequestTerminationRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None


ERROR_STATUS = {
    "SESSION_NOT_FOUND": 404,
    "OBLIGATION_NOT_FOUND": 404,
    "SESSION_TERMINATED": 409,
    "CONFLICT": 409,
    "INVALID_HYPOTHESIS_ID": 400,
}


@app.exception_handler(BeliefServerError)
async def _belief_error_handler(_, exc: BeliefServerError):
    status = ERROR_STATUS.get(exc.code, 400)
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


@app.post("/v1/sessions")
def declare_session(req: DeclareSessionRequest):
    ontology = OntologyBundle(
        hypothesis_space_id=req.ontology.hypothesis_space_id,
        hypothesis_version=req.ontology.hypothesis_version,
        causal_graph_ref=req.ontology.causal_graph_ref,
        causal_graph_version=req.ontology.causal_graph_version,
    )
    session_id, snapshot = server.declare_session(
        ontology=ontology,
        hypotheses=req.hypotheses,
        metadata=req.metadata,
    )
    return {"session_id": session_id, "snapshot": snapshot.to_dict()}


@app.post("/v1/sessions/{session_id}/eliminate")
def eliminate(session_id: str, req: EliminateRequest):
    applied, ignored, snapshot, event_id = server.eliminate(
        session_id=session_id,
        source_id=req.source_id,
        observation_id=req.observation_id,
        eliminated=req.eliminated,
        justification=req.justification,
    )
    return {
        "applied_eliminated": applied,
        "ignored_eliminated": ignored,
        "snapshot": snapshot.to_dict(),
        "audit_event_id": event_id,
    }


@app.get("/v1/sessions/{session_id}")
def query_belief(session_id: str):
    snapshot = server.query_belief(session_id=session_id)
    return snapshot.to_dict()


@app.get("/v1/sessions/{session_id}/audit")
def audit_trace(session_id: str, since_event_id: Optional[str] = None):
    events = server.audit_trace(session_id=session_id, since_event_id=since_event_id)
    return {"events": [entry.to_dict() for entry in events]}


@app.post("/v1/sessions/{session_id}/obligations")
def enter_obligation(session_id: str, req: EnterObligationRequest):
    snapshot, event_id = server.enter_obligation(
        session_id=session_id,
        obligation_id=req.obligation_id,
        min_total_eliminations=req.min_total_eliminations,
    )
    return {"snapshot": snapshot.to_dict(), "audit_event_id": event_id}


@app.post("/v1/sessions/{session_id}/obligations/{obligation_id}/exit")
def request_exit(session_id: str, obligation_id: str, req: RequestExitRequest):
    approved, reason, snapshot, event_id = server.request_exit(
        session_id=session_id,
        obligation_id=obligation_id,
        context=req.context,
    )
    return {
        "approved": approved,
        "reason": reason,
        "snapshot": snapshot.to_dict(),
        "audit_event_id": event_id,
    }


@app.post("/v1/sessions/{session_id}/conclusions")
def declare_conclusion(session_id: str, req: DeclareConclusionRequest):
    accepted, reason, snapshot, event_id = server.declare_conclusion(
        session_id=session_id,
        conclusion_id=req.conclusion_id,
        context=req.context,
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "snapshot": snapshot.to_dict(),
        "audit_event_id": event_id,
    }


@app.post("/v1/sessions/{session_id}/terminate")
def request_termination(session_id: str, req: RequestTerminationRequest):
    approved, reason, snapshot, event_id = server.request_termination(
        session_id=session_id,
        context=req.context,
    )
    return {
        "approved": approved,
        "reason": reason,
        "snapshot": snapshot.to_dict(),
        "audit_event_id": event_id,
    }
