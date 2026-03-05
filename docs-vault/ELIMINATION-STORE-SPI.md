---
tags: [cmbs/architecture, cmbs/spi, cmbs/contract]
aliases: [Elimination Store SPI, Tee SPI]
---
# Elimination Store SPI

**Date:** 2026-02-09
**Status:** Design spec (not yet implemented)
**Depends on:** v0 Core (complete), Belief Server v1 (complete)

---

## 0. Motivation

CMBS v0 stores all state in-process: `CMBSCore._survivors` is a Python `set`,
`BeliefServer._sessions` is a `dict`. This is correct for bounded use cases
(Twenty Questions, ITBench kits) where the hypothesis universe is small, static,
and ephemeral.

For the Tee + Neo4j use case, the hypothesis universe is:

- **Large** — thousands of nodes and edges in a causal graph
- **Dynamic** — the Join Phase grows the graph continuously
- **Shared** — multiple incidents reason over the same graph
- **Durable** — eliminations must survive process restarts

The Elimination Store SPI abstracts **where tombstones live** so that CMBS can
work with either backend transparently. In-memory is one implementation. Tee is
another. CMBS core invariants (INV-3, INV-5a, INV-6) are enforced locally
regardless of backend.

---

## 1. SPI Placement

CMBS already has two SPI layers on the **input** side:

| Existing SPI | Concern | Example |
|---|---|---|
| `HypothesisProvider` | Where hypotheses come from | ITBench kit, 20Q kit |
| `BeliefAdapter` | How domain events become eliminations | `ITBenchAdapter`, `TwentyQAdapter` |

The Elimination Store SPI is on the **persistence** side:

| New SPI | Concern | Example |
|---|---|---|
| `EliminationStore` | Where tombstones live | `InMemoryStore`, `TeeStore` |

### Layer diagram

```
  BeliefAdapter        (input: domain events → elimination messages)
       │
  BeliefServer         (protocol: sessions, audit, invariant enforcement)
       │
  EliminationStore     (persistence: where tombstones are stored)    ← NEW
     ┌────┴────┐
  InMemory    Tee
  (current)   (gRPC → Neo4j)
```

The SPI is injected at the `BeliefServer` level. `CMBSCore` does not change.

---

## 2. Protocol Definition

### 2.1 Types

```python
@dataclass(frozen=True)
class EliminationProvenance:
    """Opaque provenance attached to an elimination.
    TeeStore maps this to Tee's Provenance message.
    InMemoryStore ignores it (or stores it in the audit trail)."""
    source_id: str
    trigger: str


@dataclass(frozen=True)
class EliminationResult:
    """Authoritative result of a tombstone merge."""
    applied: FrozenSet[str]             # newly eliminated (first write)
    already_eliminated: FrozenSet[str]  # tombstone already existed (idempotent no-op)


@dataclass(frozen=True)
class RecoveredState:
    """Everything CMBS needs to resume a session after restart."""
    hypothesis_ids: FrozenSet[str]   # full universe at anchor
    eliminated: FrozenSet[str]       # current tombstone set
    survivors: FrozenSet[str]        # universe minus eliminated
```

### 2.2 Protocol

```python
class EliminationStore(Protocol):
    """SPI for elimination persistence.

    CMBS enforces invariants (INV-3, INV-6) locally.
    This interface handles where the tombstones live.
    """

    def create_session(
        self,
        session_id: str,
        hypothesis_ids: FrozenSet[str],
    ) -> None:
        """Register a new elimination context.

        For InMemoryStore: stores the universe and initializes empty tombstone set.
        For TeeStore: calls Tee.CreateIncident(). The hypothesis universe is
        already in the main graph; the universe_anchor is set by Tee.

        Universe anchoring (TeeStore):
        - BeliefServer already has OntologyBundle.causal_graph_version
        - Tee records this as the universe_anchor at CreateIncident() time
        - No explicit parameter is required in this SPI
        """
        ...

    def eliminate(
        self,
        session_id: str,
        eliminated: Set[str],
        provenance: EliminationProvenance,
    ) -> EliminationResult:
        """Persist tombstones. Idempotent.

        Returns authoritative classification of each ID:
        applied (new) vs already_eliminated (no-op).

        Unknown hypothesis IDs are permitted and treated as an idempotent no-op.
        """
        ...

    def get_survivors(self, session_id: str) -> FrozenSet[str]:
        """Current survivor set (universe minus tombstones)."""
        ...

    def get_eliminated(self, session_id: str) -> FrozenSet[str]:
        """Current tombstone set."""
        ...

    def recover(self, session_id: str) -> RecoveredState:
        """Recover full state for a session.

        Used on CMBS restart to reinitialize CMBSCore.
        """
        ...
```

Five methods. No optional parameters. No configuration in the protocol itself.

---

## 3. Implementations

### 3.1 `InMemoryStore`

Extracts the survivor-tracking behavior currently embedded in `CMBSCore._survivors`.

```python
class InMemoryStore:
    def __init__(self) -> None:
        self._universes: Dict[str, FrozenSet[str]] = {}
        self._tombstones: Dict[str, Set[str]] = {}

    def create_session(self, session_id, hypothesis_ids):
        self._universes[session_id] = frozenset(hypothesis_ids)
        self._tombstones[session_id] = set()

    def eliminate(self, session_id, eliminated, provenance):
        survivors = self._universes[session_id] - self._tombstones[session_id]
        applied = eliminated & survivors
        already = eliminated - survivors
        self._tombstones[session_id] |= applied
        return EliminationResult(
            applied=frozenset(applied),
            already_eliminated=frozenset(already),
        )

    def get_survivors(self, session_id):
        return frozenset(
            self._universes[session_id] - self._tombstones[session_id]
        )

    def get_eliminated(self, session_id):
        return frozenset(self._tombstones[session_id])

    def recover(self, session_id):
        return RecoveredState(
            hypothesis_ids=self._universes[session_id],
            eliminated=frozenset(self._tombstones[session_id]),
            survivors=self.get_survivors(session_id),
        )
```

This changes zero behavior. Existing tests pass without modification.

### 3.2 `TeeStore`

gRPC client to the Tee service. Depends on generated protobuf stubs.

```python
class TeeStore:
    def __init__(self, tee_endpoint: str) -> None:
        self._channel = grpc.insecure_channel(tee_endpoint)
        self._stub = TeeStub(self._channel)

    def create_session(self, session_id, hypothesis_ids):
        # CMBS session = Tee incident (1:1)
        # hypothesis_ids are already in the main graph;
        # Tee records the universe_anchor at creation time
        self._stub.CreateIncident(
            CreateIncidentRequest(incident_id=session_id)
        )

    def eliminate(self, session_id, eliminated, provenance):
        result = self._stub.MergeNodeTombstones(
            NodeTombstoneRequest(
                incident_id=session_id,
                node_ids=list(eliminated),
                provenance=Provenance(
                    source=provenance.source_id,
                    trigger=provenance.trigger,
                ),
            )
        )
        # TombstoneMergeResult is authoritative — no follow-up query needed
        return EliminationResult(
            applied=frozenset(result.applied_ids),
            already_eliminated=frozenset(result.already_tombstoned_ids),
        )

    def get_survivors(self, session_id):
        live = self._stub.GetLiveView(
            LiveViewRequest(incident_id=session_id)
        )
        return frozenset(n.id for n in live.nodes)

    def get_eliminated(self, session_id):
        tombstones = self._stub.GetTombstones(
            TombstoneRequest(incident_id=session_id)
        )
        return frozenset(t.node_id for t in tombstones.node_tombstones)

    def recover(self, session_id):
        survivors = self.get_survivors(session_id)
        eliminated = self.get_eliminated(session_id)
        return RecoveredState(
            hypothesis_ids=frozenset(survivors | eliminated),
            eliminated=eliminated,
            survivors=survivors,
        )
```

---

## 4. Integration with BeliefServer

### 4.1 Injection point

The only signature change in existing code:

```python
# Before
class BeliefServer:
    def __init__(self, validate_hypotheses=False, belief_provider=None):

# After
class BeliefServer:
    def __init__(self, validate_hypotheses=False, belief_provider=None,
                 store: Optional[EliminationStore] = None):
```

When `store` is `None`, behavior is identical to today (CMBSCore manages
survivors in-memory). When `store` is provided, BeliefServer writes through
to it.

Scope note:
- The EliminationStore applies only to CMBS-managed sessions created via
  `BeliefServer.declare_session(...)` (kernel-backed).
- Provider-side SPI sessions (`apply_probe(...)`) are adapter-local and do not
  write to the store.

### 4.2 Elimination flow (with store)

```
Agent calls BeliefServer.eliminate(session_id, source_id, observation_id, eliminated, ...)
  │
  ├─ 1. CMBSCore.submit_probe_result(probe_id, observable_id, eliminated)
  │      └─ INV-3 checked: reject if duplicate probe
  │      └─ survivors updated in-memory
  │      └─ obligation elimination counts updated
  │
  ├─ 2. if accepted AND store is not None:
  │      store.eliminate(session_id, eliminated, provenance)
  │      └─ tombstones persisted to backend
  │      └─ EliminationResult returned (authoritative for applied vs no-op)
  │
  └─ 3. audit entry appended (BeliefServer-local, as today)
```

CMBSCore is always called first. It is the invariant gatekeeper. The store is
called only after CMBSCore accepts. This means:

- INV-3 (probe non-repetition) is enforced before any write hits the backend
- INV-6 (obligation counting) is tracked locally, no backend round-trip needed
- The store never receives a write that CMBSCore rejected

### 4.3 Query flow (with store)

`BeliefServer._survivors()` currently reads from `CMBSCore.survivors` or
`BeliefState.survivors`. With a store, it could optionally read from
`store.get_survivors()` for consistency. However, for latency and simplicity,
the default is to read from CMBSCore (in-memory) since it is always kept in
sync by the write-through flow above.

The store's `get_survivors()` is used only for:
- Recovery (Section 6)
- Consistency checks (optional, diagnostic)
- External callers that bypass CMBS (e.g., Tee's own `GetLiveView`)

---

## 5. Session-to-Incident Mapping

| CMBS concept | Tee concept | Notes |
|---|---|---|
| `session_id` | `incident_id` | 1:1 mapping, same opaque string |
| `OntologyBundle.causal_graph_ref` | Main graph identity | Already in the data model |
| `OntologyBundle.causal_graph_version` | `universe_anchor` | Pins incident to graph state |
| `hypothesis_ids` | `GetMainGraph()` at creation | Frozen for this session |
| `eliminate()` | `MergeNodeTombstones()` | Tombstone write |
| `query_belief().survivors` | `GetLiveView()` | Universe minus tombstones |
| `audit_trace()` | CMBS-local | Tee does not own CMBS audit |

`OntologyBundle` already carries `causal_graph_ref` and `causal_graph_version`.
For the TeeStore path, `causal_graph_version` becomes the universe anchor — the
version of the main graph at the moment the incident was created.

---

## 6. Recovery

### 6.0 Recovery entry point (BeliefServer)

BeliefServer exposes a single explicit recovery hook to rehydrate CMBSCore
from store state plus audit history:

```python
def restore_session(
    self,
    session_id: str,
    recovered: RecoveredState,
    audit_events: Iterable[AuditEntry],
) -> None:
    """
    Rehydrate CMBSCore from persistent elimination state + audit log.
    """
```

Rules:
- `RecoveredState` comes from `EliminationStore.recover()`
- `AuditEntry` is the BeliefServer audit log entry type (or equivalent record)
- Audit replay is CMBS-local (not part of the store SPI)
- Elimination replay MUST NOT call `store.eliminate()` (no duplication)

### 6.1 What needs recovering

On CMBS restart, each active session needs:

| State | Where it lives | Recovery source |
|---|---|---|
| Survivor set | CMBSCore._survivors | `store.recover()` |
| Consumed probes | CMBSCore._consumed_probes | CMBS audit log replay |
| Obligation state | CMBSCore._obligations | CMBS audit log replay |
| Elimination history | CMBSCore._elimination_history | CMBS audit log replay |
| Stability window | CMBSCore._conclusion_history | CMBS audit log replay |
| Audit trail | BeliefServer SessionState.audit | Durable audit store |

The survivor set and tombstone set come from the `EliminationStore`. Everything
else is CMBS-local state recovered from the audit trail.

### 6.2 Recovery flow

```
CMBS starts up
  │
  ├─ For each known session_id:
  │     recovered = store.recover(session_id)
  │     kernel = CMBSCore(recovered.hypothesis_ids)
  │     replay audit log → restores consumed_probes, obligations, conclusions
  │     assert kernel.survivors == recovered.survivors  # consistency check
  │
  └─ Resume serving
```

### 6.3 Audit log durability

For InMemoryStore, the audit log is in-memory (lost on restart). This is
acceptable for ephemeral use cases.

For TeeStore, the audit log should be persisted. Options (not specified by
this SPI — orthogonal concern):

- SQLite/Postgres sidecar
- Append to a file (JSON lines)
- Store in Tee incident metadata (would require a Tee API extension)
- Store in a separate audit service

The EliminationStore SPI does not own the audit trail. It owns only
tombstones and universe state.

---

## 7. Invariant Preservation

### 7.1 No invariant changes

The EliminationStore SPI does not weaken or alter any v0 invariant.

| Invariant | Enforcement location | Changed by SPI? |
|---|---|---|
| INV-3 (Probe Non-Repetition) | CMBSCore.submit_probe_result | No — checked before store.eliminate() |
| INV-5a (Entropy) | CMBSCore.entropy property | No — computed from local survivor set |
| INV-6 (Non-Trivial Exit) | CMBSCore.request_obligation_exit | No — counts tracked locally |
| INV-2 (Stability Window) | CMBSCore.request_termination | No — conclusion history is local |

**The store is called only after invariant checks pass.** The store never
receives a write that CMBSCore would have rejected.

### 7.2 New consistency property

When a store is configured, the following must hold after every elimination:

```
store.get_survivors(session_id) == CMBSCore.survivors
```

This is maintained by the write-through flow (Section 4.2). On recovery, it
is verified by assertion (Section 6.2).

---

## 8. Opaque ID Handling: Nodes vs Edges

CMBS treats all hypothesis IDs as opaque strings. Tee has separate RPCs for
node tombstones (`MergeNodeTombstones`) and edge tombstones
(`MergeEdgeTombstones`). The SPI must bridge this without leaking Tee's
type system into CMBS core.

### 8.1 Approach: Node-only for v1

For the first implementation, CMBS eliminates **nodes only**. Edge elimination
is implicit — tombstoning a node removes all its edges from the live view.
This is already how Tee's live view query works:

```
Live edges = all edges
  MINUS edge-tombstoned
  MINUS edges whose source OR target is node-tombstoned
```

Edge tombstones exist in Tee for the rarer case of eliminating a specific
causal relationship while keeping both endpoint nodes alive. This is not
needed for the initial CMBS integration.

### 8.2 Future: Convention-based routing

When edge-level elimination is needed, `TeeStore.eliminate()` can route
based on ID format convention:

- Bare ID (`"svc-postgres"`) → `MergeNodeTombstones`
- Prefixed ID (`"edge:svc-postgres|svc-redis|DEPENDS_ON"`) → `MergeEdgeTombstones`

The convention is established by the adapter that provides the hypothesis
universe, not by CMBS core. Core continues to treat all IDs as opaque.
The `TeeStore` implementation parses the prefix for routing purposes only.

### 8.3 What this means for `hypothesis_ids`

For node-only v1:
```python
hypothesis_ids = frozenset(n.id for n in main_graph.nodes)
```

For node+edge:
```python
hypothesis_ids = (
    frozenset(n.id for n in main_graph.nodes)
    | frozenset(f"edge:{e.source}|{e.target}|{e.type}" for e in main_graph.edges)
)
```

CMBS core sees a flat set of strings either way.

---

## 9. What Changes in Existing Code

### 9.1 New files

| File | Contents |
|---|---|
| `cmbs/spi/elimination_store.py` | `EliminationStore` protocol, `EliminationResult`, `EliminationProvenance`, `RecoveredState` |
| `cmbs/stores/__init__.py` | Package |
| `cmbs/stores/memory.py` | `InMemoryStore` implementation |
| `cmbs/stores/tee.py` | `TeeStore` implementation (depends on generated gRPC stubs) |
| `tests/test_elimination_store.py` | Store protocol conformance tests (run against both implementations) |

### 9.2 Modified files

| File | Change |
|---|---|
| `cmbs/belief_server.py` | Accept optional `store` parameter. Write through to store after CMBSCore accepts. |
| `cmbs/__init__.py` | Export `EliminationStore`, `InMemoryStore`, new types. |

### 9.3 Unchanged files

| File | Why unchanged |
|---|---|
| `cmbs/core.py` | CMBSCore is the invariant enforcer. No persistence concern. |
| `cmbs/spi/hypothesis_provider.py` | Input SPI, orthogonal to persistence SPI. |
| `cmbs/adapters/types.py` | Adapter protocol unchanged. |
| `cmbs/belief_state.py` | SPI-based belief state unchanged. |
| `cmbs/belief_api.py` | Transport layer unchanged (passes `store` through to BeliefServer). |
| `tests/test_v0_core.py` | 49 core tests pass without modification. |

---

## 10. Store Protocol Conformance Tests

Both `InMemoryStore` and `TeeStore` must pass the same conformance suite:

| Test ID | Description |
|---|---|
| T-STORE-01 | `create_session` followed by `get_survivors` returns full universe |
| T-STORE-02 | `eliminate` reduces survivors by applied set |
| T-STORE-03 | `eliminate` same IDs twice → second call returns empty `applied`, full `already_eliminated` |
| T-STORE-04 | `eliminate` IDs not in universe → no effect on survivors |
| T-STORE-05 | `get_eliminated` returns cumulative tombstone set |
| T-STORE-06 | `get_survivors` ∪ `get_eliminated` ⊆ original `hypothesis_ids` |
| T-STORE-07 | `get_survivors` ∩ `get_eliminated` = ∅ |
| T-STORE-08 | `recover` returns consistent triple: `survivors = hypothesis_ids - eliminated` |
| T-STORE-09 | Concurrent eliminates from two threads produce valid final state (both applied) — **TeeStore-only / requires thread-safe store** |
| T-STORE-10 | `eliminate` with empty set is a no-op |

Tests T-STORE-01 through T-STORE-08 run against `InMemoryStore` in unit tests.
Tests T-STORE-01 through T-STORE-10 run against `TeeStore` in integration tests
(requires running Tee + Neo4j instance).

---

## 11. Acceptance Criteria

The SPI is complete when:

1. `EliminationStore` protocol is defined in `cmbs/spi/elimination_store.py`
2. BeliefServer exposes a recovery entry point (e.g., `restore_session(...)`)
3. `InMemoryStore` passes all conformance tests (T-STORE-01 through T-STORE-10)
4. `BeliefServer` accepts optional `store` parameter
5. All 49 existing v0 core tests pass without modification
6. All existing BeliefServer tests pass without modification
7. A new integration test demonstrates:
   - Create session with `InMemoryStore`
   - Eliminate hypotheses
   - Verify `store.get_survivors()` matches `CMBSCore.survivors`
   - Simulate restart: `store.recover()` → reinitialize → state matches
8. `TeeStore` passes conformance tests against a running Tee instance (integration, not required for merge)

---

## 12. Out of Scope

- **Audit log persistence.** The SPI owns tombstones, not audit entries. Audit durability is a separate concern.
- **Edge-level elimination.** v1 is node-only. Edge tombstone routing is deferred (Section 8.2).
- **Store discovery.** Unlike `HypothesisProvider` (which uses `importlib.metadata` entry points), store selection is explicit constructor injection. No plugin discovery needed.
- **Tee service implementation.** This spec covers the CMBS-side SPI only. Tee's own implementation is tracked in the Tee repo.
- **Soft belief / probabilistic elimination.** The SPI is hard-elimination only, consistent with v0 core contract.

---

## Related

- [[ARCHITECTURE]]
- [[BELIEF_SERVER_SPEC]]
- [[v0-core-contract-validation]]
- [[v0-implementation-summary]]
