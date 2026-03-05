# CMBS IR Snapshot: Integration Surface

## Minimal StateStore API (as-is)
CMBS already defines a minimal persistence SPI via `EliminationStore` (`cmbs/spi/elimination_store.py:36`):

1. `create_session(session_id: str, hypothesis_ids: FrozenSet[str]) -> None` (`cmbs/spi/elimination_store.py:39`)
2. `eliminate(session_id: str, eliminated: Set[str], provenance: EliminationProvenance) -> EliminationResult` (`cmbs/spi/elimination_store.py:46`)
3. `get_survivors(session_id: str) -> FrozenSet[str]` (`cmbs/spi/elimination_store.py:54`)
4. `get_eliminated(session_id: str) -> FrozenSet[str]` (`cmbs/spi/elimination_store.py:57`)
5. `recover(session_id: str) -> RecoveredState` (`cmbs/spi/elimination_store.py:60`)

Reference implementation: `InMemoryStore` (`cmbs/stores/memory.py:16`).

## How CMBS Uses StateStore Today
- Wiring point: `BeliefServer(..., store=...)` (`cmbs/belief_server.py:134`).
- On session declaration, store registration occurs for kernel-mode sessions (`cmbs/belief_server.py:173`-`cmbs/belief_server.py:177`).
- On elimination, store merge result becomes authoritative for applied vs ignored eliminations (`cmbs/belief_server.py:252`-`cmbs/belief_server.py:263`).
- Recovery composes store snapshot + audit stream (`restore_session`) (`cmbs/belief_server.py:286`-`cmbs/belief_server.py:362`).

## HTTP Availability
- HTTP is already available via FastAPI:
  - `app = FastAPI(...)` in `cmbs/belief_api.py:13`
  - route surface under `/v1/sessions...` starting at `cmbs/belief_api.py:80`
  - container startup command via uvicorn at `Dockerfile:12`
- Therefore, "add HTTP" is not needed for current CMBS.

## Design Notes (Smallest Later Extensions, No Behavior Change)
1. Durable store swap-in:
Implement `EliminationStore` against SQL/Redis and pass it to `BeliefServer(store=...)` without changing server method contracts (`cmbs/spi/elimination_store.py:36`, `cmbs/belief_server.py:129`).
2. Durable audit persistence:
Persist `AuditEntry.to_dict()` stream externally (current audit list is in-memory in `SessionState.audit`, `cmbs/belief_server.py:115`, `cmbs/belief_server.py:611`) and feed it back to `restore_session(...)`.
3. Optional recovery HTTP endpoints:
If remote recovery orchestration is required, expose `restore_session`/bootstrap operations in `belief_api.py` using existing server methods; this is additive and can preserve existing behavior/path semantics.
