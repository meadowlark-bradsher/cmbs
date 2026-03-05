# CMBS IR Snapshot: State And Events

## Belief/State Objects
- `CMBSCore` (`cmbs/core.py:42`): core state machine with `_survivors`, `_consumed_probes`, `_obligations`, `_obligation_elimination_counts`, `_elimination_history` (`cmbs/core.py:64`-`cmbs/core.py:71`).
- `BeliefState` (`cmbs/belief_state.py:17`): SPI-backed active map `_active` and total count `_total`; exposes `survivors` and `apply_probe` (`cmbs/belief_state.py:27`, `cmbs/belief_state.py:31`).
- `SessionState` (`cmbs/belief_server.py:105`): per-session runtime holder for kernel/SPI mode, obligations, audit list, audit head/hash, and observation dedupe index (`cmbs/belief_server.py:108`-`cmbs/belief_server.py:118`).
- Snapshot/audit DTOs:
  - `BeliefSnapshot` (`cmbs/belief_server.py:36`)
  - `AuditEntry` (`cmbs/belief_server.py:72`)
  - `ObservationRecord` (`cmbs/belief_server.py:98`)
- Persistence DTOs:
  - `RecoveredState` (`cmbs/spi/elimination_store.py:28`)
  - `EliminationResult` (`cmbs/spi/elimination_store.py:20`)
  - `EliminationProvenance` (`cmbs/spi/elimination_store.py:12`)

## Event Log Patterns
- Append-only in-memory audit list: `SessionState.audit` (`cmbs/belief_server.py:115`), appended via `_append_audit` (`cmbs/belief_server.py:580`).
- Audit entry fields include `event_id`, `ts`, `verb`, `payload`, before/after hashes, and `delta` (`cmbs/belief_server.py:73`-`cmbs/belief_server.py:80`).
- Event IDs are UUIDs from `uuid.uuid4()` (`cmbs/belief_server.py:600`).
- Verb set observed from append call sites:
  - `DECLARE_SESSION` (`cmbs/belief_server.py:193`)
  - `ELIMINATE` (`cmbs/belief_server.py:273`)
  - `APPLY_PROBE` (`cmbs/belief_server.py:406`)
  - `ENTER_OBLIGATION` (`cmbs/belief_server.py:465`)
  - `REQUEST_EXIT` (`cmbs/belief_server.py:506`)
  - `DECLARE_CONCLUSION` (`cmbs/belief_server.py:531`)
  - `REQUEST_TERMINATION` (`cmbs/belief_server.py:556`)
- Hash-chain style fields are computed using survivors, `session_id`, and previous head hash via `_hash_survivors` (`cmbs/belief_server.py:594`-`cmbs/belief_server.py:599`, `cmbs/belief_server.py:720`-`cmbs/belief_server.py:723`).
- Read pattern: `audit_trace(session_id, since_event_id)` supports whole-log or tail reads (`cmbs/belief_server.py:423`-`cmbs/belief_server.py:430`).

## Idempotency / Observation Keying
- Kernel-mode eliminate dedupe key: `observation_key = f"{source_id}::{observation_id}"` (`cmbs/belief_server.py:217`).
- SPI-mode apply_probe dedupe key uses response: `observation_id = f"{probe_id}:{response}"`, then same source+obs composite (`cmbs/belief_server.py:379`-`cmbs/belief_server.py:381`).
- Repeated observation returns previous `audit_event_id` without new effect (`cmbs/belief_server.py:218`-`cmbs/belief_server.py:225`, `cmbs/belief_server.py:381`-`cmbs/belief_server.py:388`).
- Test evidence for idempotent duplicate observation returning same event id: `tests/test_invariants.py:40`-`tests/test_invariants.py:64`.

## Persistence / Recovery Model
- Persistence SPI contract is `EliminationStore` with `create_session`, `eliminate`, `get_survivors`, `get_eliminated`, `recover` (`cmbs/spi/elimination_store.py:36`-`cmbs/spi/elimination_store.py:61`).
- Built-in store is in-memory only (`InMemoryStore` at `cmbs/stores/memory.py:16`) using `_universes` and `_tombstones` dicts (`cmbs/stores/memory.py:18`-`cmbs/stores/memory.py:19`).
- Store integration path:
  - `declare_session` calls `store.create_session(...)` in kernel mode (`cmbs/belief_server.py:173`-`cmbs/belief_server.py:177`).
  - `eliminate` calls `store.eliminate(...)` and trusts returned applied set (`cmbs/belief_server.py:252`-`cmbs/belief_server.py:263`).
- Recovery requires both recovered store state and audit event stream:
  - `restore_session(session_id, recovered, audit_events)` (`cmbs/belief_server.py:286`-`cmbs/belief_server.py:290`)
  - Rejects missing `DECLARE_SESSION` in audit (`cmbs/belief_server.py:304`-`cmbs/belief_server.py:309`)
  - Replays events via `_replay_audit_entry` (`cmbs/belief_server.py:345`-`cmbs/belief_server.py:347`, `cmbs/belief_server.py:616`)
  - Validates replayed survivors vs recovered survivors (`cmbs/belief_server.py:348`-`cmbs/belief_server.py:355`)
- Test evidence for store + audit round-trip restore: `tests/test_belief_server_store_integration.py:5`-`tests/test_belief_server_store_integration.py:44`.

## Run/Session IDs
- `session_id` is generated per declaration as UUID string (`cmbs/belief_server.py:147`).
- `audit_head_event_id` is tracked in session and snapshots (`cmbs/belief_server.py:42`, `cmbs/belief_server.py:577`, `cmbs/belief_server.py:612`).
- No explicit `run_id` symbol exists in active CMBS package; the execution identity primitive is `session_id` (verified via targeted symbol scan across `cmbs/` and `tests/`).
