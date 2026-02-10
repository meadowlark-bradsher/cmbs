# Backlog

Updated: 2026-02-10

## Elimination Store SPI - Next Steps

- Implement `TeeStore` (gRPC client, mapping to Tee RPCs, universe anchoring at `CreateIncident()`).
- Add Tee integration test harness (requires running Tee + Neo4j).
- Add recovery replay tests for obligations, conclusions, and termination.
- Decide and document audit persistence backend for durable sessions (SQLite/Postgres/file).
- Add optional store consistency check in `BeliefServer.eliminate(...)` for diagnostics.
- Add store discovery or configuration pattern (explicit injection is fine; decide if a registry is needed).
