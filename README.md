# CMBS v0 Core

CMBS is a belief-state accounting system. It tracks hypotheses, eliminations, entropy, and obligation discipline using opaque identifiers. The core is domain-agnostic and replay-auditable.

CMBS does not:
- Execute or schedule probes
- Choose actions or control workflows
- Interpret observables or hypotheses

All semantics live in adapters.

## Quick Start

```python
from cmbs.core import CMBSCore

core = CMBSCore(hypothesis_ids={"H1", "H2", "H3"})
result = core.submit_probe_result(
    probe_id="P1",
    observable_id="O1",
    eliminated={"H1"},
)
assert result.accepted
```

## Legacy Replay Shim

For audit continuity with legacy logs, use the thin adapter in `cmbs.adapters.legacy`:

```python
from cmbs.core import CMBSCore
from cmbs.adapters.legacy import LegacyReplayAdapter, LegacyEliminationEvent

core = CMBSCore(hypothesis_ids={"H1", "H2"})
adapter = LegacyReplayAdapter(core)

adapter.submit_elimination_event(
    LegacyEliminationEvent(
        probe_id="legacy:probe:001",
        observable_id="legacy:obs:alpha",
        eliminated_hypotheses={"H2"},
    )
)
```

## Docs

- `docs/ARCHITECTURE.md`
- `docs/v0-test-specification.md`
- `docs/v0-implementation-summary.md`
- `docs/v0-core-contract-validation.md`
- `docs/ARCHIVE.md`
