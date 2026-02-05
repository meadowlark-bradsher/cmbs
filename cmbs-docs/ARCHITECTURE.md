---
tags: [cmbs/architecture, cmbs/core]
aliases: [Architecture]
---
# CMBS Architecture

CMBS v0 is a belief-state accounting core with a strict boundary between mechanism and policy.

## Core vs Adapter Boundary

Core responsibilities:
- Track surviving hypotheses
- Record eliminations and audit history
- Enforce probe non-repetition (INV-3)
- Quantify entropy over hypothesis cardinality (INV-5a)
- Enforce non-trivial obligation exit (INV-6)
- Optionally enforce conclusion stability for termination

Adapter responsibilities:
- Provide hypothesis IDs and thresholds
- Translate legacy or domain events into core calls
- Decide when to declare conclusions or request termination
- Interpret observables and domain semantics

## Invariants Enforced by Core

Required (v0):
- INV-3: Probe non-repetition
- INV-5a: Entropy computed as log2(|survivors|)
- INV-6: Obligation exit requires eliminations within scope

Optional (v0):
- Stability window for termination (adapter-provided)

## Out of Scope

The core does not:
- Execute or schedule probes
- Choose actions or manage control loops
- Interpret observables, hypotheses, or conclusions
- Maintain workflow state (capabilities, repairs, affordances)
- Provide domain-specific thresholds or rules

All semantics and workflow logic live in adapters.

## Related

- [[v0-test-specification]]
- [[v0-implementation-summary]]
- [[v0-core-contract-validation]]
- [[ARCHIVE]]
