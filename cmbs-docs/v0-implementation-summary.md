---
tags: [cmbs/core, summary]
aliases: [v0 Implementation Summary]
---
# CMBS v0 Core Implementation Summary

**Date:** 2026-02-02
**Status:** Complete (49/49 tests passing)

---

## Overview

CMBS v0 Core is a domain-agnostic belief-state accounting system. It tracks hypotheses, counts eliminations, measures entropy, and enforces non-repetition. It does not know what hypotheses mean, what probes do, or when to terminate. All semantics live in adapters.

**Core provides mechanism; adapters provide policy.**

---

## Implementation Approach

The implementation followed a test-first, phased approach using the authoritative test specification (`docs/v0-test-specification.md`). Each phase targeted specific invariants and made only the minimal changes required to pass those tests.

| Phase | Target | Tests |
|-------|--------|-------|
| 1 | Core initialization, entropy, basic probe submission | 6 |
| 2 | Probe non-repetition (INV-3) | 5 |
| 3 | Non-trivial exit / obligations (INV-6) | 6 |
| 4 | Boundary integrity (opaque IDs, non-singleton, non-binary, non-execution) | 12 |
| 5 | Obligation discipline, entropy observation | 9 |
| 6 | Termination discipline (INV-2) | 6 |
| 7 | Migration support (serialization, audit trail) | 6 |

---

## API Surface

### Data Types

```python
@dataclass
class ProbeResult:
    accepted: bool
    error: Optional[str] = None

@dataclass
class ObligationExitResult:
    permitted: bool
    error: Optional[str] = None

@dataclass
class TerminationResult:
    permitted: bool
    error: Optional[str] = None

@dataclass
class EliminationEvent:
    probe_id: str
    observable_id: str
    eliminated: Set[str]
```

### CMBSCore Class

#### Constructor

```python
CMBSCore(
    hypothesis_ids: Set[str],    # Initial hypothesis universe
    stability_window: int = 0,   # 0 = disabled
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `survivors` | `Set[str]` | Current surviving hypothesis IDs |
| `entropy` | `float` | `log₂(|survivors|)`, 0 if ≤1 survivor |
| `consumed_probes` | `Set[str]` | Probe IDs already submitted |
| `active_obligations` | `Set[str]` | Currently active obligation IDs |
| `is_terminated` | `bool` | Whether termination was granted |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `submit_probe_result(probe_id, observable_id, eliminated)` | `ProbeResult` | Submit observation; rejects duplicates (INV-3) |
| `enter_obligation(obligation_id, min_eliminations)` | `None` | Enter epistemic obligation |
| `request_obligation_exit(obligation_id)` | `ObligationExitResult` | Request exit; requires eliminations (INV-6) |
| `is_obligation_active(obligation_id)` | `bool` | Check if obligation is active |
| `declare_conclusion(conclusion_id)` | `None` | Declare current conclusion for stability tracking |
| `request_termination()` | `TerminationResult` | Request termination; requires stability (INV-2) |
| `get_elimination_history()` | `List[EliminationEvent]` | Audit trail of eliminations |
| `serialize()` | `Any` | Checkpoint state |
| `deserialize(state)` | `CMBSCore` | Restore from checkpoint |

---

## Invariants Enforced

### Required (v0)

| ID | Name | Enforcement |
|----|------|-------------|
| INV-3 | Probe Non-Repetition | `submit_probe_result` rejects duplicate probe IDs |
| INV-5a | Entropy Quantification | `entropy` property computes `log₂(|survivors|)` |
| INV-6 | Non-Trivial Exit | `request_obligation_exit` requires `min_eliminations` within scope |

### Optional (v0)

| ID | Name | Enforcement |
|----|------|-------------|
| INV-2 | Stability Window | `request_termination` requires N consecutive identical conclusions |

---

## Boundary Guarantees

The implementation maintains strict boundaries per the specification:

| Guarantee | How Enforced |
|-----------|--------------|
| **Opaque identifiers** | All IDs are strings; no parsing, no special-casing |
| **No singleton assumption** | Termination permitted with any survivor count |
| **No binary conclusion assumption** | Any conclusion ID accepted |
| **No execution model** | No "attempt/success" states; vocabulary is observational |
| **No probe scheduling** | No methods to execute, run, or select probes |
| **No hardcoded thresholds** | All thresholds are adapter-provided parameters |
| **No auto-termination** | Termination requires explicit adapter request |
| **Entropy is diagnostic** | Entropy does not gate termination or obligation exit |

---

## Internal State

```python
_survivors: Set[str]                           # Current hypothesis survivors
_consumed_probes: Set[str]                     # Probe IDs already used
_stability_window: int                         # Required consecutive conclusions
_conclusion_history: List[str]                 # Conclusion sequence
_terminated: bool                              # Termination granted flag
_obligations: Dict[str, int]                   # obligation_id -> min_eliminations
_obligation_elimination_counts: Dict[str, int] # obligation_id -> elimination count
_elimination_history: List[EliminationEvent]   # Audit trail
```

---

## Test Coverage

### By Category

| Category | Count |
|----------|-------|
| INV-3 (Probe Non-Repetition) | 5 |
| INV-5a (Entropy) | 5 |
| INV-6 (Non-Trivial Exit) | 6 |
| Boundary: Opaque IDs | 5 |
| Boundary: Non-singleton | 2 |
| Boundary: Non-binary | 2 |
| Boundary: Non-execution | 3 |
| Obligation discipline | 5 |
| Entropy observation | 4 |
| Termination discipline | 6 |
| Migration support | 6 |
| **Total** | **49** |

### Key Properties Verified

1. Probe uniqueness is ID-based, not content-based
2. Entropy is observable but does not gate operations
3. Obligations require substantive change (eliminations) to exit
4. Termination requires explicit adapter request + stability
5. All identifiers are opaque to core
6. No domain semantics leak into core behavior
7. State is serializable for migration support

---

## Files

| File | Purpose |
|------|---------|
| `cmbs/core.py` | Core implementation (~180 lines) |
| `tests/test_v0_core.py` | Test suite (49 tests) |
| `docs/v0-test-specification.md` | Authoritative test specification |
| `docs/v0-core-contract-validation.md` | Contract validation and invariant analysis |

---

## What Core Does NOT Do

Per the boundary specification, the following are explicitly outside core:

- Probe execution or scheduling
- Observable interpretation
- Hypothesis semantics
- Conclusion semantics
- Probe ordering or selection
- Obligation type definitions
- Compatibility tables
- Elimination validation logic
- Threshold values (all adapter-provided)

These responsibilities belong to adapters, which implement domain-specific policy on top of core's domain-agnostic mechanism.

## Related

- [[ARCHITECTURE]]
- [[v0-test-specification]]
- [[v0-core-contract-validation]]
