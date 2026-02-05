---
tags: [cmbs/core, cmbs/invariant, spec]
aliases: [v0 Test Spec]
---
# CMBS v0 Test Specification

This document is generated from `tests/test_v0_core.py` and mirrors the test-case descriptions used to validate the v0 core.

## Overview

CMBS v0 Core Test Suite

Generated from: docs/v0-test-specification.md
Date: 2026-02-02

All tests use opaque identifiers and avoid domain semantics.
Assumes core API exists and is implemented.

## INV-3: A probe executed within an obligation scope may not be re-executed.

### T-INV3-01: Unique probes accepted (Positive)


Setup: Empty core with hypothesis set {H1, H2, H3}
Action: Submit probe results with distinct probe IDs: P1, P2, P3
Assert: All three submissions are accepted
Protects: INV-3 — core accepts unique probes

### T-INV3-02: Duplicate probe rejected (Negative)


Setup: Core with hypothesis set {H1, H2, H3}; P1 already consumed
Action: Submit probe result with probe ID P1 (duplicate)
Assert: Submission is rejected
Protects: INV-3 — core rejects duplicate probes

### T-INV3-03: Probe uniqueness is per-ID, not per-content (Positive)


Setup: Empty core with hypothesis set {H1, H2, H3}
Action: Submit two probe results:
    - P1 with observable O1, eliminating {H1}
    - P2 with observable O1, eliminating {H2}
    (Same observable, different probe IDs)
Assert: Both submissions are accepted
Protects: INV-3 — uniqueness is determined by probe ID, not observable content

### T-INV3-04: Probe rejection does not affect hypothesis state (Negative)


Setup: Core with hypothesis set {H1, H2, H3}; P1 already consumed, eliminated {H1}
Action: Submit duplicate P1, claiming to eliminate {H2}
Assert:
    - Submission is rejected
    - Survivor set remains {H2, H3} (not {H3})
Protects: INV-3 — rejected probes have no side effects

### T-INV3-05: Probe IDs are order-independent for uniqueness


Setup: Empty core with hypothesis set {H1, H2, H3, H4}
Action: Submit probes in order: P3, P1, P4, P2 (non-sequential)
Assert: All four are accepted
Protects: INV-3 — no ordering assumption on probe IDs

## INV-5a: Belief uncertainty must be computable as a scalar at each step.

### T-INV5A-01: Initial entropy matches hypothesis count (Positive)


Setup: Core initialized with hypothesis set of size N
Action: Query entropy before any eliminations
Assert: Entropy equals log2(N)
Protects: INV-5a — entropy reflects survivor count

### T-INV5A-02: Entropy decreases after elimination (Positive)


Setup: Core with hypothesis set {H1, H2, H3, H4} (entropy = 2.0)
Action: Submit probe eliminating {H1}
Assert: Entropy equals log2(3) ≈ 1.585
Protects: INV-5a — entropy updates on elimination

### T-INV5A-03: Entropy is zero with singleton survivor (Positive)


Setup: Core with hypothesis set {H1, H2}
Action: Submit probe eliminating {H1}
Assert: Entropy equals log2(1) = 0
Protects: INV-5a — entropy reaches zero at singleton

### T-INV5A-04: Entropy unchanged when no elimination occurs (Positive)


Setup: Core with hypothesis set {H1, H2, H3}
Action: Submit probe with empty elimination set {}
Assert: Entropy remains log2(3)
Protects: INV-5a — entropy reflects actual eliminations only

### T-INV5A-05: Entropy is monotonically non-increasing (Positive)


Setup: Core with hypothesis set {H1, H2, H3, H4, H5}
Action: Submit sequence of probes, each eliminating one hypothesis
Assert: Entropy after each step is <= entropy before
Protects: INV-5a + hard-elimination constraint — entropy cannot increase

## INV-6: An epistemic obligation may only be exited if belief has measurably changed.

### T-INV6-01: Obligation exit permitted after elimination (Positive)


Setup: Core with active obligation O1 (min_eliminations=1)
Action:
    1. Submit probe eliminating {H1}
    2. Request exit from O1
Assert: Exit is permitted
Protects: INV-6 — elimination satisfies obligation

### T-INV6-02: Obligation exit rejected with zero eliminations (Negative)


Setup: Core with active obligation O1 (min_eliminations=1)
Action: Request exit from O1 (no probes submitted during obligation)
Assert: Exit is rejected
Protects: INV-6 — obligations require substantive change

### T-INV6-03: Obligation exit respects threshold (Negative)


Setup: Core with active obligation O1 (min_eliminations=3)
Action:
    1. Submit probe eliminating {H1}
    2. Submit probe eliminating {H2}
    3. Request exit from O1
Assert: Exit is rejected (only 2 eliminations, threshold is 3)
Protects: INV-6 — threshold enforcement is exact

### T-INV6-04: Obligation exit permitted at threshold (Positive)


Setup: Core with active obligation O1 (min_eliminations=2)
Action:
    1. Submit probe eliminating {H1}
    2. Submit probe eliminating {H2}
    3. Request exit from O1
Assert: Exit is permitted
Protects: INV-6 — threshold is >=, not >

### T-INV6-05: Eliminations outside obligation don't count (Negative)


Setup: Core with hypothesis set {H1, H2, H3, H4}
Action:
    1. Submit probe eliminating {H1} (no obligation active)
    2. Enter obligation O1 (min_eliminations=1)
    3. Request exit from O1
Assert: Exit is rejected (elimination occurred before obligation)
Protects: INV-6 — elimination counting is scoped to obligation

### T-INV6-06: Multiple obligations track independently (Positive)


Setup: Core with hypothesis set {H1, H2, H3, H4, H5}
Action:
    1. Enter obligation O1 (min_elim=2)
    2. Eliminate H1
    3. Enter obligation O2 (min_elim=1)
    4. Eliminate H2
    5. Request exit O2 -> permitted (1 elim during O2)
    6. Request exit O1 -> permitted (2 elim during O1)
Assert: Both exits behave correctly based on their own scope
Protects: INV-6 — obligations are independent scopes

## Boundary tests: core must not interpret identifier content.

### T-BND-01: Hypothesis IDs are opaque (arbitrary strings accepted)


Setup: Initialize core with hypothesis set using arbitrary ID formats:
    {"uuid-1234-5678", "numeric_99", "🔬", "", "with spaces", "UPPERCASE"}
Action: Submit probes eliminating various hypotheses
Assert: Core operates correctly regardless of ID format
Protects: Boundary — no hypothesis ID interpretation

### T-BND-02: Probe IDs are opaque (arbitrary strings accepted)


Setup: Core with hypothesis set {H1, H2}
Action: Submit probes with arbitrary ID formats:
    "probe-alpha", "12345", "émoji🎯", ""
Assert: All unique probe IDs are accepted
Protects: Boundary — no probe ID interpretation

### T-BND-03: Observable IDs are opaque (arbitrary strings accepted)


Setup: Core with hypothesis set {H1, H2, H3}
Action: Submit probe with observable ID "anything_at_all_here"
Assert: Core accepts and processes normally
Protects: Boundary — no observable ID interpretation

### T-BND-04: Conclusion IDs are opaque (arbitrary strings accepted)


Setup: Core with stability tracking enabled
Action: Declare conclusions with arbitrary IDs:
    "conclusion_X", "42", "true", "compliant", "unknown"
Assert: Core tracks stability regardless of ID content
Protects: Boundary — no conclusion ID interpretation

### T-BND-05: IDs that "look like" domain concepts are not special-cased


Setup: Core with hypothesis set {"compliant", "non_compliant", "error"}
Action: Eliminate hypothesis "compliant"
Assert: Core treats this as ordinary elimination, no special behavior
Protects: Boundary — no domain concept recognition

## Boundary tests: termination must not assume singleton survivor.

### T-BND-06: Termination permitted with multiple survivors


Setup: Core with hypothesis set {H1, H2, H3}, stability window = 2
Action:
    1. Declare conclusion C1
    2. Declare conclusion C1 (stable for 2 steps)
    3. Request termination
Assert: Termination permitted (stability satisfied, survivor count irrelevant)
Protects: Boundary — no singleton assumption (Risk L6)

### T-BND-07: Termination permitted with zero survivors


Setup: Core with hypothesis set {H1}, stability window = 2
Action:
    1. Eliminate H1 (zero survivors)
    2. Declare conclusion C1
    3. Declare conclusion C1
    4. Request termination
Assert: Termination permitted (stability satisfied)
Protects: Boundary — zero-survivor termination is valid

## Boundary tests: conclusions must not be assumed binary.

### T-BND-08: More than two distinct conclusions supported


Setup: Core with stability tracking, window = 3
Action: Declare conclusions in sequence: C1, C2, C3, C4, C5
Assert: Core tracks all five without error
Protects: Boundary — no binary conclusion assumption (Risk L2)

### T-BND-09: Conclusion stability works for any conclusion value


Setup: Core with stability window = 3
Action:
    1. Declare C1, C1, C1 (stable)
    2. Request termination
Assert: Termination permitted
Action (continued):
    3. New session: Declare C99, C99, C99 (stable)
    4. Request termination
Assert: Termination permitted (C99 is treated same as C1)
Protects: Boundary — all conclusions are equal to core

## Boundary tests: core must not assume execution model.

### T-BND-10: No "attempt" or "success" states exist


Setup: Query core for available states/enums
Assert: No state named "attempted," "successful," "failed," "pending," or similar
Protects: Boundary — no execution model assumption (Risk L1)

### T-BND-11: Probe results are observations, not executions


Setup: Core with hypothesis set {H1, H2, H3}
Action: Submit probe result (no "execute" API exists)
Assert: Core accepts via submit_probe_result or equivalent observation-framed API
Protects: Boundary — vocabulary is observational, not agentic

### T-BND-12: Core has no probe scheduling capability


Setup: Inspect core API
Assert: No method exists to "execute probe," "run probe," "schedule probe," or "select next probe"
Protects: Boundary — probe execution is outside core

## Obligation lifecycle and scoping tests.

### T-OBL-01: Obligation entry is adapter-initiated


Setup: Core with no active obligations
Action: Adapter calls enter_obligation(O1, params)
Assert: Obligation O1 is now active
Protects: Boundary — adapter controls obligation lifecycle

### T-OBL-02: Obligation cannot self-trigger


Setup: Core with hypothesis set, no obligations
Action: Submit probes causing eliminations
Assert: No obligation becomes active without adapter explicitly entering it
Protects: Boundary — core doesn't decide when obligations trigger

### T-OBL-03: Obligation parameters are adapter-provided


Setup: Core ready for obligation
Action: Enter obligation with adapter-specified parameters:
    enter_obligation("O1", min_eliminations=5, entropy_threshold=0.5)
Assert: Core uses exactly these parameters, not defaults
Protects: Boundary — no hardcoded thresholds (Risk L4)

### T-OBL-04: Nested obligations are independent


Setup: Core with hypothesis set {H1, H2, H3, H4, H5}
Action:
    1. Enter O1 (min_elim=2)
    2. Eliminate H1
    3. Enter O2 (min_elim=1)
    4. Eliminate H2
    5. Request exit O2 -> permitted (1 elim during O2)
    6. Request exit O1 -> permitted (2 elim during O1)
Assert: Both exits behave correctly based on their own scope
Protects: INV-6 — obligation scopes are independent

### T-OBL-05: Exiting non-existent obligation fails gracefully


Setup: Core with no active obligations
Action: Request exit from "O_nonexistent"
Assert: Request fails or returns "not active" (no crash, no side effects)
Protects: Robustness — invalid requests handled cleanly

## Entropy is observable but does not gate operations.

### T-ENT-01: Entropy is queryable at any time


Setup: Core with hypothesis set
Action: Query entropy before, during, and after eliminations
Assert: Entropy value returned at each query
Protects: INV-5a — entropy is observable

### T-ENT-02: Entropy does not gate termination


Setup: Core with hypothesis set {H1, H2, H3, H4}, stability window = 2
Action:
    1. Declare C1, C1 (stable)
    2. Request termination (entropy = log2(4) = 2.0, high uncertainty)
Assert: Termination permitted (entropy is high but irrelevant)
Protects: Boundary — entropy is diagnostic, not a gate (Ambiguity A resolution)

### T-ENT-03: Entropy does not gate obligation exit


Setup: Core with active obligation (min_eliminations=1)
Action:
    1. Eliminate one hypothesis
    2. Request obligation exit (entropy still high)
Assert: Exit permitted (elimination threshold met, entropy irrelevant)
Protects: INV-6 — obligation exit is elimination-gated, not entropy-gated

### T-ENT-04: Zero entropy does not auto-terminate


Setup: Core with hypothesis set {H1, H2}, stability window = 2
Action:
    1. Eliminate H1 (entropy = 0, singleton survivor)
    2. Do NOT request termination
Assert: Core remains active, no auto-termination
Protects: Boundary — termination requires explicit adapter request

## Termination lifecycle and stability tests.

### T-TERM-01: Termination requires explicit adapter request


Setup: Core with hypothesis set, stability window = 2, conclusion stable
Action: Do not call request_termination
Assert: Core remains active indefinitely
Protects: Boundary — adapter controls termination

### T-TERM-02: Termination rejected if conclusion unstable


Setup: Core with stability window = 3
Action:
    1. Declare C1
    2. Declare C2 (changed)
    3. Request termination
Assert: Termination rejected (conclusion not stable)
Protects: INV-2 — stability is enforced

### T-TERM-03: Termination permitted after stability achieved


Setup: Core with stability window = 3
Action:
    1. Declare C1, C1, C1
    2. Request termination
Assert: Termination permitted
Protects: INV-2 — stability gating works correctly

### T-TERM-04: Stability resets on conclusion change


Setup: Core with stability window = 3
Action:
    1. Declare C1, C1 (2 stable steps)
    2. Declare C2 (change)
    3. Request termination
Assert: Termination rejected (stability reset by C2)
Protects: INV-2 — stability window is rolling, not cumulative

### T-TERM-05: Termination with stability disabled


Setup: Core initialized with stability_window = 0 (disabled)
Action:
    1. Declare C1
    2. Request termination immediately
Assert: Termination permitted (no stability requirement)
Protects: Configuration — stability is optional (INV-2 is optional)

### T-TERM-06: Termination independent of obligation state


Setup: Core with active obligation O1
Action:
    1. Declare C1, C1, C1 (stable)
    2. Request termination (obligation still active)
Assert: Termination permitted (obligation is separate from termination)
Protects: Boundary — termination and obligations are orthogonal

## Tests for strangler fig migration support.

### T-MIG-01: Core accepts legacy-format IDs


Setup: Initialize core with IDs matching legacy system format
Action: Perform standard operations
Assert: Core functions correctly
Protects: Migration — legacy ID formats are valid opaque IDs

### T-MIG-02: Core operates with partial hypothesis elimination


Setup: Core with large hypothesis set (100 hypotheses)
Action: Eliminate only 5 hypotheses, then query state
Assert: Core correctly reports 95 survivors, correct entropy
Protects: Migration — incremental elimination is valid

### T-MIG-03: Core state is serializable


Setup: Core with hypothesis set, some eliminations, active obligation
Action: Serialize core state, deserialize into new instance
Assert: New instance has identical state
Protects: Migration — state can be checkpointed and transferred

### T-MIG-04: Adapter can replay historical eliminations


Setup: Empty core
Action: Submit batch of historical elimination events in order
Assert: Core reaches expected state (correct survivors, correct entropy)
Protects: Migration — historical events can bootstrap core state

## Tests for running core alongside legacy systems.

### T-MIG-05: Core can run alongside legacy system


Setup: Core receives same elimination events as legacy system (shadow mode)
Action: Compare core entropy with legacy entropy calculation
Assert: Values match (within floating-point tolerance)
Protects: Migration — core can validate against legacy before cutover

### T-MIG-06: Core provides audit trail for comparison


Setup: Core with elimination history
Action: Query full elimination history
Assert: Core returns ordered list of (probe_id, observable_id, eliminated_hypotheses) events
Protects: Migration — audit trail enables legacy comparison

## Related

- [[ARCHITECTURE]]
- [[v0-implementation-summary]]
- [[v0-core-contract-validation]]