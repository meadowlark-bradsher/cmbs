---
tags: [cmbs/core, contract, validation]
aliases: [v0 Contract Validation]
---
# CMBS v0 Core Contract Validation (Part I)

**Context:** Final validation of CMBS v0 Core Contract before scope finalization
**Scope:** Internal consistency, boundary integrity, minimality checks
**Date:** 2026-02-02

---

## Fixed Decisions (Non-Negotiable)

- CMBS core will be open source and strictly domain-agnostic.
- Domain hypothesis graphs, probe compatibility tables, and epistemic policies are private adapter logic.
- CMBS v0 supports:
  - Finite hypothesis sets
  - Hard elimination only
  - Entropy tracking
  - Read-only termination
  - Probe execution entirely outside core

---

## 1. Internal Consistency Check

### 1.1 Mutual Compatibility of Required Invariants

| Invariant Pair | Interaction | Compatible? |
|----------------|-------------|-------------|
| INV-3 ↔ INV-5a | Non-repetition tracks probe IDs; entropy tracks hypothesis count. No interaction. | ✓ Yes |
| INV-3 ↔ INV-6 | Non-repetition ensures probe diversity; non-trivial exit requires eliminations. Diverse probes increase elimination likelihood. Mutually reinforcing. | ✓ Yes |
| INV-5a ↔ INV-6 | Entropy measures survivor count; non-trivial exit counts eliminations. Elimination → entropy decrease. Consistent. | ✓ Yes |

**Result:** All required invariants are mutually compatible.

---

### 1.2 Domain Semantics Check

| Invariant | Domain Semantics Required? | Assessment |
|-----------|---------------------------|------------|
| INV-3 | No | Compares opaque ProbeIDs for equality. No interpretation of probe content. |
| INV-5a | No | Counts elements in hypothesis set. No interpretation of hypothesis meaning. |
| INV-6 | No | Counts elimination events. No interpretation of why elimination occurred. |

**Result:** No required invariant implicitly assumes domain semantics.

---

### 1.3 Constraint Compatibility Check

| Constraint | INV-3 | INV-5a | INV-6 |
|------------|-------|--------|-------|
| Hard-elimination (no re-entry) | ✓ Probe consumption is also monotonic | ✓ Entropy can only decrease | ✓ Elimination counting is monotonic |
| Read-only (no execution) | ✓ Core rejects duplicates, doesn't execute | ✓ Entropy is observed, not caused | ✓ Eliminations are reported by adapter |

**Result:** No required invariant contradicts hard-elimination or read-only constraints.

---

### 1.4 Contradictions or Ambiguities Found

#### Ambiguity A: Termination Gating Conditions

**Issue:** The boundary documentation mentions "Entropy + stability gates" for termination permission, but the recommended model has adapter-declared termination with core-enforced stability only.

**Clarification required:** Is entropy a termination gate, or only stability?

**Resolution (based on boundary principles):** Entropy should NOT be a core termination gate.

**Rationale:**
- Risk L6 (singleton termination model) warns against core assuming entropy conditions.
- Boundary principle: "Termination condition is adapter-declared, not core-computed."
- If adapter wants entropy-gated termination, it incorporates that into its private conclusion derivation.

**Corrected model:**
1. Adapter declares conclusion at each step.
2. Core tracks stability of conclusion.
3. Adapter requests termination.
4. Core permits termination if conclusion has been stable for N steps.
5. Core does NOT gate termination on entropy level.

**Status:** This is the only ambiguity found. With the above resolution, the contract is internally consistent.

---

## 2. Boundary Integrity Check

### 2.1 Core Responsibility Verification

| Core Responsibility | Implementation Uses Only Opaque IDs? | Requires Probe/Observable/Hypothesis Semantics? | Status |
|---------------------|-------------------------------------|-----------------------------------------------|--------|
| Hypothesis set cardinality tracking | ✓ Counts `Set[HypothesisID]` | No | ✓ Clean |
| Entropy computation | ✓ `log₂(\|set\|)` on opaque ID set | No | ✓ Clean |
| Probe ID consumption tracking | ✓ `ProbeID ∈ consumed_set` | No | ✓ Clean |
| Conclusion stability window | ✓ `ConclusionID` sequence comparison | No | ✓ Clean |
| Elimination counting | ✓ Counts `HypothesisID` removal events | No | ✓ Clean |
| Obligation lifecycle | ✓ `ObligationID` + numeric parameters | No | ✓ Clean |
| Convergence computation | ✓ Slope of entropy (numeric) over time | No | ✓ Clean |
| Threshold comparison | ✓ Numeric comparison only | No | ✓ Clean |

**Result:** All core responsibilities can be implemented using only opaque identifiers. None require inspecting probe content, observable meaning, or hypothesis semantics.

---

### 2.2 Flags

**None.** No core responsibility silently assumes more than opaque identifiers and numeric parameters.

---

## 3. Minimality Check

### 3.1 Could Anything Be Removed Without Weakening Epistemic Guarantees?

| Component | Remove? | Impact if Removed | Verdict |
|-----------|---------|-------------------|---------|
| INV-3 (Probe Non-Repetition) | No | Obligations could be satisfied by looping on single probe. Symbolic compliance possible. | **Required** |
| INV-5a (Entropy) | No | No measure of epistemic state. No audit trail. INV-5c depends on it. | **Required** |
| INV-6 (Non-Trivial Exit) | No | Obligations could be exited without eliminations. Symbolic compliance possible. | **Required** |
| INV-2 (Stability Window) | Yes | Termination could occur on transient conclusion. Premature commitment possible. | **Optional** (already marked) |
| INV-5c (Convergence) | Yes | No progress rate diagnostic. Functional impact: none. | **Optional** (already marked) |

**Result:** The required set {INV-3, INV-5a, INV-6} is minimal. Nothing can be removed without weakening guarantees.

---

### 3.2 Could Anything Be Postponed to v1+ Without Loss of Correctness?

| Component | Postpone to v1+? | Assessment |
|-----------|------------------|------------|
| INV-3 | No | Core correctness depends on preventing probe loops. |
| INV-5a | No | Core observability depends on entropy. Without it, adapters have no feedback. |
| INV-6 | No | Obligation semantics depend on non-trivial exit. |
| INV-2 | Yes | Stability is valuable but not required for core function. Adapter could handle externally. |
| INV-5c | Yes | Convergence is purely diagnostic. No functional dependency. |

**Optional candidates for deferral:** INV-2, INV-5c are already correctly categorized as optional. They could be deferred to v1+ if desired, but provide value in v0 for adapters that want built-in stability/convergence tracking.

**Recommendation:** Keep INV-2 and INV-5c as optional in v0. They are cheap to implement and provide value. No strong reason to defer.

---

## Confirmation Summary

### Status

**CMBS v0 Core is internally coherent.**

### Ambiguities Identified and Resolved

| ID | Description | Resolution |
|----|-------------|------------|
| A1 | Termination gating: entropy vs stability | Core gates on stability only. Entropy-gated termination is adapter policy. |

### Contradictions Found

**None.**

---

## CMBS v0 Core Contract (Final Restatement)

1. **Core tracks a finite set of opaque hypothesis identifiers.** Adapter provides the initial universe; core tracks survivors after eliminations.

2. **Core computes entropy as log₂ of survivor count.** This is the sole measure of epistemic state. Core does not interpret what hypotheses mean.

3. **Core rejects duplicate probe results (INV-3).** Each probe ID may be consumed at most once. This prevents obligation satisfaction via repetition.

4. **Core counts eliminations within obligation scopes (INV-6).** Obligation exit requires at least one elimination (or adapter-provided threshold). This prevents symbolic compliance.

5. **Core tracks stability of adapter-declared conclusions (INV-2, optional).** Termination is permitted only if the conclusion has been stable for N steps.

6. **Core computes convergence rate from entropy history (INV-5c, optional).** This is diagnostic only; it does not gate any operation.

7. **All thresholds are adapter-provided.** Core contains no hardcoded policy values. Adapters provide elimination thresholds, stability window size, and any other numeric parameters.

8. **Core trusts adapter-provided elimination events.** Core does not validate eliminations against a compatibility model. Validation logic is private adapter responsibility.

9. **Core uses only opaque identifiers.** HypothesisID, ProbeID, ObservableID, ConclusionID, and ObligationID are opaque handles. Core never inspects their content.

10. **Core provides mechanism; adapters provide policy.** Core enforces non-repetition, counts eliminations, tracks stability, and computes entropy. Core does not decide what hypotheses exist, when obligations trigger, or when termination is appropriate.

---

## Appendix: Invariant Classification Summary

### Required (v0)

| ID | Name | Function |
|----|------|----------|
| INV-3 | Probe Non-Repetition | Reject duplicate probe results |
| INV-5a | Entropy Quantification | Compute `H = log₂(\|survivors\|)` |
| INV-6 | Non-Trivial Exit | Require eliminations for obligation exit |

### Optional (v0)

| ID | Name | Function |
|----|------|----------|
| INV-2 | Stability Window | Track conclusion stability for termination |
| INV-5c | Convergence Rate | Compute entropy slope for diagnostics |

### Deferred (v1+)

| ID | Name | Reason for Deferral |
|----|------|---------------------|
| INV-1 | Evidence Ratchet | Assumes execution lifecycle |
| INV-4 | Distributional Belief | Requires soft probability updates |
| INV-5b | Oscillation Detection | Requires soft belief dynamics |

---

## Appendix: Boundary Summary

### Core Contains (Open Source)

- Hypothesis set cardinality tracking
- Entropy computation
- Probe ID consumption tracking
- Conclusion stability window (optional)
- Elimination counting within obligation scope
- Obligation lifecycle state machine
- Convergence rate computation (optional)
- Threshold-gated obligation exit logic

### Core Does Not Contain

- Probe execution or scheduling
- Observable interpretation
- Hypothesis semantics
- Conclusion semantics
- Probe ordering
- Obligation type definitions
- Compatibility tables
- Elimination validation logic
- Threshold values

### Boundary Statement

> CMBS v0 Core is a belief-state accounting system. It tracks hypotheses, counts eliminations, measures entropy, and enforces non-repetition. It does not know what hypotheses mean, what probes do, or when to terminate. All semantics live in adapters. Core provides mechanism; adapters provide policy.

## Related

- [[ARCHITECTURE]]
- [[v0-test-specification]]
- [[v0-implementation-summary]]
