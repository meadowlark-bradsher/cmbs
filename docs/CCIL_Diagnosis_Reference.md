# CCIL Diagnosis Reference

**Purpose:** Interpret CMBS + CCIL traces consistently across IT-Bench and related tasks
**Scope:** Observational, diagnostic, non-prescriptive

---

## 0. Reading This Document

This document explains:

* **What patterns can appear** in a task run
* **What they mean epistemically**
* **What they do *not* mean**
* **How to distinguish honest failure from epistemic failure**

It does **not**:

* Judge agent quality
* Suggest actions
* Affect supervisor gating
* Define success criteria

All conclusions are **diagnostic only**.

---

## 1. Core Diagnostic Axes

Every run can be understood as movement along **five independent axes**:

1. **Evidence Progress**
2. **Posture Uncertainty**
3. **Belief Stability**
4. **Repair Dynamics**
5. **Execution Capability**

Each axis has both **discrete CMBS state** and **continuous CCIL signals**.

---

## 2. Evidence Progress

### 2.1 Discrete Evidence Mask (Authoritative)

| State        | Meaning                              |
| ------------ | ------------------------------------ |
| `none`       | No observable execution has occurred |
| `attempted`  | Artifacts or commands attempted      |
| `successful` | At least one command succeeded       |

This mask controls **legality** (posture declaration, termination).

---

### 2.2 Continuous Evidence Progress (`progress`)

**Range:** `[0, 1]`
**Meaning:** Degree of accumulated execution effort

| Pattern          | Interpretation                      |
| ---------------- | ----------------------------------- |
| Low progress     | Early exploration or hesitation     |
| Gradual increase | Normal execution attempts           |
| Rapid increase   | Significant execution events        |
| Plateau          | Effort without new execution events |

**Important:**
Progress ≠ correctness
Progress ≠ learning
Progress ≠ permission to terminate

---

## 3. Posture Uncertainty

### 3.1 Discrete Posture Mask

| State           | Meaning                       |
| --------------- | ----------------------------- |
| `unknown`       | Both postures admissible      |
| `compliant`     | Only compliant admissible     |
| `non_compliant` | Only non-compliant admissible |

Derived strictly from **observed evidence**.

---

### 3.2 Continuous Posture Entropy (`entropy_posture`)

**Range:** `[0, 1]` (normalized)

| Value | Interpretation      |
| ----- | ------------------- |
| ~1.0  | Maximum uncertainty |
| ~0.5  | Partial collapse    |
| ~0.0  | Fully determined    |

#### Common Patterns

| Pattern                       | Meaning                      |
| ----------------------------- | ---------------------------- |
| High entropy throughout       | No posture-relevant evidence |
| Entropy drops after success   | Evidence is informative      |
| Entropy drops before evidence | ⚠️ Epistemic violation       |
| Entropy oscillates            | Belief instability           |

---

## 4. Belief Stability

### 4.1 Discrete Stability Mask

| Condition | Meaning                      |
| --------- | ---------------------------- |
| Stable    | Same posture claimed N steps |
| Unstable  | Claims oscillate             |

Used **only** for termination gating.

---

### 4.2 Continuous Oscillation (`oscillation_score`)

**Meaning:** Variance of posture belief over recent steps

| Pattern                        | Interpretation                    |
| ------------------------------ | --------------------------------- |
| Low oscillation                | Belief stable (even if uncertain) |
| High oscillation               | Belief flip-flopping              |
| High oscillation + low entropy | ⚠️ Incoherent belief              |

This distinguishes:

* “Consistently uncertain” (good)
* “Unstable reasoning” (problematic)

---

## 5. Repair Dynamics

### 5.1 Discrete Repair Mask

| State             | Meaning                  |
| ----------------- | ------------------------ |
| `repair_required` | Agent must fix artifacts |

Purely procedural.

---

### 5.2 Continuous Repair Pressure (`repair_pressure`)

**Range:** `[0, 1]`

| Pattern                      | Interpretation               |
| ---------------------------- | ---------------------------- |
| Low pressure                 | Exploration or early failure |
| Rising pressure              | Active debugging             |
| High pressure                | Stalled repair loop          |
| High pressure + low progress | **Repair thrash**            |
| High pressure + new signals  | Productive struggle          |

Repair pressure answers:

> “Is the agent learning from failure, or repeating it?”

---

## 6. Execution Capability

### 6.1 Discrete Capability Indicators

Examples:

* `policy_written`
* `command_succeeded`

These are **binary facts**.

---

### 6.2 Continuous Capability Confidence (`capability_*`)

Tracked per execution domain (e.g., k8s, opa, ansible)

| Pattern                  | Interpretation                   |
| ------------------------ | -------------------------------- |
| Low capability           | Agent lacks execution competence |
| Rising capability        | Learning execution path          |
| High capability          | Robust execution understanding   |
| Success + low capability | Fragile / lucky success          |

This distinguishes:

* Procedural success
* Epistemic competence

---

## 7. Convergence Dynamics

### 7.1 Convergence Rate (`convergence_rate`)

**Meaning:** Slope of entropy over time

| Value     | Interpretation    |
| --------- | ----------------- |
| Positive  | Belief collapsing |
| Near zero | Plateau           |
| Negative  | Belief divergence |

**Critical Rule:**
Convergence ≠ permission to terminate

---

## 8. Canonical Diagnostic Patterns

### 8.1 Healthy Success

* Evidence → `successful`
* Entropy → low
* Oscillation → low
* Repair pressure → low
* Capability → high

**Interpretation:** Correct and justified conclusion.

---

### 8.2 Honest Failure (Clean)

* Evidence → `attempted` or `successful`
* Entropy → high
* Oscillation → low
* Repair pressure → high
* Timeout termination

**Interpretation:** Agent worked but lacked observability.

---

### 8.3 Premature Confidence (Epistemic Violation)

* Entropy drops before evidence
* Oscillation increases
* Posture declared early

**Interpretation:** Invalid belief collapse (should be blocked).

---

### 8.4 Repair Thrash

* Repair pressure → high
* Progress → flat
* Capability → flat or falling

**Interpretation:** Agent stuck fixing same failure mode.

---

### 8.5 Belief Instability

* High oscillation
* Entropy fluctuates
* Conflicting claims

**Interpretation:** Incoherent reasoning trajectory.

---

### 8.6 Lucky Success

* Evidence → successful
* Capability → low
* Entropy drops sharply

**Interpretation:** Outcome correct, belief fragile.

---

## 9. Termination Interpretation

### Timeout Termination

| Condition                      | Meaning                        |
| ------------------------------ | ------------------------------ |
| Timeout + high entropy         | Observability failure          |
| Timeout + low entropy          | Missed termination opportunity |
| Timeout + high repair pressure | Stalled repair                 |

Timeout is **procedural**, not epistemic.

---

## 10. What CCIL Diagnostics Must Never Do

* Trigger supervisor blocks
* Permit termination
* Override discrete masks
* Encode task semantics
* Judge correctness

CCIL answers **why**, never **what to do**.

---

## 11. One-Line Mental Model

> **Discrete CMBS enforces epistemic legality.
> CCIL reveals epistemic geometry.**

Both are required; neither replaces the other.

---

## 12. Intended Use Cases

* Debugging agent failures
* Comparing runs across models
* Identifying observability gaps
* Future EIG research
* Training-time analysis (offline)
