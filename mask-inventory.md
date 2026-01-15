# CMBS for IT-Bench CISO: Mask Inventory & Supervisor Design

This document consolidates the **belief masks**, **their scope**, and **how they apply across the CISO task lifecycle**. It is meant to be a *working design artifact* we can iterate on.

---

## 0. Design Constraints (Non-Negotiable)

**What CMBS may do**

* Gate actions based on *observed execution outcomes*
* Enforce epistemic discipline (no premature claims, no contradictions)
* Operate in an *open-world* manner
* Use only agent-visible signals

**What CMBS may NOT do**

* Encode Kyverno / Rego / Ansible schemas
* Encode CIS rules or evaluation logic
* Supply missing domain knowledge
* Collapse belief based on benchmark ontology

CMBS constrains **process**, not **content**.

---

## 1. Canonical CISO Task Flow (Authoritative)

The CISO benchmark implicitly enforces the following lifecycle:

1. Read goal
2. Select execution path (tentative)
3. Attempt artifact generation
4. Observe execution result
5. Update belief
6. Retry / refine
7. Apply successfully
8. Verify results
9. Declare posture
10. Terminate

CMBS must provide **belief masks aligned to each phase**.

---

## 2. Core Belief State (Open-World)

This is the *only* belief object the agent is allowed to emit.

```json
belief = {
  "affordances": {
    "k8s_policy": "unknown | available | unavailable",
    "opa_eval": "unknown | available | unavailable",
    "ansible_exec": "unknown | available | unavailable"
  },
  "posture": "unknown | compliant | non_compliant",
  "evidence": "none | attempted | successful"
}
```

Interpretation:

* **Affordances** = what the environment appears to support
* **Posture** = what outcomes are still admissible
* **Evidence** = how far execution has actually progressed

This belief is *claimed* by the agent but *validated* by the supervisor.

---

## 3. Belief Masks (Inventory)

Below is the complete set of masks required for CISO.

### Mask A — Affordance Mask

Tracks *what execution paths are currently possible*.

```python
affordance_mask = {
  "k8s_policy": unknown / available / unavailable,
  "opa_eval": unknown / available / unavailable,
  "ansible_exec": unknown / available / unavailable
}
```

**Updated by**

* Presence of kubeconfig
* Success/failure of kubectl
* Success/failure of `opa eval`
* Success/failure of ansible-playbook

**Used to gate**

* Policy generation attempts
* Execution commands

---

### Mask B — Evidence Progress Mask

Tracks whether *anything real has happened*.

```python
evidence_mask = {
  "state": none | attempted | successful
}
```

**Updated by**

* Artifact creation → attempted
* Successful execution → successful

**Used to gate**

* Posture declaration
* Termination

---

### Mask C — Posture Admissibility Mask

Tracks which postures are still logically possible.

```python
posture_mask = {
  "compliant": True | False,
  "non_compliant": True | False
}
```

**Updated by**

* Observed violations
* Explicit negative results

**Rules**

* Absence of evidence does NOT eliminate non_compliance
* Execution failure does NOT collapse posture

**Used to gate**

* Declared posture

---

### Mask D — Stability Mask

Ensures beliefs do not oscillate.

```python
stability_mask = {
  "posture_stable": True | False
}
```

**Updated by**

* Comparing posture across last N steps

**Used to gate**

* Termination

---

### Mask E — Termination Eligibility Mask

Derived mask (not directly updated).

```python
termination_allowed = (
  evidence == successful
  AND exactly one posture admissible
  AND posture_stable
)
```

**Used to gate**

* Terminate action

---

## 4. Masks Mapped to Task Phases

| Phase            | Relevant Masks         | What Is Enforced            |
| ---------------- | ---------------------- | --------------------------- |
| Read goal        | Affordance             | No assumptions yet          |
| Select path      | Affordance             | Tentative only              |
| Attempt artifact | Affordance, Evidence   | Must attempt before claims  |
| Observe error    | Evidence               | Errors do not imply success |
| Update belief    | All                    | Monotonic updates           |
| Retry            | Affordance, Evidence   | Retry allowed               |
| Apply            | Affordance             | Must succeed                |
| Verify           | Evidence, Posture      | Must observe output         |
| Declare posture  | Posture                | Must be admissible          |
| Terminate        | Stability, Termination | Must be earned              |

---

## 5. Supervisor Contract

**Agent responsibilities**

* Emit belief + proposed action
* Generate artifacts
* Respond to feedback

**Supervisor responsibilities**

* Validate belief against masks
* Block illegal actions
* Enforce retries
* Decide termination

Supervisor never:

* Edits artifacts
* Supplies schemas
* Infers CIS semantics

---

## 6. Integrity Checklist

Before adding any new mask, ask:

* Does this mask encode domain knowledge? (❌)
* Does it rely on evaluation logic? (❌)
* Is it derived from execution outcomes only? (✅)
* Is it monotonic or stability-based? (✅)

If the answer pattern is not ❌❌✅✅, do not add the mask.

---

## 7. Outcome

With these masks, CMBS:

* Improves reliability without cheating
* Surfaces failure modes cleanly
* Scales across all CISO scenarios
* Remains leaderboard-legitimate

This document is the authoritative reference for CMBS–CISO going forward.
