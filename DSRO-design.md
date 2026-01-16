# Design Spec: Document Search as a Repair Obligation (DSRO)

**Target system:** CMBS + CCIL–instrumented IT-Bench (CISO / OPA)
**Status:** Additive, non-disruptive feature
**Primary goal:** Prevent uninformed retries and hallucinated fixes by enforcing *meaningful document search* as a repair action

---

## 1. Motivation

Current CISO failures often arise not from lack of execution effort but from **epistemic stagnation**:

* Repeated YAML edits without new understanding
* Blind retries of `kubectl apply`
* Hallucinated Rego syntax
* No principled interaction with documentation

This design introduces **Document Search as a first-class repair option**, enforced with the same epistemic discipline as existing repair logic.

The core invariant:

> **An agent may not proceed as if it has learned unless learning has occurred.**

---

## 2. Non-Goals (Explicit)

This system does **not**:

* Teach OPA or Rego semantics
* Suggest which section to read
* Summarize documentation
* Replace the supervisor
* Require NLU, embeddings, or a learned controller
* Affect benchmark scoring logic

It enforces **process**, not **content**.

---

## 3. Conceptual Model

### 3.1 Authorities

There are **three epistemic authorities**, each with a distinct role:

| Authority                  | Role                        | Learning | Power      |
| -------------------------- | --------------------------- | -------- | ---------- |
| Supervisor                 | Enforce epistemic law       | None     | Hard block |
| Repair System              | Enforce obligation to fix   | None     | Soft block |
| Document Search Obligation | Enforce obligation to learn | None     | Soft block |

The Document Search Obligation (DSO) is *not* a supervisor.
It is a **repair-scoped obligation cone**.

---

## 4. Integration Point: Repair Loop

Document search is treated as **one repair option**.

### 4.1 Repair Menu

When repair is required, the agent may choose:

```
REPAIR OPTIONS:
1. Revise artifact
2. Retry execution
3. Perform document search
```

Selection is free.
Enforcement begins **only after selection**.

---

## 5. Document Search Obligation (DSO)

### 5.1 Entry Condition

The agent enters the DSO when:

```text
repair_action == DOCUMENT_SEARCH
```

The system does **not** force this choice.
It only enforces consequences once chosen.

---

### 5.2 Allowed Actions Inside DSO

While in DSO:

```text
ALLOWED:
- probe_document(section_id)
- probe_document(keyword)

FORBIDDEN:
- artifact revision
- execution
- posture declaration
- termination
```

This mirrors repair mode behavior.

---

## 6. Document Oracle

### 6.1 Inputs

* Structured document (text file, pre-segmented)
* Section IDs
* Keyword index (string match only)

### 6.2 Actions

```python
probe_document(
    kind: Literal["open_section", "search_keyword"],
    target: str
)
```

### 6.3 Outputs

```python
DocumentObservation = {
    "probe": Probe,
    "text": str,
}
```

The oracle performs **no reasoning**.

---

## 7. Non-Repetition Rule (Critical)

Inside DSO:

```text
A probe MAY NOT be repeated.
```

Formally:

```python
if probe in probe_history:
    BLOCK
```

This alone eliminates search thrash.

---

## 8. Meaningful Completion Criteria

Document search is considered **complete** *only if* epistemic state changes.

### 8.1 Exit Conditions

The agent may exit DSO if **any** of the following hold:

```text
entropy_posture decreased ≥ Δ
OR capability_opa increased ≥ Δ
OR repair_pressure decreased ≥ Δ
```

Agent claims are irrelevant.
Only CCIL metrics count.

---

### 8.2 Forbidden Exit

If the agent attempts to exit DSO without satisfying exit criteria:

```text
BLOCK exit
REQUIRE continued probing
```

Identical to posture-without-evidence blocking.

---

## 9. Exhaustion Semantics

If:

* All admissible probes are exhausted
* No belief change occurs

Then:

* DSO ends as **unsatisfied**
* repair_pressure remains high
* the agent may fail honestly

This is a valid outcome.

---

## 10. Relationship to CCIL

DSO relies exclusively on **existing CCIL signals**:

| Signal            | Usage                  |
| ----------------- | ---------------------- |
| entropy_posture   | Determines need & exit |
| repair_pressure   | Entry trigger & exit   |
| capability_opa    | Learning confirmation  |
| oscillation_score | Diagnostics only       |

No new belief state is required.

---

## 11. Supervisor Interaction

The supervisor:

* Allows `probe_document` actions
* Does not interpret document content
* Does not enforce probe selection
* Does not evaluate learning quality
* Does not grant termination based on DSO alone

Supervisor logic remains unchanged.

---

## 12. State Machine (Minimal)

```
[Normal Mode]
     |
     | execution failure / repair required
     v
[Repair Required]
     |
     | agent selects DOCUMENT_SEARCH
     v
[Document Search Obligation]
     |
     | probe_document (non-repeating)
     v
[Belief Update via CCIL]
     |
     | entropy / capability / pressure changes?
     |        |
     |       yes
     |        v
     |   exit DSO
     |
     |       no
     v        |
[More Probing]|
     |        |
     +--------+
```

---

## 13. Logging & Diagnostics

For each DSO episode, log:

* probes attempted
* Δ entropy per probe
* Δ capability per probe
* Δ repair pressure
* probes exhausted (yes/no)
* exit reason

This enables:

* post-hoc analysis
* MAST alignment
* future controller training (optional)

---

## 14. Why This Works

This design:

* Forces epistemic interaction with documentation
* Prevents symbolic “I read the docs” behavior
* Requires measurable learning
* Preserves agent autonomy
* Avoids hinting or semantic injection
* Requires no learning to deploy
* Is benchmark-legitimate

---

## 15. Future Extensions (Deferred)

These are **not required** for v1:

* Ranking probes by expected entropy reduction
* Learned controller over admissible probes
* EIG-based probe prioritization
* Cross-document search
* PDF segmentation

All are compatible with this design.

---

## 16. Final Invariant (Pin This)

> **Choosing document search creates an obligation to learn.
> Learning is measured by belief movement, not by claims.**

If that invariant holds, the system is correct.

---

### Ready-to-Build Checklist

* [ ] Parse documentation into section IDs
* [ ] Implement document oracle
* [ ] Add `probe_document` action
* [ ] Track probe history
* [ ] Enforce non-repetition
* [ ] Gate exit on CCIL metrics
* [ ] Log DSO episodes