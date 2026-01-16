# What to Add to `continuous-cmbs-design.md`

You already have:

* latent belief
* energy model
* ULD/HMC
* CCIL metrics
* supervisor integration

What’s missing is **support for non-execution observations** and **obligation exit semantics**.

---

## 1️⃣ Add Two New Observation Types (Very Small Change)

### Where

Section **3.3 Observation Record**

### What to add

Extend `ObservationEvent` to include **epistemic probes**:

```json
obs = {
  "step": int,
  "source": "execution" | "document_probe" | "review_probe",
  "probe_id": "string|null",
  "command": "string|null",
  "command_success": true|false|null,
  "artifact_written": true|false,
  "output_tags": ["kubectl_missing", "schema_error", ...],
  "raw_exit_code": int|null
}
```

### Why

DSRO and RO generate **observations without execution**.
CCIL must treat them as first-class belief updates.

No semantics added.
No new masks added.
Just *typed observation sources*.

---

## 2️⃣ Add Energy Terms for Document + Review Probes

### Where

Section **4.2 Energy Terms**

You already handle:

* affordance
* evidence
* posture

You now need **two additional observation-conditioned energy contributions**.

---

### 2.1 Document Probe Energy (DSRO)

Add a subsection:

#### **Document Probe Energy**

Document probes introduce **weak informational constraints**.

Rules:

* If probe returns **matching patterns** (e.g., keyword hits):

  * ↓ entropy (soft)
  * ↑ capability (soft)
* If probe returns **no matches**:

  * no belief change
* Repetition is handled upstream — **CBIL never sees repeats**

Example (conceptual):

```text
if obs.source == "document_probe":
    if obs.has_matches:
        E += w_entropy * sigmoid(z_posture)
        E -= w_capability * sigmoid(z_capability_opa)
```

Important:

* This does **not** encode Rego semantics
* Matching is string / structural only
* “has_matches” is a mechanical flag from the oracle

---

### 2.2 Review Probe Energy (RO)

Add another subsection:

#### **Review Probe Energy**

Review probes introduce **self-consistency constraints**.

Rules:

* Inspecting diffs / logs:

  * ↓ oscillation
  * ↓ repair pressure
* Revealing contradictions:

  * ↑ entropy temporarily (allowed)
* Reconciling revisions:

  * ↑ capability

Example:

```text
if obs.source == "review_probe":
    E -= w_repair * sigmoid(z_repair_pressure)
    E -= w_osc * oscillation_penalty
```

Again:

* No interpretation
* No correctness judgment
* Just belief geometry

---

## 3️⃣ Add “Belief Delta” as a First-Class Output

### Where

Section **7 Diachronic Audit Outputs**

You already log entropy and deltas, but DSRO / RO need a **boolean signal**:

> “Did belief change *enough* to exit the obligation?”

### What to add

Add a derived metric:

```text
belief_delta = {
  "entropy_drop": ΔH_posture,
  "capability_gain": Δcapability_opa,
  "repair_pressure_drop": Δrepair_pressure
}
```

And define:

```text
belief_moved = (
    ΔH_posture >= ε
 OR Δcapability >= ε
 OR Δrepair_pressure >= ε
)
```

### Why

DSRO / RO **do not interpret belief**.
They only ask CBIL: *“Did belief move?”*

This keeps authority separation intact.

---

## 4️⃣ Explicitly State Obligation Hooks (Documentation Only)

### Where

Add a **new subsection** at the end of the document:

### **13. Obligation Consumption of CCIL Metrics**

Add text like:

> CCIL does not enforce epistemic obligations directly.
> Instead, Repair, DSRO, and RO consume CCIL metrics via the following interface:
>
> * `belief_moved`
> * `entropy_posture`
> * `repair_pressure`
> * `capability_*`

And:

> Exit conditions for DSRO and RO MUST be defined exclusively in terms of CCIL deltas, not agent claims.

This is documentation-only — no code required.

---

## 5️⃣ What You Do *NOT* Need to Add

To be explicit, you **do not** need to add:

* ❌ new latent variables
* ❌ a controller
* ❌ EIG computation
* ❌ probe ranking
* ❌ NLU
* ❌ document semantics
* ❌ supervisor logic
* ❌ new masks

Everything else is already present.

---

## 6️⃣ Minimal Patch Summary

If you want a literal checklist:

**Add:**

* observation.source ∈ {execution, document_probe, review_probe}
* document probe energy terms
* review probe energy terms
* belief_delta / belief_moved metric
* short section documenting obligation hooks

**Do not change:**

* latent belief structure
* supervisor behavior
* discrete masks
* termination rules

---

## 7️⃣ One-Sentence Mental Model (Pin This)

> **HMC/ULD already tracks belief; DSRO and RO merely ask whether that belief has moved.**

That’s why this integrates cleanly.
