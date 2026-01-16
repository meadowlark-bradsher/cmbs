# Feature Spec: Continuous CMBS Inference Layer (CCIL)

## 0. Goal

Add a **latent continuous belief layer** under the existing discrete masks to support:

1. **Quantitative diachronic audits** (entropy, convergence, oscillation, premature confidence)
2. **Future extensibility** toward EIG / action selection *without changing the supervisor’s role today*.

This layer **must not**:

* suggest actions
* validate domain semantics (CIS/Kyverno/Rego/Ansible)
* alter pass/fail criteria
* replace existing discrete masks or gating rules

It is **instrumentation + inference**, not policy.

References: supervisor contract and masks remain authoritative.

---

## 1. Non-Goals

* No planner
* No reward shaping
* No EIG selection
* No schema checks beyond existing structure validation
* No new benchmark ontology
* No “helpful hints” to agent

---

## 2. Where This Fits in the Current Loop

Existing loop (simplified) :

1. Validate agent output schema
2. Observe execution results
3. Update discrete masks
4. Check belief consistency
5. Gate action
6. Execute action

**New insertion points:**

* After (2)+(3): update *continuous posterior* conditioned on latest observation
* After (4): compute *continuous audits* (but do not gate on them)
* Before returning supervisor verdict: optionally emit *diagnostic metadata* (for logs only)

Supervisor still returns the same verdicts: `ALLOW/BLOCK/CONTINUE/TERMINATE`.

---

## 3. Data Model

### 3.1 Existing Discrete Masks (unchanged)

From mask inventory :

* `affordances: {k8s_policy, opa_eval, ansible_exec} ∈ {unknown, available, unavailable}`
* `evidence ∈ {none, attempted, successful}`
* `posture ∈ {unknown, compliant, non_compliant}`
* `posture_admissible: {compliant: bool, non_compliant: bool}`
* `posture_history: list[str]` (stability)

These remain the **authoritative gating objects**.

---

### 3.2 New Continuous Latent State

Introduce `LatentBelief`:

```json
latent_belief = {
  "z_afford": { "k8s_policy": float, "opa_eval": float, "ansible_exec": float },
  "z_evidence": float,
  "z_posture": { "compliant": float, "non_compliant": float },
  "temperature": float,
  "step_index": int
}
```

Interpretation via logistic transform:

* `p_afford[path] = sigmoid(z_afford[path])` = probability path is viable
* `p_success = sigmoid(-z_evidence)` = probability evidence is “close to successful”
* `p_posture = softmax(-z_posture)` = relative plausibility of postures

**Important:** These probabilities are *diagnostic*, not gating.

---

### 3.3 Observation Record (input to inference)

Define an `ObservationEvent` from the supervisor's "observe_execution" result :

```json
obs = {
  "step": int,
  "source": "execution" | "document_probe" | "review_probe",
  "probe_id": "string|null",
  "command": "string|null",
  "command_success": true|false|null,
  "artifact_written": true|false,
  "output_tags": ["kubectl_missing", "schema_error", "policyreport_violation", ...],
  "raw_exit_code": int|null
}
```

`output_tags` is produced by simple regex/heuristic classifiers you already rely on for mask updates (no new semantics).

**Observation Sources:**

DSRO and RO generate observations without execution. CCIL treats them as first-class belief updates:

- `execution`: Standard command execution results
- `document_probe`: DSRO document searches (open_section, search_keyword)
- `review_probe`: RO self-review actions (diff, log, revision_compare)

No new semantics. No new masks. Just typed observation sources.

---

## 4. Generative/Energy Model

We need an energy function:

[
p(z \mid \text{history}) \propto \exp(-E(z; \text{history}))
]

### 4.1 Core Principle

Energy is **only** a function of:

* observed execution outcomes
* intervention-independent heuristics already used for discrete mask updates

No CIS semantics. No “understanding”.

---

### 4.2 Energy Terms (minimal and modular)

Define:

[
E(z) = E_{\text{prior}}(z) + \sum_{t} E_{\text{obs}}(z; \text{obs}_t)
]

#### Prior (stabilizer)

* Gaussian prior on logits to prevent runaway:
  [
  E_{\text{prior}} = \lambda |z|^2
  ]

#### Observation likelihood terms

**Affordance updates**

* If a path was attempted and failed with “command not found” / missing tool tag:

  * penalize high viability:
    [
    E += w \cdot \text{sigmoid}(z_{\text{path}})
    ]
* If a path was attempted and succeeded:

  * penalize low viability:
    [
    E += w \cdot \text{sigmoid}(-z_{\text{path}})
    ]

**Evidence progress**

* `artifact_written` pushes toward “attempted progress”
* `command_success` pushes toward “successful”
  A simple scheme:
* if artifact written: add term pushing `z_evidence` down slightly
* if command succeeded: push `z_evidence` down strongly
* if command failed: push `z_evidence` up mildly (failure ≠ impossibility)

**Posture**

* If "violation observed" tag: push non_compliant down (more plausible) and compliant up
* If "explicit no-violation observed" tag (rare): opposite
* Otherwise: don't update posture

This mirrors your discrete rules:

* absence of evidence does nothing

---

#### Document Probe Energy (DSRO)

Document probes introduce **weak informational constraints**.

Rules:

* If probe returns **matching patterns** (e.g., keyword hits):
  * ↓ entropy (soft)
  * ↑ capability (soft)
* If probe returns **no matches**:
  * slight increase in repair pressure (exhausting search space)
* Repetition is handled upstream — CCIL never sees repeats

```text
if obs.source == "document_probe":
    if obs.has_matches:
        E += w_entropy * sigmoid(z_posture)
        E -= w_capability * sigmoid(z_capability_opa)
```

Important:
* This does **not** encode Rego semantics
* Matching is string / structural only
* "has_matches" is a mechanical flag from the oracle

---

#### Review Probe Energy (RO)

Review probes introduce **self-consistency constraints**.

Rules:

* Inspecting diffs / logs:
  * ↓ oscillation
  * ↓ repair pressure
* Revealing contradictions:
  * ↑ entropy temporarily (allowed)
* Reconciling revisions:
  * ↑ capability

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

## 5. Inference Engine

### 5.1 Default: Underdamped Langevin (ULD)

Rationale:

* cheaper than full HMC
* mixes better than overdamped in multimodal settings
* easy to implement

Maintain momentum `r` and do K steps per observation.

Outputs:

* posterior samples ( {z^{(s)}} )
* running estimates of:

  * ( \mathbb{E}[\sigma(z)] )
  * entropy proxies
  * stability diagnostics

### 5.2 Optional: HMC mode

Enable for offline analysis runs or when you want stronger sampling diagnostics.

### 5.3 Determinism

To preserve reproducibility:

* seed sampler from (episode_id, step_index)
* fixed number of steps K
* fixed step size η, friction γ

---

## 6. Integration with Discrete Masks

### 6.1 One-way coupling (important)

Discrete masks **must not** be driven by continuous posteriors.

Coupling direction is:

**observations → discrete masks** (existing)
**observations → continuous posterior** (new)
**discrete masks → constraints on posterior** (optional hard clamps)

Optional: if a discrete affordance is `unavailable`, clamp its latent logit to a low value (or add an infinite penalty).

This ensures continuous posterior never contradicts authoritative observed facts.

---

## 7. Diachronic Audit Outputs

This is the primary deliverable.

For each step, log:

### 7.1 Convergence / entropy

* `H_afford` = entropy over affordance marginals
* `H_posture` = entropy over posture softmax
* `progress_score` = 1 - sigmoid(z_evidence) or similar

### 7.2 Stability / oscillation (continuous)

* `Δ_posture` = distance between posture distributions at t and t-1
* `Δ_afford` = L2 distance of affordance marginals
* `energy_trace` = mean energy of samples (should decrease with evidence)

### 7.3 Claim alignment (optional audit only)

When the agent emits a belief claim (discrete), compute:

* is it compatible with posterior mass?

  * e.g., if agent claims `k8s_policy=unavailable` but posterior (p_k8s>0.8), flag "underconfident"
* If agent declares posture early, show posterior posture entropy at that moment

**These do not gate.** They're logged for analysis.

---

### 7.4 Belief Delta (obligation exit signal)

DSRO and RO need a **boolean signal**: "Did belief change *enough* to exit the obligation?"

Define a derived metric:

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

Where ε is a threshold (default 0.05).

**Key principle:** DSRO and RO do **not interpret** belief. They only ask CCIL: *"Did belief move?"*

This keeps authority separation intact. The obligation systems consume CCIL metrics but do not define them.

---

## 8. Supervisor Behavior (unchanged)

The supervisor continues to gate only on:

* evidence must be `successful` before posture declaration
* termination eligibility mask 
* affordance impossibility checks (weak gating)

Continuous audits **never** block steps in v1.

---

## 9. Configuration

Add config block:

```yaml
continuous_cmbs:
  enabled: true
  sampler: uld | hmc
  num_particles: 64
  steps_per_update: 20
  step_size: 0.02
  friction: 0.1
  prior_lambda: 1e-2
  hard_clamp_to_discrete: true
  log_level: summary | full
```

---

## 10. Acceptance Criteria

### 10.1 Correctness invariants

* Pass/fail outcomes are identical with CCIL enabled/disabled (same seeds).
* No new supervisor interventions occur because of CCIL.
* CCIL uses only observation signals already available to mask updates.

### 10.2 Utility

* Logs include per-step:

  * posterior posture entropy
  * energy trace
  * convergence deltas
* In known traces (your worked example), CCIL shows:

  * high uncertainty early
  * entropy collapse after successful apply + violation observation
  * stability achieved before termination

### 10.3 Performance

* Runtime overhead bounded (e.g., <10–20% wall clock)
* CCIL can be disabled entirely

---

## 11. Testing Plan

1. **Golden trace replay**: run the worked example trace  with CCIL on/off; verify identical supervisor verdicts.
2. **Noise injection**: perturb observation tags (simulate flaky commands) and ensure:

   * discrete masks remain conservative
   * continuous posterior reflects uncertainty instead of collapsing
3. **Stability check**: create synthetic oscillation in agent claims; confirm CCIL logs increased Δ metrics while discrete gating remains unchanged.

---

## 12. Future Extension Hooks (explicitly deferred)

* Use posterior to compute "expected constraint reduction" as EIG
* Use EIG to rank allowable actions (agent still chooses)
* Add "belief sincerity" penalties for training signals

None of this is in v1.

---

## 13. Obligation Consumption of CCIL Metrics

CCIL does not enforce epistemic obligations directly. Instead, Repair, DSRO, and RO consume CCIL metrics via the following interface:

**Metrics consumed by obligations:**

* `belief_moved` — boolean signal: did belief change enough to exit?
* `entropy_posture` — uncertainty in compliance posture
* `repair_pressure` — probability agent is stuck in repair loop
* `capability_opa` — confidence in OPA/Rego execution competence

**Exit conditions:**

Exit conditions for DSRO and RO MUST be defined exclusively in terms of CCIL deltas, not agent claims.

```text
can_exit_obligation = belief_delta.belief_moved(threshold=0.05)
```

This ensures:
* Authority separation: obligations ask "did belief move?", not "what does agent claim?"
* Measurability: exit is gated by quantitative change, not qualitative interpretation
* Consistency: all obligations use the same CCIL interface

**One-sentence principle:**

> HMC/ULD already tracks belief; DSRO and RO merely ask whether that belief has moved.

---

### Practical reading of this spec

* Your current CMBS masks and supervisor stay exactly as-is.
* You add a **parallel belief inference module** that consumes the same observations and outputs **numbers you can plot**: entropy, energy, convergence, alignment.
* You get a path toward EIG later without contaminating the benchmark today.
