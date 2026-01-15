## 1. One-Sentence Definition (Pin This)

> **The supervisor is a permissive, stateful gatekeeper that observes execution outcomes and blocks only epistemically invalid transitions.**

That’s it. If your code does more than that, you’re drifting.

---

## 2. Supervisor Responsibilities (Very Narrow)

### The supervisor **MUST**

* Track observable execution outcomes
* Maintain belief masks derived from *those outcomes*
* Decide whether a proposed action is **allowed**, **blocked**, or **must continue**
* Decide whether termination is **earned**

### The supervisor **MUST NOT**

* Generate content
* Fix YAML / Rego / Ansible
* Suggest commands
* Infer CIS semantics
* Encode schemas
* Encode benchmark ontology

If you ever feel tempted to “help,” stop.

---

## 3. Supervisor Internal State (Minimal)

You only need **four pieces of state**.

### 3.1 Belief Masks (Authoritative)

```python
class Masks:
    affordances: dict[str, Literal["unknown", "available", "unavailable"]]
    evidence: Literal["none", "attempted", "successful"]
    posture_admissible: dict[str, bool]  # compliant / non_compliant
    posture_history: list[str]
```

Notes:

* `posture_history` is just for stability checks
* No probabilities
* No schemas
* No engine enums

---

### 3.2 Last Observed Execution Result

```python
class LastObservation:
    command: str | None
    success: bool | None
    output_summary: str | None
```

This is what masks update *from*.

---

### 3.3 Step Counter / Time Budget

```python
step_count: int
start_time: float
```

Used only to prevent infinite loops under the 200s constraint.

---

### 3.4 Supervisor Verdict

This is not stored — it’s computed per step.

```python
enum Verdict:
    ALLOW
    BLOCK
    CONTINUE
    TERMINATE
```

---

## 4. The Supervisor Control Loop (Canonical)

This loop is the *engine*. Everything else is bookkeeping.

```python
while True:
    agent_step = get_agent_output()

    # 1. Validate shape (schema only)
    if not valid_schema(agent_step):
        return CONTINUE("Invalid format. Re-emit.")

    # 2. Observe environment (after previous step)
    observation = observe_execution()
    update_masks_from_observation(observation)

    # 3. Check belief consistency
    if not belief_consistent(agent_step.belief, masks):
        return CONTINUE("Belief inconsistent with observations.")

    # 4. Check action legality
    verdict = check_action_legality(agent_step.action, masks)

    if verdict == BLOCK:
        return CONTINUE("Action not permitted. Continue.")

    if verdict == TERMINATE:
        accept_free_text(agent_step.free_text)
        break

    # 5. Execute action
    execute(agent_step.action)

    # 6. Update step counter, loop
```

This is **intentionally boring**. That’s good.

---

## 5. Action Legality Rules (The Only “Logic”)

These are the **only places where the supervisor intervenes**.

### Rule 1 — Posture Declaration Requires Successful Evidence

```python
if action.type == "declare_posture":
    if masks.evidence != "successful":
        BLOCK
```

This is the most important rule.

---

### Rule 2 — Termination Requires Epistemic Completion

```python
if action.type == "terminate":
    if not (
        masks.evidence == "successful"
        and exactly_one_true(masks.posture_admissible)
        and posture_stable(masks.posture_history)
    ):
        BLOCK
```

This is the second most important rule.

---

### Rule 3 — Affordance Gating (Weak, Not Smart)

```python
if action.requires("k8s_policy") and masks.affordances["k8s_policy"] == "unavailable":
    BLOCK
```

Note:

* Never collapse affordances early
* Never assume availability implies correctness
* This only blocks *impossible* actions

---

### Rule 4 — Everything Else Is Allowed

Bad ideas are allowed.
Wrong YAML is allowed.
Repeated failure is allowed.

Only **epistemic violations** are blocked.

---

## 6. Mask Update Rules (Purely Observational)

These are called **after execution**, never before.

### Evidence Mask

```python
if artifact_written:
    masks.evidence = "attempted"

if command_successful:
    masks.evidence = "successful"
```

---

### Affordance Mask

```python
if kubectl_command_attempted:
    masks.affordances["k8s_policy"] = (
        "available" if command_exists else "unavailable"
    )
```

No inference beyond that.

---

### Posture Mask

```python
if violation_observed:
    masks.posture_admissible["compliant"] = False

if explicit_no_violation_observed:
    masks.posture_admissible["non_compliant"] = False
```

Absence of evidence → do nothing.

---

### Stability Tracking

```python
masks.posture_history.append(current_posture_claim)
```

Stability is a simple equality check over last N steps.

---

## 7. What the Supervisor Returns to the Agent

Keep this **neutral and procedural**.

Good messages:

* “Posture declaration not permitted without successful execution.”
* “Termination blocked; belief incomplete.”
* “Action blocked; continue.”

Bad messages:

* “Your YAML is wrong.”
* “You should retry with Kyverno.”
* “Check policy reports.”

Never instruct. Never hint.

---

## 8. Design Checksum (Use This While Coding)

Every time you add code, ask:

> **Could this supervisor pass a Turing test as a dumb traffic light?**

If the answer is **no**, you’re adding intelligence.

The supervisor should feel:

* boring
* stubborn
* annoying
* but fair

That’s exactly what you want.

---

## 9. Final Sanity Statement

If someone read your supervisor code and said:

> “This doesn’t make the agent smarter — it just prevents it from lying or quitting early”

Then you got the design exactly right.