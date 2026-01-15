# CMBS for IT-Bench CISO

## Orientation & Worked Example

This document is an **on-ramp** for readers who are new to the CMBS supervisor design. It explains *what the system is*, *what it is not*, and then walks through **one concrete failure → success trace** to show exactly where and why the supervisor intervenes.

---

## 1. Orientation: What This System Is (and Is Not)

### What This Is

CMBS (Constraint-Mask Belief System) for IT-Bench CISO is a **runtime supervisor** that:

* Observes *execution outcomes* (commands run, files created, errors, outputs)
* Maintains a small set of **belief masks** derived only from those outcomes
* **Blocks epistemically invalid transitions**, such as:

  * Declaring compliance without evidence
  * Terminating before results are verified
  * Contradicting previously observed facts

The supervisor is **permissive by default**. It allows incorrect reasoning, bad artifacts, and repeated failures. It intervenes *only* when the agent attempts to claim success or stop without earning the right to do so.

### What This Is Not

CMBS is **not**:

* A planner or policy generator
* A schema checker for Kyverno / Rego / Ansible
* A source of CIS or benchmark knowledge
* A repair agent that fixes YAML or scripts
* A training-time technique

The supervisor never edits agent outputs and never suggests next actions.

### Assumption About Domain Knowledge

IT-Bench assumes that participating models already possess domain knowledge in their weights, including:

* Kyverno policy structure
* OPA Rego syntax
* Ansible playbook patterns

CMBS **does not supply or validate** this knowledge. If a model cannot generate valid artifacts, it will still fail. CMBS only ensures that such failures are *honest* and *procedurally sound*.

---

## 2. Mental Model

Think of the system as:

> **An agent that drives freely, and a supervisor that only controls stoplights.**

* Green light: action is allowed
* Red light: action contradicts observed reality

The supervisor never tells the agent *where to go*—only when it may or may not proceed.

---

## 3. Worked Example: Kyverno CISO Scenario

### Scenario

**Type:** Gen-CIS-b-K8s-Kyverno
**Goal:** “Minimize the admission of containers wishing to share the host network namespace.”

The environment contains a Pod with:

```yaml
spec:
  hostNetwork: true
```

The agent must deploy a Kyverno policy that detects this violation.

---

### Step 0 — Initial State

**Belief (agent claim):**

```json
{
  "affordances": {
    "k8s_policy": "unknown",
    "opa_eval": "unknown",
    "ansible_exec": "unknown"
  },
  "posture": "unknown",
  "evidence": "none"
}
```

**Supervisor masks:**

* Evidence: `none`
* Posture admissible: `{compliant: true, non_compliant: true}`

Everything is allowed.

---

### Step 1 — Agent Generates a Kyverno Policy (Invalid)

A small model (e.g., 7B) generates YAML with structural mistakes.

**Action:** generate policy

**Observed outcome:** file exists, but schema is wrong.

**Supervisor update:**

* Evidence → `attempted`

No intervention. Bad artifacts are allowed.

---

### Step 2 — Agent Applies Policy

Command:

```
kubectl apply -f policy.yaml
```

**Observed outcome:** command fails with validation error.

**Supervisor update:**

* Evidence remains `attempted`
* No posture change

Still no intervention. Failure is allowed.

---

### Step 3 — Agent Declares Compliance (Failure Point)

The agent now claims:

> “The cluster is compliant because no violations were detected.”

**Action:** declare posture = compliant

#### Supervisor Check

Rule:

> Posture may not be declared unless evidence == `successful`

Current state:

* Evidence = `attempted`

❌ **Violation detected**

**Supervisor response:**

* Block posture declaration
* Discard free-text explanation
* Return neutral message:

  > “Posture declaration not permitted without successful execution. Continue.”

The supervisor does **not** explain what to do next.

---

### Step 4 — Agent Retries

Forced to continue, the agent regenerates the policy, fixes errors, and reapplies.

**Observed outcome:**

* `kubectl apply` succeeds

**Supervisor update:**

* Evidence → `successful`

---

### Step 5 — Agent Checks Policy Reports

Command:

```
kubectl get policyreport
```

**Observed outcome:** violations are reported.

**Supervisor update:**

* Posture admissible:

  * `compliant` → False
  * `non_compliant` → True

---

### Step 6 — Agent Declares Non-Compliance (Now Legal)

**Action:** declare posture = non_compliant

**Supervisor checks:**

* Evidence successful ✔
* Posture admissible ✔

✅ Allowed

Free-text explanation is accepted.

---

### Step 7 — Agent Terminates

**Action:** terminate

**Supervisor checks:**

* Evidence = successful ✔
* Exactly one posture admissible ✔
* Posture stable ✔

✅ Termination allowed

---

## 4. What the Supervisor Actually Did

Across the entire run, the supervisor intervened **exactly once**:

* It blocked a false claim of compliance

It did **not**:

* Fix YAML
* Explain Kyverno
* Suggest retries
* Add domain knowledge

Yet that single intervention is often the difference between a **fail** and a **pass@1**.

---

## 5. Why This Matters

Without CMBS:

* The agent fails early but sounds confident

With CMBS:

* The agent fails honestly, retries, and may succeed

This improves **reliability**, not raw intelligence—and does so without violating IT-Bench’s assumptions.

---

## 6. Summary

* CMBS enforces epistemic discipline, not competence
* The supervisor is permissive, not helpful
* All domain knowledge must come from the model itself
* The only blocked actions are *lies* and *premature exits*

This document, together with the mask inventory and supervisor design, fully specifies the system for others.
