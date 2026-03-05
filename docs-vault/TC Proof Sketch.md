## 0) What you’re trying to prove

You want a theorem like:

**Theorem (Turing completeness, unbounded alternations).**
A programming model consisting of alternating monotone strata—join phases on a join-semilattice and meet phases on a meet-semilattice—connected by seal operations that can transfer unbounded state, is Turing complete (i.e., can simulate any Turing machine). 

The standard way to prove this is:

1. Pick a known Turing-complete machine model that’s easier than a full TM (e.g., **2-counter Minsky machine**, or **tag systems**, or **while programs**).
2. Show how to encode its configurations as your lattice state.
3. Show how one machine step is implemented by a bounded number of strata + seals.
4. Conclude that unbounded alternations can simulate unbounded time.

---

## 1) Formalize your computational model (minimal definitions)

You’ll need crisp definitions:

### A. Stratum

A stratum is a monotone function over a semilattice:

* Join-stratum: ( f_\lor: L_\lor \to L_\lor ) monotone w.r.t. ( \sqsubseteq ), often “accumulate facts”.
* Meet-stratum: ( f_\land: L_\land \to L_\land ) monotone w.r.t. ( \sqsupseteq ), often “eliminate hypotheses”.

### B. Seal

A seal is the only non-monotone boundary. You define it as a pair of (typed) maps between lattices:

* ( \text{seal}*{\lor\to\land}: L*\lor \to L_\land )
* ( \text{seal}*{\land\to\lor}: L*\land \to L_\lor )

Critically, your conjecture assumes **unrestricted** state transfer at seals. 
That’s what makes the proof plausible: the seal can carry an arbitrarily large encoding of machine configuration.

---

## 2) Choose the target model: 2-counter Minsky machine (recommended)

A **2-counter machine** is classic for these proofs because it’s tiny but Turing complete:

* Two unbounded natural-number registers (C_1, C_2)
* Instructions:

  * `INC(i); goto k`
  * `DECJZ(i); if Ci==0 goto k else Ci-- ; goto m`
* Program counter `pc`

Turing completeness: a 2-counter machine can simulate a TM.

So your job becomes: simulate a 2-counter machine step.

---

## 3) Encoding: represent configuration as lattice state

Configuration is ( (pc, C_1, C_2) ).

### Option 1: Encode in join-lattice facts (simplest)

Let join-lattice (L_\lor) be a set of ground facts (a powerset lattice under union):

Include facts like:

* `PC(k)` (exactly one true)
* `C1(n)` and `C2(n)` as unary numerals (or a multiset encoding)

But “exactly one PC” and “exactly one number” are *not monotone* constraints. That’s OK because enforcement happens at seals / via a meet-phase that eliminates invalid interpretations.

### Better: multiset encoding

Represent a counter value (n) by **n tokens**:

* `Tok1(t)` for each token (t) present (so count = cardinality)

Then increment = add a token (monotone in join-phase). Decrement = remove a token (monotone in meet-phase if meet is intersection / elimination).

This is extremely compatible with your duality: **join adds**, **meet removes**.

---

## 4) Show you can implement each instruction using a bounded stratum pattern

This is the heart of the proof sketch: for each instruction type, define a finite “macro” of strata + seals that transforms the encoding of one configuration into the next.

### 4.1 INC(i)

Goal: ( (pc=k, C_i=n) \mapsto (pc=k', C_i=n+1) )

**Join-stratum** can do this directly:

* Add a fresh token to counter (i)
* Add fact `PC(k')`
* (Optionally) keep old `PC(k)` around temporarily

Then a **meet-stratum** cleans up:

* eliminate all but one `PC(*)` (remove the old one)

Because meet is shrinking, “cleanup” is elimination-compatible.

So an INC step can be:

* join (accumulate: add token + new pc)
* seal
* meet (eliminate: remove old pc / enforce canonicality)
* seal

### 4.2 DECJZ(i)

This is where non-DAG / branching matters.

Goal:

* if (C_i=0) jump to (k)
* else decrement and jump to (m)

You need *a test* for zero, which is inherently non-monotone if done naively (“absence of tokens”).

This is why seals matter: they concentrate the “coordination / non-monotonicity” moment. 

Two proof routes:

#### Route A: Seal computes a decision bit (allowed under “unrestricted state transfer”)

At seal, compute:

* `IsZero(i)` based on whether any token exists for counter i
  Then next stratum uses that bit monotonically.

This is basically admitting the seal is an oracle for a non-monotone query. If seals are permitted to do that, the rest is straightforward.

#### Route B: Use dualization so “zero-test” becomes monotone in the meet world

You can store *hypotheses about a token’s existence* in a meet-lattice and eliminate them. But at some point you still need to detect “none remain” to decide the branch—again a boundary phenomenon.

In either case, your proof will say:

* **Lemma (Zero test via seal).** There exists a seal interface that produces a stable boolean summary of token presence for a counter.

Once you have `IsZero(i)`, you can implement:

* if `IsZero(i)` then set pc=k
* else remove one token and set pc=m

Remove-one-token is meet-friendly (elimination), add-new-pc is join-friendly.

So DECJZ step is still a bounded alternation macro.

---

## 5) Show sequential composition and unbounded runtime

Once you can simulate **one instruction step**, the rest is standard:

* Define a “step macro” ( \text{Step} ) that maps configuration encoding to next configuration encoding.
* Show you can apply Step repeatedly by allowing the overall computation to run for unbounded number of strata alternations (“data-dependent strata” in your doc). 

Then:

**Lemma (Iteration).** If Step is implementable, then any finite number of steps ( \text{Step}^t ) is implementable by running the alternation loop t times.

**Corollary.** The model can simulate any run of a 2-counter machine, hence is Turing complete.

---

## 6) What you must be explicit about (or reviewers will attack)

### A. Exactly what seals are allowed to do

Your conjecture says “state transfer unrestricted.” 
But “unrestricted” must be pinned down:

* Is seal a pure function of current lattice state?
* Can it compute arbitrary non-monotone predicates (like emptiness)?
* Is it allowed to “select” one branch?

If yes, TC is easy but the model becomes “monotone strata + powerful boundary operator,” and your proof basically reduces to: *seals provide the missing negation/control*.

If no, you need a more delicate construction (e.g., encode control so that the system *derives* the branch without non-monotone tests). That’s harder but more interesting.

### B. Canonicalization and uniqueness

If you enforce “exactly one PC” and a canonical counter representation, you’ll need to show:

* cleanup always succeeds
* cleanup doesn’t destroy required info
* cleanup is monotone within its stratum

### C. Unboundedness source

TC requires an unbounded resource:

* either unbounded alternations,
* or unbounded lattice universe growth (fresh tokens),
* ideally both.

---

## 7) The cleanest “paper proof” structure

Here’s a tidy outline you can drop into the doc:

1. **Define SD-machine**: alternating monotone strata + seals.
2. **Choose Minsky machine** as source formalism.
3. **Encoding**: map Minsky configuration → SD state.
4. **Instruction simulation**:

   * Construction for `INC`
   * Construction for `DECJZ` (seal yields decision / or alternative)
5. **Composition lemma**: sequencing of instruction simulations.
6. **Unbounded execution**: repeated alternations simulate arbitrary steps.
7. **Therefore** SD is Turing complete.

---

## 8) One honest sentence to include

You should explicitly say something like:

> “Turing completeness hinges on the expressive power of the seal operator; in the unrestricted transfer setting, seals can implement the non-monotone tests (e.g., emptiness) needed for conditional branching.”

That keeps it rigorous and prevents “you smuggled in the TM at the boundary” criticism.