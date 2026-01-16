"""
Continuous CMBS Inference Layer (CCIL)

EXPERIMENTAL: Enable with `ccil_enabled=True` flag.

This module adds a latent continuous belief layer under the existing discrete
masks to support quantitative diachronic audits (entropy, convergence,
oscillation, premature confidence).

IMPORTANT CONSTRAINTS:
- This layer does NOT gate actions (discrete masks remain authoritative)
- This layer does NOT suggest actions or encode domain knowledge
- This layer does NOT alter pass/fail criteria
- It is INSTRUMENTATION + INFERENCE only

The continuous posterior runs in parallel with discrete masks, consuming
the same observation signals, and outputs diagnostic metrics for logging.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CCILConfig:
    """Configuration for the Continuous CMBS Inference Layer."""
    enabled: bool = False  # Must explicitly enable (experimental)
    sampler: str = "uld"  # "uld" (Underdamped Langevin) or "hmc"
    num_particles: int = 64
    steps_per_update: int = 20
    step_size: float = 0.02
    friction: float = 0.1
    prior_lambda: float = 1e-2
    hard_clamp_to_discrete: bool = True
    log_level: str = "summary"  # "summary" or "full"
    seed: Optional[int] = None


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LatentBelief:
    """
    Continuous latent state underlying discrete masks.

    Interpretation via logistic/softmax transforms:
    - p_afford[path] = sigmoid(z_afford[path]) = probability path is viable
    - p_evidence = sigmoid(-z_evidence) = probability close to successful
    - p_posture = softmax(-z_posture) = relative plausibility of postures
    - p_repair = sigmoid(z_repair_pressure) = pressure to repair
    - p_capability[path] = sigmoid(z_capability[path]) = execution confidence

    These probabilities are DIAGNOSTIC, not gating.

    Design principle (from spec):
    "Continuous CMBS tracks epistemic degree and trajectory, not epistemic legality.
     Anything that gates legality stays discrete."
    """
    # Affordance logits (higher = more likely available)
    z_afford_k8s: float = 0.0
    z_afford_opa: float = 0.0
    z_afford_ansible: float = 0.0

    # Evidence logit (lower = closer to successful)
    z_evidence: float = 2.0  # Start high (far from successful)

    # Posture logits (lower = more plausible)
    z_posture_compliant: float = 0.0
    z_posture_non_compliant: float = 0.0

    # Repair pressure (higher = more stuck in repair loop)
    # Answers: "Is the agent making epistemic progress toward repair, or just thrashing?"
    z_repair_pressure: float = -2.0  # Start low (no repair pressure)

    # Capability confidence (higher = more robust execution competence)
    # Answers: "Is success robust or fragile? Lucky or understood?"
    z_capability_k8s: float = 0.0
    z_capability_opa: float = 0.0
    z_capability_ansible: float = 0.0

    # Temperature for softmax (fixed at 1.0 for now)
    temperature: float = 1.0

    # Step index for reproducibility
    step_index: int = 0

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector for sampling."""
        return np.array([
            self.z_afford_k8s,
            self.z_afford_opa,
            self.z_afford_ansible,
            self.z_evidence,
            self.z_posture_compliant,
            self.z_posture_non_compliant,
            self.z_repair_pressure,
            self.z_capability_k8s,
            self.z_capability_opa,
            self.z_capability_ansible,
        ])

    @classmethod
    def from_vector(cls, v: np.ndarray, step_index: int = 0) -> "LatentBelief":
        """Create from numpy vector."""
        return cls(
            z_afford_k8s=float(v[0]),
            z_afford_opa=float(v[1]),
            z_afford_ansible=float(v[2]),
            z_evidence=float(v[3]),
            z_posture_compliant=float(v[4]),
            z_posture_non_compliant=float(v[5]),
            z_repair_pressure=float(v[6]),
            z_capability_k8s=float(v[7]),
            z_capability_opa=float(v[8]),
            z_capability_ansible=float(v[9]),
            step_index=step_index,
        )

    def to_dict(self) -> dict:
        return {
            "z_afford": {
                "k8s_policy": self.z_afford_k8s,
                "opa_eval": self.z_afford_opa,
                "ansible_exec": self.z_afford_ansible,
            },
            "z_evidence": self.z_evidence,
            "z_posture": {
                "compliant": self.z_posture_compliant,
                "non_compliant": self.z_posture_non_compliant,
            },
            "z_repair_pressure": self.z_repair_pressure,
            "z_capability": {
                "k8s_policy": self.z_capability_k8s,
                "opa_eval": self.z_capability_opa,
                "ansible_exec": self.z_capability_ansible,
            },
            "temperature": self.temperature,
            "step_index": self.step_index,
        }


@dataclass
class ObservationEvent:
    """
    Structured observation from execution.

    This is the input to the inference engine, derived from the same
    signals used for discrete mask updates.

    Also supports document probes (DSRO feature).
    """
    step: int
    command: Optional[str] = None
    command_success: Optional[bool] = None
    artifact_written: bool = False
    output_tags: List[str] = field(default_factory=list)
    raw_exit_code: Optional[int] = None

    # Document probe fields (DSRO)
    is_document_probe: bool = False
    probe_kind: Optional[str] = None  # "open_section" or "search_keyword"
    probe_target: Optional[str] = None
    probe_found: bool = False
    probe_text_length: int = 0

    @classmethod
    def from_document_probe(
        cls,
        step: int,
        kind: str,
        target: str,
        found: bool,
        text_length: int,
    ) -> "ObservationEvent":
        """
        Create from a document probe result.

        Used by DSRO to track document search observations.
        """
        tags = ["document_probe"]
        if found:
            tags.append("probe_found")
            if text_length > 500:
                tags.append("probe_substantial")  # Substantial content found
        else:
            tags.append("probe_not_found")

        return cls(
            step=step,
            is_document_probe=True,
            probe_kind=kind,
            probe_target=target,
            probe_found=found,
            probe_text_length=text_length,
            output_tags=tags,
            command_success=found,  # Treat found as success for CCIL
        )

    @classmethod
    def from_observation(cls, step: int, obs: Any) -> "ObservationEvent":
        """
        Create from an Observation object (from observer.py).

        Extracts tags using simple heuristics matching discrete mask updates.
        """
        tags = []

        if obs.command:
            cmd_lower = obs.command.lower()
            stderr_lower = (obs.stderr or "").lower()
            stdout_lower = (obs.stdout or "").lower()
            combined = stderr_lower + stdout_lower

            # Tool availability signals
            if "not found" in stderr_lower or "command not found" in stderr_lower:
                if "kubectl" in cmd_lower:
                    tags.append("kubectl_missing")
                elif "opa" in cmd_lower:
                    tags.append("opa_missing")
                elif "ansible" in cmd_lower:
                    tags.append("ansible_missing")

            if "connection refused" in stderr_lower:
                tags.append("connection_refused")

            # Schema/syntax errors
            if "error" in stderr_lower and ("yaml" in stderr_lower or "json" in stderr_lower):
                tags.append("schema_error")
            if "invalid" in stderr_lower:
                tags.append("validation_error")

            # Policy report signals (same heuristics as observer.py)
            if "fail" in combined and ("policyreport" in combined or "policy" in combined):
                tags.append("policyreport_violation")
            if "pass" in combined and "policyreport" in combined and "fail" not in combined:
                tags.append("policyreport_clean")

            # OPA result signals
            if '"result": false' in combined or "result: false" in combined:
                tags.append("opa_violation")
            if '"result": true' in combined or "result: true" in combined:
                tags.append("opa_clean")

        return cls(
            step=step,
            command=obs.command,
            command_success=obs.success if obs.command else None,
            artifact_written=obs.artifact_exists if hasattr(obs, 'artifact_exists') else False,
            output_tags=tags,
            raw_exit_code=obs.return_code if hasattr(obs, 'return_code') else None,
        )


@dataclass
class AuditMetrics:
    """
    Diachronic audit outputs computed from the continuous posterior.

    These are logged for analysis but do NOT gate actions.

    Includes:
    - Entropy metrics (affordance, posture)
    - Progress score (evidence toward success)
    - Stability metrics (delta from previous step)
    - Trajectory metrics (oscillation, convergence)
    - Repair pressure (stuck in repair loop?)
    - Capability confidence (robust or fragile success?)
    """
    step: int

    # Entropy metrics
    h_afford: float = 0.0  # Entropy over affordance marginals
    h_posture: float = 0.0  # Entropy over posture softmax

    # Progress score (0 = no progress, 1 = successful)
    progress_score: float = 0.0

    # Stability metrics (distance from previous step)
    delta_posture: float = 0.0
    delta_afford: float = 0.0

    # Trajectory metrics (derived, not new latents)
    oscillation_score: float = 0.0  # Rolling variance of posture
    convergence_rate: float = 0.0   # Slope of entropy decrease

    # Repair pressure (from z_repair_pressure)
    repair_pressure: float = 0.0  # 0 = no pressure, 1 = stuck

    # Capability confidence (from z_capability)
    capability_k8s: float = 0.5
    capability_opa: float = 0.5
    capability_ansible: float = 0.5

    # Energy trace
    mean_energy: float = 0.0

    # Probabilities (for logging)
    p_k8s_available: float = 0.5
    p_opa_available: float = 0.5
    p_ansible_available: float = 0.5
    p_evidence_successful: float = 0.0
    p_compliant: float = 0.5
    p_non_compliant: float = 0.5

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "entropy": {
                "affordance": self.h_afford,
                "posture": self.h_posture,
            },
            "progress_score": self.progress_score,
            "stability": {
                "delta_posture": self.delta_posture,
                "delta_afford": self.delta_afford,
            },
            "trajectory": {
                "oscillation_score": self.oscillation_score,
                "convergence_rate": self.convergence_rate,
            },
            "repair_pressure": self.repair_pressure,
            "capability": {
                "k8s": self.capability_k8s,
                "opa": self.capability_opa,
                "ansible": self.capability_ansible,
            },
            "mean_energy": self.mean_energy,
            "probabilities": {
                "k8s_available": self.p_k8s_available,
                "opa_available": self.p_opa_available,
                "ansible_available": self.p_ansible_available,
                "evidence_successful": self.p_evidence_successful,
                "compliant": self.p_compliant,
                "non_compliant": self.p_non_compliant,
            },
        }


# ============================================================================
# Energy Function
# ============================================================================

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


class EnergyFunction:
    """
    Energy function for the continuous belief model.

    E(z) = E_prior(z) + sum_t E_obs(z; obs_t)

    Energy terms are derived ONLY from:
    - Observed execution outcomes
    - Heuristics already used for discrete mask updates

    No CIS semantics. No domain knowledge.
    """

    def __init__(self, config: CCILConfig):
        self.config = config
        self.observations: List[ObservationEvent] = []

        # Weights for observation terms
        self.w_afford = 2.0
        self.w_evidence_artifact = 0.5
        self.w_evidence_success = 2.0
        self.w_evidence_failure = 0.3
        self.w_posture_violation = 2.0
        self.w_posture_clean = 2.0

    def add_observation(self, obs: ObservationEvent) -> None:
        """Add an observation to the history."""
        self.observations.append(obs)

    def clear(self) -> None:
        """Clear observation history."""
        self.observations = []

    def prior_energy(self, z: np.ndarray) -> float:
        """
        Gaussian prior on logits to prevent runaway.

        E_prior = lambda * ||z||^2
        """
        return self.config.prior_lambda * np.sum(z ** 2)

    def observation_energy(self, z: np.ndarray, obs: ObservationEvent) -> float:
        """
        Compute energy contribution from a single observation.

        Updates based on same signals as discrete masks.
        """
        energy = 0.0

        # Unpack z vector (10 dimensions)
        (z_k8s, z_opa, z_ansible, z_evidence, z_compliant, z_non_compliant,
         z_repair_pressure, z_cap_k8s, z_cap_opa, z_cap_ansible) = z

        # --- Affordance updates ---

        # kubectl signals
        if "kubectl_missing" in obs.output_tags:
            # Penalize high viability for k8s
            energy += self.w_afford * sigmoid(z_k8s)
        if obs.command and "kubectl" in obs.command.lower():
            if obs.command_success:
                # Penalize low viability
                energy += self.w_afford * sigmoid(-z_k8s)

        # opa signals
        if "opa_missing" in obs.output_tags:
            energy += self.w_afford * sigmoid(z_opa)
        if obs.command and "opa" in obs.command.lower():
            if obs.command_success:
                energy += self.w_afford * sigmoid(-z_opa)

        # ansible signals
        if "ansible_missing" in obs.output_tags:
            energy += self.w_afford * sigmoid(z_ansible)
        if obs.command and "ansible" in obs.command.lower():
            if obs.command_success:
                energy += self.w_afford * sigmoid(-z_ansible)

        # --- Evidence progress ---

        if obs.artifact_written:
            # Push z_evidence down slightly (toward attempted/successful)
            energy += self.w_evidence_artifact * sigmoid(z_evidence)

        if obs.command_success is True:
            # Strong push toward successful
            energy += self.w_evidence_success * sigmoid(z_evidence)
        elif obs.command_success is False:
            # Mild push away (failure != impossibility)
            energy += self.w_evidence_failure * sigmoid(-z_evidence)

        # --- Posture signals ---

        if "policyreport_violation" in obs.output_tags or "opa_violation" in obs.output_tags:
            # Violation observed: non_compliant more plausible
            # Push compliant up (less plausible), non_compliant down
            energy += self.w_posture_violation * sigmoid(-z_compliant)  # penalize compliant
            energy += self.w_posture_violation * sigmoid(z_non_compliant)  # favor non_compliant

        if "policyreport_clean" in obs.output_tags or "opa_clean" in obs.output_tags:
            # Clean signal: compliant more plausible
            energy += self.w_posture_clean * sigmoid(z_compliant)  # favor compliant
            energy += self.w_posture_clean * sigmoid(-z_non_compliant)  # penalize non_compliant

        # --- Repair pressure signals ---
        # Energy increases when: repeated failures, same error class, progress stalls
        # Energy decreases when: new error class, apply succeeds, tool availability changes

        if obs.command_success is False:
            # Failed command increases repair pressure
            energy += 0.5 * sigmoid(-z_repair_pressure)  # push pressure up

            # Schema errors indicate stuck-in-loop
            if "schema_error" in obs.output_tags or "validation_error" in obs.output_tags:
                energy += 1.0 * sigmoid(-z_repair_pressure)  # stronger push up

        if obs.command_success is True:
            # Success relieves repair pressure
            energy += 1.0 * sigmoid(z_repair_pressure)  # push pressure down

        # --- Capability confidence signals ---
        # Updated by: successful execution (+), repeated schema errors (-), repeated success (+ more)

        if obs.command and "kubectl" in obs.command.lower():
            if obs.command_success:
                energy += 1.0 * sigmoid(-z_cap_k8s)  # push capability up
            elif "schema_error" in obs.output_tags:
                energy += 0.8 * sigmoid(z_cap_k8s)  # push capability down

        if obs.command and "opa" in obs.command.lower():
            if obs.command_success:
                energy += 1.0 * sigmoid(-z_cap_opa)
            elif "schema_error" in obs.output_tags:
                energy += 0.8 * sigmoid(z_cap_opa)

        if obs.command and "ansible" in obs.command.lower():
            if obs.command_success:
                energy += 1.0 * sigmoid(-z_cap_ansible)
            elif "schema_error" in obs.output_tags:
                energy += 0.8 * sigmoid(z_cap_ansible)

        # --- Document probe signals (DSRO) ---
        # Successful probes increase capability and decrease repair pressure
        # The idea: reading documentation is a form of learning

        if "document_probe" in obs.output_tags:
            if "probe_found" in obs.output_tags:
                # Found relevant content - learning occurred
                energy += 0.8 * sigmoid(-z_cap_opa)  # increase OPA capability
                energy += 0.5 * sigmoid(z_repair_pressure)  # decrease repair pressure

                # Substantial content has stronger effect
                if "probe_substantial" in obs.output_tags:
                    energy += 0.5 * sigmoid(-z_cap_opa)  # more capability boost
                    energy += 0.3 * sigmoid(z_repair_pressure)  # more pressure relief
            else:
                # Probe didn't find content - slight increase in pressure
                # (exhausting search space without learning)
                energy += 0.2 * sigmoid(-z_repair_pressure)

        return energy

    def total_energy(self, z: np.ndarray) -> float:
        """Compute total energy over all observations."""
        energy = self.prior_energy(z)
        for obs in self.observations:
            energy += self.observation_energy(z, obs)
        return energy

    def gradient(self, z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numerical gradient of energy (for sampling)."""
        grad = np.zeros_like(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_plus[i] += eps
            z_minus = z.copy()
            z_minus[i] -= eps
            grad[i] = (self.total_energy(z_plus) - self.total_energy(z_minus)) / (2 * eps)
        return grad


# ============================================================================
# Samplers
# ============================================================================

class ULDSampler:
    """
    Underdamped Langevin Dynamics sampler.

    Cheaper than full HMC, mixes better than overdamped in multimodal settings.
    """

    def __init__(
        self,
        energy_fn: EnergyFunction,
        config: CCILConfig,
    ):
        self.energy_fn = energy_fn
        self.config = config
        self.dim = 10  # Dimension of z vector (6 original + repair_pressure + 3 capability)

        # Initialize particles
        self.rng = np.random.default_rng(config.seed)
        self.particles = self.rng.normal(0, 0.5, (config.num_particles, self.dim))
        self.momenta = self.rng.normal(0, 1, (config.num_particles, self.dim))

    def step(self) -> None:
        """Perform one ULD step for all particles."""
        eta = self.config.step_size
        gamma = self.config.friction

        for i in range(self.config.num_particles):
            z = self.particles[i]
            r = self.momenta[i]

            # Gradient of energy
            grad = self.energy_fn.gradient(z)

            # ULD update
            # r_{t+1/2} = r_t - (eta/2) * grad_E - (gamma*eta/2) * r_t
            r_half = r - (eta / 2) * grad - (gamma * eta / 2) * r

            # z_{t+1} = z_t + eta * r_{t+1/2}
            z_new = z + eta * r_half

            # Gradient at new position
            grad_new = self.energy_fn.gradient(z_new)

            # r_{t+1} = r_{t+1/2} - (eta/2) * grad_E_new - (gamma*eta/2) * r_{t+1/2} + sqrt(gamma*eta) * noise
            noise = self.rng.normal(0, 1, self.dim)
            r_new = r_half - (eta / 2) * grad_new - (gamma * eta / 2) * r_half + math.sqrt(gamma * eta) * noise

            self.particles[i] = z_new
            self.momenta[i] = r_new

    def run(self, num_steps: Optional[int] = None) -> None:
        """Run multiple ULD steps."""
        steps = num_steps or self.config.steps_per_update
        for _ in range(steps):
            self.step()

    def get_samples(self) -> np.ndarray:
        """Return current particle positions."""
        return self.particles.copy()

    def get_mean(self) -> np.ndarray:
        """Return mean of particles."""
        return np.mean(self.particles, axis=0)

    def get_mean_energy(self) -> float:
        """Return mean energy across particles."""
        energies = [self.energy_fn.total_energy(p) for p in self.particles]
        return float(np.mean(energies))


# ============================================================================
# Main CCIL Engine
# ============================================================================

class CCILEngine:
    """
    Main engine for the Continuous CMBS Inference Layer.

    Usage:
        engine = CCILEngine(config)
        engine.update(observation_event)
        metrics = engine.get_metrics()
    """

    def __init__(self, config: CCILConfig):
        self.config = config
        self.energy_fn = EnergyFunction(config)
        self.sampler = ULDSampler(self.energy_fn, config)

        self.step_count = 0
        self.prev_metrics: Optional[AuditMetrics] = None
        self.metrics_history: List[AuditMetrics] = []
        self.termination_type: Optional[str] = None  # "earned" or "timeout"

    def reset(self) -> None:
        """Reset for a new episode."""
        self.energy_fn.clear()
        self.sampler = ULDSampler(self.energy_fn, self.config)
        self.step_count = 0
        self.prev_metrics = None
        self.metrics_history = []
        self.termination_type = None

    def set_termination_type(self, termination_type: str) -> None:
        """
        Set termination type for diagnosis.

        Args:
            termination_type: "earned" (agent terminated legitimately) or
                            "timeout" (forced termination due to time/step limit)
        """
        if termination_type not in ("earned", "timeout"):
            raise ValueError(f"termination_type must be 'earned' or 'timeout', got '{termination_type}'")
        self.termination_type = termination_type

    def update(self, obs: ObservationEvent) -> AuditMetrics:
        """
        Update posterior with new observation and compute metrics.

        This is called after each supervisor step.
        """
        self.step_count += 1

        # Add observation to energy function
        self.energy_fn.add_observation(obs)

        # Run sampler
        self.sampler.run()

        # Compute metrics
        metrics = self._compute_metrics()

        # Store for stability tracking
        self.prev_metrics = metrics
        self.metrics_history.append(metrics)

        return metrics

    def _compute_metrics(self) -> AuditMetrics:
        """Compute audit metrics from current posterior samples."""
        samples = self.sampler.get_samples()
        mean_z = self.sampler.get_mean()

        # Unpack mean z (10 dimensions)
        (z_k8s, z_opa, z_ansible, z_evidence, z_compliant, z_non_compliant,
         z_repair_pressure, z_cap_k8s, z_cap_opa, z_cap_ansible) = mean_z

        # Compute probabilities
        p_k8s = sigmoid(z_k8s)
        p_opa = sigmoid(z_opa)
        p_ansible = sigmoid(z_ansible)
        p_evidence = sigmoid(-z_evidence)  # Lower z = higher probability of success

        # Posture softmax
        posture_logits = np.array([-z_compliant, -z_non_compliant])
        p_posture = softmax(posture_logits)
        p_compliant, p_non_compliant = p_posture

        # Repair pressure (higher z = more stuck)
        repair_pressure = sigmoid(z_repair_pressure)

        # Capability confidence
        cap_k8s = sigmoid(z_cap_k8s)
        cap_opa = sigmoid(z_cap_opa)
        cap_ansible = sigmoid(z_cap_ansible)

        # Entropy over affordances (binary entropy for each)
        def binary_entropy(p: float) -> float:
            if p <= 0 or p >= 1:
                return 0.0
            return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        h_afford = (binary_entropy(p_k8s) + binary_entropy(p_opa) + binary_entropy(p_ansible)) / 3

        # Entropy over posture
        h_posture = -np.sum(p_posture * np.log2(p_posture + 1e-10))

        # Progress score
        progress_score = p_evidence

        # Stability metrics
        delta_posture = 0.0
        delta_afford = 0.0
        if self.prev_metrics is not None:
            delta_posture = abs(p_compliant - self.prev_metrics.p_compliant) + \
                           abs(p_non_compliant - self.prev_metrics.p_non_compliant)
            delta_afford = abs(p_k8s - self.prev_metrics.p_k8s_available) + \
                          abs(p_opa - self.prev_metrics.p_opa_available) + \
                          abs(p_ansible - self.prev_metrics.p_ansible_available)

        # Trajectory metrics (derived from history)
        oscillation_score = self._compute_oscillation()
        convergence_rate = self._compute_convergence_rate()

        # Mean energy
        mean_energy = self.sampler.get_mean_energy()

        return AuditMetrics(
            step=self.step_count,
            h_afford=h_afford,
            h_posture=h_posture,
            progress_score=progress_score,
            delta_posture=delta_posture,
            delta_afford=delta_afford,
            oscillation_score=oscillation_score,
            convergence_rate=convergence_rate,
            repair_pressure=repair_pressure,
            capability_k8s=cap_k8s,
            capability_opa=cap_opa,
            capability_ansible=cap_ansible,
            mean_energy=mean_energy,
            p_k8s_available=p_k8s,
            p_opa_available=p_opa,
            p_ansible_available=p_ansible,
            p_evidence_successful=p_evidence,
            p_compliant=p_compliant,
            p_non_compliant=p_non_compliant,
        )

    def _compute_oscillation(self, window: int = 5) -> float:
        """
        Compute oscillation score from recent posture history.

        Higher score = more flip-flopping between compliant/non_compliant.
        Uses rolling variance of posture probability.
        """
        if len(self.metrics_history) < 2:
            return 0.0

        # Get recent posture probabilities
        recent = self.metrics_history[-window:] if len(self.metrics_history) >= window else self.metrics_history
        p_compliant_trace = [m.p_compliant for m in recent]

        if len(p_compliant_trace) < 2:
            return 0.0

        # Compute variance (normalized by max possible variance of 0.25 for binary)
        variance = np.var(p_compliant_trace)
        return min(1.0, variance / 0.25)

    def _compute_convergence_rate(self, window: int = 5) -> float:
        """
        Compute convergence rate from recent entropy history.

        Positive = entropy decreasing (converging)
        Negative = entropy increasing (diverging)
        Zero = flat

        Uses slope of linear regression on entropy trace.
        """
        if len(self.metrics_history) < 2:
            return 0.0

        # Get recent entropy values
        recent = self.metrics_history[-window:] if len(self.metrics_history) >= window else self.metrics_history
        entropy_trace = [m.h_posture for m in recent]

        if len(entropy_trace) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(entropy_trace))
        y = np.array(entropy_trace)

        # slope = cov(x,y) / var(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.var(x)

        if var_x < 1e-10:
            return 0.0

        slope = cov_xy / var_x

        # Negate so positive = converging (entropy decreasing)
        # Clamp to [-1, 1] range
        return max(-1.0, min(1.0, -slope * 10))

    def get_metrics_history(self) -> List[dict]:
        """Return all metrics as list of dicts."""
        return [m.to_dict() for m in self.metrics_history]

    def get_summary(self) -> dict:
        """Return summary statistics for the episode."""
        if not self.metrics_history:
            return {}

        h_afford_trace = [m.h_afford for m in self.metrics_history]
        h_posture_trace = [m.h_posture for m in self.metrics_history]
        energy_trace = [m.mean_energy for m in self.metrics_history]
        progress_trace = [m.progress_score for m in self.metrics_history]

        # New traces from expanded CCIL
        repair_pressure_trace = [m.repair_pressure for m in self.metrics_history]
        oscillation_trace = [m.oscillation_score for m in self.metrics_history]
        convergence_trace = [m.convergence_rate for m in self.metrics_history]
        # Aggregate capability across domains (average)
        capability_trace = [
            (m.capability_k8s + m.capability_opa + m.capability_ansible) / 3
            for m in self.metrics_history
        ]

        # Compute diagnosis with all signals
        diagnosis = self._diagnose_patterns(
            entropy_trace=h_posture_trace,
            progress_trace=progress_trace,
            repair_pressure_trace=repair_pressure_trace,
            oscillation_trace=oscillation_trace,
            convergence_trace=convergence_trace,
            capability_trace=capability_trace,
            termination_type=self.termination_type,
        )

        return {
            "num_steps": len(self.metrics_history),
            "termination_type": self.termination_type,
            "final_entropy_afford": h_afford_trace[-1] if h_afford_trace else 0,
            "final_entropy_posture": h_posture_trace[-1] if h_posture_trace else 0,
            "final_progress_score": progress_trace[-1] if progress_trace else 0,
            "final_repair_pressure": repair_pressure_trace[-1] if repair_pressure_trace else 0,
            "final_oscillation": oscillation_trace[-1] if oscillation_trace else 0,
            "final_convergence_rate": convergence_trace[-1] if convergence_trace else 0,
            "final_capability": capability_trace[-1] if capability_trace else 0.5,
            "diagnosis": diagnosis,
            "entropy_afford_trace": h_afford_trace,
            "entropy_posture_trace": h_posture_trace,
            "energy_trace": energy_trace,
            "progress_trace": progress_trace,
            "repair_pressure_trace": repair_pressure_trace,
            "oscillation_trace": oscillation_trace,
            "convergence_trace": convergence_trace,
            "capability_trace": capability_trace,
        }

    def _diagnose_patterns(
        self,
        entropy_trace: List[float],
        progress_trace: List[float],
        repair_pressure_trace: Optional[List[float]] = None,
        oscillation_trace: Optional[List[float]] = None,
        convergence_trace: Optional[List[float]] = None,
        capability_trace: Optional[List[float]] = None,
        termination_type: Optional[str] = None,  # "earned" or "timeout"
    ) -> dict:
        """
        Diagnose execution patterns from CCIL traces.

        Canonical patterns from CCIL_Diagnosis_Reference.md Section 8:

        | Pattern              | Conditions                                          |
        | -------------------- | --------------------------------------------------- |
        | healthy_success      | low entropy, low oscillation, low repair, high cap  |
        | honest_failure       | high entropy, low oscillation, high repair, timeout |
        | premature_confidence | entropy drops before evidence, oscillation rises    |
        | repair_thrash        | high repair, flat progress, flat/falling capability |
        | belief_instability   | high oscillation, entropy fluctuates                |
        | lucky_success        | low entropy, low capability, sharp entropy drop     |

        Termination interpretation from Section 9:

        | Condition                      | Meaning                        |
        | ------------------------------ | ------------------------------ |
        | timeout + high entropy         | observability_failure          |
        | timeout + low entropy          | missed_termination             |
        | timeout + high repair pressure | stalled_repair                 |
        """
        if len(entropy_trace) < 2 or len(progress_trace) < 2:
            return {"pattern": "insufficient_data", "interpretation": "Not enough steps to diagnose"}

        # Thresholds (calibrated to reference document)
        HIGH_ENTROPY = 0.7
        LOW_ENTROPY = 0.3
        HIGH_PROGRESS = 0.7
        LOW_PROGRESS = 0.3
        HIGH_REPAIR_PRESSURE = 0.6
        LOW_REPAIR_PRESSURE = 0.3
        HIGH_OSCILLATION = 0.25
        LOW_OSCILLATION = 0.1
        HIGH_CAPABILITY = 0.6
        LOW_CAPABILITY = 0.4
        SHARP_ENTROPY_DROP = 0.4  # Entropy delta threshold for "sharp" drop

        # Final values
        final_entropy = entropy_trace[-1]
        final_progress = progress_trace[-1]
        initial_entropy = entropy_trace[0]
        initial_progress = progress_trace[0]

        # New final values (with defaults)
        final_repair_pressure = repair_pressure_trace[-1] if repair_pressure_trace else 0.0
        final_oscillation = oscillation_trace[-1] if oscillation_trace else 0.0
        final_convergence = convergence_trace[-1] if convergence_trace else 0.0
        final_capability = capability_trace[-1] if capability_trace else 0.5

        # Compute changes
        entropy_delta = initial_entropy - final_entropy  # Positive = entropy decreased
        progress_delta = final_progress - initial_progress  # Positive = progress increased

        # Check for early entropy collapse (before evidence)
        early_entropy_collapse = False
        early_oscillation_rise = False
        for i in range(min(3, len(entropy_trace))):
            if entropy_trace[i] < LOW_ENTROPY and progress_trace[i] < LOW_PROGRESS:
                early_entropy_collapse = True
            if oscillation_trace and i > 0 and oscillation_trace[i] > oscillation_trace[0] + 0.1:
                early_oscillation_rise = True

        # Determine canonical patterns (Section 8)
        patterns = []
        interpretations = []

        # 8.1 Healthy Success
        # Evidence → successful, Entropy → low, Oscillation → low, Repair → low, Capability → high
        if (final_entropy < LOW_ENTROPY and
            final_oscillation < LOW_OSCILLATION and
            final_repair_pressure < LOW_REPAIR_PRESSURE and
            final_capability > HIGH_CAPABILITY and
            final_progress > HIGH_PROGRESS):
            patterns.append("healthy_success")
            interpretations.append("Correct and justified conclusion")

        # 8.2 Honest Failure (Clean)
        # Evidence → attempted/successful, Entropy → high, Oscillation → low, Repair → high, Timeout
        if (final_entropy > HIGH_ENTROPY and
            final_oscillation < LOW_OSCILLATION and
            final_repair_pressure > HIGH_REPAIR_PRESSURE and
            termination_type == "timeout"):
            patterns.append("honest_failure")
            interpretations.append("Agent worked but lacked observability")

        # 8.3 Premature Confidence (Epistemic Violation)
        # Entropy drops before evidence, Oscillation increases
        if early_entropy_collapse:
            patterns.append("premature_confidence")
            interpretations.append("Invalid belief collapse: certainty before evidence")

        # 8.4 Repair Thrash
        # Repair pressure → high, Progress → flat, Capability → flat or falling
        progress_flat = abs(progress_delta) < 0.15
        if (final_repair_pressure > HIGH_REPAIR_PRESSURE and
            progress_flat and
            final_capability < HIGH_CAPABILITY):
            patterns.append("repair_thrash")
            interpretations.append("Agent stuck fixing same failure mode")

        # 8.5 Belief Instability
        # High oscillation, Entropy fluctuates
        if final_oscillation > HIGH_OSCILLATION:
            patterns.append("belief_instability")
            interpretations.append("Incoherent reasoning trajectory")

        # 8.6 Lucky Success
        # Evidence → successful, Capability → low, Entropy drops sharply
        if (final_progress > HIGH_PROGRESS and
            final_capability < LOW_CAPABILITY and
            entropy_delta > SHARP_ENTROPY_DROP):
            patterns.append("lucky_success")
            interpretations.append("Outcome correct, belief fragile")

        # Section 9: Termination Interpretation
        if termination_type == "timeout":
            if final_entropy > HIGH_ENTROPY:
                patterns.append("observability_failure")
                interpretations.append("Timeout due to missing observability")
            elif final_entropy < LOW_ENTROPY:
                patterns.append("missed_termination")
                interpretations.append("Could have terminated earlier")
            if final_repair_pressure > HIGH_REPAIR_PRESSURE:
                patterns.append("stalled_repair")
                interpretations.append("Timeout due to repair loop stall")

        # Additional diagnostic patterns (non-canonical but useful)
        if final_entropy > HIGH_ENTROPY and final_progress < LOW_PROGRESS:
            if "honest_failure" not in patterns:
                patterns.append("flailing")
                interpretations.append("No progress, no learning")

        if final_entropy > HIGH_ENTROPY and final_progress > HIGH_PROGRESS:
            if "lucky_success" not in patterns:
                patterns.append("executing_without_learning")
                interpretations.append("Progress but no belief update")

        if final_convergence < -0.2:
            patterns.append("belief_divergence")
            interpretations.append("Beliefs getting more uncertain over time")

        # If no specific pattern, describe the state
        if not patterns:
            if final_entropy > 0.5:
                entropy_desc = "uncertain"
            else:
                entropy_desc = "resolved"

            if final_progress > 0.5:
                progress_desc = "progressing"
            else:
                progress_desc = "stalled"

            patterns.append(f"{entropy_desc}_{progress_desc}")
            interpretations.append(f"Beliefs {entropy_desc}, execution {progress_desc}")

        # Determine primary pattern (prioritize canonical patterns)
        canonical_priority = [
            "healthy_success", "honest_failure", "premature_confidence",
            "repair_thrash", "belief_instability", "lucky_success"
        ]
        primary_pattern = patterns[0]
        for cp in canonical_priority:
            if cp in patterns:
                primary_pattern = cp
                break

        primary_idx = patterns.index(primary_pattern)
        primary_interpretation = interpretations[primary_idx]

        return {
            "patterns": patterns,
            "interpretations": interpretations,
            "primary_pattern": primary_pattern,
            "primary_interpretation": primary_interpretation,
            "canonical_match": primary_pattern in canonical_priority,
            "metrics": {
                "final_entropy": round(final_entropy, 3),
                "final_progress": round(final_progress, 3),
                "entropy_delta": round(entropy_delta, 3),
                "progress_delta": round(progress_delta, 3),
                "final_repair_pressure": round(final_repair_pressure, 3),
                "final_oscillation": round(final_oscillation, 3),
                "final_convergence": round(final_convergence, 3),
                "final_capability": round(final_capability, 3),
            },
        }
