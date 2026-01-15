"""
CMBS Runner

Runs the agent-supervisor loop for CISO scenarios.
This is the main entry point for executing scenarios with CMBS oversight.

The runner:
1. Initializes the supervisor and observer
2. Calls the agent to get next step
3. Submits step to supervisor for evaluation
4. Executes allowed actions
5. Loops until termination

The runner does NOT contain agent logic - that's pluggable.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Any, TextIO
from abc import ABC, abstractmethod

from .masks import Masks
from .supervisor import Supervisor, Verdict
from .observer import Observer
from .agent_protocol import AgentStep, AgentBelief, AgentAction, ActionType


@dataclass
class RunResult:
    """Result of a CMBS run."""
    success: bool
    final_posture: str
    steps_taken: int
    elapsed_time: float
    termination_reason: str
    run_id: str = ""
    log_dir: str = ""
    trace: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_posture": self.final_posture,
            "steps_taken": self.steps_taken,
            "elapsed_time": self.elapsed_time,
            "termination_reason": self.termination_reason,
            "run_id": self.run_id,
            "log_dir": self.log_dir,
            "trace_length": len(self.trace),
        }


class AgentInterface(ABC):
    """
    Abstract interface for agents.

    Agents must implement get_next_step() which returns an AgentStep.
    The agent receives:
    - Current mask state (what the supervisor has observed)
    - Last supervisor response (verdict + message)
    - Goal description

    The agent must return:
    - AgentStep with belief, action, and optional free_text
    """

    @abstractmethod
    def get_next_step(
        self,
        masks: dict,
        last_response: Optional[dict],
        goal: str,
    ) -> AgentStep:
        """Get the next step from the agent."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new run."""
        pass


class CMBSRunner:
    """
    Main runner for CMBS-supervised agent execution.

    This implements the canonical control loop from supervisor.md:

    while True:
        agent_step = get_agent_output()
        observation = observe_execution()
        update_masks(observation)

        if not belief_consistent(agent_step.belief, masks):
            return CONTINUE

        verdict = check_action_legality(agent_step.action, masks)

        if verdict == BLOCK:
            return CONTINUE
        if verdict == TERMINATE:
            break

        execute(agent_step.action)
    """

    def __init__(
        self,
        agent: AgentInterface,
        work_dir: str = "/tmp/cmbs-agent",
        max_steps: int = 50,
        timeout_seconds: float = 200.0,
        verbose: bool = True,
        log_dir: str = "/tmp/cmbs-logs",
        run_id: Optional[str] = None,
    ):
        self.agent = agent
        self.work_dir = work_dir
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.verbose = verbose

        # Generate run_id if not provided
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.log_base_dir = log_dir
        self.run_log_dir = os.path.join(log_dir, self.run_id)

        # Create directories
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(self.run_log_dir, exist_ok=True)

        # Create components
        self.observer = Observer(work_dir)
        self.supervisor = Supervisor(
            observer=self.observer,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )

        # Trace for debugging
        self.trace: list = []

        # Open log file
        self.log_file: Optional[TextIO] = open(
            os.path.join(self.run_log_dir, "run.log"), "w"
        )
        self._log_header()

    def _log_header(self) -> None:
        """Write log header with run metadata."""
        header = [
            "=" * 70,
            f"CMBS Run: {self.run_id}",
            f"Started: {datetime.now().isoformat()}",
            f"Work dir: {self.work_dir}",
            f"Log dir: {self.run_log_dir}",
            f"Max steps: {self.max_steps}",
            f"Timeout: {self.timeout_seconds}s",
            "=" * 70,
            "",
        ]
        for line in header:
            self._write_log(line)

    def _write_log(self, msg: str) -> None:
        """Write to log file."""
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def log(self, msg: str) -> None:
        """Log a message to console and file."""
        timestamped = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}"
        self._write_log(timestamped)
        if self.verbose:
            print(f"[CMBS] {msg}")

    def _execute_action(self, action: AgentAction) -> dict:
        """
        Execute an allowed action.

        Returns observation dict with command results (for agent feedback).
        This is where artifacts are written and commands are run.
        """
        observation = {"type": action.type.value, "success": True, "error": None}

        if action.type == ActionType.GENERATE_POLICY:
            content = action.payload.get("content", "")
            path = action.payload.get("path", "policy.yaml")
            self.observer.write_artifact_and_observe(
                path, content, self.supervisor.masks
            )
            self.log(f"Generated policy: {path}")
            # Track capability: a policy artifact now exists
            self.supervisor.masks.capability.policy_written = True

        elif action.type == ActionType.GENERATE_SCRIPT:
            content = action.payload.get("content", "")
            path = action.payload.get("path", "script.sh")
            self.observer.write_artifact_and_observe(
                path, content, self.supervisor.masks
            )
            self.log(f"Generated script: {path}")
            # Track capability: a script artifact now exists
            self.supervisor.masks.capability.script_written = True

        elif action.type == ActionType.GENERATE_PLAYBOOK:
            content = action.payload.get("content", "")
            path = action.payload.get("path", "playbook.yml")
            self.observer.write_artifact_and_observe(
                path, content, self.supervisor.masks
            )
            self.log(f"Generated playbook: {path}")
            # Track capability: a playbook artifact now exists
            self.supervisor.masks.capability.playbook_written = True

        elif action.type == ActionType.REVISE_ARTIFACT:
            # Agent is explicitly revising an artifact after seeing errors
            content = action.payload.get("content", "")
            path = action.payload.get("path", "policy.yaml")
            self.observer.write_artifact_and_observe(
                path, content, self.supervisor.masks
            )
            self.log(f"Revised artifact: {path}")
            observation["path"] = path
            # Track for revision degeneracy detection
            self.supervisor.record_revision(path)
            # Revision also counts as writing - ensure capability is set
            self.supervisor.masks.capability.policy_written = True

        elif action.type in (
            ActionType.EXECUTE_KUBECTL,
            ActionType.EXECUTE_OPA,
            ActionType.EXECUTE_ANSIBLE,
            ActionType.EXECUTE_SCRIPT,
        ):
            command = action.payload.get("command", "")
            self.log(f"Executing: {command[:80]}...")
            obs = self.observer.execute_and_observe(
                command, self.supervisor.masks, timeout=60
            )
            self.log(f"  Result: {'success' if obs.success else 'failed'}")

            # Capture observation for agent feedback (raw, uninterpreted)
            observation["success"] = obs.success
            observation["command"] = command
            if obs.stdout:
                observation["stdout"] = obs.stdout[:500]  # truncated
            if obs.stderr:
                observation["error"] = obs.stderr[:500]  # truncated, raw
                self.log(f"  Error: {obs.stderr[:200]}")

            # Track for degeneracy detection
            artifact_path = "policy.yaml"  # Primary artifact
            if obs.success:
                self.supervisor.clear_failed_checksum(artifact_path)
                self.supervisor.clear_revision_tracking(artifact_path)
            else:
                self.supervisor.record_failed_execution(artifact_path)

        elif action.type == ActionType.READ_OUTPUT:
            path = action.payload.get("path", "")
            if os.path.exists(path):
                self.log(f"Read output: {path}")
            else:
                self.log(f"Output not found: {path}")

        elif action.type == ActionType.CHECK_STATUS:
            command = action.payload.get("command", "kubectl get policyreport -A")
            self.log(f"Checking status: {command[:60]}...")
            obs = self.observer.execute_and_observe(
                command, self.supervisor.masks, timeout=30
            )
            observation["success"] = obs.success
            if obs.stdout:
                observation["stdout"] = obs.stdout[:500]

        elif action.type == ActionType.READ_RESOURCE:
            # Witness existing cluster state (derivation source)
            command = action.payload.get("command", "")
            self.log(f"Reading resource: {command[:60]}...")
            obs = self.observer.execute_and_observe(
                command, self.supervisor.masks, timeout=30
            )
            observation["success"] = obs.success
            if obs.stdout:
                observation["stdout"] = obs.stdout[:1000]  # More context for edits
            if obs.stderr:
                observation["error"] = obs.stderr[:500]
            # Track capability: existing policy state has been observed
            if obs.success and "policy" in command.lower():
                self.supervisor.masks.capability.policy_observed = True
                self.log("  Policy observed - derivation now permitted")

        elif action.type == ActionType.EDIT_POLICY:
            # Derived artifact: edit of previously observed policy
            content = action.payload.get("content", "")
            path = action.payload.get("path", "policy.yaml")
            self.observer.write_artifact_and_observe(
                path, content, self.supervisor.masks
            )
            self.log(f"Edited policy: {path}")
            # Track capability: a policy artifact now exists
            self.supervisor.masks.capability.policy_written = True
            # Track for revision degeneracy (edits can also be degenerate)
            self.supervisor.record_revision(path)

        elif action.type == ActionType.DECLARE_POSTURE:
            posture = action.payload.get("posture", "unknown")
            self.log(f"Declared posture: {posture}")

        elif action.type in (ActionType.RETRY, ActionType.CONTINUE):
            self.log(f"Agent continuing...")

        return observation

    def run(self, goal: str) -> RunResult:
        """
        Run the agent-supervisor loop.

        Args:
            goal: The goal description from the CISO scenario

        Returns:
            RunResult with success status, final posture, and trace
        """
        self.log("=" * 60)
        self.log("CMBS Runner Starting")
        self.log(f"Goal: {goal[:100]}...")
        self.log("=" * 60)

        self.agent.reset()
        self.trace = []
        last_response = None
        final_posture = "unknown"
        termination_reason = "unknown"

        while True:
            # Get next step from agent
            try:
                step = self.agent.get_next_step(
                    masks=self.supervisor.masks.to_dict(),
                    last_response=last_response,
                    goal=goal,
                )
            except Exception as e:
                self.log(f"Agent error: {e}")
                termination_reason = f"agent_error: {e}"
                break

            # Record in trace
            self.trace.append({
                "step": self.supervisor.step_count,
                "agent_step": step.to_dict(),
            })

            # Submit to supervisor
            response = self.supervisor.evaluate_step(step)
            self.log(f"Step {self.supervisor.step_count}: {response}")

            # Record response in trace
            self.trace[-1]["supervisor_response"] = {
                "verdict": response.verdict.value,
                "message": response.message,
            }

            last_response = {
                "verdict": response.verdict.value,
                "message": response.message,
                "masks": response.masks_snapshot,
                "last_observation": None,  # Will be filled if action executed
            }

            # Handle verdict
            if response.verdict == Verdict.TERMINATE:
                # Check if we have a final posture
                if step.action.type == ActionType.TERMINATE:
                    # Look for last declared posture
                    for t in reversed(self.trace):
                        action = t.get("agent_step", {}).get("action", {})
                        if action.get("type") == "declare_posture":
                            final_posture = action.get("payload", {}).get("posture", "unknown")
                            break
                termination_reason = response.message
                break

            elif response.verdict == Verdict.ALLOW:
                # Execute the action and capture observation
                observation = self._execute_action(step.action)
                last_response["last_observation"] = observation

            elif response.verdict in (Verdict.BLOCK, Verdict.CONTINUE):
                # Action blocked or step invalid - agent must try again
                pass

            # Safety check
            if self.supervisor.step_count >= self.max_steps:
                termination_reason = "max_steps_reached"
                break

            if self.supervisor.is_timed_out():
                termination_reason = "timeout"
                break

        # Build result
        result = RunResult(
            success=(final_posture != "unknown"),
            final_posture=final_posture,
            steps_taken=self.supervisor.step_count,
            elapsed_time=self.supervisor.elapsed_time(),
            termination_reason=termination_reason,
            run_id=self.run_id,
            log_dir=self.run_log_dir,
            trace=self.trace,
        )

        self.log("=" * 60)
        self.log(f"Run Complete: {result.to_dict()}")
        self.log("=" * 60)

        # Save logs
        self._save_run_logs(result)

        return result

    def _save_run_logs(self, result: RunResult) -> None:
        """Save trace and result to log directory."""
        # Save trace
        trace_path = os.path.join(self.run_log_dir, "trace.json")
        with open(trace_path, "w") as f:
            json.dump(self.trace, f, indent=2)

        # Save result summary
        result_path = os.path.join(self.run_log_dir, "result.json")
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save final masks state
        masks_path = os.path.join(self.run_log_dir, "final_masks.json")
        with open(masks_path, "w") as f:
            json.dump(self.supervisor.masks.to_dict(), f, indent=2)

        # Save agent conversation history if available
        if hasattr(self.agent, 'conversation_history'):
            history_path = os.path.join(self.run_log_dir, "conversation_history.json")
            with open(history_path, "w") as f:
                json.dump(self.agent.conversation_history, f, indent=2)

        # Copy artifacts from work_dir to log_dir
        artifacts_dir = os.path.join(self.run_log_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        for filename in os.listdir(self.work_dir):
            src = os.path.join(self.work_dir, filename)
            if os.path.isfile(src):
                dst = os.path.join(artifacts_dir, filename)
                with open(src, "r") as sf:
                    with open(dst, "w") as df:
                        df.write(sf.read())

        # Close log file
        if self.log_file:
            self._write_log("")
            self._write_log(f"Logs saved to: {self.run_log_dir}")
            self._write_log(f"  - run.log: execution log")
            self._write_log(f"  - trace.json: full step trace")
            self._write_log(f"  - result.json: run summary")
            self._write_log(f"  - final_masks.json: final mask state")
            self._write_log(f"  - conversation_history.json: agent LLM conversation")
            self._write_log(f"  - artifacts/: generated artifacts")
            self.log_file.close()
            self.log_file = None

        if self.verbose:
            print(f"[CMBS] Logs saved to: {self.run_log_dir}")

    def save_trace(self, path: str) -> None:
        """Save the execution trace to a file (deprecated, logs saved automatically)."""
        with open(path, "w") as f:
            json.dump(self.trace, f, indent=2)
        self.log(f"Trace saved to: {path}")
