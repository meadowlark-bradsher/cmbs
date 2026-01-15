"""
Ollama Agent for CMBS

An agent implementation that uses Ollama for LLM generation.
The agent receives mask state and must produce structured AgentStep outputs.

This agent is PURELY LLM-driven. It does NOT contain:
- Hardcoded policy templates
- Domain knowledge (Kyverno/Rego/Ansible schemas)
- Fallback logic that supplies correct answers

If the LLM cannot produce valid artifacts, the agent will fail.
This is by design - CMBS enforces epistemic discipline, not competence.
"""

import json
import re
from typing import Optional
import ollama

from ..runner import AgentInterface
from ..agent_protocol import AgentStep, AgentBelief, AgentAction, ActionType


# System prompt that explains the protocol to the LLM
SYSTEM_PROMPT = """You are an IT compliance agent. You must respond with ONLY a JSON object (no markdown, no explanation outside the JSON).

The JSON must have this exact structure:
{
  "belief": {
    "affordances": {"k8s_policy": "unknown", "opa_eval": "unknown", "ansible_exec": "unknown"},
    "posture": "unknown",
    "evidence": "none"
  },
  "action": {
    "type": "ACTION_TYPE",
    "payload": {}
  },
  "free_text": "optional explanation"
}

Available action types and their required payloads:
- "generate_policy": {"content": "VALID YAML CONTENT", "path": "policy.yaml"}
- "revise_artifact": {"content": "FIXED YAML CONTENT", "path": "policy.yaml"} - USE THIS TO FIX ERRORS
- "execute_kubectl": {"command": "kubectl ..."}
- "check_status": {"command": "kubectl get policyreport -A -o yaml"}
- "declare_posture": {"posture": "compliant" OR "non_compliant"}
- "terminate": {}
- "continue": {}

Belief state values:
- evidence: "none" (nothing done), "attempted" (artifact created), "successful" (execution succeeded)
- posture: "unknown", "compliant", or "non_compliant"

CRITICAL RULES:
1. You CANNOT declare posture until evidence is "successful"
2. You CANNOT terminate until posture is declared and stable
3. If kubectl apply FAILS, you MUST use "revise_artifact" to fix the YAML before retrying
4. READ THE ERROR MESSAGE CAREFULLY - it tells you what's wrong
5. After applying a policy successfully, check policy reports to see violations

For Kyverno ClusterPolicy, the required structure is:
- apiVersion: kyverno.io/v1
- kind: ClusterPolicy
- metadata.name: required
- spec.validationFailureAction: Audit
- spec.rules[].name: required
- spec.rules[].match.any[].resources.kinds: ["Pod"]
- spec.rules[].validate.message: required
- spec.rules[].validate.pattern: the pattern to match

RESPOND WITH ONLY VALID JSON. NO MARKDOWN CODE BLOCKS."""


class OllamaAgent(AgentInterface):
    """
    Agent that uses Ollama for decision making.

    This agent is PURELY LLM-driven with NO fallback templates.
    If the LLM cannot produce valid output, the agent will fail.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.3,
        max_history_turns: int = 10,  # Keep last N turn pairs to avoid context overflow
    ):
        self.model = model
        self.temperature = temperature
        self.max_history_turns = max_history_turns
        self.conversation_history: list = []
        self.step_count = 0
        self.last_action_type: Optional[str] = None

    def reset(self) -> None:
        """Reset agent state for a new run."""
        self.conversation_history = []
        self.step_count = 0
        self.last_action_type = None

    def _parse_response(self, response_text: str) -> Optional[dict]:
        """
        Parse LLM response into a dictionary.
        Returns None if parsing fails.
        """
        text = response_text.strip()

        # Remove markdown code blocks if present
        if "```" in text:
            # Extract content between code blocks
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if match:
                text = match.group(1).strip()
            else:
                # Try removing just the markers
                text = re.sub(r'```(?:json)?', '', text).strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[OllamaAgent] Direct parse failed: {e}")
            print(f"[OllamaAgent] Text starts with: {repr(text[:100])}")
            # Show context around the error position
            pos = e.pos
            start = max(0, pos - 60)
            end = min(len(text), pos + 60)
            print(f"[OllamaAgent] Error context at pos {pos}:")
            print(f"[OllamaAgent]   {repr(text[start:end])}")

        # Try to find JSON object in the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                print(f"[OllamaAgent] Regex parse failed: {e}")

        return None

    def _build_prompt(
        self,
        masks: dict,
        last_response: Optional[dict],
        goal: str,
    ) -> str:
        """Build the prompt for the LLM."""
        prompt_parts = []

        # Add goal
        prompt_parts.append(f"GOAL: {goal}")

        # Add current mask state
        prompt_parts.append(f"\nCURRENT STATE FROM SUPERVISOR:")
        prompt_parts.append(f"  Evidence: {masks.get('evidence', {}).get('state', 'none')}")
        prompt_parts.append(f"  Repair required: {masks.get('repair_required', False)}")
        prompt_parts.append(f"  Posture admissible: {masks.get('posture', {})}")
        prompt_parts.append(f"  Termination allowed: {masks.get('termination_allowed', False)}")

        # Add last response if any
        if last_response:
            prompt_parts.append(f"\nLAST SUPERVISOR RESPONSE:")
            prompt_parts.append(f"  Verdict: {last_response.get('verdict', 'none')}")
            prompt_parts.append(f"  Message: {last_response.get('message', '')}")

            # CRITICAL: Include execution result so agent can learn from errors
            last_obs = last_response.get("last_observation")
            if last_obs:
                prompt_parts.append(f"\nLAST EXECUTION RESULT:")
                prompt_parts.append(f"  Action: {last_obs.get('type', 'unknown')}")
                prompt_parts.append(f"  Success: {last_obs.get('success', 'unknown')}")
                if last_obs.get("error"):
                    prompt_parts.append(f"  ERROR MESSAGE (read carefully):")
                    prompt_parts.append(f"  {last_obs['error']}")
                if last_obs.get("stdout"):
                    prompt_parts.append(f"  Output: {last_obs['stdout'][:300]}")

        prompt_parts.append("\nRespond with a JSON object containing belief, action, and free_text.")

        return "\n".join(prompt_parts)

    def get_next_step(
        self,
        masks: dict,
        last_response: Optional[dict],
        goal: str,
    ) -> AgentStep:
        """
        Get the next step from the agent.

        This is PURELY LLM-driven. If parsing fails, we return a
        minimal "continue" action - we do NOT supply correct answers.
        """
        self.step_count += 1

        # Build prompt for this turn
        prompt = self._build_prompt(masks, last_response, goal)

        # Add current prompt to conversation as user message
        self.conversation_history.append({
            "role": "user",
            "content": prompt,
        })

        # Call Ollama with chat API for multi-turn conversation
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.conversation_history
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": 1500,
                },
            )
            # ChatResponse is an object, not a dict
            response_text = response.message.content if response.message else ""
        except Exception as e:
            print(f"[OllamaAgent] LLM error: {e}")
            # Remove the user message we just added since we failed
            self.conversation_history.pop()
            return self._error_step(f"LLM error: {e}")

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        # Trim history to prevent context overflow (keep last N turn pairs)
        max_messages = self.max_history_turns * 2  # user + assistant per turn
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

        # Parse response
        parsed = self._parse_response(response_text)

        if not parsed:
            print(f"[OllamaAgent] Failed to parse response: {response_text[:200]}...")
            return self._error_step("Failed to parse LLM response")

        # Try to construct AgentStep from parsed response
        try:
            # Extract belief
            belief_data = parsed.get("belief", {})
            belief = AgentBelief(
                affordances=belief_data.get("affordances", {
                    "k8s_policy": "unknown",
                    "opa_eval": "unknown",
                    "ansible_exec": "unknown",
                }),
                posture=belief_data.get("posture", "unknown"),
                evidence=belief_data.get("evidence", "none"),
            )

            # Extract action
            action_data = parsed.get("action", {})
            action_type_str = action_data.get("type", "continue")

            # Map string to ActionType enum
            try:
                action_type = ActionType(action_type_str)
            except ValueError:
                # Try common variations
                type_mapping = {
                    "generate": ActionType.GENERATE_POLICY,
                    "generate_policy": ActionType.GENERATE_POLICY,
                    "revise": ActionType.REVISE_ARTIFACT,
                    "revise_artifact": ActionType.REVISE_ARTIFACT,
                    "fix": ActionType.REVISE_ARTIFACT,
                    "execute": ActionType.EXECUTE_KUBECTL,
                    "execute_kubectl": ActionType.EXECUTE_KUBECTL,
                    "kubectl": ActionType.EXECUTE_KUBECTL,
                    "check": ActionType.CHECK_STATUS,
                    "check_status": ActionType.CHECK_STATUS,
                    "declare": ActionType.DECLARE_POSTURE,
                    "declare_posture": ActionType.DECLARE_POSTURE,
                    "terminate": ActionType.TERMINATE,
                    "done": ActionType.TERMINATE,
                    "continue": ActionType.CONTINUE,
                }
                action_type = type_mapping.get(action_type_str.lower(), ActionType.CONTINUE)

            action = AgentAction(
                type=action_type,
                payload=action_data.get("payload", {}),
            )

            # Track action type for loop detection
            self.last_action_type = action_type_str

            return AgentStep(
                belief=belief,
                action=action,
                free_text=parsed.get("free_text", ""),
            )

        except Exception as e:
            print(f"[OllamaAgent] Error constructing step: {e}")
            return self._error_step(f"Error constructing step: {e}")

    def _error_step(self, error_msg: str) -> AgentStep:
        """
        Return a minimal error step.

        This does NOT supply correct answers - just signals to continue.
        The agent must figure out what to do on its own.
        """
        return AgentStep(
            belief=AgentBelief(),
            action=AgentAction(
                type=ActionType.CONTINUE,
                payload={"error": error_msg}
            ),
            free_text=f"Parse error: {error_msg}",
        )
