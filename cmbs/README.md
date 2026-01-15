# CMBS - Constraint-Mask Belief System

A runtime supervisor for IT-Bench CISO scenarios that enforces epistemic discipline without supplying domain knowledge.

## Quick Start

```bash
# Run the test scenario
python -m cmbs.test_scenario1
```

## Directory Structure

```
cmbs/
├── __init__.py          # Package exports
├── README.md            # This file
├── masks.py             # The five belief masks
├── observer.py          # Execution outcome observation
├── agent_protocol.py    # Agent communication protocol
├── supervisor.py        # Main control loop
├── runner.py            # CISO scenario runner
├── test_scenario1.py    # Test script for Scenario 1
└── agents/
    ├── __init__.py
    └── ollama_agent.py  # Example Ollama-based agent
```

## Architecture

### The Five Masks (`masks.py`)

| Mask | Purpose | Values |
|------|---------|--------|
| **Affordance** | What execution paths exist | `unknown/available/unavailable` |
| **Evidence** | How far execution progressed | `none/attempted/successful` |
| **Posture** | Which outcomes are admissible | `{compliant: T/F, non_compliant: T/F}` |
| **Stability** | Belief oscillation tracking | `posture_stable: T/F` |
| **Termination** | Derived eligibility | Boolean |

### Supervisor Rules (`supervisor.py`)

1. **Posture requires evidence**: Cannot declare compliant/non_compliant unless `evidence == successful`
2. **Termination requires completion**: Must have successful evidence + exactly one posture admissible + stable beliefs
3. **Affordance gating**: Blocks actions requiring unavailable capabilities
4. **Everything else is allowed**: Bad YAML, wrong commands, repeated failures - all permitted

### Control Loop (`runner.py`)

```python
while True:
    agent_step = agent.get_next_step(masks, last_response, goal)
    response = supervisor.evaluate_step(agent_step)

    if response.verdict == TERMINATE:
        break
    elif response.verdict == ALLOW:
        execute_action(agent_step.action)
    # BLOCK/CONTINUE: agent must try again
```

## Key Design Principles

1. **Permissive by default** - only blocks epistemic violations
2. **No domain knowledge** - doesn't know Kyverno/Rego/Ansible schemas
3. **No hints or suggestions** - never tells agent what to do
4. **Only blocks lies and premature exits**

## Usage

### Creating a Custom Agent

```python
from cmbs.runner import AgentInterface, CMBSRunner
from cmbs.agent_protocol import AgentStep, AgentBelief, AgentAction, ActionType

class MyAgent(AgentInterface):
    def get_next_step(self, masks, last_response, goal):
        # Your logic here
        return AgentStep(
            belief=AgentBelief(...),
            action=AgentAction(type=ActionType.GENERATE_POLICY, payload={...}),
            free_text="optional explanation"
        )

    def reset(self):
        pass

# Run with CMBS supervision
agent = MyAgent()
runner = CMBSRunner(agent, work_dir="/tmp/my-run")
result = runner.run("Your goal here")
```

### Available Actions

| Action | Payload | Description |
|--------|---------|-------------|
| `generate_policy` | `{content, path}` | Create Kyverno policy |
| `generate_script` | `{content, path}` | Create shell script |
| `generate_playbook` | `{content, path}` | Create Ansible playbook |
| `execute_kubectl` | `{command}` | Run kubectl command |
| `execute_opa` | `{command}` | Run OPA command |
| `execute_ansible` | `{command}` | Run ansible-playbook |
| `check_status` | `{command}` | Check policy reports |
| `declare_posture` | `{posture}` | Declare compliant/non_compliant |
| `terminate` | `{}` | End the task |

## References

- `CMBS-for-ITBench.md` - Design rationale and worked example
- `supervisor.md` - Supervisor implementation guide
- `mask-inventory.md` - Complete mask specification
