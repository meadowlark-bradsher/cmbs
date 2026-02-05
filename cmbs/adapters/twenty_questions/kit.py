"""
Twenty Questions reference kit loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    question: str
    keep: Dict[str, List[str]]


@dataclass(frozen=True)
class TwentyQKit:
    hypotheses: List[str]
    actions: Dict[str, ActionSpec]
    actions_order: List[str]


def load_kit(path: str) -> TwentyQKit:
    data = _load_yaml(path)
    hypotheses = list(data.get("hypotheses", []))
    actions_raw = list(data.get("actions", []))
    actions: Dict[str, ActionSpec] = {}
    order: List[str] = []
    for item in actions_raw:
        action_id = item["id"]
        question = item["question"]
        keep = item["keep"]
        actions[action_id] = ActionSpec(action_id=action_id, question=question, keep=keep)
        order.append(action_id)
    return TwentyQKit(hypotheses=hypotheses, actions=actions, actions_order=order)


def _load_yaml(path: str) -> Dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "pyyaml is required to load kit files. Install with `pip install pyyaml`."
        ) from exc
    return yaml.safe_load(Path(path).read_text())
