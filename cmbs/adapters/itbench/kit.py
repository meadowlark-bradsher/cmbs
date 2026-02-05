"""
ITBench reference kit loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    description: str
    outcomes: List[str]


@dataclass(frozen=True)
class ITBenchKit:
    hypotheses: List[str]
    actions: Dict[str, ActionSpec]
    actions_order: List[str]
    oracle_table: Dict[str, Dict[str, Dict[str, List[str]]]]


def load_kit(path: str) -> ITBenchKit:
    data = _load_yaml(path)
    hypotheses = list(data.get("hypotheses", []))
    actions_raw = list(data.get("actions", []))
    oracle_table = data.get("oracle_table", {})
    actions: Dict[str, ActionSpec] = {}
    order: List[str] = []
    for item in actions_raw:
        action_id = item["id"]
        description = item["description"]
        outcomes = list(item["outcomes"])
        actions[action_id] = ActionSpec(
            action_id=action_id,
            description=description,
            outcomes=outcomes,
        )
        order.append(action_id)
    return ITBenchKit(
        hypotheses=hypotheses,
        actions=actions,
        actions_order=order,
        oracle_table=oracle_table,
    )


def _load_yaml(path: str) -> Dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "pyyaml is required to load kit files. Install with `pip install pyyaml`."
        ) from exc
    return yaml.safe_load(Path(path).read_text())
