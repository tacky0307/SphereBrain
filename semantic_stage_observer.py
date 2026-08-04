from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from semantic_encoder_v2 import _context_tail, component_nodes, load_brain
from semantic_raw_output import observe_once


@dataclass
class StageState:
    name: str
    top_nodes: list[tuple[int, float]]
    active_count: int
    peak_node: int | None
    peak_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def _top_nodes(activation: np.ndarray, limit: int = 12) -> list[tuple[int, float]]:
    indexes = np.flatnonzero(activation > 0)
    ranked = sorted(
        ((int(index), float(activation[index])) for index in indexes),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: max(1, int(limit))]


def _stage(name: str, activation: np.ndarray, limit: int = 12) -> StageState:
    top = _top_nodes(activation, limit)
    return StageState(
        name=name,
        top_nodes=top,
        active_count=int(np.count_nonzero(activation > 0)),
        peak_node=top[0][0] if top else None,
        peak_value=top[0][1] if top else 0.0,
    )


def _context_activation(brain, subject_result) -> np.ndarray:
    """Reproduce the current context_nodes behavior without changing Core.

    The current Core converts every inherited context node to activation 0.34.
    This observer exposes that flattened hand-off as stage B.
    """
    activation = np.zeros(brain.node_count, dtype=float)
    for node in _context_tail(subject_result):
        activation[int(node)] = 0.34
    return activation


def observe_stages(
    subject: str,
    relation: str,
    *,
    output_steps: int = 24,
    adaptive_ratio: float = 0.35,
    top_nodes: int = 12,
) -> dict:
    subject = subject.strip()
    relation = relation.strip()
    if not subject or not relation:
        raise ValueError("主体と関係を入力してください。")

    brain = load_brain()

    subject_sources = (
        component_nodes(brain, "role:subject", "subject", 2)
        + component_nodes(brain, "entity", subject, 3)
    )
    subject_result = brain.propagate(
        subject_sources,
        steps=8,
        threshold=0.18,
        noise=0.0,
        learn=False,
    )

    context_activation = _context_activation(brain, subject_result)

    relation_sources = (
        component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", relation, 3)
    )
    relation_result = brain.propagate(
        relation_sources,
        steps=10,
        threshold=0.18,
        noise=0.0,
        learn=False,
        context_nodes=_context_tail(subject_result),
    )

    raw = observe_once(
        subject,
        relation,
        output_steps=output_steps,
        mode="adaptive",
        adaptive_ratio=adaptive_ratio,
        save=False,
    )["raw_output"]
    output_activation = np.zeros(brain.node_count, dtype=float)
    for node, value in raw["active_nodes"]:
        output_activation[int(node)] = float(value)

    stages = [
        _stage("A 主体刺激直後", subject_result.final_activation, top_nodes),
        _stage("B 関係へ渡す主体文脈", context_activation, top_nodes),
        _stage("C 関係刺激直後", relation_result.final_activation, top_nodes),
        _stage("D 自由伝播終了後", output_activation, top_nodes),
    ]

    return {
        "subject": subject,
        "relation": relation,
        "stages": [stage.to_dict() for stage in stages],
        "raw_output": raw,
    }


def _signature(stage: dict) -> dict[int, float]:
    return {int(node): float(value) for node, value in stage["top_nodes"]}


def weighted_similarity(left: dict, right: dict) -> float:
    a = _signature(left)
    b = _signature(right)
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    denominator = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def compare_inputs(
    left_subject: str,
    left_relation: str,
    right_subject: str,
    right_relation: str,
    *,
    output_steps: int = 24,
    adaptive_ratio: float = 0.35,
    top_nodes: int = 12,
) -> dict:
    left = observe_stages(
        left_subject,
        left_relation,
        output_steps=output_steps,
        adaptive_ratio=adaptive_ratio,
        top_nodes=top_nodes,
    )
    right = observe_stages(
        right_subject,
        right_relation,
        output_steps=output_steps,
        adaptive_ratio=adaptive_ratio,
        top_nodes=top_nodes,
    )

    comparisons = []
    for left_stage, right_stage in zip(left["stages"], right["stages"]):
        left_nodes = set(_signature(left_stage))
        right_nodes = set(_signature(right_stage))
        comparisons.append(
            {
                "name": left_stage["name"],
                "similarity": weighted_similarity(left_stage, right_stage),
                "common_count": len(left_nodes & right_nodes),
                "left_only_count": len(left_nodes - right_nodes),
                "right_only_count": len(right_nodes - left_nodes),
            }
        )

    return {
        "left": left,
        "right": right,
        "comparisons": comparisons,
    }
