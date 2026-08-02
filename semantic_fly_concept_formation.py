from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from contextual_core import ContextualSphereBrain
from semantic_encoder_v2 import StructuredInput, load_brain
from semantic_encoder_v2_contextual import encode_and_experience_contextual


@dataclass(frozen=True)
class ProbeItem:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


FLY_A = ProbeItem("飛行機", "動作", "飛ぶ")
FLY_B = ProbeItem("鳥", "動作", "飛ぶ")
SUPPORT_A = ProbeItem("飛行機", "場所", "空")
SUPPORT_B = ProbeItem("鳥", "場所", "空")

OLD_ACTIONS = [
    ProbeItem("車", "動作", "走る"),
    ProbeItem("バス", "動作", "止まる"),
    ProbeItem("馬", "動作", "歩く"),
]


def _clone_brain() -> ContextualSphereBrain:
    return ContextualSphereBrain.from_brain(load_brain())


def _observe(brain: ContextualSphereBrain, item: ProbeItem):
    return encode_and_experience_contextual(
        brain,
        StructuredInput(item.subject, item.relation, item.content),
        learn=False,
        context_anchor=0.58,
        context_decay=0.94,
        resonance=True,
    )


def _learn(brain: ContextualSphereBrain, item: ProbeItem) -> None:
    encode_and_experience_contextual(
        brain,
        StructuredInput(item.subject, item.relation, item.content),
        learn=True,
        context_anchor=0.58,
        context_decay=0.94,
        resonance=True,
    )


def _weighted_similarity(left: np.ndarray, right: np.ndarray) -> float:
    a = {int(i): float(v) for i, v in enumerate(left) if float(v) > 0}
    b = {int(i): float(v) for i, v in enumerate(right) if float(v) > 0}
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def _jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _content_signature(experience) -> dict:
    result = experience.content_result
    return {
        "activation": result.final_activation,
        "nodes": set(int(v) for v in result.activated_nodes),
        "edges": set(tuple(int(x) for x in edge) for edge in result.traversed_edges),
    }


def _pair_metrics(left, right) -> dict:
    a = _content_signature(left)
    b = _content_signature(right)
    common_edges = a["edges"] & b["edges"]
    return {
        "activation_similarity": _weighted_similarity(a["activation"], b["activation"]),
        "node_similarity": _jaccard(a["nodes"], b["nodes"]),
        "edge_similarity": _jaccard(a["edges"], b["edges"]),
        "common_edges": len(common_edges),
        "left_only_edges": len(a["edges"] - b["edges"]),
        "right_only_edges": len(b["edges"] - a["edges"]),
        "left_edge_count": len(a["edges"]),
        "right_edge_count": len(b["edges"]),
    }


def _similarity_to_reference(probe, reference) -> dict:
    a = _content_signature(probe)
    b = _content_signature(reference)
    return {
        "activation": _weighted_similarity(a["activation"], b["activation"]),
        "edges": _jaccard(a["edges"], b["edges"]),
    }


def _snapshot(brain: ContextualSphereBrain, cycle: int) -> dict:
    fly_a = _observe(brain, FLY_A)
    fly_b = _observe(brain, FLY_B)
    pair = _pair_metrics(fly_a, fly_b)

    references = [(item, _observe(brain, item)) for item in OLD_ACTIONS]
    attraction_rows = []
    for item, reference in references:
        a_score = _similarity_to_reference(fly_a, reference)
        b_score = _similarity_to_reference(fly_b, reference)
        attraction_rows.append({
            "label": item.label,
            "airplane_activation": a_score["activation"],
            "bird_activation": b_score["activation"],
            "airplane_edges": a_score["edges"],
            "bird_edges": b_score["edges"],
            "average_activation": (a_score["activation"] + b_score["activation"]) / 2.0,
            "average_edges": (a_score["edges"] + b_score["edges"]) / 2.0,
        })

    attraction_rows.sort(key=lambda row: (-row["average_edges"], -row["average_activation"], row["label"]))
    old_action_average = sum(row["average_edges"] for row in attraction_rows) / len(attraction_rows)
    old_action_peak = max((row["average_edges"] for row in attraction_rows), default=0.0)

    return {
        "cycle": cycle,
        "pair": pair,
        "old_action_average": old_action_average,
        "old_action_peak": old_action_peak,
        "attractions": attraction_rows,
    }


def run_experiment(
    checkpoints: list[int] | None = None,
    *,
    include_support: bool = True,
) -> dict:
    checkpoints = checkpoints or [0, 1, 3, 5, 10]
    checkpoints = sorted(set(max(0, int(v)) for v in checkpoints))
    if not checkpoints or checkpoints[0] != 0:
        checkpoints.insert(0, 0)
    if checkpoints[-1] > 30:
        raise ValueError("最大チェックポイントは30回です。")

    brain = _clone_brain()
    snapshots = [_snapshot(brain, 0)]
    current = 0

    curriculum = [FLY_A, FLY_B]
    if include_support:
        curriculum.extend([SUPPORT_A, SUPPORT_B])

    for target in checkpoints[1:]:
        while current < target:
            for item in curriculum:
                _learn(brain, item)
            current += 1
        snapshots.append(_snapshot(brain, current))

    baseline = snapshots[0]
    for snap in snapshots:
        snap["delta_common_edges"] = snap["pair"]["common_edges"] - baseline["pair"]["common_edges"]
        snap["delta_edge_similarity"] = snap["pair"]["edge_similarity"] - baseline["pair"]["edge_similarity"]
        snap["delta_old_action_average"] = snap["old_action_average"] - baseline["old_action_average"]
        snap["concept_margin"] = snap["pair"]["edge_similarity"] - snap["old_action_average"]

    return {
        "checkpoints": checkpoints,
        "include_support": include_support,
        "curriculum": [item.label for item in curriculum],
        "snapshots": snapshots,
        "base_core_unchanged": True,
    }
