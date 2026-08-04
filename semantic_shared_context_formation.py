from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from contextual_core import ContextualSphereBrain
from semantic_encoder_v2 import StructuredInput, component_nodes, load_brain
from semantic_encoder_v2_contextual import encode_and_experience_contextual


@dataclass
class PairMetrics:
    activation_similarity: float
    node_similarity: float
    edge_similarity: float
    shared_edges: int
    left_only_edges: int
    right_only_edges: int

    def to_dict(self) -> dict:
        return asdict(self)


def _jaccard(left: Iterable, right: Iterable) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _weighted_similarity(left: np.ndarray, right: np.ndarray) -> float:
    indexes = set(np.flatnonzero(left > 0).tolist()) | set(np.flatnonzero(right > 0).tolist())
    if not indexes:
        return 1.0
    numerator = sum(min(float(left[i]), float(right[i])) for i in indexes)
    denominator = sum(max(float(left[i]), float(right[i])) for i in indexes)
    return numerator / denominator if denominator else 0.0


def _direct_nodes(brain, item: StructuredInput) -> set[int]:
    return set(
        component_nodes(brain, "role:subject", "subject", 2)
        + component_nodes(brain, "entity", item.subject, 3)
        + component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", item.relation, 3)
        + component_nodes(brain, "role:content", "content", 2)
        + component_nodes(brain, "content", item.content, 3)
    )


def _filtered_signature(brain, item: StructuredInput):
    experience = encode_and_experience_contextual(brain, item, learn=False)
    result = experience.content_result
    direct = _direct_nodes(brain, item)
    nodes = {int(v) for v in result.activated_nodes if int(v) not in direct}
    edges = {
        tuple(int(x) for x in edge)
        for edge in result.traversed_edges
        if int(edge[0]) not in direct and int(edge[1]) not in direct
    }
    activation = result.final_activation.copy()
    for node in direct:
        activation[int(node)] = 0.0
    return activation, nodes, edges


def _compare(brain, left_item: StructuredInput, right_item: StructuredInput) -> PairMetrics:
    left_activation, left_nodes, left_edges = _filtered_signature(brain, left_item)
    right_activation, right_nodes, right_edges = _filtered_signature(brain, right_item)
    return PairMetrics(
        activation_similarity=_weighted_similarity(left_activation, right_activation),
        node_similarity=_jaccard(left_nodes, right_nodes),
        edge_similarity=_jaccard(left_edges, right_edges),
        shared_edges=len(left_edges & right_edges),
        left_only_edges=len(left_edges - right_edges),
        right_only_edges=len(right_edges - left_edges),
    )


def _learn(brain, item: StructuredInput) -> None:
    encode_and_experience_contextual(brain, item, learn=True)


def _clone_brain() -> ContextualSphereBrain:
    return ContextualSphereBrain.from_brain(load_brain())


def _measure(brain) -> dict:
    bird_action = StructuredInput("鳥", "動作", "羽ばたく")
    plane_action = StructuredInput("飛行機", "動作", "飛行する")
    butterfly_action = StructuredInput("蝶", "動作", "羽ばたく")
    drone_action = StructuredInput("ドローン", "動作", "飛行する")

    target = _compare(brain, bird_action, plane_action)
    butterfly_to_plane = _compare(brain, butterfly_action, plane_action)
    drone_to_bird = _compare(brain, drone_action, bird_action)

    return {
        "target": target.to_dict(),
        "transfer": {
            "butterfly_to_plane": butterfly_to_plane.to_dict(),
            "drone_to_bird": drone_to_bird.to_dict(),
            "average_edge_similarity": (
                butterfly_to_plane.edge_similarity + drone_to_bird.edge_similarity
            ) / 2.0,
        },
    }


def run_experiment(checkpoints: list[int] | None = None) -> dict:
    checkpoints = sorted(set(checkpoints or [0, 1, 3, 5, 10]))
    if not checkpoints or checkpoints[0] < 0 or checkpoints[-1] > 30:
        raise ValueError("チェックポイントは0〜30で指定してください。")

    experimental = _clone_brain()
    control = _clone_brain()

    experimental_curriculum = [
        StructuredInput("鳥", "動作", "羽ばたく"),
        StructuredInput("飛行機", "動作", "飛行する"),
        StructuredInput("鳥", "場所", "空"),
        StructuredInput("飛行機", "場所", "空"),
    ]
    control_curriculum = [
        StructuredInput("鳥", "動作", "羽ばたく"),
        StructuredInput("飛行機", "動作", "飛行する"),
        StructuredInput("鳥", "場所", "森"),
        StructuredInput("飛行機", "場所", "空港"),
    ]

    records: list[dict] = []
    baseline_shared_exp = None
    baseline_shared_control = None
    max_cycle = checkpoints[-1]

    for cycle in range(max_cycle + 1):
        if cycle in checkpoints:
            exp = _measure(experimental)
            ctrl = _measure(control)
            if baseline_shared_exp is None:
                baseline_shared_exp = exp["target"]["shared_edges"]
                baseline_shared_control = ctrl["target"]["shared_edges"]
            records.append({
                "cycle": cycle,
                "experimental": exp,
                "control": ctrl,
                "context_effect": exp["target"]["edge_similarity"] - ctrl["target"]["edge_similarity"],
                "shared_edge_effect": exp["target"]["shared_edges"] - ctrl["target"]["shared_edges"],
                "experimental_shared_growth": exp["target"]["shared_edges"] - baseline_shared_exp,
                "control_shared_growth": ctrl["target"]["shared_edges"] - baseline_shared_control,
                "transfer_effect": exp["transfer"]["average_edge_similarity"] - ctrl["transfer"]["average_edge_similarity"],
            })

        if cycle == max_cycle:
            break
        for item in experimental_curriculum:
            _learn(experimental, item)
        for item in control_curriculum:
            _learn(control, item)

    final = records[-1]
    confirmed = (
        final["context_effect"] > 0.02
        and final["shared_edge_effect"] > 0
        and final["transfer_effect"] > 0.01
    )
    return {
        "checkpoints": checkpoints,
        "experimental_curriculum": [item.label for item in experimental_curriculum],
        "control_curriculum": [item.label for item in control_curriculum],
        "records": records,
        "confirmed": confirmed,
        "verdict": (
            "共有文脈による橋形成を確認" if confirmed
            else "共有文脈による橋形成は未確認"
        ),
    }
