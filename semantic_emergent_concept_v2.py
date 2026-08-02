from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from contextual_core import ContextualSphereBrain
from semantic_encoder_v2 import StructuredInput, component_nodes, load_brain
from semantic_encoder_v2_contextual import encode_and_experience_contextual


@dataclass(frozen=True)
class Probe:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


def _copy_brain() -> ContextualSphereBrain:
    return ContextualSphereBrain.from_brain(load_brain())


def _direct_nodes(brain: ContextualSphereBrain, probe: Probe) -> set[int]:
    return set(
        component_nodes(brain, "role:subject", "subject", 2)
        + component_nodes(brain, "entity", probe.subject, 3)
        + component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", probe.relation, 3)
        + component_nodes(brain, "role:content", "content", 2)
        + component_nodes(brain, "content", probe.content, 3)
    )


def _filtered_signature(brain: ContextualSphereBrain, probe: Probe) -> dict:
    exp = encode_and_experience_contextual(
        brain,
        StructuredInput(probe.subject, probe.relation, probe.content),
        learn=False,
    )
    direct = _direct_nodes(brain, probe)
    edges = {
        tuple(int(v) for v in edge)
        for edge in exp.content_result.traversed_edges
        if int(edge[0]) not in direct and int(edge[1]) not in direct
    }
    nodes = {
        int(node)
        for edge in edges
        for node in edge
    }
    activation = {
        int(index): float(value)
        for index, value in enumerate(exp.content_result.final_activation)
        if float(value) > 0 and int(index) not in direct
    }
    return {"probe": probe.label, "edges": edges, "nodes": nodes, "activation": activation}


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _weighted(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def _compare(left: dict, right: dict) -> dict:
    common_edges = left["edges"] & right["edges"]
    return {
        "left": left["probe"],
        "right": right["probe"],
        "activation_similarity": _weighted(left["activation"], right["activation"]),
        "node_similarity": _jaccard(left["nodes"], right["nodes"]),
        "edge_similarity": _jaccard(left["edges"], right["edges"]),
        "common_edges": len(common_edges),
        "left_only_edges": len(left["edges"] - right["edges"]),
        "right_only_edges": len(right["edges"] - left["edges"]),
    }


def _learn_cycle(brain: ContextualSphereBrain) -> None:
    curriculum = [
        Probe("飛行機", "動作", "ルーク"),
        Probe("鳥", "動作", "ネラ"),
        Probe("飛行機", "場所", "空"),
        Probe("鳥", "場所", "空"),
        Probe("ルーク", "場所", "空"),
        Probe("ネラ", "場所", "空"),
    ]
    for item in curriculum:
        encode_and_experience_contextual(
            brain,
            StructuredInput(item.subject, item.relation, item.content),
            learn=True,
        )


def run_experiment(checkpoints: Iterable[int] = (0, 1, 3, 5, 10)) -> dict:
    points = sorted({max(0, int(value)) for value in checkpoints})
    if not points:
        raise ValueError("チェックポイントを1件以上指定してください。")
    if points[-1] > 30:
        raise ValueError("最大30サイクルです。")

    brain = _copy_brain()
    target_a = Probe("飛行機", "動作", "ルーク")
    target_b = Probe("鳥", "動作", "ネラ")
    swap_a = Probe("飛行機", "動作", "ネラ")
    swap_b = Probe("鳥", "動作", "ルーク")
    old_a = Probe("飛行機", "動作", "走る")
    old_b = Probe("鳥", "動作", "止まる")

    rows = []
    current = 0
    for checkpoint in points:
        while current < checkpoint:
            _learn_cycle(brain)
            current += 1

        target = _compare(_filtered_signature(brain, target_a), _filtered_signature(brain, target_b))
        swapped = _compare(_filtered_signature(brain, swap_a), _filtered_signature(brain, swap_b))
        legacy = _compare(_filtered_signature(brain, old_a), _filtered_signature(brain, old_b))
        specificity = target["edge_similarity"] - max(swapped["edge_similarity"], legacy["edge_similarity"])
        rows.append({
            "cycle": checkpoint,
            "target": target,
            "swapped": swapped,
            "legacy": legacy,
            "specificity_margin": specificity,
            "shared_gain": target["common_edges"],
        })

    baseline_common = rows[0]["target"]["common_edges"]
    baseline_edge = rows[0]["target"]["edge_similarity"]
    for row in rows:
        row["shared_gain"] = row["target"]["common_edges"] - baseline_common
        row["edge_gain"] = row["target"]["edge_similarity"] - baseline_edge

    return {
        "rows": rows,
        "curriculum": [
            "飛行機｜動作｜ルーク",
            "鳥｜動作｜ネラ",
            "飛行機｜場所｜空",
            "鳥｜場所｜空",
            "ルーク｜場所｜空",
            "ネラ｜場所｜空",
        ],
        "note": "直接入力ノードに接するEdgeを評価から除外しています。保存済みCoreとDBは変更しません。",
    }
