from __future__ import annotations

from dataclasses import dataclass

from semantic_encoder_v2 import StructuredInput, encode_and_experience, load_brain
from semantic_encoder_v2_contextual import observe_contextual


@dataclass(frozen=True)
class ExperienceInput:
    subject: str
    relation: str
    content: str

    @property
    def label(self) -> str:
        return f"{self.subject}｜{self.relation}｜{self.content}"


def _edge_set(result) -> set[tuple[int, int]]:
    return {tuple(sorted((int(a), int(b)))) for a, b in result.traversed_edges}


def _node_set(result) -> set[int]:
    return {int(v) for v in result.activated_nodes}


def _active_map(result) -> dict[int, float]:
    return {
        int(index): float(value)
        for index, value in enumerate(result.final_activation)
        if float(value) > 0
    }


def _weighted_similarity(left, right) -> float:
    a = _active_map(left)
    b = _active_map(right)
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _stage_summary(left, right, name: str, left_text: str, right_text: str) -> dict:
    left_edges = _edge_set(left)
    right_edges = _edge_set(right)
    left_nodes = _node_set(left)
    right_nodes = _node_set(right)
    common_edges = sorted(left_edges & right_edges)
    left_only = sorted(left_edges - right_edges)
    right_only = sorted(right_edges - left_edges)
    common_nodes = sorted(left_nodes & right_nodes)

    activation_similarity = _weighted_similarity(left, right)
    edge_similarity = _jaccard(left_edges, right_edges)
    node_similarity = _jaccard(left_nodes, right_nodes)

    if activation_similarity >= 0.995:
        reading = "最終状態はほぼ同じ"
    elif activation_similarity >= 0.80:
        reading = "共通性を保ちながら差が残る"
    else:
        reading = "経験ごとの違いが明確に残る"

    return {
        "name": name,
        "left_text": left_text,
        "right_text": right_text,
        "activation_similarity": activation_similarity,
        "edge_similarity": edge_similarity,
        "node_similarity": node_similarity,
        "common_edges": len(common_edges),
        "left_only_edges": len(left_only),
        "right_only_edges": len(right_only),
        "common_nodes": len(common_nodes),
        "left_nodes": len(left_nodes),
        "right_nodes": len(right_nodes),
        "reading": reading,
        "common_edge_sample": common_edges[:12],
        "left_edge_sample": left_only[:12],
        "right_edge_sample": right_only[:12],
    }


def _pipeline(experience, left_item: ExperienceInput, right_item: ExperienceInput):
    stages = [
        ("主体", experience[0].subject_result, experience[1].subject_result, left_item.subject, right_item.subject),
        ("関係", experience[0].relation_result, experience[1].relation_result, left_item.relation, right_item.relation),
        ("内容", experience[0].content_result, experience[1].content_result, left_item.content, right_item.content),
    ]
    return [_stage_summary(a, b, name, lt, rt) for name, a, b, lt, rt in stages]


def build_view(
    left_subject: str,
    left_relation: str,
    left_content: str,
    right_subject: str,
    right_relation: str,
    right_content: str,
) -> dict:
    left_item = ExperienceInput(left_subject.strip(), left_relation.strip(), left_content.strip())
    right_item = ExperienceInput(right_subject.strip(), right_relation.strip(), right_content.strip())
    if not all((left_item.subject, left_item.relation, left_item.content, right_item.subject, right_item.relation, right_item.content)):
        raise ValueError("2つの経験について、主体・関係・内容をすべて入力してください。")

    old_brain = load_brain()
    old_left = encode_and_experience(old_brain, StructuredInput(left_item.subject, left_item.relation, left_item.content), learn=False)
    old_right = encode_and_experience(old_brain, StructuredInput(right_item.subject, right_item.relation, right_item.content), learn=False)

    new_left = observe_contextual(left_item.subject, left_item.relation, left_item.content)
    new_right = observe_contextual(right_item.subject, right_item.relation, right_item.content)

    old_stages = _pipeline((old_left, old_right), left_item, right_item)
    new_stages = _pipeline((new_left, new_right), left_item, right_item)

    old_final = old_stages[-1]["activation_similarity"]
    new_final = new_stages[-1]["activation_similarity"]
    return {
        "left": left_item,
        "right": right_item,
        "old": old_stages,
        "new": new_stages,
        "headline": {
            "old_final": old_final,
            "new_final": new_final,
            "difference_retained": max(0.0, old_final - new_final),
            "old_message": "旧v2では、経験差が最終状態で消えています。" if old_final >= 0.995 else "旧v2でも一部の差が残っています。",
            "new_message": "v2.1では、それぞれの経験を踏まえた差が最終状態まで残っています。" if new_final < 0.995 else "v2.1でも最終状態が同一です。",
        },
    }
