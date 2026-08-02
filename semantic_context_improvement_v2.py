from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from semantic_context_preservation import _jaccard, _subject_context, _weighted_similarity
from semantic_encoder_v2 import component_nodes, load_brain


@dataclass
class Result:
    mode: str
    subject: str
    relation: str
    context_nodes: list[tuple[int, float]]
    active_nodes: list[tuple[int, float]]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    subject_edge_count: int
    relation_edge_count: int
    mixed_edge_count: int
    steps: int
    stop_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ranked(activation: np.ndarray) -> list[tuple[int, float]]:
    return sorted(
        ((int(i), float(activation[i])) for i in np.flatnonzero(activation > 0)),
        key=lambda item: (-item[1], item[0]),
    )


def _normalized_context(context: dict[int, float], anchor: float) -> dict[int, float]:
    """Keep relative strengths while anchoring the strongest trace above threshold."""
    if not context:
        return {}
    peak = max(context.values())
    if peak <= 0:
        return {}
    return {
        int(node): float(np.clip((value / peak) * anchor, 0.0, 1.0))
        for node, value in context.items()
    }


def _propagate(
    brain,
    relation_sources: list[int],
    raw_context: dict[int, float],
    *,
    steps: int,
    mode: str,
    threshold: float,
    context_anchor: float,
    context_decay: float,
) -> tuple[np.ndarray, set[int], set[tuple[int, int]], dict[str, int], int, str]:
    threshold = max(float(threshold), 0.18)
    context = _normalized_context(raw_context, max(float(context_anchor), threshold))

    activation = np.zeros(brain.node_count, dtype=float)
    origin: dict[int, set[str]] = {}
    for index, node in enumerate(relation_sources):
        node = int(node)
        activation[node] = max(activation[node], 1.0 - index * 0.08)
        origin.setdefault(node, set()).add("relation")
    for node, value in context.items():
        activation[node] = max(activation[node], value)
        origin.setdefault(node, set()).add("subject")

    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed_edges: set[tuple[int, int]] = set()
    edge_origins: dict[tuple[int, int], set[str]] = {}
    executed = 0
    stop_reason = "step_limit"

    for step in range(max(0, int(steps))):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            stop_reason = "no_active_sources"
            break

        contributions: dict[int, list[tuple[float, int, set[str]]]] = {}
        for source in active_sources:
            source = int(source)
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            for local_index in best_indices:
                target = int(neighbors[local_index])
                value = float(scores[local_index]) * brain.signal_decay
                # v2: resonance collects weak routes before thresholding.
                if mode != "resonance" and value < threshold:
                    continue
                if value > 0:
                    contributions.setdefault(target, []).append(
                        (value, source, set(origin.get(source, {"unknown"})))
                    )

        candidates: list[tuple[int, float, list[tuple[float, int, set[str]]], set[str]]] = []
        for target, items in contributions.items():
            if mode == "resonance":
                value = min(1.0, sum(item[0] for item in items))
                if value < threshold:
                    continue
                used = items
            else:
                strongest = max(items, key=lambda item: item[0])
                value = strongest[0]
                used = [strongest]
            labels: set[str] = set()
            for _, _, source_labels in used:
                labels.update(source_labels)
            candidates.append((target, value, used, labels))

        candidates.sort(key=lambda item: item[1], reverse=True)
        remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
        limit = min(brain.max_active_per_step, len(candidates))
        selected = []
        new_count = 0
        for item in candidates:
            target = item[0]
            is_new = target not in activated_nodes
            if is_new and new_count >= remaining_capacity:
                continue
            selected.append(item)
            if is_new:
                new_count += 1
            if len(selected) >= limit:
                break

        next_activation = np.zeros(brain.node_count, dtype=float)
        next_origin: dict[int, set[str]] = {}
        for target, value, used, labels in selected:
            next_activation[target] = max(next_activation[target], float(np.clip(value, 0.0, 1.0)))
            next_origin.setdefault(target, set()).update(labels)
            for _, source, source_labels in used:
                edge = tuple(sorted((int(source), int(target))))
                traversed_edges.add(edge)
                edge_origins.setdefault(edge, set()).update(source_labels)

        if mode in {"persistent", "resonance"}:
            retained_scale = float(context_decay) ** (step + 1)
            for node, value in context.items():
                retained = value * retained_scale
                if retained >= threshold:
                    next_activation[node] = max(next_activation[node], retained)
                    next_origin.setdefault(node, set()).add("subject")

        active_now = set(np.flatnonzero(next_activation > 0).tolist())
        if not active_now:
            stop_reason = "below_threshold"
            break

        activation = next_activation
        origin = next_origin
        activated_nodes.update(active_now)
        executed += 1
        if len(activated_nodes) >= brain.max_total_active_nodes:
            stop_reason = "node_limit"
            break

    counts = {"subject": 0, "relation": 0, "mixed": 0}
    for labels in edge_origins.values():
        if "subject" in labels and "relation" in labels:
            counts["mixed"] += 1
        elif "subject" in labels:
            counts["subject"] += 1
        elif "relation" in labels:
            counts["relation"] += 1

    return activation, activated_nodes, traversed_edges, counts, executed, stop_reason


def observe(subject: str, relation: str, mode: str, *, subject_steps: int = 8,
            relation_steps: int = 10, context_anchor: float = 0.58,
            context_decay: float = 0.94) -> dict:
    subject = subject.strip()
    relation = relation.strip()
    if not subject or not relation:
        raise ValueError("主体と関係を入力してください。")

    brain = load_brain()
    _, raw_context = _subject_context(subject, "trace_decay", subject_steps)
    relation_sources = (
        component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", relation, 3)
    )
    final, activated, edges, counts, executed, stop_reason = _propagate(
        brain,
        relation_sources,
        raw_context,
        steps=relation_steps,
        mode=mode,
        threshold=0.18,
        context_anchor=context_anchor,
        context_decay=context_decay,
    )
    normalized = _normalized_context(raw_context, context_anchor)
    return Result(
        mode=mode,
        subject=subject,
        relation=relation,
        context_nodes=sorted(normalized.items(), key=lambda item: (-item[1], item[0])),
        active_nodes=_ranked(final),
        activated_nodes=sorted(activated),
        traversed_edges=sorted(edges),
        subject_edge_count=counts["subject"],
        relation_edge_count=counts["relation"],
        mixed_edge_count=counts["mixed"],
        steps=executed,
        stop_reason=stop_reason,
    ).to_dict()


def compare(left_subject: str, right_subject: str, relation: str, *,
            subject_steps: int = 8, relation_steps: int = 10,
            context_anchor: float = 0.58, context_decay: float = 0.94) -> dict:
    modes = [
        ("baseline", "A 基準：正規化文脈を最初だけ渡す"),
        ("persistent", "B 改善：正規化文脈を各stepで持続"),
        ("resonance", "C 改善：持続文脈＋閾値前の共鳴合成"),
    ]
    comparisons = []
    for mode, label in modes:
        left = observe(left_subject, relation, mode, subject_steps=subject_steps,
                       relation_steps=relation_steps, context_anchor=context_anchor,
                       context_decay=context_decay)
        right = observe(right_subject, relation, mode, subject_steps=subject_steps,
                        relation_steps=relation_steps, context_anchor=context_anchor,
                        context_decay=context_decay)
        left_active = {int(n) for n, _ in left["active_nodes"]}
        right_active = {int(n) for n, _ in right["active_nodes"]}
        left_all = {int(n) for n in left["activated_nodes"]}
        right_all = {int(n) for n in right["activated_nodes"]}
        left_edges = {tuple(edge) for edge in left["traversed_edges"]}
        right_edges = {tuple(edge) for edge in right["traversed_edges"]}
        comparisons.append({
            "mode": mode,
            "label": label,
            "left": left,
            "right": right,
            "context_similarity": _weighted_similarity(left["context_nodes"], right["context_nodes"]),
            "final_similarity": _weighted_similarity(left["active_nodes"], right["active_nodes"]),
            "final_node_jaccard": _jaccard(left_active, right_active),
            "all_node_jaccard": _jaccard(left_all, right_all),
            "edge_jaccard": _jaccard(left_edges, right_edges),
            "left_only_edges": len(left_edges - right_edges),
            "right_only_edges": len(right_edges - left_edges),
        })
    return {
        "left_subject": left_subject,
        "right_subject": right_subject,
        "relation": relation,
        "subject_steps": subject_steps,
        "relation_steps": relation_steps,
        "context_anchor": context_anchor,
        "context_decay": context_decay,
        "comparisons": comparisons,
    }
