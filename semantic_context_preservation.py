from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from semantic_encoder_v2 import _context_tail, component_nodes, load_brain
from semantic_step_observer import trace_subject


@dataclass
class RelationState:
    mode: str
    subject: str
    relation: str
    context_nodes: list[tuple[int, float]]
    active_nodes: list[tuple[int, float]]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    steps: int
    stop_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ranked(activation: np.ndarray) -> list[tuple[int, float]]:
    return sorted(
        ((int(i), float(activation[i])) for i in np.flatnonzero(activation > 0)),
        key=lambda x: (-x[1], x[0]),
    )


def _subject_context(subject: str, mode: str, subject_steps: int) -> tuple[dict, dict[int, float]]:
    trace = trace_subject(subject, steps=subject_steps, threshold=0.18)
    states = trace["states"]
    context: dict[int, float] = {}

    if mode == "flat_tail":
        # Current implementation: nodes near the end of subject propagation,
        # with all strength and temporal information flattened to 0.34.
        final_history_like = []
        for state in states:
            final_history_like.append([node for node, _ in state["active_nodes"]])
        ordered: list[int] = []
        for step_nodes in reversed(final_history_like):
            for node in step_nodes:
                if node not in ordered:
                    ordered.append(int(node))
                if len(ordered) >= 18:
                    break
            if len(ordered) >= 18:
                break
        context = {node: 0.34 for node in ordered}

    elif mode == "final_activation":
        for node, value in states[-1]["active_nodes"]:
            context[int(node)] = float(value)

    elif mode == "trace_decay":
        # Preserve route and time. Recent states remain stronger, while early
        # entry traces are retained at lower strength instead of being erased.
        last_index = max(1, len(states) - 1)
        for index, state in enumerate(states):
            recency = 0.30 + 0.70 * (index / last_index)
            for node, value in state["active_nodes"]:
                preserved = float(value) * recency * 0.72
                context[int(node)] = max(context.get(int(node), 0.0), preserved)
    else:
        raise ValueError(f"未知の文脈方式です: {mode}")

    return trace, context


def _propagate_relation(
    brain,
    relation_sources: Iterable[int],
    context: dict[int, float],
    *,
    steps: int,
    threshold: float = 0.18,
) -> tuple[np.ndarray, set[int], set[tuple[int, int]], int, str]:
    sources = list(relation_sources)
    activation = np.zeros(brain.node_count, dtype=float)
    for index, node in enumerate(sources):
        activation[int(node)] = max(activation[int(node)], 1.0 - index * 0.08)
    for node, value in context.items():
        activation[int(node)] = max(activation[int(node)], float(np.clip(value, 0.0, 1.0)))

    activated_nodes = set(np.flatnonzero(activation > 0).tolist())
    traversed_edges: set[tuple[int, int]] = set()
    stop_reason = "step_limit"
    executed = 0
    threshold = max(float(threshold), 0.18)

    for _ in range(max(0, int(steps))):
        active_sources = np.flatnonzero(activation > 0)
        if active_sources.size == 0:
            stop_reason = "no_active_sources"
            break

        candidates: dict[int, tuple[float, int]] = {}
        for source in active_sources:
            neighbors = np.flatnonzero(brain.adjacency[source])
            if neighbors.size == 0:
                continue
            scores = activation[source] * brain.weights[source, neighbors]
            branch_count = min(brain.max_branches, neighbors.size)
            best_indices = np.argpartition(scores, -branch_count)[-branch_count:]
            for local_index in best_indices:
                target = int(neighbors[local_index])
                value = float(scores[local_index]) * brain.signal_decay
                if value < threshold:
                    continue
                previous = candidates.get(target)
                if previous is None or value > previous[0]:
                    candidates[target] = (value, int(source))

        if not candidates:
            stop_reason = "below_threshold"
            break

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining_capacity = max(0, brain.max_total_active_nodes - len(activated_nodes))
        step_limit = min(brain.max_active_per_step, len(ranked))
        selected = []
        new_count = 0
        for target, payload in ranked:
            is_new = target not in activated_nodes
            if is_new and new_count >= remaining_capacity:
                continue
            selected.append((target, payload))
            if is_new:
                new_count += 1
            if len(selected) >= step_limit:
                break

        if not selected:
            stop_reason = "capacity_limit"
            break

        next_activation = np.zeros(brain.node_count, dtype=float)
        for target, (value, source) in selected:
            value = float(np.clip(value, 0.0, 1.0))
            if value < threshold:
                continue
            next_activation[target] = max(next_activation[target], value)
            traversed_edges.add(tuple(sorted((int(source), int(target)))))

        active_now = set(np.flatnonzero(next_activation > 0).tolist())
        if not active_now:
            stop_reason = "below_threshold"
            break
        activation = next_activation
        activated_nodes.update(active_now)
        executed += 1
        if len(activated_nodes) >= brain.max_total_active_nodes:
            stop_reason = "node_limit"
            break

    return activation, activated_nodes, traversed_edges, executed, stop_reason


def observe(subject: str, relation: str, mode: str, *, subject_steps: int = 8, relation_steps: int = 10) -> dict:
    subject = subject.strip()
    relation = relation.strip()
    if not subject or not relation:
        raise ValueError("主体と関係を入力してください。")

    brain = load_brain()
    subject_trace, context = _subject_context(subject, mode, subject_steps)
    relation_sources = (
        component_nodes(brain, "role:relation", "relation", 2)
        + component_nodes(brain, "relation", relation, 3)
    )
    final_activation, activated, edges, executed, stop_reason = _propagate_relation(
        brain,
        relation_sources,
        context,
        steps=relation_steps,
    )
    result = RelationState(
        mode=mode,
        subject=subject,
        relation=relation,
        context_nodes=sorted(context.items(), key=lambda x: (-x[1], x[0])),
        active_nodes=_ranked(final_activation),
        activated_nodes=sorted(activated),
        traversed_edges=sorted(edges),
        steps=executed,
        stop_reason=stop_reason,
    )
    payload = result.to_dict()
    payload["subject_trace"] = subject_trace
    payload["relation_sources"] = relation_sources
    return payload


def _weighted_similarity(left: list[tuple[int, float]], right: list[tuple[int, float]]) -> float:
    a = {int(n): float(v) for n, v in left}
    b = {int(n): float(v) for n, v in right}
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    denominator = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_modes(
    left_subject: str,
    right_subject: str,
    relation: str,
    *,
    subject_steps: int = 8,
    relation_steps: int = 10,
) -> dict:
    modes = ["flat_tail", "final_activation", "trace_decay"]
    labels = {
        "flat_tail": "A 現行：末尾ノード一律0.34",
        "final_activation": "B 最終活性値を保持",
        "trace_decay": "C 時系列・経路痕跡を保持",
    }
    comparisons = []
    for mode in modes:
        left = observe(left_subject, relation, mode, subject_steps=subject_steps, relation_steps=relation_steps)
        right = observe(right_subject, relation, mode, subject_steps=subject_steps, relation_steps=relation_steps)
        left_active = set(int(n) for n, _ in left["active_nodes"])
        right_active = set(int(n) for n, _ in right["active_nodes"])
        left_all = set(int(n) for n in left["activated_nodes"])
        right_all = set(int(n) for n in right["activated_nodes"])
        left_edges = {tuple(v) for v in left["traversed_edges"]}
        right_edges = {tuple(v) for v in right["traversed_edges"]}
        comparisons.append({
            "mode": mode,
            "label": labels[mode],
            "left": left,
            "right": right,
            "final_similarity": _weighted_similarity(left["active_nodes"], right["active_nodes"]),
            "final_node_jaccard": _jaccard(left_active, right_active),
            "all_node_jaccard": _jaccard(left_all, right_all),
            "edge_jaccard": _jaccard(left_edges, right_edges),
            "left_only_edges": len(left_edges - right_edges),
            "right_only_edges": len(right_edges - left_edges),
            "context_similarity": _weighted_similarity(left["context_nodes"], right["context_nodes"]),
        })
    return {
        "left_subject": left_subject,
        "right_subject": right_subject,
        "relation": relation,
        "subject_steps": subject_steps,
        "relation_steps": relation_steps,
        "comparisons": comparisons,
    }
