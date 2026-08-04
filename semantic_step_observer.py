from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from semantic_encoder_v2 import component_nodes, load_brain


@dataclass
class StepState:
    step: int
    active_nodes: list[tuple[int, float]]
    active_count: int
    peak_node: int | None
    peak_value: float
    new_nodes: list[int]
    step_edges: list[tuple[int, int]]
    cumulative_nodes: list[int]
    cumulative_edges: list[tuple[int, int]]

    def to_dict(self) -> dict:
        return asdict(self)


def _ranked_activation(activation: np.ndarray) -> list[tuple[int, float]]:
    indexes = np.flatnonzero(activation > 0)
    return sorted(
        ((int(index), float(activation[index])) for index in indexes),
        key=lambda item: (-item[1], item[0]),
    )


def _snapshot(
    step: int,
    activation: np.ndarray,
    previous_nodes: set[int],
    step_edges: set[tuple[int, int]],
    cumulative_nodes: set[int],
    cumulative_edges: set[tuple[int, int]],
) -> StepState:
    ranked = _ranked_activation(activation)
    current_nodes = {node for node, _ in ranked}
    return StepState(
        step=step,
        active_nodes=ranked,
        active_count=len(ranked),
        peak_node=ranked[0][0] if ranked else None,
        peak_value=ranked[0][1] if ranked else 0.0,
        new_nodes=sorted(current_nodes - previous_nodes),
        step_edges=sorted(step_edges),
        cumulative_nodes=sorted(cumulative_nodes),
        cumulative_edges=sorted(cumulative_edges),
    )


def trace_subject(subject: str, *, steps: int = 8, threshold: float = 0.18) -> dict:
    """Observe subject propagation without learning or noise.

    This reproduces SphereBrain._propagate_focused step by step, but never
    reinforces edges or changes the saved Core.
    """
    subject = subject.strip()
    if not subject:
        raise ValueError("主体を入力してください。")

    brain = load_brain()
    role_nodes = component_nodes(brain, "role:subject", "subject", 2)
    entity_nodes = component_nodes(brain, "entity", subject, 3)
    source_nodes = role_nodes + entity_nodes

    sources, activation = brain._initial_activation(source_nodes, None)
    cumulative_nodes = set(np.flatnonzero(activation > 0).tolist())
    cumulative_edges: set[tuple[int, int]] = set()
    states = [
        _snapshot(
            0,
            activation,
            set(),
            set(),
            cumulative_nodes,
            cumulative_edges,
        )
    ]

    stop_reason = "step_limit"
    for step in range(1, max(0, int(steps)) + 1):
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
                if value < max(float(threshold), 0.18):
                    continue
                previous = candidates.get(target)
                if previous is None or value > previous[0]:
                    candidates[target] = (value, int(source))

        if not candidates:
            stop_reason = "below_threshold"
            break

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
        remaining_capacity = max(0, brain.max_total_active_nodes - len(cumulative_nodes))
        step_limit = min(brain.max_active_per_step, len(ranked))

        selected: list[tuple[int, tuple[float, int]]] = []
        new_nodes_selected = 0
        for target, payload in ranked:
            is_new = target not in cumulative_nodes
            if is_new and new_nodes_selected >= remaining_capacity:
                continue
            selected.append((target, payload))
            if is_new:
                new_nodes_selected += 1
            if len(selected) >= step_limit:
                break

        if not selected:
            stop_reason = "capacity_limit"
            break

        previous_nodes = set(np.flatnonzero(activation > 0).tolist())
        next_activation = np.zeros(brain.node_count, dtype=float)
        step_edges: set[tuple[int, int]] = set()
        for target, (value, source) in selected:
            value = float(np.clip(value, 0.0, 1.0))
            if value < max(float(threshold), 0.18):
                continue
            next_activation[target] = max(next_activation[target], value)
            step_edges.add(tuple(sorted((source, target))))

        active_now = set(np.flatnonzero(next_activation > 0).tolist())
        if not active_now:
            stop_reason = "below_threshold"
            break

        cumulative_nodes.update(active_now)
        cumulative_edges.update(step_edges)
        activation = next_activation
        states.append(
            _snapshot(
                step,
                activation,
                previous_nodes,
                step_edges,
                cumulative_nodes,
                cumulative_edges,
            )
        )

        if len(cumulative_nodes) >= brain.max_total_active_nodes:
            stop_reason = "node_limit"
            break

    return {
        "subject": subject,
        "role_nodes": role_nodes,
        "entity_nodes": entity_nodes,
        "source_nodes": sources,
        "threshold": max(float(threshold), 0.18),
        "requested_steps": max(0, int(steps)),
        "executed_steps": len(states) - 1,
        "stop_reason": stop_reason,
        "states": [state.to_dict() for state in states],
    }


def _activation_map(state: dict) -> dict[int, float]:
    return {int(node): float(value) for node, value in state["active_nodes"]}


def weighted_similarity(left: dict, right: dict) -> float:
    a = _activation_map(left)
    b = _activation_map(right)
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    numerator = sum(min(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    denominator = sum(max(a.get(key, 0.0), b.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_subjects(
    left_subject: str,
    right_subject: str,
    *,
    steps: int = 8,
    threshold: float = 0.18,
) -> dict:
    left = trace_subject(left_subject, steps=steps, threshold=threshold)
    right = trace_subject(right_subject, steps=steps, threshold=threshold)

    max_states = max(len(left["states"]), len(right["states"]))
    comparisons: list[dict] = []
    first_identical_step: int | None = None

    for index in range(max_states):
        left_state = left["states"][index] if index < len(left["states"]) else None
        right_state = right["states"][index] if index < len(right["states"]) else None
        if left_state is None or right_state is None:
            comparisons.append(
                {
                    "step": index,
                    "comparable": False,
                    "activation_similarity": 0.0,
                    "current_node_jaccard": 0.0,
                    "cumulative_node_jaccard": 0.0,
                    "cumulative_edge_jaccard": 0.0,
                    "common_current_nodes": 0,
                    "left_only_current_nodes": 0,
                    "right_only_current_nodes": 0,
                    "left_only_edges": [],
                    "right_only_edges": [],
                }
            )
            continue

        left_current = set(_activation_map(left_state))
        right_current = set(_activation_map(right_state))
        left_cumulative_nodes = {int(v) for v in left_state["cumulative_nodes"]}
        right_cumulative_nodes = {int(v) for v in right_state["cumulative_nodes"]}
        left_edges = {tuple(edge) for edge in left_state["cumulative_edges"]}
        right_edges = {tuple(edge) for edge in right_state["cumulative_edges"]}
        similarity = weighted_similarity(left_state, right_state)

        comparisons.append(
            {
                "step": index,
                "comparable": True,
                "activation_similarity": similarity,
                "current_node_jaccard": _jaccard(left_current, right_current),
                "cumulative_node_jaccard": _jaccard(left_cumulative_nodes, right_cumulative_nodes),
                "cumulative_edge_jaccard": _jaccard(left_edges, right_edges),
                "common_current_nodes": len(left_current & right_current),
                "left_only_current_nodes": len(left_current - right_current),
                "right_only_current_nodes": len(right_current - left_current),
                "left_only_edges": sorted(left_edges - right_edges),
                "right_only_edges": sorted(right_edges - left_edges),
            }
        )

        if first_identical_step is None and similarity >= 0.999999:
            first_identical_step = index

    return {
        "left": left,
        "right": right,
        "comparisons": comparisons,
        "first_identical_step": first_identical_step,
        "source_overlap": {
            "common": sorted(set(left["source_nodes"]) & set(right["source_nodes"])),
            "left_only": sorted(set(left["source_nodes"]) - set(right["source_nodes"])),
            "right_only": sorted(set(right["source_nodes"]) - set(left["source_nodes"])),
        },
    }
