from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from surface_flow import SurfaceFlowBrain


@dataclass(frozen=True)
class Example:
    x: float
    y: float


TRAIN_X = np.arange(0.00, 0.5001, 0.05)
TEST_X = np.arange(0.025, 0.5000, 0.05)
EPOCHS = 40
PATTERN_WIDTH = 5
CANDIDATE_Y = np.linspace(0.0, 1.0, 101)


def ordered_surface_nodes(brain: SurfaceFlowBrain, nodes: list[int]) -> list[int]:
    """Give the surface a stable one-dimensional continuous coordinate.

    The brain itself still operates on the spherical graph. This ordering is only
    an experimental sensor/actuator interface: nearby numeric values stimulate
    overlapping nearby surface patterns instead of unrelated hash-selected nodes.
    """
    return sorted(
        nodes,
        key=lambda node: (
            float(brain.positions[node, 1]),
            float(brain.positions[node, 2]),
            float(brain.positions[node, 0]),
        ),
    )


def scalar_pattern(value: float, ordered_nodes: list[int], width: int = PATTERN_WIDTH) -> list[int]:
    """Encode a scalar in [0, 1] as an overlapping population pattern."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"value must be in [0, 1], got {value}")
    if width < 1 or width > len(ordered_nodes):
        raise ValueError("invalid pattern width")

    center = int(round(value * (len(ordered_nodes) - 1)))
    half = width // 2
    start = max(0, min(center - half, len(ordered_nodes) - width))
    return ordered_nodes[start : start + width]


def shortest_path(brain: SurfaceFlowBrain, start: int, goal: int) -> list[int]:
    queue: list[tuple[float, int]] = [(0.0, start)]
    costs = {start: 0.0}
    previous: dict[int, int] = {}

    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if cost != costs.get(node):
            continue

        for neighbor_raw in brain.adjacency[node].nonzero()[0]:
            neighbor = int(neighbor_raw)
            weight = float(brain.weights[node, neighbor])
            new_cost = cost + 1.0 / max(weight, 1e-9)
            if new_cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = new_cost
                previous[neighbor] = node
                heapq.heappush(queue, (new_cost, neighbor))

    if goal not in costs:
        return []

    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def teacher_edges(
    brain: SurfaceFlowBrain,
    input_pattern: list[int],
    target_pattern: list[int],
) -> set[tuple[int, int]]:
    """Create routes for one numeric input-output experience.

    This keeps the same teacher-guided mechanism as the first associative test so
    this experiment isolates one question: can overlapping continuous patterns
    produce useful responses for values that were never trained?
    """
    edges: set[tuple[int, int]] = set()
    for source, target in zip(input_pattern, target_pattern, strict=True):
        path = shortest_path(brain, source, target)
        edges.update(zip(path, path[1:]))
    return edges


def output_node_energy(brain: SurfaceFlowBrain, result) -> np.ndarray:
    index = {node: i for i, node in enumerate(brain.output_nodes)}
    energy = np.zeros(len(brain.output_nodes), dtype=float)
    for step in result.output_history:
        for node, value in step.items():
            energy[index[node]] += value
    return energy


def observe(brain: SurfaceFlowBrain, input_pattern: list[int]):
    return brain.propagate(
        input_pattern,
        learn=False,
        noise=0.0,
        steps=40,
        threshold=0.04,
    )


def candidate_score(
    gain_by_output_node: dict[int, float],
    target_pattern: list[int],
) -> float:
    return sum(max(0.0, gain_by_output_node.get(node, 0.0)) for node in target_pattern)


def predict_value(
    brain: SurfaceFlowBrain,
    x: float,
    input_axis: list[int],
    output_axis: list[int],
    baseline_energy: np.ndarray,
) -> tuple[float, float, list[tuple[float, float]]]:
    result = observe(brain, scalar_pattern(x, input_axis))
    current = output_node_energy(brain, result)
    gain = current - baseline_energy
    gain_by_node = {
        node: float(gain[index])
        for index, node in enumerate(brain.output_nodes)
    }

    scored = [
        (
            float(y),
            candidate_score(gain_by_node, scalar_pattern(float(y), output_axis)),
        )
        for y in CANDIDATE_Y
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    predicted, best_score = ranked[0]
    return predicted, best_score, ranked[:3]


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    input_axis = ordered_surface_nodes(brain, brain.input_nodes)
    output_axis = ordered_surface_nodes(brain, brain.output_nodes)

    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    print("continuous-value experiment: y = 2x")
    print(f"training examples: {len(training)} (x step=0.05)")
    print(f"unseen midpoint tests: {len(testing)}")
    print(f"population pattern width: {PATTERN_WIDTH}")
    print()

    baseline_by_test: dict[float, np.ndarray] = {}
    for example in testing:
        result = observe(brain, scalar_pattern(example.x, input_axis))
        baseline_by_test[example.x] = output_node_energy(brain, result)

    route_counts: list[int] = []
    for example in training:
        routes = teacher_edges(
            brain,
            scalar_pattern(example.x, input_axis),
            scalar_pattern(example.y, output_axis),
        )
        route_counts.append(len(routes))

    print(
        "teacher-guided route edges per example: "
        f"min={min(route_counts)} mean={np.mean(route_counts):.1f} max={max(route_counts)}"
    )
    print()

    checkpoints = {1, 5, 10, 20, EPOCHS}
    for epoch in range(1, EPOCHS + 1):
        # Present every known numeric experience once per epoch. No language or
        # text hashing is involved anywhere in this experiment.
        for example in training:
            routes = teacher_edges(
                brain,
                scalar_pattern(example.x, input_axis),
                scalar_pattern(example.y, output_axis),
            )
            brain._reinforce(routes)

        if epoch not in checkpoints:
            continue

        print(f"--- epoch {epoch:>2} ---")
        absolute_errors: list[float] = []
        for example in testing:
            predicted, score, top3 = predict_value(
                brain,
                example.x,
                input_axis,
                output_axis,
                baseline_by_test[example.x],
            )
            error = abs(predicted - example.y)
            absolute_errors.append(error)
            candidates = ", ".join(f"{value:.2f}:{value_score:.4f}" for value, value_score in top3)
            print(
                f"x={example.x:.3f} expected={example.y:.3f} "
                f"predicted={predicted:.3f} error={error:.3f} "
                f"gain_score={score:.4f} top=[{candidates}]"
            )

        print(
            f"mean absolute error: {np.mean(absolute_errors):.4f} | "
            f"max error: {np.max(absolute_errors):.4f}"
        )
        print()


if __name__ == "__main__":
    main()
