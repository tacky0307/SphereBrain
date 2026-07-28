from __future__ import annotations

import heapq

from surface_flow import SurfaceFlowBrain

CUE = "空が"
TARGET = "青い"
DISTRACTORS = ["広い", "暗い", "雨"]
CHECKPOINTS = {0, 1, 2, 5, 10, 20, 50, 75, 100}


def shortest_path(brain: SurfaceFlowBrain, start: int, goal: int) -> list[int]:
    """Find a strong existing route through the sphere for teacher-guided learning."""
    queue: list[tuple[float, int]] = [(0.0, start)]
    costs = {start: 0.0}
    previous: dict[int, int] = {}

    while queue:
        cost, node = heapq.heappop(queue)
        if node == goal:
            break
        if cost != costs.get(node):
            continue

        neighbors = brain.adjacency[node].nonzero()[0]
        for neighbor_raw in neighbors:
            neighbor = int(neighbor_raw)
            weight = float(brain.weights[node, neighbor])
            edge_cost = 1.0 / max(weight, 1e-9)
            new_cost = cost + edge_cost
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
    cue_nodes: list[int],
    target_nodes: list[int],
) -> set[tuple[int, int]]:
    """Create directed cue-to-answer routes used only during training."""
    edges: set[tuple[int, int]] = set()
    for source, target in zip(cue_nodes, target_nodes, strict=True):
        path = shortest_path(brain, source, target)
        edges.update(zip(path, path[1:]))
    return edges


def score_label(brain: SurfaceFlowBrain, result, label: str) -> float:
    return brain.target_score(result, brain.concept_to_outputs(label))


def observe_scores(
    brain: SurfaceFlowBrain,
    cue_nodes: list[int],
) -> tuple[object, dict[str, float]]:
    # Recall is always free: no target or teacher signal is supplied here.
    observed = brain.propagate(cue_nodes, learn=False, noise=0.0, steps=40, threshold=0.04)
    labels = [TARGET, *DISTRACTORS]
    scores = {label: score_label(brain, observed, label) for label in labels}
    return observed, scores


def print_checkpoint(
    brain: SurfaceFlowBrain,
    experience_no: int,
    cue_nodes: list[int],
    baseline_scores: dict[str, float],
) -> None:
    observed, raw_scores = observe_scores(brain, cue_nodes)
    gains = {
        label: raw_scores[label] - baseline_scores[label]
        for label in raw_scores
    }
    ranked = sorted(gains.items(), key=lambda item: item[1], reverse=True)

    print(f"--- experience {experience_no:>3} ---")
    print("observed output nodes:", observed.output_nodes)
    print("learned gain over experience 0:")
    for rank, (label, gain) in enumerate(ranked, start=1):
        marker = " <- target" if label == TARGET else ""
        print(
            f"  {rank}. {label}: gain={gain:+.4f} "
            f"raw={raw_scores[label]:.4f} baseline={baseline_scores[label]:.4f}{marker}"
        )

    best_label, best_gain = ranked[0]
    recalled = best_label if best_gain > 1e-6 else "none"
    print("recalled word:", recalled)
    print()


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    cue_nodes = brain.stimulus_to_inputs(CUE)
    target_nodes = brain.concept_to_outputs(TARGET)

    print(f"training pair: {CUE} -> {TARGET}")
    print("cue input nodes:", cue_nodes)
    print("target output nodes:", target_nodes)
    for label in DISTRACTORS:
        print(f"distractor '{label}' output nodes:", brain.concept_to_outputs(label))
    print()

    initial_teacher_edges = teacher_edges(brain, cue_nodes, target_nodes)
    print("teacher-guided route edges:", len(initial_teacher_edges))
    print()

    _, baseline_scores = observe_scores(brain, cue_nodes)
    print_checkpoint(brain, 0, cue_nodes, baseline_scores)

    for experience_no in range(1, 101):
        # During experience, the correct continuation is present. The current
        # strongest cue-to-target routes are reinforced. No answer is injected
        # during checkpoint recall.
        guided_edges = teacher_edges(brain, cue_nodes, target_nodes)
        brain._reinforce(guided_edges)

        if experience_no in CHECKPOINTS:
            print_checkpoint(brain, experience_no, cue_nodes, baseline_scores)


if __name__ == "__main__":
    main()
