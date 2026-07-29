from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain

EPOCHS = 5
MAX_REPLAY_STEPS = 48
THRESHOLD = 0.055
EDGE_ACTIVITY_RATIO = 0.35
PERSISTENCE = 0.18
CONVERGENCE_TOLERANCE = 0.0008
CONVERGENCE_PATIENCE = 5

EXPERIENCE = ("空", "青い", "昼", "風")
DISTRACTOR_EXPERIENCES = (
    ("夜", "暗い", "星", "静か"),
    ("海", "広い", "波", "潮風"),
)

Edge = tuple[int, int]


@dataclass
class PathReplayResult:
    edge_history: list[set[Edge]]
    active_counts: list[int]
    differences: list[float]
    stop_reason: str
    elapsed_seconds: float


def pattern_vector(pattern: dict[int, float], node_count: int) -> np.ndarray:
    vector = np.zeros(node_count, dtype=float)
    for node, value in pattern.items():
        vector[int(node)] = float(value)
    return vector


def build_brain() -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=2501,
        learning_rate=0.035,
        decay_rate=0.0002,
        fatigue_gain=0.24,
        fatigue_decay=0.78,
        transmission_gain=0.90,
    )

    words = sorted(
        {
            word
            for experience in (EXPERIENCE, *DISTRACTOR_EXPERIENCES)
            for word in experience
        }
    )
    input_encoder = TextSurfaceEncoder(brain.input_nodes, width=5)
    output_encoder = TextSurfaceEncoder(brain.output_nodes, width=5)
    input_patterns = {word: input_encoder.encode(word) for word in words}
    output_patterns = {word: output_encoder.encode(word) for word in words}
    return brain, input_patterns, output_patterns


def transitions(experience: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(zip(experience, experience[1:]))


def train(
    brain: SurfaceFlowBrain,
    input_patterns: dict[str, dict[int, float]],
    output_patterns: dict[str, dict[int, float]],
) -> tuple[int, float, dict[tuple[str, str], set[Edge]], set[Edge]]:
    pairs = [
        pair
        for experience in (EXPERIENCE, *DISTRACTOR_EXPERIENCES)
        for pair in transitions(experience)
    ]
    total = EPOCHS * len(pairs)
    completed = 0
    started = time.perf_counter()
    learned_paths: dict[tuple[str, str], set[Edge]] = defaultdict(set)

    print(f"Training: {EPOCHS} epochs / {len(pairs)} transitions per epoch")
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        for source, target in pairs:
            edges = brain.experience(input_patterns[source], output_patterns[target])
            learned_paths[(source, target)].update(edges)
            completed += 1
        print(
            f"  epoch {epoch}/{EPOCHS}: {completed}/{total} transitions "
            f"({100.0 * completed / total:5.1f}%) in "
            f"{time.perf_counter() - epoch_started:.2f}s"
        )

    target_path: set[Edge] = set()
    for pair in transitions(EXPERIENCE):
        target_path.update(learned_paths[pair])

    return completed, time.perf_counter() - started, dict(learned_paths), target_path


def replay(
    brain: SurfaceFlowBrain,
    cue_pattern: dict[int, float],
) -> PathReplayResult:
    started = time.perf_counter()
    activation = pattern_vector(cue_pattern, brain.node_count)
    fatigue = np.zeros(brain.node_count, dtype=float)
    edge_history: list[set[Edge]] = []
    active_counts = [int(np.count_nonzero(activation))]
    differences: list[float] = []
    stable_steps = 0
    stop_reason = f"reached maximum replay steps ({MAX_REPLAY_STEPS})"

    for _ in range(MAX_REPLAY_STEPS):
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        contributions = effective[:, None] * brain.weights * brain.transmission_gain
        contributions[~brain.edge_enabled] = 0.0
        contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)

        edge_threshold = THRESHOLD * EDGE_ACTIVITY_RATIO
        active_edges = {
            (int(source), int(target))
            for source, target in np.argwhere(contributions >= edge_threshold)
        }
        edge_history.append(active_edges)

        propagated = 1.0 - np.prod(1.0 - contributions, axis=0)
        next_activation = 1.0 - (1.0 - propagated) * (1.0 - PERSISTENCE * activation)
        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < THRESHOLD] = 0.0

        difference = float(np.mean(np.abs(next_activation - activation)))
        differences.append(difference)
        active_counts.append(int(np.count_nonzero(next_activation)))

        fatigue = fatigue * brain.fatigue_decay
        fatigue += next_activation * brain.fatigue_gain
        fatigue = np.clip(fatigue, 0.0, 0.95)
        activation = next_activation

        if active_counts[-1] == 0:
            stop_reason = "internal activity vanished"
            break
        if difference < CONVERGENCE_TOLERANCE:
            stable_steps += 1
            if stable_steps >= CONVERGENCE_PATIENCE:
                stop_reason = (
                    f"converged for {CONVERGENCE_PATIENCE} consecutive steps "
                    f"(mean delta < {CONVERGENCE_TOLERANCE})"
                )
                break
        else:
            stable_steps = 0

    return PathReplayResult(
        edge_history=edge_history,
        active_counts=active_counts,
        differences=differences,
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - started,
    )


def jaccard(left: set[Edge], right: set[Edge]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def recall_rate(replayed: set[Edge], learned: set[Edge]) -> float:
    return 0.0 if not learned else len(replayed & learned) / len(learned)


def precision(replayed: set[Edge], learned: set[Edge]) -> float:
    return 0.0 if not replayed else len(replayed & learned) / len(replayed)


def longest_directed_path(edges: set[Edge]) -> int:
    if not edges:
        return 0

    outgoing: dict[int, list[int]] = defaultdict(list)
    indegree: Counter[int] = Counter()
    nodes: set[int] = set()
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
        nodes.add(source)
        nodes.add(target)

    queue = deque(node for node in nodes if indegree[node] == 0)
    distance = {node: 0 for node in nodes}
    visited = 0
    best = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for target in outgoing[node]:
            distance[target] = max(distance[target], distance[node] + 1)
            best = max(best, distance[target])
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if visited == len(nodes):
        return best

    # Cycles can appear during autonomous replay. In that case report the
    # longest simple walk we can find with a bounded DFS.
    limit = min(len(nodes), 64)
    best = 0
    for start in nodes:
        stack = [(start, {start}, 0)]
        while stack:
            node, seen, length = stack.pop()
            best = max(best, length)
            if length >= limit:
                continue
            for target in outgoing[node]:
                if target not in seen:
                    stack.append((target, seen | {target}, length + 1))
    return best


def main() -> None:
    total_started = time.perf_counter()
    print("SphereBrain v25 — Path Replay")
    print("The first cue is injected once. Replay is observed as edge flow, not words.\n")

    print("Target experience:")
    print("  " + " -> ".join(EXPERIENCE))
    print("Distractor experiences:")
    for experience in DISTRACTOR_EXPERIENCES:
        print("  " + " -> ".join(experience))
    print()

    brain, inputs, outputs = build_brain()
    events, training_seconds, learned_paths, target_path = train(brain, inputs, outputs)

    result = replay(brain, inputs[EXPERIENCE[0]])
    replayed_edges = set().union(*result.edge_history) if result.edge_history else set()
    heat = Counter(edge for step in result.edge_history for edge in step)

    print("\nPath Replay")
    print(f"  steps          : {len(result.edge_history)}")
    print(f"  final active   : {result.active_counts[-1]}")
    print(f"  unique edges   : {len(replayed_edges)}")
    print(f"  stop           : {result.stop_reason}")
    print(f"  replay time    : {result.elapsed_seconds:.3f}s")

    overlap = replayed_edges & target_path
    print("\nTarget-path comparison")
    print(f"  learned target edges : {len(target_path)}")
    print(f"  replay overlap       : {len(overlap)}")
    print(f"  recall rate          : {recall_rate(replayed_edges, target_path):.3f}")
    print(f"  precision            : {precision(replayed_edges, target_path):.3f}")
    print(f"  jaccard similarity   : {jaccard(replayed_edges, target_path):.3f}")

    print("\nPer-transition overlap")
    for pair in transitions(EXPERIENCE):
        learned = learned_paths[pair]
        shared = replayed_edges & learned
        print(
            f"  {pair[0]} -> {pair[1]}: "
            f"shared={len(shared):3d}/{len(learned):3d} "
            f"recall={recall_rate(replayed_edges, learned):.3f}"
        )

    print("\nReplay shape by selected step")
    selected = sorted(
        {0, 1, 2, 4, 8, 12, 16, 24, len(result.edge_history) - 1}
        & set(range(len(result.edge_history)))
    )
    for step in selected:
        edges = result.edge_history[step]
        longest = longest_directed_path(edges)
        ratio = 0.0 if not edges else longest / len(edges)
        delta = result.differences[step]
        print(
            f"  step {step + 1:2d}: edges={len(edges):4d} "
            f"longest_path={longest:3d} path_ratio={ratio:.3f} "
            f"active_nodes={result.active_counts[step + 1]:3d} delta={delta:.6f}"
        )

    print("\nMost replayed pathways")
    for (source, target), count in heat.most_common(20):
        marker = "target" if (source, target) in target_path else "other"
        print(f"  {source:3d} -> {target:3d}  count={count:2d}  {marker}")

    print("\nSummary")
    print("-" * 72)
    print(f"learned transitions : {events}")
    print(f"training time       : {training_seconds:.2f}s")
    print(f"total experiment    : {time.perf_counter() - total_started:.2f}s")

    print("\nReading the result")
    print("1. High recall with low precision means the remembered path was reached, but flow spread too widely.")
    print("2. High precision with low recall means replay stayed narrow, but recovered only part of the memory.")
    print("3. High values for both support the idea that learned pathways are being replayed.")
    print("4. path_ratio near 1 suggests a narrow route; near 0 suggests broad branching.")
    print("5. No word is decoded or fed back during replay.")


if __name__ == "__main__":
    main()
