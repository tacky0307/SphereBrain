from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain

EPOCHS = 5
MAX_REPLAY_STEPS = 36
THRESHOLD = 0.035
PERSISTENCE = 0.12
ENERGY_BUDGET = 8.0
CONVERGENCE_TOLERANCE = 0.0005
CONVERGENCE_PATIENCE = 4
TOP_K_TRIALS = (24, 48, 96)

EXPERIENCE = ("空", "青い", "昼", "風")
DISTRACTOR_EXPERIENCES = (
    ("夜", "暗い", "星", "静か"),
    ("海", "広い", "波", "潮風"),
)

Edge = tuple[int, int]


@dataclass
class CompetitiveReplayResult:
    top_k: int
    activation_history: list[np.ndarray]
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
        seed=2601,
        learning_rate=0.035,
        decay_rate=0.0002,
        fatigue_gain=0.30,
        fatigue_decay=0.74,
        transmission_gain=0.92,
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
    inputs = {word: input_encoder.encode(word) for word in words}
    outputs = {word: output_encoder.encode(word) for word in words}
    return brain, inputs, outputs


def transitions(experience: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(zip(experience, experience[1:]))


def train(
    brain: SurfaceFlowBrain,
    inputs: dict[str, dict[int, float]],
    outputs: dict[str, dict[int, float]],
) -> tuple[int, float, dict[tuple[str, str], set[Edge]]]:
    pairs = [
        pair
        for experience in (EXPERIENCE, *DISTRACTOR_EXPERIENCES)
        for pair in transitions(experience)
    ]
    learned_paths: dict[tuple[str, str], set[Edge]] = defaultdict(set)
    total = EPOCHS * len(pairs)
    completed = 0
    started = time.perf_counter()

    print(f"Training: {EPOCHS} epochs / {len(pairs)} transitions per epoch")
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        for source, target in pairs:
            learned_paths[(source, target)].update(
                brain.experience(inputs[source], outputs[target])
            )
            completed += 1
        print(
            f"  epoch {epoch}/{EPOCHS}: {completed}/{total} transitions "
            f"({100.0 * completed / total:5.1f}%) in "
            f"{time.perf_counter() - epoch_started:.2f}s"
        )

    return completed, time.perf_counter() - started, dict(learned_paths)


def keep_strongest(vector: np.ndarray, top_k: int) -> np.ndarray:
    positive = np.flatnonzero(vector > 0.0)
    if positive.size <= top_k:
        return vector
    strongest = positive[np.argpartition(vector[positive], -top_k)[-top_k:]]
    mask = np.zeros(vector.size, dtype=bool)
    mask[strongest] = True
    return np.where(mask, vector, 0.0)


def normalize_energy(vector: np.ndarray, budget: float) -> np.ndarray:
    total = float(np.sum(vector))
    if total <= 0.0 or total <= budget:
        return vector
    return vector * (budget / total)


def replay(
    brain: SurfaceFlowBrain,
    cue_pattern: dict[int, float],
    top_k: int,
) -> CompetitiveReplayResult:
    started = time.perf_counter()
    activation = pattern_vector(cue_pattern, brain.node_count)
    activation = normalize_energy(activation, ENERGY_BUDGET)
    fatigue = np.zeros(brain.node_count, dtype=float)

    activation_history = [activation.copy()]
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

        propagated = 1.0 - np.prod(1.0 - contributions, axis=0)
        next_activation = propagated + PERSISTENCE * activation
        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < THRESHOLD] = 0.0

        # Global competition: only the strongest local activity survives.
        next_activation = keep_strongest(next_activation, top_k)
        next_activation = normalize_energy(next_activation, ENERGY_BUDGET)

        survivor_mask = next_activation > 0.0
        active_edges = {
            (int(source), int(target))
            for source, target in np.argwhere(
                (contributions >= THRESHOLD * 0.35)
                & survivor_mask[None, :]
            )
        }
        edge_history.append(active_edges)

        difference = float(np.mean(np.abs(next_activation - activation)))
        differences.append(difference)
        active_counts.append(int(np.count_nonzero(next_activation)))
        activation_history.append(next_activation.copy())

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

    return CompetitiveReplayResult(
        top_k=top_k,
        activation_history=activation_history,
        edge_history=edge_history,
        active_counts=active_counts,
        differences=differences,
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - started,
    )


def edge_metrics(replayed: set[Edge], learned: set[Edge]) -> tuple[float, float]:
    shared = len(replayed & learned)
    recall = 0.0 if not learned else shared / len(learned)
    precision = 0.0 if not replayed else shared / len(replayed)
    return recall, precision


def first_hit_steps(
    edge_history: list[set[Edge]],
    learned_paths: dict[tuple[str, str], set[Edge]],
) -> dict[tuple[str, str], int | None]:
    hits: dict[tuple[str, str], int | None] = {}
    for pair in transitions(EXPERIENCE):
        learned = learned_paths[pair]
        hits[pair] = next(
            (
                step
                for step, edges in enumerate(edge_history, start=1)
                if learned and len(edges & learned) / len(learned) >= 0.20
            ),
            None,
        )
    return hits


def ordered_hits(hits: dict[tuple[str, str], int | None]) -> bool:
    values = [hits[pair] for pair in transitions(EXPERIENCE)]
    return all(value is not None for value in values) and values == sorted(values)


def main() -> None:
    total_started = time.perf_counter()
    print("SphereBrain v26 — Competitive Replay")
    print("Only the first cue is injected. Strong internal activity competes for limited space.")
    print("No decoded word is fed back and replay does not learn.\n")

    print("Target experience:")
    print("  " + " -> ".join(EXPERIENCE))
    print("Distractor experiences:")
    for experience in DISTRACTOR_EXPERIENCES:
        print("  " + " -> ".join(experience))
    print()

    brain, inputs, outputs = build_brain()
    events, training_seconds, learned_paths = train(brain, inputs, outputs)

    target_edges = set().union(*(learned_paths[p] for p in transitions(EXPERIENCE)))
    distractor_pairs = [
        pair for experience in DISTRACTOR_EXPERIENCES for pair in transitions(experience)
    ]
    distractor_edges = set().union(*(learned_paths[p] for p in distractor_pairs))

    print("\nCompetitive replay trials")
    for top_k in TOP_K_TRIALS:
        result = replay(brain, inputs[EXPERIENCE[0]], top_k)
        replayed = set().union(*result.edge_history) if result.edge_history else set()
        target_recall, target_precision = edge_metrics(replayed, target_edges)
        distractor_recall, distractor_precision = edge_metrics(replayed, distractor_edges)
        hits = first_hit_steps(result.edge_history, learned_paths)

        print(f"\n  top_k={top_k}")
        print(f"    steps              : {len(result.edge_history)}")
        print(f"    max active nodes   : {max(result.active_counts)}")
        print(f"    final active nodes : {result.active_counts[-1]}")
        print(f"    unique replay edges: {len(replayed)}")
        print(f"    target recall      : {target_recall:.3f}")
        print(f"    target precision   : {target_precision:.3f}")
        print(f"    distractor recall  : {distractor_recall:.3f}")
        print(f"    distractor precision: {distractor_precision:.3f}")
        print(f"    selectivity gap    : {target_precision - distractor_precision:+.3f}")
        print(f"    ordered target hits: {ordered_hits(hits)}")
        for pair in transitions(EXPERIENCE):
            shown = "none" if hits[pair] is None else str(hits[pair])
            print(f"      {pair[0]} -> {pair[1]} first_hit={shown}")
        print(f"    stop               : {result.stop_reason}")
        print(f"    replay time        : {result.elapsed_seconds:.3f}s")

        heat = Counter(edge for step in result.edge_history for edge in step)
        target_hot = sum(count for edge, count in heat.items() if edge in target_edges)
        other_hot = sum(count for edge, count in heat.items() if edge not in target_edges)
        print(f"    repeated target flow: {target_hot}")
        print(f"    repeated other flow : {other_hot}")

    print("\nSummary")
    print("-" * 72)
    print(f"learned transitions : {events}")
    print(f"training time       : {training_seconds:.2f}s")
    print(f"total experiment    : {time.perf_counter() - total_started:.2f}s")

    print("\nHow to read v26")
    print("1. max active nodes must stay near top_k, not expand to all 600 nodes.")
    print("2. A positive selectivity gap means target flow is favored over distractor flow.")
    print("3. ordered target hits asks whether the three learned transitions emerge in order.")
    print("4. Narrow replay alone is not success; it must also prefer the learned target path.")
    print("5. This experiment measures pathway selection, not word-answer accuracy.")


if __name__ == "__main__":
    main()
