from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path(__file__).resolve().parent
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain
from run_adaptive_path_selection_v19 import (
    ACTIVE_THRESHOLD,
    EXPERIENCES,
    MAX_BRANCHES,
    PROBES,
    STEPS,
    TraceResult,
    adjacent_pairs,
    cosine,
    decode,
    input_vector,
    normalized_entropy,
    peak_output,
    selected_destinations,
    successor_margin,
)


EXPECTED: dict[str, set[str]] = {
    "空": {"青い", "見える"},
    "青い": {"空"},
    "今日": {"楽しい", "雨"},
    "雨": {"降る"},
    "私": {"うれしい"},
}


@dataclass(frozen=True)
class BoundaryMode:
    key: str
    label: str
    broad_steps: int
    medium_steps: int
    residual_fraction: float = 0.0

    def threshold_for_step(self, step: int) -> float:
        if step <= self.broad_steps:
            return 0.80
        if step <= self.broad_steps + self.medium_steps:
            return 0.90
        return 0.95

    @property
    def schedule_text(self) -> str:
        broad_end = self.broad_steps
        medium_end = self.broad_steps + self.medium_steps
        return (
            f"1-{broad_end}:0.80/"
            f"{broad_end + 1}-{medium_end}:0.90/"
            f"{medium_end + 1}-{STEPS}:0.95"
        )


MODES: tuple[BoundaryMode, ...] = (
    BoundaryMode("boundary1", "Broad exploration for 1 step", 1, 4),
    BoundaryMode("boundary2", "Broad exploration for 2 steps", 2, 4),
    BoundaryMode("boundary3", "Broad exploration for 3 steps", 3, 4),
    BoundaryMode("boundary4", "Broad exploration for 4 steps", 4, 6),
    BoundaryMode("boundary5", "Broad exploration for 5 steps", 5, 6),
    BoundaryMode("boundary6", "Broad exploration for 6 steps", 6, 6),
)


def temporal_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    mode: BoundaryMode,
    steps: int = STEPS,
) -> TraceResult:
    activation = input_vector(brain, pattern)
    initial_energy = float(np.sum(activation))
    fatigue = np.zeros(brain.node_count, dtype=float)

    history = [activation.copy()]
    energy = [initial_energy]
    mean_branches = [0.0]
    branch_entropy = [normalized_entropy(activation)]
    selected_edges = [0]

    for step in range(1, steps + 1):
        relative_threshold = mode.threshold_for_step(step)
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        next_activation = np.zeros(brain.node_count, dtype=float)
        active_sources = np.flatnonzero(effective > ACTIVE_THRESHOLD)
        step_branch_counts: list[int] = []
        step_selected_edges = 0

        for source in active_sources:
            source_energy = float(effective[source])
            selected, selected_weights, branch_count = selected_destinations(
                brain,
                int(source),
                "adaptive",
                relative_threshold,
            )
            if selected.size == 0:
                continue

            step_branch_counts.append(branch_count)
            step_selected_edges += branch_count
            selected_total = float(np.sum(selected_weights))
            if selected_total <= 0.0:
                continue

            main_energy = source_energy * (1.0 - mode.residual_fraction)
            next_activation[selected] += main_energy * (selected_weights / selected_total)

        next_activation[next_activation < ACTIVE_THRESHOLD] = 0.0
        current_total = float(np.sum(next_activation))
        if current_total > 0.0:
            next_activation *= initial_energy / current_total

        fatigue *= brain.fatigue_decay
        fatigue += np.clip(next_activation, 0.0, 1.0) * brain.fatigue_gain
        fatigue = np.clip(fatigue, 0.0, 0.95)

        activation = next_activation
        history.append(activation.copy())
        energy.append(float(np.sum(activation)))
        mean_branches.append(
            float(np.mean(step_branch_counts)) if step_branch_counts else 0.0
        )
        branch_entropy.append(normalized_entropy(activation))
        selected_edges.append(step_selected_edges)

    return TraceResult(history, mean_branches, branch_entropy, energy, selected_edges)


def mean_pairwise_by_step(traces: dict[str, TraceResult]) -> list[float]:
    means: list[float] = []
    for step in range(STEPS + 1):
        similarities = [
            cosine(traces[left].history[step], traces[right].history[step])
            for left, right in combinations(PROBES, 2)
        ]
        means.append(float(np.mean(similarities)))
    return means


def decode_at_step(
    brain: SurfaceFlowBrain,
    trace: TraceResult,
    output_patterns: dict[str, dict[int, float]],
    step: int,
) -> str:
    vector = trace.history[step].copy()
    mask = np.zeros(brain.node_count, dtype=bool)
    mask[brain.output_nodes] = True
    vector *= mask
    return decode(vector, output_patterns, brain.node_count, limit=1)[0][0]


def stable_recall_step(sequence: list[str], expected: set[str]) -> int | None:
    for index, candidate in enumerate(sequence, start=1):
        if candidate in expected and all(item in expected for item in sequence[index - 1 :]):
            return index
    return None


def persistence(sequence: list[str], expected: set[str]) -> float:
    return float(np.mean([candidate in expected for candidate in sequence]))


def identity_score(traces: dict[str, TraceResult]) -> float:
    similarities = [
        cosine(traces[left].history[STEPS], traces[right].history[STEPS])
        for left, right in combinations(PROBES, 2)
    ]
    return 1.0 - float(np.mean(similarities))


def expected_top1_count(
    brain: SurfaceFlowBrain,
    traces: dict[str, TraceResult],
    output_patterns: dict[str, dict[int, float]],
) -> int:
    hits = 0
    for source in PROBES:
        candidate = decode(
            peak_output(brain, traces[source]),
            output_patterns,
            brain.node_count,
            limit=1,
        )[0][0]
        if candidate in EXPECTED[source]:
            hits += 1
    return hits


def main() -> None:
    print("SphereBrain v19c — Early Narrowing Boundary")
    print("How many broad-exploration steps are actually needed for stable recall?")
    print("Residual flow is fixed at 0.000 so only narrowing timing changes.\n")

    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=1801,
        learning_rate=0.035,
        decay_rate=0.0004,
    )
    input_encoder = TextSurfaceEncoder(brain.input_nodes, width=5)
    output_encoder = TextSurfaceEncoder(brain.output_nodes, width=5)

    vocabulary = sorted({word for experience in EXPERIENCES for word in experience})
    input_patterns = {word: input_encoder.encode(word) for word in vocabulary}
    output_patterns = {word: output_encoder.encode(word) for word in vocabulary}

    rng = np.random.default_rng(1802)
    pairs = adjacent_pairs()
    epochs = 75
    for _ in range(epochs):
        for pair_index in rng.permutation(len(pairs)):
            source, target = pairs[int(pair_index)]
            brain.experience(input_patterns[source], output_patterns[target])

    print(f"Training: epochs={epochs}, events={epochs * len(pairs)}")
    print(f"Max adaptive branches={MAX_BRANCHES}")
    print("-" * 148)
    for mode in MODES:
        print(
            f"{mode.key:<12} {mode.label:<38} "
            f"schedule={mode.schedule_text:<34} residual={mode.residual_fraction:.3f}"
        )

    traces_by_mode: dict[str, dict[str, TraceResult]] = {
        mode.key: {
            word: temporal_trace(brain, input_patterns[word], mode)
            for word in PROBES
        }
        for mode in MODES
    }

    metrics: dict[str, dict[str, float]] = {}
    recall_details: dict[str, dict[str, tuple[int | None, float]]] = {}

    print("\nBoundary comparison")
    print("-" * 148)
    print(
        f"{'mode':<12} {'nodes@5':>8} {'nodes@24':>9} {'branch1-4':>11} "
        f"{'branch11-24':>13} {'sim@5':>8} {'sim@24':>9} {'identity':>9} "
        f"{'stable':>10} {'persist':>9} {'margin':>9} {'top1':>6}"
    )

    for mode in MODES:
        traces = traces_by_mode[mode.key]
        similarities = mean_pairwise_by_step(traces)
        expected_mean, unrelated_mean, margin = successor_margin(
            brain,
            traces,
            output_patterns,
            vocabulary,
        )
        top1 = expected_top1_count(brain, traces, output_patterns)
        nodes5 = float(np.mean([
            np.count_nonzero(traces[word].history[5]) for word in PROBES
        ]))
        nodes24 = float(np.mean([
            np.count_nonzero(traces[word].history[24]) for word in PROBES
        ]))
        early_branch = float(np.mean([
            traces[word].mean_branches[step]
            for word in PROBES
            for step in range(1, 5)
        ]))
        late_branch = float(np.mean([
            traces[word].mean_branches[step]
            for word in PROBES
            for step in range(11, 25)
        ]))

        per_probe: dict[str, tuple[int | None, float]] = {}
        stable_values: list[int] = []
        persistence_values: list[float] = []
        for word in PROBES:
            sequence = [
                decode_at_step(brain, traces[word], output_patterns, step)
                for step in range(1, STEPS + 1)
            ]
            stable = stable_recall_step(sequence, EXPECTED[word])
            persist = persistence(sequence, EXPECTED[word])
            per_probe[word] = (stable, persist)
            if stable is not None:
                stable_values.append(stable)
            persistence_values.append(persist)

        stable_mean = (
            float(np.mean(stable_values)) if len(stable_values) == len(PROBES) else float("inf")
        )
        persistence_mean = float(np.mean(persistence_values))
        identity = identity_score(traces)

        stable_text = "never" if not np.isfinite(stable_mean) else f"{stable_mean:.1f}"
        print(
            f"{mode.key:<12} {nodes5:8.1f} {nodes24:9.1f} {early_branch:11.2f} "
            f"{late_branch:13.2f} {similarities[5]:8.3f} {similarities[24]:9.3f} "
            f"{identity:9.3f} {stable_text:>10} {persistence_mean:9.3f} "
            f"{margin:9.3f} {top1:6d}"
        )

        metrics[mode.key] = {
            "top1": float(top1),
            "stable": stable_mean,
            "persistence": persistence_mean,
            "margin": margin,
            "final_similarity": similarities[24],
            "identity": identity,
            "expected": expected_mean,
            "unrelated": unrelated_mean,
        }
        recall_details[mode.key] = per_probe

    print("\nPer-probe stable recall and persistence")
    print("-" * 148)
    for mode in MODES:
        print(f"\n{mode.label} | schedule={mode.schedule_text}")
        for word in PROBES:
            stable, persist = recall_details[mode.key][word]
            stable_text = "never" if stable is None else str(stable)
            candidates = decode(
                peak_output(brain, traces_by_mode[mode.key][word]),
                output_patterns,
                brain.node_count,
            )
            candidate_text = ", ".join(
                f"{candidate}:{score:.3f}" for candidate, score in candidates
            )
            print(
                f"  {word:<4} stable={stable_text:<5} persistence={persist:.1%} "
                f"final={candidate_text}"
            )

    best = max(
        MODES,
        key=lambda mode: (
            metrics[mode.key]["top1"],
            -metrics[mode.key]["stable"],
            metrics[mode.key]["persistence"],
            metrics[mode.key]["margin"],
            -metrics[mode.key]["final_similarity"],
        ),
    )
    result = metrics[best.key]

    print("\nEarly narrowing recommendation")
    print("-" * 148)
    stable_text = "never" if not np.isfinite(result["stable"]) else f"{result['stable']:.1f}"
    print(
        f"Best boundary in this run: {best.key} "
        f"(top1={int(result['top1'])}/5, stable={stable_text}, "
        f"persistence={result['persistence']:.1%}, margin={result['margin']:+.3f}, "
        f"sim@24={result['final_similarity']:.3f}, identity={result['identity']:.3f})"
    )
    print("Selection priority: top-1 recall, earlier stable recall, persistence, margin, then lower final convergence.")

    print("\nInterpretation guardrails")
    print("-" * 148)
    print("1. Only the duration of broad exploration changes between conditions.")
    print("2. Threshold changes depend on propagation time, never on word labels.")
    print("3. Residual flow remains zero to avoid reintroducing late weak-route convergence.")
    print("4. Stable recall means the expected successor stays top-1 for every remaining step.")
    print("5. This is the final scheduled-gating boundary experiment before self-adjusting v20 gating.")


if __name__ == "__main__":
    main()
