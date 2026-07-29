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
    first_step_at_or_above,
    input_vector,
    normalized_entropy,
    peak_output,
    selected_destinations,
    successor_margin,
)


@dataclass(frozen=True)
class TemporalMode:
    key: str
    label: str
    schedule: tuple[tuple[int, float], ...]
    residual_fraction: float

    def threshold_for_step(self, step: int) -> float:
        for last_step, threshold in self.schedule:
            if step <= last_step:
                return threshold
        return self.schedule[-1][1]

    @property
    def schedule_text(self) -> str:
        start = 1
        parts: list[str] = []
        for last_step, threshold in self.schedule:
            parts.append(f"{start}-{last_step}:{threshold:.2f}")
            start = last_step + 1
        return "/".join(parts)


RESIDUALS: tuple[float, ...] = (0.000, 0.002, 0.005, 0.010, 0.020)

STATIC_80 = tuple((STEPS, 0.80) for _ in range(1))
STATIC_95 = tuple((STEPS, 0.95) for _ in range(1))
GENTLE = ((4, 0.80), (10, 0.90), (STEPS, 0.95))
EARLY_FOCUS = ((2, 0.80), (6, 0.90), (STEPS, 0.95))
LATE_FOCUS = ((6, 0.80), (14, 0.90), (STEPS, 0.95))

MODES: tuple[TemporalMode, ...] = (
    TemporalMode("static80_r020", "Static 0.80 control", STATIC_80, 0.020),
    TemporalMode("static95_r020", "Static 0.95 control", STATIC_95, 0.020),
    *tuple(
        TemporalMode(
            f"gentle_r{int(residual * 1000):03d}",
            f"Temporal 0.80→0.90→0.95 residual={residual:.3f}",
            GENTLE,
            residual,
        )
        for residual in RESIDUALS
    ),
    TemporalMode("early_r005", "Earlier narrowing", EARLY_FOCUS, 0.005),
    TemporalMode("late_r005", "Later narrowing", LATE_FOCUS, 0.005),
)


def temporal_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    mode: TemporalMode,
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

            if mode.residual_fraction > 0.0:
                enabled = np.flatnonzero(brain.edge_enabled[source])
                residual = np.setdiff1d(enabled, selected, assume_unique=False)
                if residual.size > 0:
                    residual_weights = np.asarray(brain.weights[source, residual], dtype=float)
                    residual_weights = np.clip(residual_weights, 0.0, None)
                    residual_total = float(np.sum(residual_weights))
                    if residual_total > 0.0:
                        next_activation[residual] += (
                            source_energy
                            * mode.residual_fraction
                            * (residual_weights / residual_total)
                        )

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


def expected_top1_count(
    brain: SurfaceFlowBrain,
    traces: dict[str, TraceResult],
    output_patterns: dict[str, dict[int, float]],
) -> int:
    expected: dict[str, set[str]] = {
        "空": {"青い", "見える"},
        "青い": {"空"},
        "今日": {"楽しい", "雨"},
        "雨": {"降る"},
        "私": {"うれしい"},
    }
    hits = 0
    for source in PROBES:
        candidate = decode(
            peak_output(brain, traces[source]),
            output_patterns,
            brain.node_count,
            limit=1,
        )[0][0]
        if candidate in expected[source]:
            hits += 1
    return hits


def main() -> None:
    print("SphereBrain v19b — Temporal Adaptive Gating")
    print("Adaptive routing becomes more selective as propagation continues.")
    print("Question: can early recall remain broad while late convergence is suppressed?\n")

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
    print("-" * 128)
    for mode in MODES:
        print(
            f"{mode.key:<17} {mode.label:<53} "
            f"schedule={mode.schedule_text:<28} residual={mode.residual_fraction:.3f}"
        )

    traces_by_mode: dict[str, dict[str, TraceResult]] = {
        mode.key: {
            word: temporal_trace(brain, input_patterns[word], mode)
            for word in PROBES
        }
        for mode in MODES
    }

    print("\nTemporal gating summary")
    print("-" * 128)
    print(
        f"{'mode':<17} {'nodes@5':>8} {'nodes@24':>9} {'branch 1-4':>11} "
        f"{'branch 11-24':>13} {'sim@5':>8} {'sim@24':>9} {'first>=.90':>11} "
        f"{'margin':>9} {'top1':>6}"
    )

    metrics: dict[str, dict[str, float]] = {}
    for mode in MODES:
        traces = traces_by_mode[mode.key]
        similarities = mean_pairwise_by_step(traces)
        first = first_step_at_or_above(similarities, 0.90)
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
        first_text = "never" if first is None else str(first)
        print(
            f"{mode.key:<17} {nodes5:8.1f} {nodes24:9.1f} {early_branch:11.2f} "
            f"{late_branch:13.2f} {similarities[5]:8.3f} {similarities[24]:9.3f} "
            f"{first_text:>11} {margin:9.3f} {top1:6d}"
        )
        metrics[mode.key] = {
            "margin": margin,
            "final_similarity": similarities[24],
            "top1": float(top1),
            "nodes24": nodes24,
            "expected": expected_mean,
            "unrelated": unrelated_mean,
        }

    print("\nPer-probe decoded candidates")
    print("-" * 128)
    for mode in MODES:
        print(f"\n{mode.label} | schedule={mode.schedule_text}")
        for word in PROBES:
            candidates = decode(
                peak_output(brain, traces_by_mode[mode.key][word]),
                output_patterns,
                brain.node_count,
            )
            text = ", ".join(f"{candidate}:{score:.3f}" for candidate, score in candidates)
            print(f"  {word:<4} -> {text}")

    eligible = [
        mode for mode in MODES
        if mode.key not in {"static80_r020", "static95_r020"}
    ]
    best = max(
        eligible,
        key=lambda mode: (
            metrics[mode.key]["top1"],
            metrics[mode.key]["margin"],
            -metrics[mode.key]["final_similarity"],
        ),
    )

    print("\nTemporal gating recommendation")
    print("-" * 128)
    result = metrics[best.key]
    print(
        f"Best temporal condition in this run: {best.key} "
        f"(top1={int(result['top1'])}/5, margin={result['margin']:+.3f}, "
        f"final similarity={result['final_similarity']:.3f})"
    )
    print("Selection priority: correct top-1 recall, successor margin, then lower final convergence.")

    print("\nInterpretation guardrails")
    print("-" * 128)
    print("1. Threshold changes depend only on propagation time, never on word labels.")
    print("2. Static 0.80 and 0.95 controls show whether timing itself adds value.")
    print("3. Residual flow is swept because weak-route leakage may cause late convergence.")
    print("4. A useful result should keep all or most learned top-1 successors while lowering similarity@24.")
    print("5. This remains an observation-layer experiment; the core SurfaceFlowBrain is unchanged.")


if __name__ == "__main__":
    main()
