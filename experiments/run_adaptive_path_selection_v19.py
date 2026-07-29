from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain


EXPERIENCES: tuple[tuple[str, ...], ...] = (
    ("空", "青い"),
    ("青い", "空", "見える"),
    ("今日", "楽しい"),
    ("雨", "降る"),
    ("今日", "雨"),
    ("私", "うれしい"),
)
PROBES: tuple[str, ...] = ("空", "青い", "今日", "雨", "私")
RELATIVE_THRESHOLDS: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)
STEPS = 24
ACTIVE_THRESHOLD = 1e-5
RESIDUAL_FRACTION = 0.02
MAX_BRANCHES = 8


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    kind: str
    relative_threshold: float | None = None


@dataclass
class TraceResult:
    history: list[np.ndarray]
    mean_branches: list[float]
    branch_entropy: list[float]
    energy: list[float]
    selected_edges: list[int]


MODES: tuple[Mode, ...] = (
    Mode("baseline", "A: current diffusion", "baseline"),
    Mode("strongest2", "B: fixed strongest-2 routing", "strongest2"),
    *tuple(
        Mode(
            f"adaptive_{int(threshold * 100):02d}",
            f"Adaptive relative threshold {threshold:.2f}",
            "adaptive",
            threshold,
        )
        for threshold in RELATIVE_THRESHOLDS
    ),
)


def adjacent_pairs() -> list[tuple[str, str]]:
    return [
        (words[index], words[index + 1])
        for words in EXPERIENCES
        for index in range(len(words) - 1)
    ]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(np.dot(left, right) / denominator)


def normalized_entropy(values: np.ndarray) -> float:
    positive = values[values > 0.0]
    if positive.size <= 1:
        return 0.0
    probabilities = positive / float(np.sum(positive))
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-15)))
    return entropy / math.log(float(positive.size))


def input_vector(brain: SurfaceFlowBrain, pattern: dict[int, float]) -> np.ndarray:
    vector = np.zeros(brain.node_count, dtype=float)
    for node, value in pattern.items():
        vector[int(node)] = float(value)
    return vector


def baseline_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    steps: int = STEPS,
    threshold: float = 0.08,
) -> TraceResult:
    activation = input_vector(brain, pattern)
    fatigue = np.zeros(brain.node_count, dtype=float)
    history = [activation.copy()]
    energy = [float(np.sum(activation))]
    mean_branches = [0.0]
    branch_entropy = [0.0]
    selected_edges = [0]

    for _ in range(steps):
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        contributions = effective[:, None] * brain.weights * brain.transmission_gain
        contributions[~brain.edge_enabled] = 0.0
        contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)
        next_activation = 1.0 - np.prod(1.0 - contributions, axis=0)
        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < threshold] = 0.0

        active_sources = np.flatnonzero(effective > 0.0)
        branch_counts = [
            int(np.count_nonzero(contributions[source] > 0.0))
            for source in active_sources
        ]
        mean_branches.append(float(np.mean(branch_counts)) if branch_counts else 0.0)
        branch_entropy.append(normalized_entropy(next_activation))
        selected_edges.append(int(sum(branch_counts)))

        fatigue *= brain.fatigue_decay
        fatigue += next_activation * brain.fatigue_gain
        fatigue = np.clip(fatigue, 0.0, 0.95)
        activation = next_activation
        history.append(activation.copy())
        energy.append(float(np.sum(activation)))

    return TraceResult(history, mean_branches, branch_entropy, energy, selected_edges)


def selected_destinations(
    brain: SurfaceFlowBrain,
    source: int,
    kind: str,
    relative_threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    enabled = np.flatnonzero(brain.edge_enabled[source])
    if enabled.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0

    weights = np.asarray(brain.weights[source, enabled], dtype=float)
    positive_mask = weights > 0.0
    enabled = enabled[positive_mask]
    weights = weights[positive_mask]
    if enabled.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0

    if kind == "strongest2":
        keep = min(2, enabled.size)
        indices = np.argpartition(weights, -keep)[-keep:]
        selected = enabled[indices]
        selected_weights = weights[indices]
        return selected, selected_weights, int(selected.size)

    if kind != "adaptive" or relative_threshold is None:
        raise ValueError(f"Unsupported routing kind: {kind}")

    best = float(np.max(weights))
    selected_mask = weights >= best * relative_threshold
    selected = enabled[selected_mask]
    selected_weights = weights[selected_mask]

    if selected.size > MAX_BRANCHES:
        indices = np.argpartition(selected_weights, -MAX_BRANCHES)[-MAX_BRANCHES:]
        selected = selected[indices]
        selected_weights = selected_weights[indices]

    if selected.size == 0:
        best_index = int(np.argmax(weights))
        selected = np.asarray([enabled[best_index]], dtype=int)
        selected_weights = np.asarray([weights[best_index]], dtype=float)

    return selected, selected_weights, int(selected.size)


def routed_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    mode: Mode,
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

    for _ in range(steps):
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
                mode.kind,
                mode.relative_threshold,
            )
            if selected.size == 0:
                continue

            step_branch_counts.append(branch_count)
            step_selected_edges += branch_count

            selected_total = float(np.sum(selected_weights))
            if selected_total <= 0.0:
                continue

            main_energy = source_energy
            if mode.kind == "adaptive":
                main_energy *= 1.0 - RESIDUAL_FRACTION

            next_activation[selected] += main_energy * (selected_weights / selected_total)

            if mode.kind == "adaptive" and RESIDUAL_FRACTION > 0.0:
                enabled = np.flatnonzero(brain.edge_enabled[source])
                residual = np.setdiff1d(enabled, selected, assume_unique=False)
                if residual.size > 0:
                    residual_weights = np.asarray(brain.weights[source, residual], dtype=float)
                    residual_weights = np.clip(residual_weights, 0.0, None)
                    residual_total = float(np.sum(residual_weights))
                    if residual_total > 0.0:
                        next_activation[residual] += (
                            source_energy
                            * RESIDUAL_FRACTION
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


def run_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    mode: Mode,
) -> TraceResult:
    if mode.kind == "baseline":
        return baseline_trace(brain, pattern)
    return routed_trace(brain, pattern, mode)


def peak_output(brain: SurfaceFlowBrain, trace: TraceResult) -> np.ndarray:
    peak = np.max(np.stack(trace.history, axis=0), axis=0)
    mask = np.zeros(brain.node_count, dtype=bool)
    mask[brain.output_nodes] = True
    return peak * mask


def decode(
    vector: np.ndarray,
    output_patterns: dict[str, dict[int, float]],
    node_count: int,
    limit: int = 3,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for word, pattern in output_patterns.items():
        target = np.zeros(node_count, dtype=float)
        for node, value in pattern.items():
            target[node] = value
        ranked.append((word, cosine(vector, target)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def first_step_at_or_above(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value >= threshold:
            return index
    return None


def mean_pairwise_by_step(
    traces: dict[str, TraceResult],
    steps: int = STEPS,
) -> list[float]:
    means: list[float] = []
    for step in range(steps + 1):
        similarities = [
            cosine(traces[left].history[step], traces[right].history[step])
            for left, right in combinations(PROBES, 2)
        ]
        means.append(float(np.mean(similarities)))
    return means


def successor_margin(
    brain: SurfaceFlowBrain,
    traces: dict[str, TraceResult],
    output_patterns: dict[str, dict[int, float]],
    vocabulary: list[str],
) -> tuple[float, float, float]:
    expected: dict[str, tuple[str, ...]] = {
        "空": ("青い", "見える"),
        "青い": ("空",),
        "今日": ("楽しい", "雨"),
        "雨": ("降る",),
        "私": ("うれしい",),
    }
    expected_scores: list[float] = []
    unrelated_scores: list[float] = []

    for source in PROBES:
        scores = dict(
            decode(
                peak_output(brain, traces[source]),
                output_patterns,
                brain.node_count,
                limit=len(vocabulary),
            )
        )
        expected_set = set(expected[source])
        expected_scores.extend(scores[word] for word in expected_set)
        unrelated_scores.extend(
            score
            for word, score in scores.items()
            if word not in expected_set and word != source
        )

    expected_mean = float(np.mean(expected_scores))
    unrelated_mean = float(np.mean(unrelated_scores))
    return expected_mean, unrelated_mean, expected_mean - unrelated_mean


def main() -> None:
    print("SphereBrain v19 — Adaptive Path Selection")
    print("A learned brain chooses a variable number of outgoing pathways at each node.")
    print("Question: can adaptive branching preserve identity, energy, and learned successors?\n")

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
    print(f"Adaptive residual flow={RESIDUAL_FRACTION:.3f}, max branches={MAX_BRANCHES}")
    print("-" * 112)
    for mode in MODES:
        threshold_text = "-" if mode.relative_threshold is None else f"{mode.relative_threshold:.2f}"
        print(f"{mode.key:<14} {mode.label:<48} relative-threshold={threshold_text}")

    traces_by_mode: dict[str, dict[str, TraceResult]] = {
        mode.key: {
            word: run_trace(brain, input_patterns[word], mode)
            for word in PROBES
        }
        for mode in MODES
    }

    print("\nActivation, energy, and branching summary")
    print("-" * 112)
    print(
        f"{'mode':<14} {'nodes@5':>8} {'nodes@24':>9} {'energy@24':>11} "
        f"{'mean branch':>12} {'entropy':>9} {'edges/step':>11}"
    )
    for mode in MODES:
        traces = traces_by_mode[mode.key]
        nodes5 = np.mean([
            np.count_nonzero(traces[word].history[5]) for word in PROBES
        ])
        nodes24 = np.mean([
            np.count_nonzero(traces[word].history[24]) for word in PROBES
        ])
        energy24 = np.mean([traces[word].energy[24] for word in PROBES])
        branches = np.mean([
            value
            for word in PROBES
            for value in traces[word].mean_branches[1:]
        ])
        entropy = np.mean([
            value
            for word in PROBES
            for value in traces[word].branch_entropy[1:]
        ])
        edges_per_step = np.mean([
            value
            for word in PROBES
            for value in traces[word].selected_edges[1:]
        ])
        print(
            f"{mode.key:<14} {nodes5:8.1f} {nodes24:9.1f} {energy24:11.3f} "
            f"{branches:12.2f} {entropy:9.3f} {edges_per_step:11.1f}"
        )

    print("\nConvergence and learned-successor separation")
    print("-" * 112)
    print(
        f"{'mode':<14} {'first mean>=.90':>17} {'mean@5':>9} {'mean@24':>10} "
        f"{'expected':>10} {'unrelated':>10} {'margin':>9}"
    )
    margins: dict[str, float] = {}
    final_similarities: dict[str, float] = {}
    for mode in MODES:
        traces = traces_by_mode[mode.key]
        similarities = mean_pairwise_by_step(traces)
        first = first_step_at_or_above(similarities, 0.90)
        first_text = "never" if first is None else str(first)
        expected_mean, unrelated_mean, margin = successor_margin(
            brain,
            traces,
            output_patterns,
            vocabulary,
        )
        margins[mode.key] = margin
        final_similarities[mode.key] = similarities[24]
        print(
            f"{mode.key:<14} {first_text:>17} {similarities[5]:9.3f} "
            f"{similarities[24]:10.3f} {expected_mean:10.3f} "
            f"{unrelated_mean:10.3f} {margin:9.3f}"
        )

    print("\nPer-probe decoded candidates")
    print("-" * 112)
    for mode in MODES:
        print(f"\n{mode.label}")
        for word in PROBES:
            candidates = decode(
                peak_output(brain, traces_by_mode[mode.key][word]),
                output_patterns,
                brain.node_count,
            )
            text = ", ".join(
                f"{candidate}:{score:.3f}" for candidate, score in candidates
            )
            print(f"  {word:<4} -> {text}")

    adaptive_modes = [mode for mode in MODES if mode.kind == "adaptive"]
    best_mode = max(
        adaptive_modes,
        key=lambda mode: (
            margins[mode.key],
            -final_similarities[mode.key],
        ),
    )

    print("\nAdaptive threshold recommendation")
    print("-" * 112)
    print(
        f"Best adaptive condition in this run: {best_mode.key} "
        f"(margin={margins[best_mode.key]:+.3f}, "
        f"final similarity={final_similarities[best_mode.key]:.3f})"
    )
    print("This is an experimental winner, not yet a permanent SphereBrain rule.")

    print("\nInterpretation guardrails")
    print("-" * 112)
    print("1. Adaptive selection uses relative local weights; it does not read word labels.")
    print("2. Energy is renormalized after every routed step so route quality is not confused with decay.")
    print("3. Residual flow keeps weak alternatives alive, but too much residual flow may recreate diffusion.")
    print("4. A useful condition needs positive successor margin without rapid cross-input convergence.")
    print("5. If adaptive branching still dies or diffuses, the next experiment should tune residual flow and fatigue.")


if __name__ == "__main__":
    main()
