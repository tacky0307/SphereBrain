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
from run_early_narrowing_boundary_v19c import EXPECTED


@dataclass(frozen=True)
class GateMode:
    key: str
    label: str
    medium_change: float
    narrow_change: float
    required_stable_steps: int = 1


@dataclass
class SelfTrace:
    trace: TraceResult
    thresholds: list[float]
    changes: list[float]
    transitions: list[tuple[int, float, float]]


MODES: tuple[GateMode, ...] = (
    GateMode("change_loose", "Self gate: change < .20 / .08", 0.20, 0.08),
    GateMode("change_mid", "Self gate: change < .15 / .05", 0.15, 0.05),
    GateMode("change_strict", "Self gate: change < .10 / .03", 0.10, 0.03),
    GateMode("change_mid2", "Self gate: 2 stable steps", 0.15, 0.05, 2),
)


def scheduled_boundary1_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    steps: int = STEPS,
) -> SelfTrace:
    """v19c boundary1 control: 1x.80, 4x.90, then .95."""
    schedule = [0.80] + [0.90] * 4 + [0.95] * max(0, steps - 5)
    return propagate_with_thresholds(brain, pattern, schedule[:steps])


def propagate_with_thresholds(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    thresholds: list[float],
) -> SelfTrace:
    activation = input_vector(brain, pattern)
    initial_energy = float(np.sum(activation))
    fatigue = np.zeros(brain.node_count, dtype=float)

    history = [activation.copy()]
    energy = [initial_energy]
    mean_branches = [0.0]
    branch_entropy = [normalized_entropy(activation)]
    selected_edges = [0]
    changes = [0.0]

    for threshold in thresholds:
        previous = activation.copy()
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        next_activation = np.zeros(brain.node_count, dtype=float)
        active_sources = np.flatnonzero(effective > ACTIVE_THRESHOLD)
        branch_counts: list[int] = []
        edge_count = 0

        for source in active_sources:
            source_energy = float(effective[source])
            selected, weights, branch_count = selected_destinations(
                brain, int(source), "adaptive", threshold
            )
            if selected.size == 0:
                continue
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            branch_counts.append(branch_count)
            edge_count += branch_count
            next_activation[selected] += source_energy * (weights / total)

        next_activation[next_activation < ACTIVE_THRESHOLD] = 0.0
        total = float(np.sum(next_activation))
        if total > 0.0:
            next_activation *= initial_energy / total

        fatigue *= brain.fatigue_decay
        fatigue += np.clip(next_activation, 0.0, 1.0) * brain.fatigue_gain
        fatigue = np.clip(fatigue, 0.0, 0.95)

        activation = next_activation
        history.append(activation.copy())
        energy.append(float(np.sum(activation)))
        mean_branches.append(float(np.mean(branch_counts)) if branch_counts else 0.0)
        branch_entropy.append(normalized_entropy(activation))
        selected_edges.append(edge_count)
        changes.append(1.0 - cosine(previous, activation))

    trace = TraceResult(history, mean_branches, branch_entropy, energy, selected_edges)
    return SelfTrace(trace, [0.0] + thresholds, changes, [])


def self_adjusting_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
    mode: GateMode,
    steps: int = STEPS,
) -> SelfTrace:
    """
    Start broad, then narrow only from internal activity stability.

    The gate never reads words, expected answers, decoder scores, or labels.
    It observes only 1 - cosine(previous_activity, current_activity).
    Selectivity is monotonic: .80 -> .90 -> .95.
    """
    activation = input_vector(brain, pattern)
    initial_energy = float(np.sum(activation))
    fatigue = np.zeros(brain.node_count, dtype=float)

    history = [activation.copy()]
    energy = [initial_energy]
    mean_branches = [0.0]
    branch_entropy = [normalized_entropy(activation)]
    selected_edges = [0]
    thresholds = [0.0]
    changes = [0.0]
    transitions: list[tuple[int, float, float]] = []

    threshold = 0.80
    stable_count = 0

    for step in range(1, steps + 1):
        previous = activation.copy()
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        next_activation = np.zeros(brain.node_count, dtype=float)
        active_sources = np.flatnonzero(effective > ACTIVE_THRESHOLD)
        branch_counts: list[int] = []
        edge_count = 0

        for source in active_sources:
            source_energy = float(effective[source])
            selected, weights, branch_count = selected_destinations(
                brain, int(source), "adaptive", threshold
            )
            if selected.size == 0:
                continue
            total = float(np.sum(weights))
            if total <= 0.0:
                continue
            branch_counts.append(branch_count)
            edge_count += branch_count
            next_activation[selected] += source_energy * (weights / total)

        next_activation[next_activation < ACTIVE_THRESHOLD] = 0.0
        total = float(np.sum(next_activation))
        if total > 0.0:
            next_activation *= initial_energy / total

        fatigue *= brain.fatigue_decay
        fatigue += np.clip(next_activation, 0.0, 1.0) * brain.fatigue_gain
        fatigue = np.clip(fatigue, 0.0, 0.95)

        activation = next_activation
        change = 1.0 - cosine(previous, activation)

        history.append(activation.copy())
        energy.append(float(np.sum(activation)))
        mean_branches.append(float(np.mean(branch_counts)) if branch_counts else 0.0)
        branch_entropy.append(normalized_entropy(activation))
        selected_edges.append(edge_count)
        thresholds.append(threshold)
        changes.append(change)

        target = threshold
        boundary = None
        if threshold == 0.80 and change <= mode.medium_change:
            target = 0.90
            boundary = mode.medium_change
        elif threshold == 0.90 and change <= mode.narrow_change:
            target = 0.95
            boundary = mode.narrow_change

        if target > threshold:
            stable_count += 1
            if stable_count >= mode.required_stable_steps:
                transitions.append((step, target, change))
                threshold = target
                stable_count = 0
        else:
            stable_count = 0

    trace = TraceResult(history, mean_branches, branch_entropy, energy, selected_edges)
    return SelfTrace(trace, thresholds, changes, transitions)


def mean_pairwise_by_step(traces: dict[str, TraceResult]) -> list[float]:
    values: list[float] = []
    for step in range(STEPS + 1):
        similarities = [
            cosine(traces[a].history[step], traces[b].history[step])
            for a, b in combinations(PROBES, 2)
        ]
        values.append(float(np.mean(similarities)))
    return values


def expected_top1_count(
    brain: SurfaceFlowBrain,
    traces: dict[str, TraceResult],
    output_patterns: dict[str, dict[int, float]],
) -> int:
    hits = 0
    for source in PROBES:
        candidate = decode(
            peak_output(brain, traces[source]), output_patterns, brain.node_count, limit=1
        )[0][0]
        hits += int(candidate in EXPECTED[source])
    return hits


def identity_score(traces: dict[str, TraceResult]) -> float:
    similarities = [
        cosine(traces[a].history[STEPS], traces[b].history[STEPS])
        for a, b in combinations(PROBES, 2)
    ]
    return 1.0 - float(np.mean(similarities))


def main() -> None:
    print("SphereBrain v20 — Self-Adjusting Gating")
    print("The brain narrows from its own activity change, not from propagation time.")
    print("Gate input: 1 - cosine(previous activity, current activity).")
    print("No word labels, expected successors, or decoder scores enter the gate.\n")

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
    print("-" * 142)
    print("boundary1     Scheduled control: 1x.80 / 4x.90 / rest .95")
    for mode in MODES:
        print(
            f"{mode.key:<13} {mode.label:<34} "
            f"medium<={mode.medium_change:.3f} narrow<={mode.narrow_change:.3f} "
            f"stable_steps={mode.required_stable_steps}"
        )

    self_traces: dict[str, dict[str, SelfTrace]] = {
        "boundary1": {
            word: scheduled_boundary1_trace(brain, input_patterns[word]) for word in PROBES
        }
    }
    for mode in MODES:
        self_traces[mode.key] = {
            word: self_adjusting_trace(brain, input_patterns[word], mode) for word in PROBES
        }

    print("\nSelf-adjusting comparison")
    print("-" * 142)
    print(
        f"{'mode':<13} {'nodes@24':>9} {'sim@24':>9} {'identity':>9} "
        f"{'margin':>9} {'top1':>6} {'mean .90':>9} {'mean .95':>9} {'final gate':>11}"
    )

    metrics: dict[str, dict[str, float]] = {}
    keys = ["boundary1"] + [mode.key for mode in MODES]
    for key in keys:
        traces = {word: self_traces[key][word].trace for word in PROBES}
        similarities = mean_pairwise_by_step(traces)
        _, _, margin = successor_margin(brain, traces, output_patterns, vocabulary)
        top1 = expected_top1_count(brain, traces, output_patterns)
        identity = identity_score(traces)
        nodes24 = float(np.mean([
            np.count_nonzero(traces[word].history[STEPS]) for word in PROBES
        ]))

        to90: list[int] = []
        to95: list[int] = []
        final_gates: list[float] = []
        for word in PROBES:
            item = self_traces[key][word]
            for step, target, _change in item.transitions:
                if target == 0.90:
                    to90.append(step)
                elif target == 0.95:
                    to95.append(step)
            final_gates.append(item.thresholds[-1])

        mean90 = float(np.mean(to90)) if to90 else float("inf")
        mean95 = float(np.mean(to95)) if to95 else float("inf")
        text90 = "never" if not np.isfinite(mean90) else f"{mean90:.1f}"
        text95 = "never" if not np.isfinite(mean95) else f"{mean95:.1f}"
        final_gate = float(np.mean(final_gates))

        print(
            f"{key:<13} {nodes24:9.1f} {similarities[STEPS]:9.3f} {identity:9.3f} "
            f"{margin:9.3f} {top1:6d} {text90:>9} {text95:>9} {final_gate:11.3f}"
        )
        metrics[key] = {
            "top1": float(top1),
            "margin": margin,
            "similarity": similarities[STEPS],
            "identity": identity,
        }

    print("\nPer-probe gate transitions and decoded candidates")
    print("-" * 142)
    for key in keys:
        print(f"\n{key}")
        for word in PROBES:
            item = self_traces[key][word]
            transition_text = ", ".join(
                f"step{step}->.{int(target * 100):02d}(change={change:.3f})"
                for step, target, change in item.transitions
            ) or "scheduled/no self-transition"
            candidates = decode(
                peak_output(brain, item.trace), output_patterns, brain.node_count
            )
            candidate_text = ", ".join(
                f"{candidate}:{score:.3f}" for candidate, score in candidates
            )
            print(f"  {word:<4} {transition_text:<58} final={candidate_text}")

    self_keys = [mode.key for mode in MODES]
    best = max(
        self_keys,
        key=lambda key: (
            metrics[key]["top1"],
            metrics[key]["margin"],
            metrics[key]["identity"],
            -metrics[key]["similarity"],
        ),
    )
    print("\nSelf-adjusting gating recommendation")
    print("-" * 142)
    print(
        f"Best self-adjusting condition in this run: {best} "
        f"(top1={int(metrics[best]['top1'])}/5, margin={metrics[best]['margin']:+.3f}, "
        f"sim@24={metrics[best]['similarity']:.3f}, identity={metrics[best]['identity']:.3f})"
    )
    print("Selection priority: correct recall, successor margin, identity, then lower convergence.")

    print("\nInterpretation guardrails")
    print("-" * 142)
    print("1. The self gate reads only consecutive internal activation patterns.")
    print("2. It never reads text, labels, expected answers, or decoded candidates.")
    print("3. Gate selectivity can only increase (.80 -> .90 -> .95); it cannot oscillate.")
    print("4. The scheduled boundary1 result is retained as the v19c control.")
    print("5. This is an observation-layer experiment; SurfaceFlowBrain remains unchanged.")


if __name__ == "__main__":
    main()
