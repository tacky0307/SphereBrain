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

# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------

EPOCHS = 5
MAX_REPLAY_STEPS = 36
THRESHOLD = 0.035
PERSISTENCE = 0.10
ENERGY_BUDGET = 8.0
CONVERGENCE_TOLERANCE = 0.0005
CONVERGENCE_PATIENCE = 4
TOP_K = 24

# Long-term weights are sharpened before temporary flow bias is applied.
WEIGHT_POWER = 2.35

# Temporary flow memory.
FLOW_BIAS_GAIN = 0.28
FLOW_BIAS_DECAY = 0.90
FLOW_BIAS_MAXIMUM = 3.25

# When A -> B flows, lightly prepare a few learned edges leaving B.
# This is an internal pathway operation: no word label or decoder is used.
LOOKAHEAD_GAIN = 0.08
LOOKAHEAD_EDGES_PER_NODE = 2

EXPERIENCE = ("空", "青い", "昼", "風")
DISTRACTOR_EXPERIENCES = (
    ("夜", "暗い", "星", "静か"),
    ("海", "広い", "波", "潮風"),
)

Edge = tuple[int, int]
Transition = tuple[str, str]


@dataclass
class FlowReplayResult:
    activation_history: list[np.ndarray]
    edge_history: list[set[Edge]]
    prepared_edge_history: list[set[Edge]]
    active_counts: list[int]
    differences: list[float]
    bias_max_history: list[float]
    bias_energy_history: list[float]
    stop_reason: str
    elapsed_seconds: float


def pattern_vector(pattern: dict[int, float], node_count: int) -> np.ndarray:
    vector = np.zeros(node_count, dtype=float)
    for node, value in pattern.items():
        vector[int(node)] = float(value)
    return vector


def transitions(experience: tuple[str, ...]) -> list[Transition]:
    return list(zip(experience, experience[1:]))


def build_brain() -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=2701,
        learning_rate=0.035,
        decay_rate=0.0002,
        fatigue_gain=0.30,
        fatigue_decay=0.74,
        transmission_gain=0.92,
        flow_bias_enabled=True,
        flow_bias_gain=FLOW_BIAS_GAIN,
        flow_bias_decay=FLOW_BIAS_DECAY,
        flow_bias_maximum=FLOW_BIAS_MAXIMUM,
        flow_bias_weight_power=WEIGHT_POWER,
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


def train(
    brain: SurfaceFlowBrain,
    inputs: dict[str, dict[int, float]],
    outputs: dict[str, dict[int, float]],
) -> tuple[int, float, dict[Transition, set[Edge]]]:
    pairs = [
        pair
        for experience in (EXPERIENCE, *DISTRACTOR_EXPERIENCES)
        for pair in transitions(experience)
    ]
    learned_paths: dict[Transition, set[Edge]] = defaultdict(set)
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

    # Replay must begin without temporary state left over from another run.
    brain.reset_flow_bias()
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


def learned_lookahead_edges(
    brain: SurfaceFlowBrain,
    active_edges: set[Edge],
) -> set[Edge]:
    """Choose a small number of learned outgoing edges after current flow.

    Selection uses only numeric internal state: reached nodes, usage counts,
    enabled edges, and long-term weights. It never uses word labels.
    """
    reached_nodes = {target for _, target in active_edges}
    prepared: set[Edge] = set()

    for node in reached_nodes:
        candidates = np.flatnonzero(
            brain.edge_enabled[node] & (brain.usage[node] > 0)
        )
        if candidates.size == 0:
            continue

        ranked = sorted(
            (int(target) for target in candidates),
            key=lambda target: (
                int(brain.usage[node, target]),
                float(brain.weights[node, target]),
            ),
            reverse=True,
        )
        for target in ranked[:LOOKAHEAD_EDGES_PER_NODE]:
            prepared.add((node, target))

    return prepared


def replay(
    brain: SurfaceFlowBrain,
    cue_pattern: dict[int, float],
    *,
    use_flow_bias: bool,
) -> FlowReplayResult:
    started = time.perf_counter()
    brain.reset_flow_bias()

    activation = pattern_vector(cue_pattern, brain.node_count)
    activation = normalize_energy(activation, ENERGY_BUDGET)
    fatigue = np.zeros(brain.node_count, dtype=float)

    activation_history = [activation.copy()]
    edge_history: list[set[Edge]] = []
    prepared_edge_history: list[set[Edge]] = []
    active_counts = [int(np.count_nonzero(activation))]
    differences: list[float] = []
    bias_max_history: list[float] = []
    bias_energy_history: list[float] = []
    stable_steps = 0
    stop_reason = f"reached maximum replay steps ({MAX_REPLAY_STEPS})"

    for _ in range(MAX_REPLAY_STEPS):
        if use_flow_bias:
            brain.flow_bias.decay()

        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        step_weights = brain.effective_weights(use_flow_bias=use_flow_bias)
        contributions = effective[:, None] * step_weights * brain.transmission_gain
        contributions[~brain.edge_enabled] = 0.0
        contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)

        propagated = 1.0 - np.prod(1.0 - contributions, axis=0)
        next_activation = propagated + PERSISTENCE * activation
        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < THRESHOLD] = 0.0
        next_activation = keep_strongest(next_activation, TOP_K)
        next_activation = normalize_energy(next_activation, ENERGY_BUDGET)

        survivor_mask = next_activation > 0.0
        active_edges = {
            (int(source), int(target))
            for source, target in np.argwhere(
                (contributions >= THRESHOLD * brain.edge_activity_ratio)
                & survivor_mask[None, :]
            )
        }
        edge_history.append(active_edges)

        prepared_edges: set[Edge] = set()
        if use_flow_bias and active_edges:
            brain.flow_bias.reinforce(active_edges)
            prepared_edges = learned_lookahead_edges(brain, active_edges)
            if prepared_edges:
                brain.flow_bias.reinforce(prepared_edges, amount=LOOKAHEAD_GAIN)
        prepared_edge_history.append(prepared_edges)

        bias_stats = brain.flow_bias_stats()
        bias_max_history.append(bias_stats.maximum)
        bias_energy_history.append(bias_stats.total_energy)

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

    return FlowReplayResult(
        activation_history=activation_history,
        edge_history=edge_history,
        prepared_edge_history=prepared_edge_history,
        active_counts=active_counts,
        differences=differences,
        bias_max_history=bias_max_history,
        bias_energy_history=bias_energy_history,
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - started,
    )


def edge_metrics(replayed: set[Edge], learned: set[Edge]) -> tuple[float, float]:
    shared = len(replayed & learned)
    recall = 0.0 if not learned else shared / len(learned)
    precision = 0.0 if not replayed else shared / len(replayed)
    return recall, precision


def transition_strength_by_step(
    edge_history: list[set[Edge]],
    learned_paths: dict[Transition, set[Edge]],
) -> dict[Transition, list[float]]:
    strengths: dict[Transition, list[float]] = {}
    for pair, learned in learned_paths.items():
        if not learned:
            strengths[pair] = [0.0 for _ in edge_history]
            continue
        strengths[pair] = [
            len(edges & learned) / len(learned)
            for edges in edge_history
        ]
    return strengths


def first_hit_steps(
    strengths: dict[Transition, list[float]],
    threshold: float = 0.20,
) -> dict[Transition, int | None]:
    hits: dict[Transition, int | None] = {}
    for pair in transitions(EXPERIENCE):
        hits[pair] = next(
            (
                step
                for step, value in enumerate(strengths[pair], start=1)
                if value >= threshold
            ),
            None,
        )
    return hits


def peak_steps(
    strengths: dict[Transition, list[float]],
) -> dict[Transition, tuple[int | None, float]]:
    peaks: dict[Transition, tuple[int | None, float]] = {}
    for pair in transitions(EXPERIENCE):
        values = strengths[pair]
        if not values or max(values) <= 0.0:
            peaks[pair] = (None, 0.0)
            continue
        index = int(np.argmax(values))
        peaks[pair] = (index + 1, float(values[index]))
    return peaks


def ordered_steps(values: list[int | None]) -> bool:
    return (
        all(value is not None for value in values)
        and all(left <= right for left, right in zip(values, values[1:]))
        and len(set(values)) > 1
    )


def dominant_target_flow(
    strengths: dict[Transition, list[float]],
) -> list[str]:
    target_pairs = transitions(EXPERIENCE)
    sequence: list[str] = []
    for step in range(len(next(iter(strengths.values()), []))):
        pair = max(target_pairs, key=lambda item: strengths[item][step])
        value = strengths[pair][step]
        sequence.append("none" if value <= 0.0 else f"{pair[0]}->{pair[1]}")
    return sequence


def compress_sequence(sequence: list[str]) -> str:
    if not sequence:
        return "none"
    groups: list[tuple[str, int, int]] = []
    start = 1
    current = sequence[0]
    for step, item in enumerate(sequence[1:], start=2):
        if item != current:
            groups.append((current, start, step - 1))
            current = item
            start = step
    groups.append((current, start, len(sequence)))
    return " | ".join(
        f"{name}@{first}" if first == last else f"{name}@{first}-{last}"
        for name, first, last in groups
    )


def print_trial(
    name: str,
    result: FlowReplayResult,
    learned_paths: dict[Transition, set[Edge]],
    target_edges: set[Edge],
    distractor_edges: set[Edge],
) -> dict[str, float | bool]:
    replayed = set().union(*result.edge_history) if result.edge_history else set()
    target_recall, target_precision = edge_metrics(replayed, target_edges)
    distractor_recall, distractor_precision = edge_metrics(replayed, distractor_edges)
    strengths = transition_strength_by_step(result.edge_history, learned_paths)
    hits = first_hit_steps(strengths)
    peaks = peak_steps(strengths)

    hit_order = ordered_steps([hits[pair] for pair in transitions(EXPERIENCE)])
    peak_order = ordered_steps([peaks[pair][0] for pair in transitions(EXPERIENCE)])
    dominant = dominant_target_flow(strengths)

    heat = Counter(edge for step in result.edge_history for edge in step)
    target_hot = sum(count for edge, count in heat.items() if edge in target_edges)
    distractor_hot = sum(
        count for edge, count in heat.items() if edge in distractor_edges
    )
    other_hot = sum(
        count
        for edge, count in heat.items()
        if edge not in target_edges and edge not in distractor_edges
    )

    print(f"\n{name}")
    print(f"  steps                 : {len(result.edge_history)}")
    print(f"  max active nodes      : {max(result.active_counts)}")
    print(f"  final active nodes    : {result.active_counts[-1]}")
    print(f"  unique replay edges   : {len(replayed)}")
    print(f"  target recall         : {target_recall:.3f}")
    print(f"  target precision      : {target_precision:.3f}")
    print(f"  distractor recall     : {distractor_recall:.3f}")
    print(f"  distractor precision  : {distractor_precision:.3f}")
    print(f"  selectivity gap       : {target_precision - distractor_precision:+.3f}")
    print(f"  ordered first hits    : {hit_order}")
    print(f"  ordered peak flow     : {peak_order}")
    for pair in transitions(EXPERIENCE):
        first = "none" if hits[pair] is None else str(hits[pair])
        peak_step, peak_value = peaks[pair]
        peak_shown = "none" if peak_step is None else str(peak_step)
        print(
            f"    {pair[0]} -> {pair[1]} "
            f"first_hit={first}, peak_step={peak_shown}, peak={peak_value:.3f}"
        )
    print(f"  dominant target flow  : {compress_sequence(dominant)}")
    print(f"  repeated target flow  : {target_hot}")
    print(f"  repeated distractor   : {distractor_hot}")
    print(f"  repeated other flow   : {other_hot}")
    print(
        f"  final bias max/energy : "
        f"{result.bias_max_history[-1] if result.bias_max_history else 1.0:.3f} / "
        f"{result.bias_energy_history[-1] if result.bias_energy_history else 0.0:.3f}"
    )
    print(f"  stop                  : {result.stop_reason}")
    print(f"  replay time           : {result.elapsed_seconds:.3f}s")

    return {
        "target_precision": target_precision,
        "distractor_precision": distractor_precision,
        "selectivity_gap": target_precision - distractor_precision,
        "ordered_first_hits": hit_order,
        "ordered_peaks": peak_order,
    }


def main() -> None:
    total_started = time.perf_counter()
    print("SphereBrain v27 — Flow-Biased Competitive Replay")
    print("Only the first cue is injected.")
    print("No decoded word is fed back, and replay does not alter long-term weights.")
    print("Temporary edge bias is created only by internal numeric flow.\n")

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

    print("\nReplay comparison")
    print(f"  competition top_k : {TOP_K}")
    print(f"  weight power      : {WEIGHT_POWER}")
    print(f"  flow gain/decay   : {FLOW_BIAS_GAIN} / {FLOW_BIAS_DECAY}")
    print(f"  lookahead gain    : {LOOKAHEAD_GAIN}")

    baseline = replay(brain, inputs[EXPERIENCE[0]], use_flow_bias=False)
    baseline_summary = print_trial(
        "Baseline — sharpened weights, no temporary flow bias",
        baseline,
        learned_paths,
        target_edges,
        distractor_edges,
    )

    biased = replay(brain, inputs[EXPERIENCE[0]], use_flow_bias=True)
    biased_summary = print_trial(
        "v27 — temporary flow bias + learned numeric lookahead",
        biased,
        learned_paths,
        target_edges,
        distractor_edges,
    )

    print("\nComparison")
    print("-" * 76)
    print(
        "target precision change : "
        f"{float(biased_summary['target_precision']) - float(baseline_summary['target_precision']):+.3f}"
    )
    print(
        "selectivity gap change  : "
        f"{float(biased_summary['selectivity_gap']) - float(baseline_summary['selectivity_gap']):+.3f}"
    )
    print(
        "ordered peaks           : "
        f"baseline={baseline_summary['ordered_peaks']} / "
        f"v27={biased_summary['ordered_peaks']}"
    )

    print("\nSummary")
    print("-" * 76)
    print(f"learned transitions : {events}")
    print(f"training time       : {training_seconds:.2f}s")
    print(f"total experiment    : {time.perf_counter() - total_started:.2f}s")

    print("\nHow to read v27")
    print("1. Activity must remain near top_k instead of expanding to all 600 nodes.")
    print("2. Flow bias should improve target precision or the selectivity gap.")
    print("3. ordered peak flow is stronger evidence than merely touching all paths.")
    print("4. A useful replay should move across target transitions, then settle or vanish.")
    print("5. Failure is informative: it means temporary inertia alone is insufficient.")
    print("6. The observer decodes only after replay; it never controls the internal flow.")


if __name__ == "__main__":
    main()
