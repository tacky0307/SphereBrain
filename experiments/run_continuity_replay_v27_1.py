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
# SphereBrain v27.1
#
# v27 added temporary edge bias and numeric lookahead.
# v27.1 tests whether replay fails because successive memories are disconnected:
#
#   input("空") -> output("青い")
#                         |
#                         v  continuity bridge
#                  input("青い") -> output("昼")
#
# The bridge is learned as an ordinary numeric pathway. During replay no word,
# decoder, or external feedback is used.
# -----------------------------------------------------------------------------

EPOCHS = 5
BRIDGE_EPOCHS = 5
MAX_REPLAY_STEPS = 36

THRESHOLD = 0.035
PERSISTENCE = 0.10
ENERGY_BUDGET = 8.0
CONVERGENCE_TOLERANCE = 0.0005
CONVERGENCE_PATIENCE = 4
TOP_K = 24

WEIGHT_POWER = 2.35

FLOW_BIAS_GAIN = 0.28
FLOW_BIAS_DECAY = 0.90
FLOW_BIAS_MAXIMUM = 3.25

LOOKAHEAD_GAIN = 0.08
LOOKAHEAD_EDGES_PER_NODE = 2

# Bridge training is intentionally weaker than ordinary transition training.
# It should connect consecutive memories without becoming the dominant memory.
BRIDGE_REINFORCE_SCALE = 0.55

EXPERIENCE = ("空", "青い", "昼", "風")
DISTRACTOR_EXPERIENCES = (
    ("夜", "暗い", "星", "静か"),
    ("海", "広い", "波", "潮風"),
)

Edge = tuple[int, int]
Transition = tuple[str, str]


@dataclass
class ReplayResult:
    activation_history: list[np.ndarray]
    edge_history: list[set[Edge]]
    prepared_edge_history: list[set[Edge]]
    active_counts: list[int]
    differences: list[float]
    bias_max_history: list[float]
    bias_energy_history: list[float]
    stop_reason: str
    elapsed_seconds: float


@dataclass
class TrainingResult:
    events: int
    elapsed_seconds: float
    learned_paths: dict[Transition, set[Edge]]
    bridge_paths: dict[str, set[Edge]]


def transitions(experience: tuple[str, ...]) -> list[Transition]:
    return list(zip(experience, experience[1:]))


def all_experiences() -> tuple[tuple[str, ...], ...]:
    return (EXPERIENCE, *DISTRACTOR_EXPERIENCES)


def pattern_vector(pattern: dict[int, float], node_count: int) -> np.ndarray:
    vector = np.zeros(node_count, dtype=float)
    for node, value in pattern.items():
        vector[int(node)] = float(value)
    return vector


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


def build_brain(seed: int = 2711) -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=seed,
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
            for experience in all_experiences()
            for word in experience
        }
    )

    # Input and output encoders intentionally use different surface regions.
    # v27.1 learns continuity between the resulting numeric patterns.
    input_encoder = TextSurfaceEncoder(brain.input_nodes, width=5)
    output_encoder = TextSurfaceEncoder(brain.output_nodes, width=5)

    inputs = {word: input_encoder.encode(word) for word in words}
    outputs = {word: output_encoder.encode(word) for word in words}
    return brain, inputs, outputs


def ranked_nodes(pattern: dict[int, float]) -> list[int]:
    return sorted(pattern, key=lambda node: pattern[node], reverse=True)


def continuity_words() -> list[str]:
    """Words that appear as both an output and the next transition's input."""
    words: list[str] = []
    for experience in all_experiences():
        words.extend(experience[1:-1])
    return words


def learn_continuity_bridge(
    brain: SurfaceFlowBrain,
    output_pattern: dict[int, float],
    input_pattern: dict[int, float],
) -> set[Edge]:
    """Learn output-pattern -> input-pattern continuity as numeric pathways.

    This deliberately uses the same routing and reinforcement machinery as
    ordinary experience. The only difference is that both endpoint patterns
    are already validated by their respective external encoders.

    No label is stored in the brain. The label is used only by this experiment
    to choose which two externally generated numeric patterns co-occur.
    """
    sources = ranked_nodes(output_pattern)
    targets = ranked_nodes(input_pattern)
    pair_count = max(len(sources), len(targets))
    edges: set[Edge] = set()

    for index in range(pair_count):
        source = sources[index % len(sources)]
        target = targets[index % len(targets)]
        path = brain._shortest_path(source, target, temporarily_used=edges)
        edges.update(zip(path, path[1:]))

    # Apply weaker reinforcement than an ordinary transition.
    old_rate = brain.learning_rate
    brain.learning_rate = old_rate * BRIDGE_REINFORCE_SCALE
    try:
        brain._reinforce(edges)
    finally:
        brain.learning_rate = old_rate

    return edges


def train(
    brain: SurfaceFlowBrain,
    inputs: dict[str, dict[int, float]],
    outputs: dict[str, dict[int, float]],
    *,
    with_bridges: bool,
) -> TrainingResult:
    pairs = [
        pair
        for experience in all_experiences()
        for pair in transitions(experience)
    ]

    learned_paths: dict[Transition, set[Edge]] = defaultdict(set)
    bridge_paths: dict[str, set[Edge]] = defaultdict(set)
    total = EPOCHS * len(pairs)
    completed = 0
    started = time.perf_counter()

    print(f"Transition training: {EPOCHS} epochs / {len(pairs)} transitions per epoch")
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

    if with_bridges:
        words = continuity_words()
        print(
            f"Continuity training: {BRIDGE_EPOCHS} epochs / "
            f"{len(words)} bridges per epoch"
        )
        for epoch in range(1, BRIDGE_EPOCHS + 1):
            epoch_started = time.perf_counter()
            for word in words:
                bridge_paths[word].update(
                    learn_continuity_bridge(
                        brain,
                        outputs[word],
                        inputs[word],
                    )
                )
            print(
                f"  bridge epoch {epoch}/{BRIDGE_EPOCHS}: "
                f"{len(words)} continuity events in "
                f"{time.perf_counter() - epoch_started:.2f}s"
            )

    brain.reset_flow_bias()
    return TrainingResult(
        events=completed,
        elapsed_seconds=time.perf_counter() - started,
        learned_paths=dict(learned_paths),
        bridge_paths=dict(bridge_paths),
    )


def learned_lookahead_edges(
    brain: SurfaceFlowBrain,
    active_edges: set[Edge],
) -> set[Edge]:
    """Prepare a few learned outgoing edges from numerically reached nodes."""
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
) -> ReplayResult:
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
                brain.flow_bias.reinforce(
                    prepared_edges,
                    amount=LOOKAHEAD_GAIN,
                )
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

    return ReplayResult(
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


def bridge_strength_by_step(
    edge_history: list[set[Edge]],
    bridge_paths: dict[str, set[Edge]],
) -> dict[str, list[float]]:
    strengths: dict[str, list[float]] = {}
    for word, learned in bridge_paths.items():
        if not learned:
            strengths[word] = [0.0 for _ in edge_history]
            continue
        strengths[word] = [
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


def bridge_peaks(
    strengths: dict[str, list[float]],
) -> dict[str, tuple[int | None, float]]:
    peaks: dict[str, tuple[int | None, float]] = {}
    for word, values in strengths.items():
        if not values or max(values) <= 0.0:
            peaks[word] = (None, 0.0)
            continue
        index = int(np.argmax(values))
        peaks[word] = (index + 1, float(values[index]))
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
    step_count = len(next(iter(strengths.values()), []))

    for step in range(step_count):
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
    result: ReplayResult,
    learned_paths: dict[Transition, set[Edge]],
    bridge_paths: dict[str, set[Edge]],
    target_edges: set[Edge],
    distractor_edges: set[Edge],
) -> dict[str, float | bool]:
    replayed = set().union(*result.edge_history) if result.edge_history else set()
    target_recall, target_precision = edge_metrics(replayed, target_edges)
    distractor_recall, distractor_precision = edge_metrics(
        replayed,
        distractor_edges,
    )

    transition_strengths = transition_strength_by_step(
        result.edge_history,
        learned_paths,
    )
    hits = first_hit_steps(transition_strengths)
    peaks = peak_steps(transition_strengths)

    hit_order = ordered_steps([hits[pair] for pair in transitions(EXPERIENCE)])
    peak_order = ordered_steps(
        [peaks[pair][0] for pair in transitions(EXPERIENCE)]
    )
    dominant = dominant_target_flow(transition_strengths)

    target_bridge_words = list(EXPERIENCE[1:-1])
    target_bridge_edges = set().union(
        *(bridge_paths.get(word, set()) for word in target_bridge_words)
    )
    bridge_recall, bridge_precision = edge_metrics(
        replayed,
        target_bridge_edges,
    )
    bridge_strengths = bridge_strength_by_step(
        result.edge_history,
        {
            word: bridge_paths.get(word, set())
            for word in target_bridge_words
        },
    )
    target_bridge_peaks = bridge_peaks(bridge_strengths)

    heat = Counter(edge for step in result.edge_history for edge in step)
    target_hot = sum(count for edge, count in heat.items() if edge in target_edges)
    distractor_hot = sum(
        count for edge, count in heat.items() if edge in distractor_edges
    )
    bridge_hot = sum(
        count for edge, count in heat.items() if edge in target_bridge_edges
    )
    other_hot = sum(
        count
        for edge, count in heat.items()
        if edge not in target_edges
        and edge not in distractor_edges
        and edge not in target_bridge_edges
    )

    print(f"\n{name}")
    print(f"  steps                  : {len(result.edge_history)}")
    print(f"  max active nodes       : {max(result.active_counts)}")
    print(f"  final active nodes     : {result.active_counts[-1]}")
    print(f"  unique replay edges    : {len(replayed)}")
    print(f"  target recall          : {target_recall:.3f}")
    print(f"  target precision       : {target_precision:.3f}")
    print(f"  distractor recall      : {distractor_recall:.3f}")
    print(f"  distractor precision   : {distractor_precision:.3f}")
    print(f"  selectivity gap        : {target_precision - distractor_precision:+.3f}")
    print(f"  target bridge recall   : {bridge_recall:.3f}")
    print(f"  target bridge precision: {bridge_precision:.3f}")
    print(f"  ordered first hits     : {hit_order}")
    print(f"  ordered peak flow      : {peak_order}")

    for pair in transitions(EXPERIENCE):
        first = "none" if hits[pair] is None else str(hits[pair])
        peak_step, peak_value = peaks[pair]
        peak_shown = "none" if peak_step is None else str(peak_step)
        print(
            f"    {pair[0]} -> {pair[1]} "
            f"first_hit={first}, peak_step={peak_shown}, peak={peak_value:.3f}"
        )

    for word in target_bridge_words:
        peak_step, peak_value = target_bridge_peaks[word]
        peak_shown = "none" if peak_step is None else str(peak_step)
        print(
            f"    bridge output({word}) -> input({word}) "
            f"peak_step={peak_shown}, peak={peak_value:.3f}"
        )

    print(f"  dominant target flow   : {compress_sequence(dominant)}")
    print(f"  repeated target flow   : {target_hot}")
    print(f"  repeated target bridges: {bridge_hot}")
    print(f"  repeated distractor    : {distractor_hot}")
    print(f"  repeated other flow    : {other_hot}")
    print(
        "  final bias max/energy  : "
        f"{result.bias_max_history[-1] if result.bias_max_history else 1.0:.3f} / "
        f"{result.bias_energy_history[-1] if result.bias_energy_history else 0.0:.3f}"
    )
    print(f"  stop                   : {result.stop_reason}")
    print(f"  replay time            : {result.elapsed_seconds:.3f}s")

    return {
        "target_precision": target_precision,
        "selectivity_gap": target_precision - distractor_precision,
        "bridge_recall": bridge_recall,
        "ordered_first_hits": hit_order,
        "ordered_peaks": peak_order,
    }


def run_condition(
    name: str,
    *,
    with_bridges: bool,
) -> tuple[dict[str, float | bool], TrainingResult, float]:
    condition_started = time.perf_counter()
    brain, inputs, outputs = build_brain()
    training = train(
        brain,
        inputs,
        outputs,
        with_bridges=with_bridges,
    )

    target_edges = set().union(
        *(training.learned_paths[pair] for pair in transitions(EXPERIENCE))
    )
    distractor_pairs = [
        pair
        for experience in DISTRACTOR_EXPERIENCES
        for pair in transitions(experience)
    ]
    distractor_edges = set().union(
        *(training.learned_paths[pair] for pair in distractor_pairs)
    )

    replay_result = replay(
        brain,
        inputs[EXPERIENCE[0]],
        use_flow_bias=True,
    )
    summary = print_trial(
        name,
        replay_result,
        training.learned_paths,
        training.bridge_paths,
        target_edges,
        distractor_edges,
    )
    return summary, training, time.perf_counter() - condition_started


def main() -> None:
    total_started = time.perf_counter()

    print("SphereBrain v27.1 — Internal Continuity Bridge Replay")
    print("Only the first cue is injected.")
    print("No decoded word is fed back during replay.")
    print("Long-term transition paths and continuity bridges are numeric pathways.\n")

    print("Target experience:")
    print("  " + " -> ".join(EXPERIENCE))
    print("Distractor experiences:")
    for experience in DISTRACTOR_EXPERIENCES:
        print("  " + " -> ".join(experience))

    print("\nContinuity hypothesis:")
    print("  output(青い) -> input(青い)")
    print("  output(昼)   -> input(昼)")
    print("  The same principle is also trained for distractor continuity.\n")

    print("Experiment settings")
    print(f"  competition top_k      : {TOP_K}")
    print(f"  weight power           : {WEIGHT_POWER}")
    print(f"  flow gain/decay        : {FLOW_BIAS_GAIN} / {FLOW_BIAS_DECAY}")
    print(f"  lookahead gain         : {LOOKAHEAD_GAIN}")
    print(f"  bridge epochs          : {BRIDGE_EPOCHS}")
    print(f"  bridge reinforce scale : {BRIDGE_REINFORCE_SCALE}")

    print("\n" + "=" * 78)
    print("Condition A: v27 control — no continuity bridges")
    print("=" * 78)
    control, control_training, control_seconds = run_condition(
        "v27 control — flow bias and lookahead, no continuity bridges",
        with_bridges=False,
    )

    print("\n" + "=" * 78)
    print("Condition B: v27.1 — learned internal continuity bridges")
    print("=" * 78)
    bridged, bridged_training, bridged_seconds = run_condition(
        "v27.1 — flow bias, lookahead, and continuity bridges",
        with_bridges=True,
    )

    print("\nComparison")
    print("-" * 78)
    print(
        "target precision change : "
        f"{float(bridged['target_precision']) - float(control['target_precision']):+.3f}"
    )
    print(
        "selectivity gap change  : "
        f"{float(bridged['selectivity_gap']) - float(control['selectivity_gap']):+.3f}"
    )
    print(
        "bridge recall           : "
        f"control={float(control['bridge_recall']):.3f} / "
        f"v27.1={float(bridged['bridge_recall']):.3f}"
    )
    print(
        "ordered first hits      : "
        f"control={control['ordered_first_hits']} / "
        f"v27.1={bridged['ordered_first_hits']}"
    )
    print(
        "ordered peaks           : "
        f"control={control['ordered_peaks']} / "
        f"v27.1={bridged['ordered_peaks']}"
    )

    print("\nSummary")
    print("-" * 78)
    print(f"control transition events : {control_training.events}")
    print(f"v27.1 transition events   : {bridged_training.events}")
    print(
        "v27.1 learned bridges    : "
        f"{sum(len(edges) for edges in bridged_training.bridge_paths.values())}"
    )
    print(f"control condition time    : {control_seconds:.2f}s")
    print(f"v27.1 condition time      : {bridged_seconds:.2f}s")
    print(f"total experiment          : {time.perf_counter() - total_started:.2f}s")

    print("\nHow to read v27.1")
    print("1. 青い->昼 must become non-zero before claiming continuity improved.")
    print("2. Continuity bridges should activate between successive target transitions.")
    print("3. Ordered peaks are stronger evidence than touching all target paths.")
    print("4. Target precision should not improve only by increasing distractor flow.")
    print("5. If bridges activate but order remains false, competition or timing is next.")
    print("6. The observer uses labels only after replay to measure numeric pathway flow.")
    print("7. This file uses private routing methods experimentally; core promotion comes later.")


if __name__ == "__main__":
    main()
