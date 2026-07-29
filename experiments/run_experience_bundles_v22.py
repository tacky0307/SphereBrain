from __future__ import annotations

import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain

STEPS = 24
EPOCHS = 5
ACTIVE_THRESHOLD = 1e-5
RESIDUAL_FRACTION = 0.02
MAX_BRANCHES = 8
RELATIVE_THRESHOLD = 0.80


@dataclass(frozen=True)
class Experience:
    name: str
    elements: tuple[str, ...]


@dataclass
class TraceResult:
    history: list[np.ndarray]
    mean_branches: list[float]
    branch_entropy: list[float]
    energy: list[float]
    selected_edges: list[int]


EXPERIENCES: tuple[Experience, ...] = (
    Experience(
        "clear_sky_1",
        ("空", "青い", "広がっている", "澄んでいる", "昼", "太陽", "明るい", "風", "鳥", "見上げる"),
    ),
    Experience(
        "clear_sky_2",
        ("空", "青い", "広い", "澄んでいる", "晴れ", "昼", "暖かい", "雲", "風"),
    ),
    Experience(
        "cloudy_sky",
        ("空", "灰色", "曇り", "広がっている", "雲", "風", "雨", "寒い"),
    ),
    Experience(
        "night_sky",
        ("空", "暗い", "夜", "星", "月", "広がっている", "静か", "見上げる"),
    ),
    Experience(
        "sunset_sky",
        ("空", "赤い", "橙色", "夕方", "夕焼け", "雲", "広がっている", "見上げる"),
    ),
    Experience(
        "blue_sea",
        ("海", "青い", "広い", "澄んでいる", "波", "魚", "潮風", "光る"),
    ),
    Experience(
        "open_field",
        ("草原", "緑", "広い", "広がっている", "風", "鳥", "暖かい", "歩く"),
    ),
    Experience(
        "clear_water",
        ("水", "透明", "澄んでいる", "冷たい", "流れる", "光る"),
    ),
)

QUERY = ("空", "青い", "澄んでいる", "広がっている", "海")


def vocabulary() -> list[str]:
    return sorted({element for experience in EXPERIENCES for element in experience.elements})


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


def selected_destinations(
    brain: SurfaceFlowBrain,
    source: int,
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

    best = float(np.max(weights))
    selected_mask = weights >= best * RELATIVE_THRESHOLD
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


def adaptive_trace(
    brain: SurfaceFlowBrain,
    pattern: dict[int, float],
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

        for source_raw in active_sources:
            source = int(source_raw)
            source_energy = float(effective[source])
            selected, selected_weights, branch_count = selected_destinations(brain, source)
            if selected.size == 0:
                continue

            step_branch_counts.append(branch_count)
            step_selected_edges += branch_count
            selected_total = float(np.sum(selected_weights))
            if selected_total <= 0.0:
                continue

            main_energy = source_energy * (1.0 - RESIDUAL_FRACTION)
            next_activation[selected] += main_energy * (selected_weights / selected_total)

            if RESIDUAL_FRACTION > 0.0:
                enabled = np.flatnonzero(brain.edge_enabled[source])
                residual = np.setdiff1d(enabled, selected, assume_unique=False)
                if residual.size > 0:
                    residual_weights = np.clip(
                        np.asarray(brain.weights[source, residual], dtype=float), 0.0, None
                    )
                    residual_total = float(np.sum(residual_weights))
                    if residual_total > 0.0:
                        next_activation[residual] += (
                            source_energy
                            * RESIDUAL_FRACTION
                            * (residual_weights / residual_total)
                        )

        next_activation[next_activation < ACTIVE_THRESHOLD] = 0.0
        current_total = float(np.sum(next_activation))
        if current_total > 0.0 and initial_energy > 0.0:
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

        if not np.any(activation > 0.0):
            break

    return TraceResult(history, mean_branches, branch_entropy, energy, selected_edges)


def peak_output(brain: SurfaceFlowBrain, trace: TraceResult) -> np.ndarray:
    peak = np.max(np.stack(trace.history, axis=0), axis=0)
    mask = np.zeros(brain.node_count, dtype=bool)
    mask[brain.output_nodes] = True
    return peak * mask


def decode(
    vector: np.ndarray,
    output_patterns: dict[str, dict[int, float]],
    node_count: int,
    limit: int = 8,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for word, pattern in output_patterns.items():
        target = np.zeros(node_count, dtype=float)
        for node, value in pattern.items():
            target[int(node)] = float(value)
        ranked.append((word, cosine(vector, target)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def pair_counts(mode: str) -> Counter[tuple[str, str]]:
    """Collect repeated word pairs before path search.

    Each unique source-target pair is searched only once per epoch. Its original
    repetition count is then preserved by reinforcing the selected route again.
    """
    pairs: Counter[tuple[str, str]] = Counter()
    for experience in EXPERIENCES:
        if mode == "bundle":
            pairs.update(
                (source, target)
                for source in experience.elements
                for target in experience.elements
                if source != target
            )
        elif mode == "chain":
            pairs.update(zip(experience.elements, experience.elements[1:]))
        else:
            raise ValueError(f"Unsupported training mode: {mode}")
    return pairs


def reinforce_pair(
    brain: SurfaceFlowBrain,
    source_pattern: dict[int, float],
    target_pattern: dict[int, float],
    repetitions: int,
) -> None:
    """Search routes once, then reuse those routes for repeated co-occurrence."""
    inputs = brain._validate_pattern(source_pattern, brain.input_nodes, "input_pattern")
    targets = brain._validate_pattern(target_pattern, brain.output_nodes, "target_pattern")
    input_rank = sorted(inputs, key=lambda node: inputs[node], reverse=True)
    target_rank = sorted(targets, key=lambda node: targets[node], reverse=True)
    pair_count = max(len(input_rank), len(target_rank))
    edges: set[tuple[int, int]] = set()

    for index in range(pair_count):
        source = input_rank[index % len(input_rank)]
        target = target_rank[index % len(target_rank)]
        path = brain._shortest_path(source, target, temporarily_used=edges)
        edges.update(zip(path, path[1:]))

    for _ in range(repetitions):
        brain._reinforce(edges)


def train(
    mode: str,
    epochs: int = EPOCHS,
) -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    int,
    int,
    float,
]:
    started = time.perf_counter()
    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=2201,
        learning_rate=0.025,
        decay_rate=0.0002,
    )
    words = vocabulary()
    input_encoder = TextSurfaceEncoder(brain.input_nodes, width=5)
    output_encoder = TextSurfaceEncoder(brain.output_nodes, width=5)
    input_patterns = {word: input_encoder.encode(word) for word in words}
    output_patterns = {word: output_encoder.encode(word) for word in words}

    counts = pair_counts(mode)
    unique_pairs = list(counts.items())
    raw_events_per_epoch = sum(counts.values())
    total_raw_events = raw_events_per_epoch * epochs
    total_searches = len(unique_pairs) * epochs
    rng = np.random.default_rng(2202)

    print(
        f"\n[{mode}] {epochs} epochs / "
        f"{len(unique_pairs)} unique pairs per epoch / "
        f"{raw_events_per_epoch} original events per epoch"
    )

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        order = rng.permutation(len(unique_pairs))
        for position, index_raw in enumerate(order, start=1):
            (source, target), repetitions = unique_pairs[int(index_raw)]
            reinforce_pair(
                brain,
                input_patterns[source],
                output_patterns[target],
                repetitions,
            )
            if position % 50 == 0 or position == len(unique_pairs):
                percent = position / len(unique_pairs) * 100.0
                print(
                    f"  epoch {epoch}/{epochs}: "
                    f"{position}/{len(unique_pairs)} pairs ({percent:5.1f}%)",
                    end="\r" if position < len(unique_pairs) else "\n",
                    flush=True,
                )
        epoch_elapsed = time.perf_counter() - epoch_started
        total_elapsed = time.perf_counter() - started
        print(
            f"  completed epoch {epoch}/{epochs} in {epoch_elapsed:.2f}s "
            f"(total {total_elapsed:.2f}s)"
        )

    elapsed = time.perf_counter() - started
    return (
        brain,
        input_patterns,
        output_patterns,
        total_raw_events,
        total_searches,
        elapsed,
    )


def trace_all(
    brain: SurfaceFlowBrain,
    input_patterns: dict[str, dict[int, float]],
) -> dict[str, TraceResult]:
    return {word: adaptive_trace(brain, input_patterns[word]) for word in QUERY}


def top_matches(
    brain: SurfaceFlowBrain,
    trace: TraceResult,
    output_patterns: dict[str, dict[int, float]],
    limit: int = 8,
) -> list[tuple[str, float]]:
    return decode(
        peak_output(brain, trace),
        output_patterns,
        brain.node_count,
        limit=limit,
    )


def shared_context_score(traces: dict[str, TraceResult]) -> float:
    pairs = (
        ("空", "青い"),
        ("空", "澄んでいる"),
        ("空", "広がっている"),
        ("青い", "海"),
    )
    similarities = []
    for left, right in pairs:
        similarities.append(cosine(traces[left].history[-1], traces[right].history[-1]))
    return float(np.mean(similarities))


def main() -> None:
    whole_started = time.perf_counter()
    print("SphereBrain v22b — Fast Experience Bundles")
    print("Words remain observer labels; the core receives numeric surface patterns only.")
    print(f"Quick observation setting: {EPOCHS} epochs instead of 30.")
    print("Repeated source-target pairs reuse one route search per epoch.\n")

    results = {}
    for mode in ("chain", "bundle"):
        brain, inputs, outputs, events, searches, elapsed = train(mode)
        print(f"  tracing {len(QUERY)} probes...", flush=True)
        trace_started = time.perf_counter()
        traces = trace_all(brain, inputs)
        trace_elapsed = time.perf_counter() - trace_started
        print(f"  trace completed in {trace_elapsed:.2f}s")
        results[mode] = (brain, outputs, traces, events, searches, elapsed + trace_elapsed)

    print(
        f"\n{'mode':<10} {'events':>8} {'searches':>10} "
        f"{'seconds':>10} {'shared context':>16} {'mean active':>13}"
    )
    print("-" * 76)
    for mode in ("chain", "bundle"):
        _brain, _outputs, traces, events, searches, elapsed = results[mode]
        shared = shared_context_score(traces)
        active = float(
            np.mean([np.count_nonzero(trace.history[-1]) for trace in traces.values()])
        )
        print(
            f"{mode:<10} {events:>8} {searches:>10} {elapsed:>10.2f} "
            f"{shared:>16.3f} {active:>13.1f}"
        )

    print("\nHuman-observer decoding after propagation")
    print("-" * 110)
    for mode in ("chain", "bundle"):
        brain, outputs, traces, _events, _searches, _elapsed = results[mode]
        print(f"\n{mode}")
        for word in QUERY:
            matches = top_matches(brain, traces[word], outputs)
            shown = ", ".join(f"{candidate}:{score:.3f}" for candidate, score in matches)
            print(f"  {word:<8} -> {shown}")

    print("\nExperience definitions")
    print("-" * 110)
    for experience in EXPERIENCES:
        print(f"  {experience.name:<16} " + " / ".join(experience.elements))

    print("\nGuardrails")
    print("1. Bundle order has no meaning: every element is paired with every other element.")
    print("2. SphereBrain is not told noun, verb, adjective, truth, or expected-answer labels.")
    print("3. Decoded words are observer labels used only after internal propagation.")
    print("4. Route reuse reduces computation; it does not claim biological equivalence.")
    print("5. This tests concept-like co-occurrence structure, not sensory grounding yet.")
    print(f"\nTotal experiment time: {time.perf_counter() - whole_started:.2f}s")


if __name__ == "__main__":
    main()
