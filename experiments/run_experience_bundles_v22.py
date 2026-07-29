from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain

STEPS = 24
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


def learn_bundle(
    brain: SurfaceFlowBrain,
    experience: Experience,
    input_patterns: dict[str, dict[int, float]],
    output_patterns: dict[str, dict[int, float]],
) -> int:
    events = 0
    for source in experience.elements:
        for target in experience.elements:
            if source == target:
                continue
            brain.experience(input_patterns[source], output_patterns[target])
            events += 1
    return events


def learn_chain_control(
    brain: SurfaceFlowBrain,
    experience: Experience,
    input_patterns: dict[str, dict[int, float]],
    output_patterns: dict[str, dict[int, float]],
) -> int:
    events = 0
    for source, target in zip(experience.elements, experience.elements[1:]):
        brain.experience(input_patterns[source], output_patterns[target])
        events += 1
    return events


def train(
    mode: str,
    epochs: int = 30,
) -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    int,
]:
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

    rng = np.random.default_rng(2202)
    total_events = 0
    for _ in range(epochs):
        for index in rng.permutation(len(EXPERIENCES)):
            experience = EXPERIENCES[int(index)]
            if mode == "bundle":
                total_events += learn_bundle(
                    brain, experience, input_patterns, output_patterns
                )
            elif mode == "chain":
                total_events += learn_chain_control(
                    brain, experience, input_patterns, output_patterns
                )
            else:
                raise ValueError(f"Unsupported training mode: {mode}")
    return brain, input_patterns, output_patterns, total_events


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
        left_final = traces[left].history[-1]
        right_final = traces[right].history[-1]
        similarities.append(cosine(left_final, right_final))
    return float(np.mean(similarities))


def main() -> None:
    print("SphereBrain v22a — Independent Experience Bundles")
    print("One experience contains nouns, qualities, actions, sensations, and context together.")
    print("The experiment is self-contained and does not import earlier experiment scripts.\n")

    results = {}
    for mode in ("chain", "bundle"):
        brain, inputs, outputs, events = train(mode)
        traces = trace_all(brain, inputs)
        results[mode] = (brain, outputs, traces, events)

    print(f"{'mode':<10} {'events':>8} {'shared context':>16} {'mean active':>13}")
    print("-" * 54)
    for mode in ("chain", "bundle"):
        _brain, _outputs, traces, events = results[mode]
        shared = shared_context_score(traces)
        active = float(
            np.mean([np.count_nonzero(trace.history[-1]) for trace in traces.values()])
        )
        print(f"{mode:<10} {events:>8} {shared:>16.3f} {active:>13.1f}")

    print("\nHuman-observer decoding after propagation")
    print("-" * 110)
    for mode in ("chain", "bundle"):
        brain, outputs, traces, _events = results[mode]
        print(f"\n{mode}")
        for word in QUERY:
            matches = top_matches(brain, traces[word], outputs)
            shown = ", ".join(
                f"{candidate}:{score:.3f}" for candidate, score in matches
            )
            print(f"  {word:<8} -> {shown}")

    print("\nExperience definitions")
    print("-" * 110)
    for experience in EXPERIENCES:
        print(f"  {experience.name:<16} " + " / ".join(experience.elements))

    print("\nGuardrails")
    print("1. Bundle order has no meaning: every element is paired with every other element.")
    print("2. SphereBrain is not told noun, verb, adjective, truth, or expected-answer labels.")
    print("3. Decoded words are observer labels used only after internal propagation.")
    print("4. This tests concept-like co-occurrence structure, not real sensory grounding yet.")


if __name__ == "__main__":
    main()
