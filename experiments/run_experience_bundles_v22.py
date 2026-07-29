from __future__ import annotations

import sys
from dataclasses import dataclass
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
    PROBES,
    TraceResult,
    adaptive_trace,
    cosine,
    decode,
    peak_output,
)


@dataclass(frozen=True)
class Experience:
    name: str
    elements: tuple[str, ...]


EXPERIENCES: tuple[Experience, ...] = (
    Experience("clear_sky_1", ("空", "青い", "広がっている", "澄んでいる", "昼", "太陽", "明るい", "風", "鳥", "見上げる")),
    Experience("clear_sky_2", ("空", "青い", "広い", "澄んでいる", "晴れ", "昼", "暖かい", "雲", "風")),
    Experience("cloudy_sky", ("空", "灰色", "曇り", "広がっている", "雲", "風", "雨", "寒い")),
    Experience("night_sky", ("空", "暗い", "夜", "星", "月", "広がっている", "静か", "見上げる")),
    Experience("sunset_sky", ("空", "赤い", "橙色", "夕方", "夕焼け", "雲", "広がっている", "見上げる")),
    Experience("blue_sea", ("海", "青い", "広い", "澄んでいる", "波", "魚", "潮風", "光る")),
    Experience("open_field", ("草原", "緑", "広い", "広がっている", "風", "鳥", "暖かい", "歩く")),
    Experience("clear_water", ("水", "透明", "澄んでいる", "冷たい", "流れる", "光る")),
)

# Words used only by the human observer after propagation.
OBSERVE = ("青い", "広がっている", "澄んでいる", "昼", "太陽", "雲", "風", "鳥", "暗い", "夜", "海", "広い")
QUERY = ("空", "青い", "澄んでいる", "広がっている", "海")


def vocabulary() -> list[str]:
    return sorted({element for experience in EXPERIENCES for element in experience.elements})


def learn_bundle(
    brain: SurfaceFlowBrain,
    experience: Experience,
    input_patterns: dict[str, dict[int, float]],
    output_patterns: dict[str, dict[int, float]],
) -> int:
    """Treat one experience as an unordered co-occurrence event.

    Every element can evoke every other element. The brain receives only numeric
    patterns; names and parts of speech remain outside the core.
    """
    events = 0
    elements = experience.elements
    for source in elements:
        for target in elements:
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
    """Control: learn only the written order of each experience."""
    events = 0
    for source, target in zip(experience.elements, experience.elements[1:]):
        brain.experience(input_patterns[source], output_patterns[target])
        events += 1
    return events


def train(mode: str, epochs: int = 30) -> tuple[SurfaceFlowBrain, dict[str, dict[int, float]], dict[str, dict[int, float]], int]:
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
                total_events += learn_bundle(brain, experience, input_patterns, output_patterns)
            elif mode == "chain":
                total_events += learn_chain_control(brain, experience, input_patterns, output_patterns)
            else:
                raise ValueError(mode)
    return brain, input_patterns, output_patterns, total_events


def trace_all(
    brain: SurfaceFlowBrain,
    input_patterns: dict[str, dict[int, float]],
) -> dict[str, TraceResult]:
    return {
        word: adaptive_trace(brain, input_patterns[word], threshold=0.80)
        for word in QUERY
    }


def top_matches(
    brain: SurfaceFlowBrain,
    trace: TraceResult,
    output_patterns: dict[str, dict[int, float]],
    limit: int = 8,
) -> list[tuple[str, float]]:
    return decode(peak_output(brain, trace), output_patterns, brain.node_count, limit=limit)


def shared_context_score(
    brain: SurfaceFlowBrain,
    traces: dict[str, TraceResult],
) -> float:
    pairs = (("空", "青い"), ("空", "澄んでいる"), ("空", "広がっている"), ("青い", "海"))
    return float(np.mean([cosine(traces[a].history[-1], traces[b].history[-1]) for a, b in pairs]))


def main() -> None:
    print("SphereBrain v22 — Experience Bundles")
    print("One experience contains nouns, qualities, actions, sensations, and context together.")
    print("The core receives only numeric patterns; grammar labels are not supplied.\n")

    results = {}
    for mode in ("chain", "bundle"):
        brain, inputs, outputs, events = train(mode)
        traces = trace_all(brain, inputs)
        results[mode] = (brain, outputs, traces, events)

    print(f"{'mode':<10} {'events':>8} {'shared context':>16} {'mean active':>13}")
    print("-" * 54)
    for mode in ("chain", "bundle"):
        brain, _outputs, traces, events = results[mode]
        shared = shared_context_score(brain, traces)
        active = float(np.mean([np.count_nonzero(trace.history[-1]) for trace in traces.values()]))
        print(f"{mode:<10} {events:>8} {shared:>16.3f} {active:>13.1f}")

    print("\nHuman-observer decoding after propagation")
    print("-" * 110)
    for mode in ("chain", "bundle"):
        brain, outputs, traces, _events = results[mode]
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
    print("2. SphereBrain is not told noun, verb, adjective, truth, or expected answer labels.")
    print("3. Decoded words are observer labels used only after internal propagation.")
    print("4. This tests concept-like co-occurrence structure, not real sensory grounding yet.")


if __name__ == "__main__":
    main()
