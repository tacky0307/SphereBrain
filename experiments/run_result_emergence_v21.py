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
    ACTIVE_THRESHOLD, EXPERIENCES, PROBES, adjacent_pairs, cosine, decode,
    input_vector, selected_destinations,
)
from run_early_narrowing_boundary_v19c import EXPECTED

MAX_STEPS = 48


@dataclass(frozen=True)
class Mode:
    key: str
    similarity: float
    top5_share: float
    output_share: float
    stable_steps: int


@dataclass
class Trace:
    history: list[np.ndarray]
    result_step: int | None
    similarities: list[float]
    output_shares: list[float]
    top5_shares: list[float]


MODES = (
    Mode("emerge_loose", 0.970, 0.22, 0.08, 2),
    Mode("emerge_mid", 0.985, 0.28, 0.10, 2),
    Mode("emerge_strict", 0.995, 0.34, 0.12, 3),
)


def threshold(step: int) -> float:
    if step == 1:
        return 0.80
    if step <= 5:
        return 0.90
    return 0.95


def output_vector(brain: SurfaceFlowBrain, activation: np.ndarray) -> np.ndarray:
    result = np.zeros(brain.node_count, dtype=float)
    result[brain.output_nodes] = activation[brain.output_nodes]
    return result


def top_share(vector: np.ndarray, width: int = 5) -> float:
    values = vector[vector > ACTIVE_THRESHOLD]
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    width = min(width, values.size)
    return float(np.sum(np.partition(values, values.size - width)[-width:]) / total)


def step_once(brain, activation, fatigue, initial_energy, gate):
    effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
    next_activation = np.zeros(brain.node_count, dtype=float)
    for source in np.flatnonzero(effective > ACTIVE_THRESHOLD):
        selected, weights, _ = selected_destinations(brain, int(source), "adaptive", gate)
        total = float(np.sum(weights))
        if selected.size and total > 0.0:
            next_activation[selected] += float(effective[source]) * (weights / total)
    next_activation[next_activation < ACTIVE_THRESHOLD] = 0.0
    total = float(np.sum(next_activation))
    if total > 0.0:
        next_activation *= initial_energy / total
    fatigue *= brain.fatigue_decay
    fatigue += np.clip(next_activation, 0.0, 1.0) * brain.fatigue_gain
    return next_activation, np.clip(fatigue, 0.0, 0.95)


def run_trace(brain, pattern, mode: Mode | None, max_steps=MAX_STEPS) -> Trace:
    activation = input_vector(brain, pattern)
    initial_energy = float(np.sum(activation))
    fatigue = np.zeros(brain.node_count, dtype=float)
    history = [activation.copy()]
    similarities = [0.0]
    output_shares = [0.0]
    top5_shares = [0.0]
    previous_output = output_vector(brain, activation)
    stable = 0
    result_step = None

    limit = 24 if mode is None else max_steps
    for step in range(1, limit + 1):
        activation, fatigue = step_once(
            brain, activation, fatigue, initial_energy, threshold(step)
        )
        current_output = output_vector(brain, activation)
        similarity = cosine(previous_output, current_output)
        output_share = float(np.sum(current_output)) / initial_energy
        concentration = top_share(current_output)
        history.append(activation.copy())
        similarities.append(similarity)
        output_shares.append(output_share)
        top5_shares.append(concentration)

        if mode is not None:
            formed = (
                similarity >= mode.similarity
                and output_share >= mode.output_share
                and concentration >= mode.top5_share
            )
            stable = stable + 1 if formed else 0
            if stable >= mode.stable_steps:
                result_step = step
                break
        previous_output = current_output

    if mode is None:
        result_step = 24
    return Trace(history, result_step, similarities, output_shares, top5_shares)


def final_output(brain, trace):
    return output_vector(brain, trace.history[-1])


def identity(brain, traces):
    sims = [
        cosine(final_output(brain, traces[a]), final_output(brain, traces[b]))
        for a, b in combinations(PROBES, 2)
    ]
    return 1.0 - float(np.mean(sims))


def top1_count(brain, traces, output_patterns):
    hits = 0
    for source in PROBES:
        candidate = decode(final_output(brain, traces[source]), output_patterns,
                           brain.node_count, limit=1)[0][0]
        hits += int(candidate in EXPECTED[source])
    return hits


def margin(brain, traces, output_patterns, vocabulary):
    values = []
    for source in PROBES:
        scores = dict(decode(final_output(brain, traces[source]), output_patterns,
                             brain.node_count, limit=len(vocabulary)))
        expected = [scores[word] for word in EXPECTED[source]]
        other = [scores[word] for word in vocabulary if word not in EXPECTED[source]]
        values.append(float(np.mean(expected) - np.mean(other)))
    return float(np.mean(values))


def main():
    print("SphereBrain v21 — Result Emergence")
    print("The run ends because an internal result forms, not because time expires.")
    print("The detector reads only output stability, output energy, and concentration.\n")

    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=1801,
                             learning_rate=0.035, decay_rate=0.0004)
    input_encoder = TextSurfaceEncoder(brain.input_nodes, width=5)
    output_encoder = TextSurfaceEncoder(brain.output_nodes, width=5)
    vocabulary = sorted({word for experience in EXPERIENCES for word in experience})
    input_patterns = {word: input_encoder.encode(word) for word in vocabulary}
    output_patterns = {word: output_encoder.encode(word) for word in vocabulary}

    rng = np.random.default_rng(1802)
    pairs = adjacent_pairs()
    for _ in range(75):
        for pair_index in rng.permutation(len(pairs)):
            source, target = pairs[int(pair_index)]
            brain.experience(input_patterns[source], output_patterns[target])

    traces_by_mode = {
        "fixed24": {word: run_trace(brain, input_patterns[word], None) for word in PROBES}
    }
    for mode in MODES:
        traces_by_mode[mode.key] = {
            word: run_trace(brain, input_patterns[word], mode) for word in PROBES
        }

    print("mode           formed  mean step   min   max  identity   margin  top1")
    print("-" * 82)
    for key, traces in traces_by_mode.items():
        steps = [t.result_step for t in traces.values() if t.result_step is not None]
        mean_step = float(np.mean(steps)) if steps else float("nan")
        print(f"{key:<14} {len(steps):>2}/{len(PROBES):<2} {mean_step:10.1f} "
              f"{min(steps) if steps else 0:5d} {max(steps) if steps else 0:5d} "
              f"{identity(brain, traces):9.3f} {margin(brain, traces, output_patterns, vocabulary):8.3f} "
              f"{top1_count(brain, traces, output_patterns):5d}")

    print("\nPer-probe result formation")
    print("-" * 110)
    for key, traces in traces_by_mode.items():
        print(f"\n{key}")
        for word in PROBES:
            trace = traces[word]
            step_text = "not formed" if trace.result_step is None else f"step {trace.result_step}"
            candidates = decode(final_output(brain, trace), output_patterns,
                                brain.node_count, limit=3)
            text = ", ".join(f"{name}:{score:.3f}" for name, score in candidates)
            print(f"  {word:<4} {step_text:<11} sim={trace.similarities[-1]:.3f} "
                  f"output={trace.output_shares[-1]:.3f} top5={trace.top5_shares[-1]:.3f} "
                  f"final={text}")

    print("\nGuardrails")
    print("1. Decoder output is inspected only after the detector has already stopped the run.")
    print("2. Stable emptiness is not a result: output energy and concentration are also required.")
    print("3. No result by step 48 is reported as 'not formed'; step 48 is only a safety ceiling.")
    print("4. SurfaceFlowBrain itself remains unchanged in this observation-layer experiment.")


if __name__ == "__main__":
    main()
