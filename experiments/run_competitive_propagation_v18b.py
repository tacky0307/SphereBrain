from __future__ import annotations

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


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    top_k_nodes: int | None
    strongest_edges_per_source: int | None


MODES: tuple[Mode, ...] = (
    Mode("baseline", "A: current propagation", None, None),
    Mode("topk", "B: keep global top-40 activations", 40, None),
    Mode("routes", "C: strongest 2 outgoing edges per active node", None, 2),
    Mode("combined", "D: top-40 plus strongest 2 edges", 40, 2),
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


def competitive_trace(
    brain: SurfaceFlowBrain,
    input_pattern: dict[int, float],
    mode: Mode,
    steps: int = 24,
    threshold: float = 0.08,
) -> list[np.ndarray]:
    activation = np.zeros(brain.node_count, dtype=float)
    fatigue = np.zeros(brain.node_count, dtype=float)
    for node, value in input_pattern.items():
        activation[int(node)] = float(value)

    history = [activation.copy()]

    for _ in range(steps):
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        contributions = effective[:, None] * brain.weights * brain.transmission_gain
        contributions[~brain.edge_enabled] = 0.0

        if mode.strongest_edges_per_source is not None:
            keep = mode.strongest_edges_per_source
            for source in np.flatnonzero(effective > 0.0):
                row = contributions[source]
                positive = np.flatnonzero(row > 0.0)
                if positive.size <= keep:
                    continue
                strongest = positive[np.argpartition(row[positive], -keep)[-keep:]]
                mask = np.zeros(brain.node_count, dtype=bool)
                mask[strongest] = True
                row[~mask] = 0.0

        contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)
        next_activation = 1.0 - np.prod(1.0 - contributions, axis=0)
        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < threshold] = 0.0

        if mode.top_k_nodes is not None:
            active = np.flatnonzero(next_activation > 0.0)
            if active.size > mode.top_k_nodes:
                keep_nodes = active[
                    np.argpartition(next_activation[active], -mode.top_k_nodes)[-mode.top_k_nodes:]
                ]
                keep_mask = np.zeros(brain.node_count, dtype=bool)
                keep_mask[keep_nodes] = True
                next_activation[~keep_mask] = 0.0

        fatigue *= brain.fatigue_decay
        fatigue += next_activation * brain.fatigue_gain
        fatigue = np.clip(fatigue, 0.0, 0.95)
        activation = next_activation
        history.append(activation.copy())
        if not np.any(activation):
            break

    while len(history) < steps + 1:
        history.append(np.zeros(brain.node_count, dtype=float))
    return history


def peak_output(brain: SurfaceFlowBrain, trace: list[np.ndarray]) -> np.ndarray:
    peak = np.max(np.stack(trace, axis=0), axis=0)
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


def main() -> None:
    print("SphereBrain v18b — Competitive Propagation Experiment")
    print("The same trained brain is tested under four propagation rules.")
    print("Question: can competition preserve distinct internal activity without destroying recall?\n")

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

    print(f"Training fixed for every condition: epochs={epochs}, events={epochs * len(pairs)}")
    print("-" * 100)
    for mode in MODES:
        print(
            f"{mode.label:<48} top-k={str(mode.top_k_nodes):<5} "
            f"strongest-edges={mode.strongest_edges_per_source}"
        )

    mode_traces: dict[str, dict[str, list[np.ndarray]]] = {}
    for mode in MODES:
        mode_traces[mode.key] = {
            word: competitive_trace(brain, input_patterns[word], mode)
            for word in PROBES
        }

    print("\nActivation spread summary")
    print("-" * 100)
    print(f"{'mode':<12} {'mean@1':>8} {'mean@5':>8} {'mean@10':>9} {'mean@24':>9} {'max':>7}")
    for mode in MODES:
        traces = mode_traces[mode.key]
        counts = {
            step: [int(np.count_nonzero(traces[word][step])) for word in PROBES]
            for step in (1, 5, 10, 24)
        }
        maximum = max(
            int(np.count_nonzero(vector))
            for word in PROBES
            for vector in traces[word]
        )
        print(
            f"{mode.key:<12} "
            f"{np.mean(counts[1]):8.1f} {np.mean(counts[5]):8.1f} "
            f"{np.mean(counts[10]):9.1f} {np.mean(counts[24]):9.1f} {maximum:7d}"
        )

    print("\nConvergence summary")
    print("-" * 100)
    print(f"{'mode':<12} {'first mean>=.90':>17} {'mean@5':>9} {'mean@10':>10} {'final mean':>11}")
    for mode in MODES:
        traces = mode_traces[mode.key]
        means: list[float] = []
        for step in range(25):
            similarities = [
                cosine(traces[left][step], traces[right][step])
                for left, right in combinations(PROBES, 2)
            ]
            means.append(float(np.mean(similarities)))
        first = first_step_at_or_above(means, 0.90)
        first_text = "never" if first is None else str(first)
        print(
            f"{mode.key:<12} {first_text:>17} {means[5]:9.3f} "
            f"{means[10]:10.3f} {means[24]:11.3f}"
        )

    print("\nPer-probe decoded candidates")
    print("-" * 100)
    for mode in MODES:
        print(f"\n{mode.label}")
        for word in PROBES:
            peak = peak_output(brain, mode_traces[mode.key][word])
            candidates = decode(peak, output_patterns, brain.node_count)
            text = ", ".join(f"{candidate}:{score:.3f}" for candidate, score in candidates)
            print(f"  {word:<4} -> {text}")

    expected: dict[str, tuple[str, ...]] = {
        "空": ("青い", "見える"),
        "青い": ("空",),
        "今日": ("楽しい", "雨"),
        "雨": ("降る",),
        "私": ("うれしい",),
    }

    print("\nExpected-successor separation")
    print("-" * 100)
    print("Positive margin means experienced successors outrank unrelated words on average.")
    print(f"{'mode':<12} {'mean expected':>14} {'mean unrelated':>15} {'margin':>10}")
    for mode in MODES:
        expected_scores: list[float] = []
        unrelated_scores: list[float] = []
        for source in PROBES:
            peak = peak_output(brain, mode_traces[mode.key][source])
            scores = dict(decode(peak, output_patterns, brain.node_count, limit=len(vocabulary)))
            expected_set = set(expected[source])
            expected_scores.extend(scores[word] for word in expected_set)
            unrelated_scores.extend(
                score
                for word, score in scores.items()
                if word not in expected_set and word != source
            )
        expected_mean = float(np.mean(expected_scores))
        unrelated_mean = float(np.mean(unrelated_scores))
        print(
            f"{mode.key:<12} {expected_mean:14.3f} {unrelated_mean:15.3f} "
            f"{expected_mean - unrelated_mean:10.3f}"
        )

    print("\nInterpretation guardrails")
    print("-" * 100)
    print("1. Lower convergence is useful only if experienced successor signals remain measurable.")
    print("2. Top-k is global inhibition; strongest-edge routing is local pathway competition.")
    print("3. This experiment changes propagation during observation, not the learned pathway weights.")
    print("4. The best condition should preserve input differences and improve expected-successor margin.")


if __name__ == "__main__":
    main()
