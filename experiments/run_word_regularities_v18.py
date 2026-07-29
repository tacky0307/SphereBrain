from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concept_observer_v17 import UnknownStateObserver
from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain, SurfaceFlowResult


# Human-readable labels exist only in this experiment adapter.
# SphereBrain receives numeric node/activity patterns and never receives these strings.
EXPERIENCES: tuple[tuple[str, ...], ...] = (
    ("空", "青い"),
    ("青い", "空", "見える"),
    ("今日", "楽しい"),
    ("雨", "降る"),
    ("今日", "雨"),
    ("私", "うれしい"),
)

PROBES: tuple[str, ...] = ("空", "青い", "今日", "雨", "私")


def adjacent_pairs(experiences: tuple[tuple[str, ...], ...]) -> list[tuple[str, str]]:
    return [
        (words[index], words[index + 1])
        for words in experiences
        for index in range(len(words) - 1)
    ]


def output_peak(result: SurfaceFlowResult) -> dict[int, float]:
    peak: dict[int, float] = {}
    for step in result.output_history:
        for node, value in step.items():
            peak[node] = max(peak.get(node, 0.0), float(value))
    return peak


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    nodes = set(left) | set(right)
    if not nodes:
        return 0.0
    a = np.array([left.get(node, 0.0) for node in nodes], dtype=float)
    b = np.array([right.get(node, 0.0) for node in nodes], dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def decode_candidates(
    activation: dict[int, float],
    output_patterns: dict[str, dict[int, float]],
    limit: int = 3,
) -> list[tuple[str, float]]:
    ranked = sorted(
        ((word, cosine(activation, pattern)) for word, pattern in output_patterns.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:limit]


def probe(
    brain: SurfaceFlowBrain,
    input_encoder: TextSurfaceEncoder,
    output_patterns: dict[str, dict[int, float]],
    word: str,
) -> tuple[SurfaceFlowResult, dict[int, float], list[tuple[str, float]]]:
    result = brain.propagate(
        input_encoder.encode(word),
        steps=24,
        threshold=0.08,
        noise=0.0,
    )
    activation = output_peak(result)
    return result, activation, decode_candidates(activation, output_patterns)


def format_candidates(candidates: list[tuple[str, float]]) -> str:
    return ", ".join(f"{word}:{score:.3f}" for word, score in candidates)


def main() -> None:
    print("SphereBrain v18 — First Word Regularity Experiment")
    print("Words are labels outside the brain. The brain receives numbers only.")
    print("Question: can repeated numeric experiences form reproducible word-to-word routes?\n")

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
    pairs = adjacent_pairs(EXPERIENCES)

    print("External adapter map (shown for observation only)")
    print("-" * 88)
    for word in vocabulary:
        input_nodes = sorted(input_patterns[word])
        output_nodes = sorted(output_patterns[word])
        print(f"{word:<6} input={input_nodes} output={output_nodes}")

    print("\nBefore experience")
    print("-" * 88)
    before: dict[str, list[tuple[str, float]]] = {}
    for word in PROBES:
        _, _, candidates = probe(brain, input_encoder, output_patterns, word)
        before[word] = candidates
        print(f"{word:<6} -> {format_candidates(candidates)}")

    rng = np.random.default_rng(1802)
    pair_usage: defaultdict[tuple[str, str], int] = defaultdict(int)
    epochs = 75
    for _ in range(epochs):
        order = rng.permutation(len(pairs))
        for pair_index in order:
            source, target = pairs[int(pair_index)]
            brain.experience(input_patterns[source], output_patterns[target])
            pair_usage[(source, target)] += 1

    print(f"\nExperience completed: epochs={epochs} adjacent-events={epochs * len(pairs)}")
    print("Experienced transitions")
    print("-" * 88)
    for (source, target), count in sorted(pair_usage.items()):
        print(f"{source:<6} -> {target:<6} count={count}")

    observer = UnknownStateObserver(similarity_threshold=0.82)
    print("\nAfter experience")
    print("-" * 88)
    for word in PROBES:
        _, activation, candidates = probe(brain, input_encoder, output_patterns, word)
        observation = observer.observe(activation, context=[word])
        before_top = before[word][0][0] if before[word] else "-"
        after_top = candidates[0][0] if candidates else "-"
        status = "NEW" if observation.is_new else "RECURRED"
        print(
            f"{word:<6} -> {format_candidates(candidates):<38} "
            f"top {before_top}->{after_top} state={observation.state_id} {status}"
        )

    # Repeat without additional learning. Similar internal responses should recur.
    print("\nImmediate recurrence check (no additional experience)")
    print("-" * 88)
    for word in PROBES:
        _, activation, _ = probe(brain, input_encoder, output_patterns, word)
        observation = observer.observe(activation, context=[word, "再試行"])
        status = "NEW" if observation.is_new else "RECURRED"
        print(
            f"{word:<6} state={observation.state_id:<3} {status:<8} "
            f"similarity={observation.similarity:.3f} occurrences={observation.occurrences}"
        )

    output_path = Path("data/word_regularities_v18_unknown_states.json")
    observer.save(output_path)

    print("\nInterpretation guardrails")
    print("-" * 88)
    print("1. A learned target becoming stronger supports route formation, not language understanding.")
    print("2. '今日' has two experienced successors; preserving both is more informative than forcing one answer.")
    print("3. Unknown states are retained even when no human label fits them.")
    print(f"4. Observer memory saved to: {output_path}")


if __name__ == "__main__":
    main()
