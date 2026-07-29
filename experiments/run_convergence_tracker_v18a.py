from __future__ import annotations

import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain, SurfaceFlowResult


EXPERIENCES: tuple[tuple[str, ...], ...] = (
    ("空", "青い"),
    ("青い", "空", "見える"),
    ("今日", "楽しい"),
    ("雨", "降る"),
    ("今日", "雨"),
    ("私", "うれしい"),
)
PROBES: tuple[str, ...] = ("空", "青い", "今日", "雨", "私")
MAX_STEPS = 24
CONVERGENCE_THRESHOLD = 0.90


def adjacent_pairs(experiences: tuple[tuple[str, ...], ...]) -> list[tuple[str, str]]:
    return [
        (words[index], words[index + 1])
        for words in experiences
        for index in range(len(words) - 1)
    ]


def binary_cosine(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(np.sqrt(len(left) * len(right)))


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def padded_history(result: SurfaceFlowResult, length: int = MAX_STEPS + 1) -> list[set[int]]:
    history = [set(step) for step in result.activation_history]
    if not history:
        history = [set()]
    while len(history) < length:
        history.append(set())
    return history[:length]


def first_convergence(left: list[set[int]], right: list[set[int]]) -> tuple[int | None, float]:
    for step, (a, b) in enumerate(zip(left, right)):
        similarity = binary_cosine(a, b)
        if similarity >= CONVERGENCE_THRESHOLD and a and b:
            return step, similarity
    final_similarity = binary_cosine(left[-1], right[-1])
    return None, final_similarity


def train_brain() -> tuple[SurfaceFlowBrain, TextSurfaceEncoder]:
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
    rng = np.random.default_rng(1802)
    for _ in range(75):
        for pair_index in rng.permutation(len(pairs)):
            source, target = pairs[int(pair_index)]
            brain.experience(input_patterns[source], output_patterns[target])
    return brain, input_encoder


def probe_all(
    brain: SurfaceFlowBrain,
    encoder: TextSurfaceEncoder,
) -> tuple[dict[str, SurfaceFlowResult], dict[str, list[set[int]]]]:
    results: dict[str, SurfaceFlowResult] = {}
    histories: dict[str, list[set[int]]] = {}
    for word in PROBES:
        result = brain.propagate(
            encoder.encode(word),
            steps=MAX_STEPS,
            threshold=0.08,
            noise=0.0,
        )
        results[word] = result
        histories[word] = padded_history(result)
    return results, histories


def print_spread(histories: dict[str, list[set[int]]]) -> None:
    print("\nActivation spread by step")
    print("-" * 92)
    print("step  " + "  ".join(f"{word:>6}" for word in PROBES) + "   union  common")
    for step in range(MAX_STEPS + 1):
        sets = [histories[word][step] for word in PROBES]
        union = set().union(*sets)
        common = set.intersection(*sets) if all(sets) else set()
        counts = "  ".join(f"{len(nodes):6d}" for nodes in sets)
        print(f"{step:>4}  {counts}  {len(union):6d}  {len(common):6d}")


def print_pairwise(histories: dict[str, list[set[int]]]) -> None:
    print("\nPairwise convergence")
    print("-" * 92)
    for left_word, right_word in combinations(PROBES, 2):
        left = histories[left_word]
        right = histories[right_word]
        step, similarity = first_convergence(left, right)
        peak_step, peak_similarity = max(
            enumerate(binary_cosine(a, b) for a, b in zip(left, right)),
            key=lambda item: item[1],
        )
        convergence = "none" if step is None else str(step)
        print(
            f"{left_word:>3} vs {right_word:<3} "
            f"first>={CONVERGENCE_THRESHOLD:.2f}: {convergence:>4}  "
            f"peak={peak_similarity:.3f}@{peak_step:<2}  "
            f"final={similarity:.3f}  "
            f"final-jaccard={jaccard(left[-1], right[-1]):.3f}"
        )


def print_step_matrix(histories: dict[str, list[set[int]]]) -> None:
    print("\nMean pairwise similarity by step")
    print("-" * 92)
    previous = None
    largest_jump = (-1.0, -1)
    for step in range(MAX_STEPS + 1):
        values = [
            binary_cosine(histories[a][step], histories[b][step])
            for a, b in combinations(PROBES, 2)
        ]
        mean = float(np.mean(values))
        minimum = min(values)
        maximum = max(values)
        jump = 0.0 if previous is None else mean - previous
        if jump > largest_jump[0]:
            largest_jump = (jump, step)
        print(
            f"step={step:02d} mean={mean:.3f} min={minimum:.3f} "
            f"max={maximum:.3f} jump={jump:+.3f}"
        )
        previous = mean
    print(
        f"\nLargest mean-similarity jump: step {largest_jump[1]} "
        f"({largest_jump[0]:+.3f})"
    )


def print_common_nodes(
    brain: SurfaceFlowBrain,
    results: dict[str, SurfaceFlowResult],
    histories: dict[str, list[set[int]]],
) -> None:
    first_seen: dict[int, dict[str, int]] = defaultdict(dict)
    for word in PROBES:
        for step, nodes in enumerate(histories[word]):
            for node in nodes:
                first_seen[node].setdefault(word, step)

    shared = [
        node for node, appearances in first_seen.items()
        if len(appearances) == len(PROBES)
    ]
    shared.sort(
        key=lambda node: (
            max(first_seen[node].values()),
            -int(brain.node_usage[node]),
            node,
        )
    )

    traversed_by_word: dict[str, Counter[int]] = {}
    for word, result in results.items():
        counter: Counter[int] = Counter()
        for source, target in result.traversed_edges:
            counter[source] += 1
            counter[target] += 1
        traversed_by_word[word] = counter

    print("\nEarliest nodes reached by every probe")
    print("-" * 92)
    if not shared:
        print("No node was activated by every probe within the observed steps.")
        return

    for node in shared[:15]:
        steps = ", ".join(f"{word}:{first_seen[node][word]}" for word in PROBES)
        route_hits = sum(traversed_by_word[word][node] for word in PROBES)
        enabled_degree = int(np.count_nonzero(brain.edge_enabled[node]))
        print(
            f"node={node:<4} latest-first-step={max(first_seen[node].values()):<2} "
            f"usage={int(brain.node_usage[node]):<5} degree={enabled_degree:<3} "
            f"probe-route-hits={route_hits:<4} [{steps}]"
        )


def print_route_hubs(brain: SurfaceFlowBrain, results: dict[str, SurfaceFlowResult]) -> None:
    word_nodes: dict[str, set[int]] = {}
    total_hits: Counter[int] = Counter()
    for word, result in results.items():
        nodes: set[int] = set()
        for source, target in result.traversed_edges:
            nodes.add(source)
            nodes.add(target)
            total_hits[source] += 1
            total_hits[target] += 1
        word_nodes[word] = nodes

    print("\nShared route hubs during probing")
    print("-" * 92)
    candidates = set().union(*word_nodes.values())
    ranked = sorted(
        candidates,
        key=lambda node: (
            sum(node in word_nodes[word] for word in PROBES),
            total_hits[node],
            int(brain.node_usage[node]),
        ),
        reverse=True,
    )
    for node in ranked[:15]:
        words = [word for word in PROBES if node in word_nodes[word]]
        print(
            f"node={node:<4} probes={len(words)}/{len(PROBES)} "
            f"route-hits={total_hits[node]:<4} usage={int(brain.node_usage[node]):<5} "
            f"weight-out-max={float(np.max(brain.weights[node])):.3f} "
            f"words={','.join(words)}"
        )


def main() -> None:
    print("SphereBrain v18a — Convergence Tracker")
    print("Different numeric stimuli are followed step by step inside the brain.")
    print("Question: when and where do distinct activations become the same?\n")

    brain, encoder = train_brain()
    results, histories = probe_all(brain, encoder)

    print("Probe input patterns (external labels shown only for observation)")
    print("-" * 92)
    for word in PROBES:
        print(f"{word:<6} numeric-input-nodes={sorted(encoder.encode(word))}")

    print_spread(histories)
    print_pairwise(histories)
    print_step_matrix(histories)
    print_common_nodes(brain, results, histories)
    print_route_hubs(brain, results)

    print("\nInterpretation guardrails")
    print("-" * 92)
    print("1. High similarity means overlapping active nodes, not shared human meaning.")
    print("2. Rapid node-count growth suggests diffusion; shared hubs suggest route capture.")
    print("3. The earliest large similarity jump is a candidate convergence stage, not proof of one cause.")
    print("4. Node IDs are internal numeric observation points; words remain external labels.")


if __name__ == "__main__":
    main()
