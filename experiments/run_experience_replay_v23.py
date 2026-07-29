from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain


EPOCHS = 5
PROPAGATION_STEPS = 24
MAX_REPLAY_STEPS = 8
MIN_SCORE = 0.16
MIN_MARGIN = 0.01

EXPERIENCE = ("空", "青い", "昼", "風")
DISTRACTOR_EXPERIENCES = (
    ("夜", "暗い", "星", "静か"),
    ("海", "広い", "波", "潮風"),
)


@dataclass
class ReplayStep:
    index: int
    cue: str
    recalled: str | None
    score: float
    margin: float
    active_output_nodes: int
    elapsed_seconds: float
    reason: str = ""


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def output_peak(brain: SurfaceFlowBrain, output_history: list[dict[int, float]]) -> np.ndarray:
    vector = np.zeros(brain.node_count, dtype=float)
    for visible in output_history:
        for node, value in visible.items():
            vector[int(node)] = max(vector[int(node)], float(value))
    return vector


def pattern_vector(pattern: dict[int, float], node_count: int) -> np.ndarray:
    vector = np.zeros(node_count, dtype=float)
    for node, value in pattern.items():
        vector[int(node)] = float(value)
    return vector


def decode_ranked(
    vector: np.ndarray,
    output_vectors: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    ranked = [(word, cosine(vector, target)) for word, target in output_vectors.items()]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def build_brain() -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[str, np.ndarray],
]:
    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=2301,
        learning_rate=0.035,
        decay_rate=0.0002,
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
    input_patterns = {word: input_encoder.encode(word) for word in words}
    output_patterns = {word: output_encoder.encode(word) for word in words}
    output_vectors = {
        word: pattern_vector(pattern, brain.node_count)
        for word, pattern in output_patterns.items()
    }
    return brain, input_patterns, output_patterns, output_vectors


def transitions(experience: tuple[str, ...]) -> list[tuple[str, str]]:
    return list(zip(experience, experience[1:]))


def train(
    brain: SurfaceFlowBrain,
    input_patterns: dict[str, dict[int, float]],
    output_patterns: dict[str, dict[int, float]],
) -> tuple[int, float]:
    all_experiences = (EXPERIENCE, *DISTRACTOR_EXPERIENCES)
    pairs = [pair for experience in all_experiences for pair in transitions(experience)]
    total = EPOCHS * len(pairs)
    completed = 0
    started = time.perf_counter()

    print(f"Training: {EPOCHS} epochs / {len(pairs)} transitions per epoch")
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        for source, target in pairs:
            brain.experience(input_patterns[source], output_patterns[target])
            completed += 1
        epoch_elapsed = time.perf_counter() - epoch_started
        print(
            f"  epoch {epoch}/{EPOCHS}: {completed}/{total} transitions "
            f"({100.0 * completed / total:5.1f}%) in {epoch_elapsed:.2f}s"
        )

    elapsed = time.perf_counter() - started
    return completed, elapsed


def recall_once(
    brain: SurfaceFlowBrain,
    cue: str,
    input_patterns: dict[str, dict[int, float]],
    output_vectors: dict[str, np.ndarray],
) -> tuple[str | None, float, float, int, float, list[tuple[str, float]]]:
    started = time.perf_counter()
    result = brain.propagate(
        input_patterns[cue],
        steps=PROPAGATION_STEPS,
        threshold=0.08,
        noise=0.0,
    )
    peak = output_peak(brain, result.output_history)
    ranked = decode_ranked(peak, output_vectors)
    elapsed = time.perf_counter() - started

    if not ranked:
        return None, 0.0, 0.0, 0, elapsed, []

    best_word, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score
    active_output_nodes = int(np.count_nonzero(peak))
    return best_word, best_score, margin, active_output_nodes, elapsed, ranked[:5]


def replay(
    brain: SurfaceFlowBrain,
    start_cue: str,
    input_patterns: dict[str, dict[int, float]],
    output_vectors: dict[str, np.ndarray],
) -> list[ReplayStep]:
    steps: list[ReplayStep] = []
    seen = {start_cue}
    cue = start_cue

    print("\nExperience Replay")
    print(f"  step 0: {start_cue}  (external cue only)")

    for index in range(1, MAX_REPLAY_STEPS + 1):
        recalled, score, margin, active, elapsed, ranked = recall_once(
            brain, cue, input_patterns, output_vectors
        )
        candidates = ", ".join(f"{word}:{value:.3f}" for word, value in ranked)

        reason = ""
        accepted = recalled
        if recalled is None or active == 0:
            accepted = None
            reason = "no internal output activity"
        elif score < MIN_SCORE:
            accepted = None
            reason = f"score below {MIN_SCORE:.2f}"
        elif margin < MIN_MARGIN:
            accepted = None
            reason = f"top candidates too close (margin < {MIN_MARGIN:.2f})"
        elif recalled in seen:
            accepted = None
            reason = "recalled state already appeared"

        step = ReplayStep(
            index=index,
            cue=cue,
            recalled=accepted,
            score=score,
            margin=margin,
            active_output_nodes=active,
            elapsed_seconds=elapsed,
            reason=reason,
        )
        steps.append(step)

        shown = accepted if accepted is not None else "STOP"
        print(
            f"  step {index}: {cue} -> {shown} "
            f"score={score:.3f} margin={margin:.3f} "
            f"active={active} time={elapsed:.2f}s"
        )
        print(f"           observer candidates: {candidates}")

        if accepted is None:
            print(f"           stop reason: {reason}")
            break

        seen.add(accepted)
        cue = accepted
    else:
        print(f"  stop reason: reached maximum replay length {MAX_REPLAY_STEPS}")

    return steps


def main() -> None:
    total_started = time.perf_counter()
    print("SphereBrain v23a — Experience Replay")
    print("Learn an experience, later provide only its first cue, then close the recall loop.")
    print("The core receives numeric patterns; words below are observer labels.")
    print("Important: v23a uses an external observer bridge to feed each recalled state back")
    print("to the input surface. Pure uninterrupted internal replay is the next core change.\n")

    print("Target experience:")
    print("  " + " -> ".join(EXPERIENCE))
    print("Distractor experiences:")
    for experience in DISTRACTOR_EXPERIENCES:
        print("  " + " -> ".join(experience))
    print()

    brain, inputs, outputs, output_vectors = build_brain()
    events, training_seconds = train(brain, inputs, outputs)
    replay_started = time.perf_counter()
    steps = replay(brain, EXPERIENCE[0], inputs, output_vectors)
    replay_seconds = time.perf_counter() - replay_started

    recalled_sequence = [EXPERIENCE[0]] + [
        step.recalled for step in steps if step.recalled is not None
    ]
    expected_prefix = list(EXPERIENCE[: len(recalled_sequence)])
    prefix_matches = recalled_sequence == expected_prefix

    print("\nSummary")
    print("-" * 72)
    print(f"learned transitions : {events}")
    print(f"training time       : {training_seconds:.2f}s")
    print(f"replay time         : {replay_seconds:.2f}s")
    print(f"recalled sequence   : {' -> '.join(recalled_sequence)}")
    print(f"expected prefix     : {' -> '.join(expected_prefix)}")
    print(f"prefix match        : {prefix_matches}")
    print(f"total experiment    : {time.perf_counter() - total_started:.2f}s")

    print("\nInterpretation guardrails")
    print("1. No expected answer is supplied during replay.")
    print("2. Observer decoding converts numeric output activity into readable labels.")
    print("3. v23a feeds the decoded state back as the next numeric cue outside the core.")
    print("4. Therefore this is a closed-loop replay prototype, not yet autonomous internal replay.")


if __name__ == "__main__":
    main()
