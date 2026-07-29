from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain

EPOCHS = 5
MAX_INTERNAL_STEPS = 48
THRESHOLD = 0.055
PERSISTENCE_VALUES = (0.10, 0.20, 0.30)
CONVERGENCE_TOLERANCE = 0.0008
CONVERGENCE_PATIENCE = 5

EXPERIENCE = ("空", "青い", "昼", "風")
DISTRACTOR_EXPERIENCES = (
    ("夜", "暗い", "星", "静か"),
    ("海", "広い", "波", "潮風"),
)


@dataclass
class InternalReplayResult:
    persistence: float
    states: list[np.ndarray]
    differences: list[float]
    active_counts: list[int]
    stop_reason: str
    elapsed_seconds: float


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def pattern_vector(pattern: dict[int, float], node_count: int) -> np.ndarray:
    vector = np.zeros(node_count, dtype=float)
    for node, value in pattern.items():
        vector[int(node)] = float(value)
    return vector


def build_brain() -> tuple[
    SurfaceFlowBrain,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[str, np.ndarray],
]:
    brain = SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=2401,
        learning_rate=0.035,
        decay_rate=0.0002,
        fatigue_gain=0.24,
        fatigue_decay=0.78,
        transmission_gain=0.90,
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
    pairs = [
        pair
        for experience in (EXPERIENCE, *DISTRACTOR_EXPERIENCES)
        for pair in transitions(experience)
    ]
    total = EPOCHS * len(pairs)
    completed = 0
    started = time.perf_counter()

    print(f"Training: {EPOCHS} epochs / {len(pairs)} transitions per epoch")
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        for source, target in pairs:
            brain.experience(input_patterns[source], output_patterns[target])
            completed += 1
        print(
            f"  epoch {epoch}/{EPOCHS}: {completed}/{total} transitions "
            f"({100.0 * completed / total:5.1f}%) in "
            f"{time.perf_counter() - epoch_started:.2f}s"
        )
    return completed, time.perf_counter() - started


def internal_replay(
    brain: SurfaceFlowBrain,
    cue_pattern: dict[int, float],
    persistence: float,
) -> InternalReplayResult:
    started = time.perf_counter()
    activation = pattern_vector(cue_pattern, brain.node_count)
    fatigue = np.zeros(brain.node_count, dtype=float)
    states = [activation.copy()]
    differences: list[float] = []
    active_counts = [int(np.count_nonzero(activation))]
    stable_steps = 0
    stop_reason = f"reached maximum internal steps ({MAX_INTERNAL_STEPS})"

    for _ in range(MAX_INTERNAL_STEPS):
        effective = activation * (1.0 - np.clip(fatigue, 0.0, 0.95))
        contributions = effective[:, None] * brain.weights * brain.transmission_gain
        contributions[~brain.edge_enabled] = 0.0
        contributions = np.clip(contributions, 0.0, 1.0 - 1e-12)
        propagated = 1.0 - np.prod(1.0 - contributions, axis=0)

        # Residual activity is internal only: no decoder and no external feedback.
        next_activation = 1.0 - (1.0 - propagated) * (1.0 - persistence * activation)
        next_activation = np.clip(next_activation, 0.0, 1.0)
        next_activation[next_activation < THRESHOLD] = 0.0

        difference = float(np.mean(np.abs(next_activation - activation)))
        states.append(next_activation.copy())
        differences.append(difference)
        active_counts.append(int(np.count_nonzero(next_activation)))

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

    return InternalReplayResult(
        persistence=persistence,
        states=states,
        differences=differences,
        active_counts=active_counts,
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - started,
    )


def decode_state_history(
    result: InternalReplayResult,
    output_vectors: dict[str, np.ndarray],
) -> list[list[tuple[str, float]]]:
    # This function is intentionally called only after replay has completely ended.
    decoded: list[list[tuple[str, float]]] = []
    for state in result.states:
        ranked = [(word, cosine(state, vector)) for word, vector in output_vectors.items()]
        ranked.sort(key=lambda item: item[1], reverse=True)
        decoded.append(ranked[:5])
    return decoded


def compact_observer_timeline(
    decoded: list[list[tuple[str, float]]],
    minimum_score: float = 0.10,
) -> list[tuple[int, str, float]]:
    timeline: list[tuple[int, str, float]] = []
    previous: str | None = None
    for step, ranked in enumerate(decoded):
        if not ranked:
            continue
        word, score = ranked[0]
        if score < minimum_score or word == previous:
            continue
        timeline.append((step, word, score))
        previous = word
    return timeline


def main() -> None:
    total_started = time.perf_counter()
    print("SphereBrain v24 — Autonomous Internal Replay")
    print("Only the first cue is externally injected.")
    print("During replay there is no decoder, no word selection, no feedback, and no learning.")
    print("Numeric state history is decoded only after the internal process has stopped.\n")

    print("Target experience:")
    print("  " + " -> ".join(EXPERIENCE))
    print("Distractor experiences:")
    for experience in DISTRACTOR_EXPERIENCES:
        print("  " + " -> ".join(experience))
    print()

    brain, inputs, outputs, output_vectors = build_brain()
    events, training_seconds = train(brain, inputs, outputs)

    print("\nAutonomous replay trials")
    results: list[InternalReplayResult] = []
    for persistence in PERSISTENCE_VALUES:
        result = internal_replay(brain, inputs[EXPERIENCE[0]], persistence)
        results.append(result)
        final_delta = result.differences[-1] if result.differences else 0.0
        print(
            f"  persistence={persistence:.2f}: steps={len(result.states) - 1:2d} "
            f"final_active={result.active_counts[-1]:3d} "
            f"final_delta={final_delta:.6f} time={result.elapsed_seconds:.2f}s"
        )
        print(f"    stop: {result.stop_reason}")

    print("\nPost-hoc Observer analysis")
    for result in results:
        decoded = decode_state_history(result, output_vectors)
        timeline = compact_observer_timeline(decoded)
        print(f"\n  persistence={result.persistence:.2f}")
        if timeline:
            print("    dominant-state changes:")
            for step, word, score in timeline:
                print(f"      internal step {step:2d}: {word} score={score:.3f}")
        else:
            print("    dominant-state changes: none above observer threshold")

        print("    selected snapshots:")
        snapshot_steps = sorted(
            {0, 1, 2, 4, 8, 12, 16, 24, len(decoded) - 1}
            & set(range(len(decoded)))
        )
        for step in snapshot_steps:
            candidates = ", ".join(
                f"{word}:{score:.3f}" for word, score in decoded[step]
            )
            delta = 0.0 if step == 0 else result.differences[step - 1]
            print(
                f"      step {step:2d}: active={result.active_counts[step]:3d} "
                f"delta={delta:.6f} | {candidates}"
            )

    print("\nSummary")
    print("-" * 72)
    print(f"learned transitions : {events}")
    print(f"training time       : {training_seconds:.2f}s")
    print(f"total experiment    : {time.perf_counter() - total_started:.2f}s")

    print("\nInterpretation guardrails")
    print("1. Only '空' is supplied from outside at replay step 0.")
    print("2. No decoded word is fed back into the core.")
    print("3. Replay does not change pathway weights or usage counts.")
    print("4. Observer labels are applied only after all numeric states were recorded.")
    print("5. A readable sequence is evidence to investigate, not proof of human-like memory.")


if __name__ == "__main__":
    main()
