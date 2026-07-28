from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surface_encoders import ScalarSurfaceEncoder, ordered_surface_nodes
from surface_flow import SurfaceFlowBrain


@dataclass(frozen=True)
class Example:
    x: float
    y: float


TRAIN_X = np.arange(0.00, 0.5001, 0.05)
TEST_X = np.arange(0.025, 0.5000, 0.05)
EPOCHS = 40
PATTERN_WIDTH = 5
CANDIDATE_Y = np.linspace(0.0, 1.0, 101)


def output_node_energy(brain: SurfaceFlowBrain, result) -> np.ndarray:
    index = {node: i for i, node in enumerate(brain.output_nodes)}
    energy = np.zeros(len(brain.output_nodes), dtype=float)
    for step in result.output_history:
        for node, value in step.items():
            energy[index[node]] += value
    return energy


def observe(brain: SurfaceFlowBrain, input_pattern):
    return brain.propagate(input_pattern, noise=0.0, steps=40, threshold=0.04)


def candidate_score(gain_by_output_node: dict[int, float], target_pattern) -> float:
    return sum(
        max(0.0, gain_by_output_node.get(node, 0.0)) * activity
        for node, activity in target_pattern.items()
    )


def predict_value(
    brain: SurfaceFlowBrain,
    x: float,
    input_encoder: ScalarSurfaceEncoder,
    output_encoder: ScalarSurfaceEncoder,
    baseline_energy: np.ndarray,
):
    result = observe(brain, input_encoder.encode(x))
    current = output_node_energy(brain, result)
    gain = current - baseline_energy
    gain_by_node = {
        node: float(gain[index])
        for index, node in enumerate(brain.output_nodes)
    }

    scored = [
        (
            float(y),
            candidate_score(gain_by_node, output_encoder.encode(float(y))),
        )
        for y in CANDIDATE_Y
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    predicted, best_score = ranked[0]
    return predicted, best_score, ranked[:3]


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    input_encoder = ScalarSurfaceEncoder(
        ordered_surface_nodes(brain.positions, brain.input_nodes),
        width=PATTERN_WIDTH,
    )
    output_encoder = ScalarSurfaceEncoder(
        ordered_surface_nodes(brain.positions, brain.output_nodes),
        width=PATTERN_WIDTH,
    )

    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    print("continuous-value experiment: y = 2x")
    print(f"training examples: {len(training)} (x step=0.05)")
    print(f"unseen midpoint tests: {len(testing)}")
    print(f"population pattern width: {PATTERN_WIDTH}")
    print("SphereBrain core receives numeric surface patterns only.")
    print()

    baseline_by_test: dict[float, np.ndarray] = {}
    for example in testing:
        result = observe(brain, input_encoder.encode(example.x))
        baseline_by_test[example.x] = output_node_energy(brain, result)

    checkpoints = {1, 5, 10, 20, EPOCHS}
    for epoch in range(1, EPOCHS + 1):
        route_counts: list[int] = []
        for example in training:
            reinforced = brain.experience(
                input_pattern=input_encoder.encode(example.x),
                target_pattern=output_encoder.encode(example.y),
            )
            route_counts.append(len(reinforced))

        if epoch not in checkpoints:
            continue

        print(f"--- epoch {epoch:>2} ---")
        print(
            "reinforced route edges per example: "
            f"min={min(route_counts)} mean={np.mean(route_counts):.1f} max={max(route_counts)}"
        )
        absolute_errors: list[float] = []
        for example in testing:
            predicted, score, top3 = predict_value(
                brain,
                example.x,
                input_encoder,
                output_encoder,
                baseline_by_test[example.x],
            )
            error = abs(predicted - example.y)
            absolute_errors.append(error)
            candidates = ", ".join(
                f"{value:.2f}:{value_score:.4f}"
                for value, value_score in top3
            )
            print(
                f"x={example.x:.3f} expected={example.y:.3f} "
                f"predicted={predicted:.3f} error={error:.3f} "
                f"gain_score={score:.4f} top=[{candidates}]"
            )

        print(
            f"mean absolute error: {np.mean(absolute_errors):.4f} | "
            f"max error: {np.max(absolute_errors):.4f}"
        )
        print()


if __name__ == "__main__":
    main()
