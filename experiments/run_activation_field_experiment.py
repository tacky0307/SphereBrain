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
CANDIDATE_Y = np.linspace(0.0, 1.0, 101)
EPOCHS = 40
PATTERN_WIDTH = 5
DECODER_POWER = 2.0


def output_node_energy(brain: SurfaceFlowBrain, result) -> np.ndarray:
    index = {node: i for i, node in enumerate(brain.output_nodes)}
    energy = np.zeros(len(brain.output_nodes), dtype=float)
    for step in result.output_history:
        for node, value in step.items():
            energy[index[node]] += value
    return energy


def observe(brain: SurfaceFlowBrain, input_pattern, use_field: bool):
    return brain.propagate(
        input_pattern,
        noise=0.0,
        steps=40,
        threshold=0.04,
        use_activation_field=use_field,
        update_activation_field=False,
    )


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
    use_field: bool,
) -> tuple[float, float]:
    result = observe(brain, input_encoder.encode(x), use_field=use_field)
    current = output_node_energy(brain, result)
    gain = current - baseline_energy
    gain_by_node = {
        node: float(gain[index])
        for index, node in enumerate(brain.output_nodes)
    }

    scored = np.array(
        [
            candidate_score(gain_by_node, output_encoder.encode(float(y)))
            for y in CANDIDATE_Y
        ],
        dtype=float,
    )
    positive = np.clip(scored, 0.0, None)
    weights = positive**DECODER_POWER
    total = float(np.sum(weights))
    if total <= 1e-12:
        return 0.5, 0.0
    predicted = float(np.sum(CANDIDATE_Y * weights) / total)
    confidence = float(np.max(weights) / total)
    return predicted, confidence


def build_brain(field_enabled: bool) -> SurfaceFlowBrain:
    return SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        activation_field_enabled=field_enabled,
        activation_field_influence=0.05,
        activation_field_decay=0.90,
        activation_field_gain=0.10,
    )


def build_encoders(brain: SurfaceFlowBrain):
    input_encoder = ScalarSurfaceEncoder(
        ordered_surface_nodes(brain.positions, brain.input_nodes),
        width=PATTERN_WIDTH,
    )
    output_encoder = ScalarSurfaceEncoder(
        ordered_surface_nodes(brain.positions, brain.output_nodes),
        width=PATTERN_WIDTH,
    )
    return input_encoder, output_encoder


def main() -> None:
    plain = build_brain(field_enabled=False)
    field = build_brain(field_enabled=True)
    plain_input, plain_output = build_encoders(plain)
    field_input, field_output = build_encoders(field)

    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    plain_baseline: dict[float, np.ndarray] = {}
    field_baseline: dict[float, np.ndarray] = {}
    for example in testing:
        plain_baseline[example.x] = output_node_energy(
            plain,
            observe(plain, plain_input.encode(example.x), use_field=False),
        )
        field_baseline[example.x] = output_node_energy(
            field,
            observe(field, field_input.encode(example.x), use_field=False),
        )

    print("activation-field experiment: y = 2x")
    print("comparison: identical pathway learning, field OFF versus field ON")
    print("field parameters: influence=0.05 decay=0.90 gain=0.10")
    print("decoder: full-distribution weighted mean (power=2.0)")
    print()

    checkpoints = {1, 5, 10, 20, EPOCHS}
    for epoch in range(1, EPOCHS + 1):
        for example in training:
            plain.experience(
                input_pattern=plain_input.encode(example.x),
                target_pattern=plain_output.encode(example.y),
                update_activation_field=False,
            )
            field.experience(
                input_pattern=field_input.encode(example.x),
                target_pattern=field_output.encode(example.y),
                update_activation_field=True,
            )

        if epoch not in checkpoints:
            continue

        plain_errors: list[float] = []
        field_errors: list[float] = []
        print(f"--- epoch {epoch:>2} ---")
        for example in testing:
            plain_prediction, plain_confidence = predict_value(
                plain,
                example.x,
                plain_input,
                plain_output,
                plain_baseline[example.x],
                use_field=False,
            )
            field_prediction, field_confidence = predict_value(
                field,
                example.x,
                field_input,
                field_output,
                field_baseline[example.x],
                use_field=True,
            )
            plain_error = abs(plain_prediction - example.y)
            field_error = abs(field_prediction - example.y)
            plain_errors.append(plain_error)
            field_errors.append(field_error)
            print(
                f"x={example.x:.3f} expected={example.y:.3f} "
                f"plain={plain_prediction:.3f} error={plain_error:.3f} "
                f"field={field_prediction:.3f} error={field_error:.3f} "
                f"plain_conf={plain_confidence:.4f} field_conf={field_confidence:.4f}"
            )

        stats = field.activation_field_stats()
        print(
            f"plain MAE: {np.mean(plain_errors):.4f} | "
            f"field MAE: {np.mean(field_errors):.4f} | "
            f"difference(field-plain): "
            f"{np.mean(field_errors) - np.mean(plain_errors):+.4f}"
        )
        print(
            "activation field: "
            f"mean={stats['mean']:.4f} max={stats['max']:.4f} "
            f"active_ratio={stats['active_ratio']:.3f} energy={stats['energy']:.2f}"
        )
        print()


if __name__ == "__main__":
    main()
