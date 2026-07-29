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
PRETRAIN_EPOCHS = 20
RECOVERY_EPOCHS = 10
LESION_FRACTION = 0.05
PATTERN_WIDTH = 5
DECODER_POWER = 2.0


def build_brain(plasticity: bool) -> SurfaceFlowBrain:
    return SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        bidirectional_plasticity_enabled=plasticity,
        recent_activity_decay=0.90,
        unused_activity_threshold=0.015,
        unused_weakening_rate=0.00015,
        overuse_activity_threshold=0.35,
        overuse_weakening_rate=0.0020,
        minimum_edge_weight=0.02,
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


def observe(brain: SurfaceFlowBrain, input_pattern):
    return brain.propagate(
        input_pattern,
        noise=0.0,
        steps=40,
        threshold=0.04,
        use_activation_field=False,
        update_activation_field=False,
    )


def output_node_energy(brain: SurfaceFlowBrain, result) -> np.ndarray:
    index = {node: i for i, node in enumerate(brain.output_nodes)}
    energy = np.zeros(len(brain.output_nodes), dtype=float)
    for step in result.output_history:
        for node, value in step.items():
            energy[index[node]] += value
    return energy


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
) -> tuple[float, float, set[tuple[int, int]]]:
    result = observe(brain, input_encoder.encode(x))
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
        prediction = 0.5
        confidence = 0.0
    else:
        prediction = float(np.sum(CANDIDATE_Y * weights) / total)
        confidence = float(np.max(weights) / total)
    return prediction, confidence, set(result.traversed_edges)


def train(
    brain: SurfaceFlowBrain,
    input_encoder: ScalarSurfaceEncoder,
    output_encoder: ScalarSurfaceEncoder,
    examples: list[Example],
    epochs: int,
) -> None:
    for _ in range(epochs):
        for example in examples:
            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )


def evaluate(
    label: str,
    brain: SurfaceFlowBrain,
    input_encoder: ScalarSurfaceEncoder,
    output_encoder: ScalarSurfaceEncoder,
    baselines: dict[float, np.ndarray],
    examples: list[Example],
) -> tuple[float, set[tuple[int, int]]]:
    errors: list[float] = []
    traversed: set[tuple[int, int]] = set()
    print(label)
    for example in examples:
        prediction, confidence, edges = predict_value(
            brain,
            example.x,
            input_encoder,
            output_encoder,
            baselines[example.x],
        )
        error = abs(prediction - example.y)
        errors.append(error)
        traversed.update(edges)
        print(
            f"x={example.x:.3f} expected={example.y:.3f} "
            f"predicted={prediction:.3f} error={error:.3f} "
            f"confidence={confidence:.4f}"
        )
    mae = float(np.mean(errors))
    print(f"MAE: {mae:.4f} | traversed unique edges: {len(traversed)}")
    print()
    return mae, traversed


def run_condition(name: str, plasticity: bool) -> None:
    brain = build_brain(plasticity=plasticity)
    input_encoder, output_encoder = build_encoders(brain)
    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    baselines = {
        example.x: output_node_energy(
            brain,
            observe(brain, input_encoder.encode(example.x)),
        )
        for example in testing
    }

    print("=" * 72)
    print(f"condition: {name}")
    print(f"bidirectional plasticity: {'ON' if plasticity else 'OFF'}")
    print("=" * 72)

    train(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    pre_mae, pre_edges = evaluate(
        "--- before lesion ---",
        brain,
        input_encoder,
        output_encoder,
        baselines,
        testing,
    )

    disabled = brain.lesion_most_used_edges(
        fraction=LESION_FRACTION,
        bidirectional=True,
    )
    stats = brain.pathway_stats()
    print(
        f"lesion: disabled {len(disabled)} directed edges "
        f"({LESION_FRACTION:.0%} of used-edge ranking, reverse directions included)"
    )
    print(
        f"pathways now: enabled={int(stats['enabled_edges'])} "
        f"disabled={int(stats['disabled_edges'])} "
        f"mean_enabled_weight={stats['mean_enabled_weight']:.4f}"
    )
    print()

    damaged_mae, damaged_edges = evaluate(
        "--- immediately after lesion ---",
        brain,
        input_encoder,
        output_encoder,
        baselines,
        testing,
    )

    train(brain, input_encoder, output_encoder, training, RECOVERY_EPOCHS)
    recovered_mae, recovered_edges = evaluate(
        f"--- after {RECOVERY_EPOCHS} recovery epochs ---",
        brain,
        input_encoder,
        output_encoder,
        baselines,
        testing,
    )

    damage = damaged_mae - pre_mae
    recovered_amount = damaged_mae - recovered_mae
    recovery_ratio = 0.0 if damage <= 1e-12 else recovered_amount / damage
    retained_routes = len(pre_edges & recovered_edges)
    new_routes = len(recovered_edges - pre_edges)

    print("summary")
    print(f"before lesion MAE:       {pre_mae:.4f}")
    print(f"immediately damaged MAE: {damaged_mae:.4f}")
    print(f"recovered MAE:           {recovered_mae:.4f}")
    print(f"damage increase:         {damage:+.4f}")
    print(f"recovered amount:        {recovered_amount:+.4f}")
    print(f"recovery ratio:          {recovery_ratio:.1%}")
    print(f"retained pre-lesion routes: {retained_routes}")
    print(f"new recovery routes:        {new_routes}")
    print()


def main() -> None:
    print("SphereBrain pathway lesion and recovery experiment")
    print("task: y = 2x")
    print(
        f"pretrain={PRETRAIN_EPOCHS} epochs, lesion={LESION_FRACTION:.0%}, "
        f"recovery={RECOVERY_EPOCHS} epochs"
    )
    print()
    run_condition("ordinary reinforcement", plasticity=False)
    run_condition("bidirectional plasticity", plasticity=True)


if __name__ == "__main__":
    main()
