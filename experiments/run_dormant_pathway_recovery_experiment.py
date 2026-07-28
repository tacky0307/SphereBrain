from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dormant_surface_flow import DormantSurfaceFlowBrain
from dormant_surface_flow_v3 import RecoveryGatedDormantBrain
from surface_flow import SurfaceFlowBrain
from experiments.run_pathway_recovery_experiment import (
    CANDIDATE_Y,
    DECODER_POWER,
    LESION_FRACTION,
    PRETRAIN_EPOCHS,
    RECOVERY_EPOCHS,
    TEST_X,
    TRAIN_X,
    Example,
    build_encoders,
    candidate_score,
    observe,
    output_node_energy,
    train,
)


def build_ordinary_brain() -> SurfaceFlowBrain:
    return SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        bidirectional_plasticity_enabled=False,
    )


def build_dormant_brain() -> RecoveryGatedDormantBrain:
    return RecoveryGatedDormantBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        dormancy_after=160,
        protection_period=36,
        dormant_transmission=0.40,
        dormant_search_penalty=1.2,
        reactivation_boost=0.025,
        auto_reactivation_traversals=2,
        state_activity_decay=0.92,
        overuse_threshold=0.55,
        # Isolate dormancy/reactivation in this experiment. Homeostatic
        # suppression previously made trained output lower than the fixed
        # untrained baseline and caused every decoder score to be clipped to 0.
        overuse_penalty_gain=0.0,
        overuse_penalty_decay=0.82,
    )


def print_state_stats(brain: DormantSurfaceFlowBrain, label: str) -> None:
    stats = brain.pathway_state_stats()
    print(label)
    print(
        f"normal={int(stats['normal'])} "
        f"protected={int(stats['protected'])} "
        f"dormant={int(stats['dormant'])} "
        f"reactivations={int(stats['reactivations'])} "
        f"auto_reactivations={int(stats['auto_reactivations'])} "
        f"recovery_mode={'ON' if stats['recovery_mode'] else 'OFF'} "
        f"mean_homeostatic_penalty={stats['mean_homeostatic_penalty']:.4f}"
    )
    print()


def predict_absolute(brain, x, input_encoder, output_encoder):
    """Decode current output energy without an obsolete untrained baseline."""
    result = observe(brain, input_encoder.encode(x))
    energy = output_node_energy(brain, result)
    energy_by_node = {
        node: float(energy[index])
        for index, node in enumerate(brain.output_nodes)
    }
    scored = np.array(
        [
            candidate_score(energy_by_node, output_encoder.encode(float(y)))
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


def evaluate_absolute(label, brain, input_encoder, output_encoder, examples):
    errors: list[float] = []
    traversed: set[tuple[int, int]] = set()
    print(label)
    for example in examples:
        prediction, confidence, edges = predict_absolute(
            brain,
            example.x,
            input_encoder,
            output_encoder,
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


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    """Let strong flow wake standby routes, then reinforce useful alternatives."""
    for _ in range(epochs):
        for example in examples:
            if isinstance(brain, RecoveryGatedDormantBrain):
                observe(brain, input_encoder.encode(example.x))
            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )


def run_condition(name: str, brain: SurfaceFlowBrain) -> None:
    input_encoder, output_encoder = build_encoders(brain)
    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    print("=" * 76)
    print(f"condition: {name}")
    print("=" * 76)

    train(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    pre_mae, pre_edges = evaluate_absolute(
        "--- before lesion ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )
    if isinstance(brain, DormantSurfaceFlowBrain):
        print_state_stats(brain, "pathway states before lesion")
        before_stats = brain.pathway_state_stats()
        reactivations_before = int(before_stats["reactivations"])
        auto_before = int(before_stats["auto_reactivations"])
    else:
        reactivations_before = 0
        auto_before = 0

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

    damaged_mae, damaged_edges = evaluate_absolute(
        "--- immediately after lesion ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )

    if isinstance(brain, DormantSurfaceFlowBrain):
        brain.set_recovery_mode(True)
        print("recovery mode: ON (new dormancy suspended; strong-flow waking enabled)\n")

    recovery_train(
        brain,
        input_encoder,
        output_encoder,
        training,
        RECOVERY_EPOCHS,
    )

    if isinstance(brain, DormantSurfaceFlowBrain):
        brain.set_recovery_mode(False)

    recovered_mae, recovered_edges = evaluate_absolute(
        f"--- after {RECOVERY_EPOCHS} recovery epochs ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )

    if isinstance(brain, DormantSurfaceFlowBrain):
        print_state_stats(brain, "pathway states after recovery")
        after_stats = brain.pathway_state_stats()
        reactivations_after = int(after_stats["reactivations"])
        auto_after = int(after_stats["auto_reactivations"])
    else:
        reactivations_after = 0
        auto_after = 0

    damage = damaged_mae - pre_mae
    recovered_amount = damaged_mae - recovered_mae
    recovery_ratio = 0.0 if damage <= 1e-12 else recovered_amount / damage

    print("summary")
    print(f"before lesion MAE:       {pre_mae:.4f}")
    print(f"immediately damaged MAE: {damaged_mae:.4f}")
    print(f"recovered MAE:           {recovered_mae:.4f}")
    print(f"damage increase:         {damage:+.4f}")
    print(f"recovered amount:        {recovered_amount:+.4f}")
    print(f"recovery ratio:          {recovery_ratio:.1%}")
    print(f"retained pre-lesion routes: {len(pre_edges & recovered_edges)}")
    print(f"newly traversed routes:     {len(recovered_edges - pre_edges)}")
    if isinstance(brain, DormantSurfaceFlowBrain):
        print(
            "reactivated dormant routes during recovery: "
            f"{reactivations_after - reactivations_before}"
        )
        print(
            "automatic signal-driven reactivations:      "
            f"{auto_after - auto_before}"
        )
    print()


def main() -> None:
    print("SphereBrain protected/dormant/reactivating pathway experiment v3")
    print("task: y = 2x")
    print(
        f"pretrain={PRETRAIN_EPOCHS} epochs, lesion={LESION_FRACTION:.0%}, "
        f"recovery={RECOVERY_EPOCHS} epochs"
    )
    print("dormant transmission=40%, dormancy delay=4x, protection=2x")
    print("evaluation is read-only; automatic waking is recovery-gated")
    print("decoder uses current absolute output energy, not a stale baseline")
    print("homeostatic penalty is disabled to isolate dormancy/reactivation")
    print()

    run_condition("ordinary reinforcement", build_ordinary_brain())
    run_condition("protected + dormant + reactivation v3", build_dormant_brain())


if __name__ == "__main__":
    main()
