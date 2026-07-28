from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dormant_surface_flow import DormantSurfaceFlowBrain
from surface_flow import SurfaceFlowBrain
from experiments.run_pathway_recovery_experiment import (
    LESION_FRACTION,
    PRETRAIN_EPOCHS,
    RECOVERY_EPOCHS,
    TEST_X,
    TRAIN_X,
    Example,
    build_encoders,
    evaluate,
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


def build_dormant_brain() -> DormantSurfaceFlowBrain:
    return DormantSurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        dormancy_after=40,
        protection_period=18,
        dormant_transmission=0.18,
        dormant_search_penalty=1.8,
        reactivation_boost=0.025,
        state_activity_decay=0.92,
        overuse_threshold=0.55,
        overuse_penalty_gain=0.55,
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
        f"mean_homeostatic_penalty={stats['mean_homeostatic_penalty']:.4f}"
    )
    print()


def run_condition(name: str, brain: SurfaceFlowBrain) -> None:
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

    print("=" * 76)
    print(f"condition: {name}")
    print("=" * 76)

    train(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    pre_mae, pre_edges = evaluate(
        "--- before lesion ---",
        brain,
        input_encoder,
        output_encoder,
        baselines,
        testing,
    )
    if isinstance(brain, DormantSurfaceFlowBrain):
        print_state_stats(brain, "pathway states before lesion")
        reactivations_before = int(brain.pathway_state_stats()["reactivations"])
    else:
        reactivations_before = 0

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

    if isinstance(brain, DormantSurfaceFlowBrain):
        print_state_stats(brain, "pathway states after recovery")
        reactivations_after = int(brain.pathway_state_stats()["reactivations"])
    else:
        reactivations_after = 0

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
        print(f"reactivated dormant routes during recovery: {reactivations_after - reactivations_before}")
    print()


def main() -> None:
    print("SphereBrain protected/dormant/reactivating pathway experiment")
    print("task: y = 2x")
    print(
        f"pretrain={PRETRAIN_EPOCHS} epochs, lesion={LESION_FRACTION:.0%}, "
        f"recovery={RECOVERY_EPOCHS} epochs"
    )
    print("dormant pathways preserve weight and can be reactivated by relearning")
    print()

    run_condition("ordinary reinforcement", build_ordinary_brain())
    run_condition("protected + dormant + reactivation", build_dormant_brain())


if __name__ == "__main__":
    main()
