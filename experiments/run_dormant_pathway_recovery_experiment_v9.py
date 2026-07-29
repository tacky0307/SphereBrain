from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment as v7
from dormant_surface_flow import DormantSurfaceFlowBrain
from dormant_surface_flow_v9 import SameRegionDistinctInputRecoveryBrain


MAX_PROMOTIONS_PER_EPOCH = 10
MAX_PROMOTIONS_TOTAL = 100
DISTINCT_INPUTS_REQUIRED = 2


def build_dormant_brain() -> SameRegionDistinctInputRecoveryBrain:
    return SameRegionDistinctInputRecoveryBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        dormancy_after=160,
        protection_period=36,
        dormant_transmission=0.40,
        dormant_search_penalty=1.2,
        reactivation_boost=0.025,
        state_activity_decay=0.92,
        overuse_threshold=0.55,
        overuse_penalty_gain=0.0,
        overuse_penalty_decay=0.82,
        strong_contribution_threshold=0.04,
        activity_increase_ratio=2.0,
        activity_increase_margin=0.02,
        candidate_required_experiences=DISTINCT_INPUTS_REQUIRED,
        max_candidates_per_experience=20,
        max_promotions_per_epoch=MAX_PROMOTIONS_PER_EPOCH,
        max_promotions_total=MAX_PROMOTIONS_TOTAL,
        distinct_inputs_required=DISTINCT_INPUTS_REQUIRED,
    )


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    minimum = min(example.x for example in examples)
    maximum = max(example.x for example in examples)

    for epoch in range(epochs):
        if isinstance(brain, SameRegionDistinctInputRecoveryBrain):
            brain.begin_recovery_epoch()

        for example in examples:
            if isinstance(brain, SameRegionDistinctInputRecoveryBrain):
                region = v7.input_region(example.x, minimum, maximum)
                brain.set_experience(region, example.x)
                # Candidate selection precedes teacher reinforcement.
                v7.observe(brain, input_encoder.encode(example.x))

            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )

        if isinstance(brain, SameRegionDistinctInputRecoveryBrain):
            measured = brain.recovery_measurement_stats()
            print(
                f"epoch {epoch + 1:02d}: "
                f"promotions={int(measured['promotions_this_epoch'])} "
                f"total={int(measured['selective_promotions_total'])} "
                f"pending={int(measured['pending_candidate_edges'])} "
                f"two_input_paths={int(measured['candidate_paths_two_or_more_distinct_inputs'])}"
            )

    if isinstance(brain, SameRegionDistinctInputRecoveryBrain):
        brain.set_experience(None, None)
    print()


def print_state_stats(brain: DormantSurfaceFlowBrain, label: str) -> None:
    stats = brain.pathway_state_stats()
    print(label)
    print(
        f"normal={int(stats['normal'])} protected={int(stats['protected'])} "
        f"dormant={int(stats['dormant'])} reactivations={int(stats['reactivations'])} "
        f"recovery_mode={'ON' if stats['recovery_mode'] else 'OFF'}"
    )

    if isinstance(brain, SameRegionDistinctInputRecoveryBrain):
        measured = brain.recovery_measurement_stats()
        print(
            f"candidate_events={int(measured['candidate_selection_events_total'])} "
            f"unique_candidates={int(measured['candidate_unique_edges_total'])} "
            f"promotions={int(measured['selective_promotions_total'])} "
            f"pending={int(measured['pending_candidate_edges'])}"
        )
        print(
            f"same_input_repeats_ignored={int(measured['duplicate_input_selections_ignored'])} "
            f"epoch_cap_blocks={int(measured['epoch_promotion_cap_blocks'])} "
            f"total_cap_blocks={int(measured['total_promotion_cap_blocks'])}"
        )
        print(
            f"one_input_paths={int(measured['candidate_paths_one_distinct_input'])} "
            f"two_or_more_input_paths={int(measured['candidate_paths_two_or_more_distinct_inputs'])} "
            f"max_inputs_same_region={int(measured['max_distinct_inputs_same_region'])}"
        )
        print(
            f"promotion_eligible_by_region: "
            f"low={int(measured['low_region_promotion_eligible'])} "
            f"middle={int(measured['middle_region_promotion_eligible'])} "
            f"high={int(measured['high_region_promotion_eligible'])}"
        )
        print(
            f"teacher_direct_wakes={int(measured['teacher_direct_reactivations_total'])} "
            f"teacher_blocked_dormant={int(measured['teacher_blocked_dormant_total'])} "
            f"baseline_mean_contribution={measured['baseline_mean_contribution']:.4f}"
        )
    print()


def main() -> None:
    # Reuse the established v7 evaluation flow while replacing the recovery
    # brain, training loop, and diagnostics for the v9 hypothesis.
    v7.DiverseRegionCappedRecoveryBrain = SameRegionDistinctInputRecoveryBrain
    v7.build_dormant_brain = build_dormant_brain
    v7.recovery_train = recovery_train
    v7.print_state_stats = print_state_stats

    print("SphereBrain same-region distinct-input dormant recovery experiment v9")
    print("task: y = 2x")
    print(
        f"pretrain={v7.PRETRAIN_EPOCHS} epochs, lesion={v7.LESION_FRACTION:.0%}, "
        f"recovery={v7.RECOVERY_EPOCHS} epochs"
    )
    print("candidate threshold=max(0.04, pre-lesion mean + 0.02)")
    print("top 20 candidates per experience")
    print("same input repeats count once")
    print("promotion requires 2 distinct input values inside the same region")
    print("promotion caps: max 10 per epoch, max 100 total")
    print("teacher reinforcement only after promotion; dormant direct wake is blocked\n")

    v7.run_condition("ordinary reinforcement", v7.build_ordinary_brain())
    v7.run_condition(
        "same-region distinct-input dormant recovery v9",
        build_dormant_brain(),
    )


if __name__ == "__main__":
    main()
