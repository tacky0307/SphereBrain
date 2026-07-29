from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment as base
from dormant_surface_flow import DormantSurfaceFlowBrain
from dormant_surface_flow_v10 import CandidateWidthOverlapMeasurementBrain


CANDIDATE_WIDTHS = (20, 50, 100, 200)
MAX_PROMOTIONS_PER_EPOCH = 10
MAX_PROMOTIONS_TOTAL = 100
DISTINCT_INPUTS_REQUIRED = 2


def build_dormant_brain() -> CandidateWidthOverlapMeasurementBrain:
    return CandidateWidthOverlapMeasurementBrain(
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
        candidate_widths=CANDIDATE_WIDTHS,
    )


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    minimum = min(example.x for example in examples)
    maximum = max(example.x for example in examples)

    for epoch in range(epochs):
        if isinstance(brain, CandidateWidthOverlapMeasurementBrain):
            brain.begin_recovery_epoch()

        for example in examples:
            if isinstance(brain, CandidateWidthOverlapMeasurementBrain):
                region = base.input_region(example.x, minimum, maximum)
                brain.set_experience(region, example.x)
                base.observe(brain, input_encoder.encode(example.x))

            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )

        if isinstance(brain, CandidateWidthOverlapMeasurementBrain):
            measured = brain.recovery_measurement_stats()
            overlap = brain.candidate_width_overlap_stats()
            compact = " ".join(
                f"top{width}:shared2+={stats['at_least_2_inputs']}"
                for width, stats in overlap.items()
            )
            print(
                f"epoch {epoch + 1:02d}: "
                f"actual_v9_promotions={int(measured['promotions_this_epoch'])} "
                f"total={int(measured['selective_promotions_total'])} "
                f"{compact}"
            )

    if isinstance(brain, CandidateWidthOverlapMeasurementBrain):
        brain.set_experience(None, None)
    print()


def print_overlap_table(brain: CandidateWidthOverlapMeasurementBrain) -> None:
    print("candidate-width overlap distribution")
    print("width | unique | exactly1 | inputs>=2 | inputs>=3 | inputs>=4 | max inputs")
    for width, stats in brain.candidate_width_overlap_stats().items():
        print(
            f"{width:>5} | "
            f"{stats['unique_paths']:>6} | "
            f"{stats['exactly_1_input']:>8} | "
            f"{stats['at_least_2_inputs']:>9} | "
            f"{stats['at_least_3_inputs']:>9} | "
            f"{stats['at_least_4_inputs']:>9} | "
            f"{stats['max_distinct_inputs']:>10}"
        )
    print()


def print_state_stats(brain: DormantSurfaceFlowBrain, label: str) -> None:
    stats = brain.pathway_state_stats()
    print(label)
    print(
        f"normal={int(stats['normal'])} protected={int(stats['protected'])} "
        f"dormant={int(stats['dormant'])} reactivations={int(stats['reactivations'])} "
        f"recovery_mode={'ON' if stats['recovery_mode'] else 'OFF'}"
    )

    if isinstance(brain, CandidateWidthOverlapMeasurementBrain):
        measured = brain.recovery_measurement_stats()
        print(
            f"actual_top20_candidate_events={int(measured['candidate_selection_events_total'])} "
            f"actual_top20_unique={int(measured['candidate_unique_edges_total'])} "
            f"actual_v9_promotions={int(measured['selective_promotions_total'])}"
        )
        print(
            f"same_input_repeats_ignored={int(measured['duplicate_input_selections_ignored'])} "
            f"teacher_direct_wakes={int(measured['teacher_direct_reactivations_total'])} "
            f"teacher_blocked_dormant={int(measured['teacher_blocked_dormant_total'])}"
        )
        print_overlap_table(brain)
    else:
        print()


def main() -> None:
    base.DiverseRegionCappedRecoveryBrain = CandidateWidthOverlapMeasurementBrain
    base.build_dormant_brain = build_dormant_brain
    base.recovery_train = recovery_train
    base.print_state_stats = print_state_stats

    print("SphereBrain candidate-width overlap distribution experiment v10")
    print("behavior remains v9 at actual top 20; top 50/100/200 are shadow measurements only")
    print("task: y = 2x")
    print(
        f"pretrain={base.PRETRAIN_EPOCHS} epochs, lesion={base.LESION_FRACTION:.0%}, "
        f"recovery={base.RECOVERY_EPOCHS} epochs"
    )
    print("candidate threshold=max(0.04, pre-lesion mean + 0.02)")
    print("measure candidate widths: 20, 50, 100, 200")
    print("same input repeats count once")
    print("reports pathways shared by 2+, 3+, and 4+ distinct input values")
    print("teacher reinforcement remains allowed only after actual v9 promotion")
    print("dormant direct wake remains blocked\n")

    base.run_condition("ordinary reinforcement", base.build_ordinary_brain())
    base.run_condition(
        "candidate-width overlap distribution v10",
        build_dormant_brain(),
    )


if __name__ == "__main__":
    main()
