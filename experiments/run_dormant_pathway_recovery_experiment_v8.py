from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment as v7
from dormant_surface_flow import DormantSurfaceFlowBrain
from dormant_surface_flow_v8 import RegionDistributionMeasuredRecoveryBrain


# Reuse the verified v7 experiment flow while replacing only the measured brain
# and reporting. Candidate selection, promotion, caps, and teacher blocking are
# deliberately unchanged.
v7.DiverseRegionCappedRecoveryBrain = RegionDistributionMeasuredRecoveryBrain


def build_dormant_brain() -> RegionDistributionMeasuredRecoveryBrain:
    return RegionDistributionMeasuredRecoveryBrain(
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
        candidate_required_experiences=3,
        max_candidates_per_experience=20,
        max_promotions_per_epoch=v7.MAX_PROMOTIONS_PER_EPOCH,
        max_promotions_total=v7.MAX_PROMOTIONS_TOTAL,
    )


def print_region_distribution(brain: RegionDistributionMeasuredRecoveryBrain) -> None:
    measured = brain.recovery_measurement_stats()
    print("candidate region-reach distribution")
    print(f"reached exactly 1 region: {int(measured['reached_1_region'])}")
    print(f"reached exactly 2 regions: {int(measured['reached_2_regions'])}")
    print(f"reached exactly 3 regions: {int(measured['reached_3_regions'])}")
    print("candidate pathways containing each region")
    print(f"low region:    {int(measured['region_low_candidates'])}")
    print(f"middle region: {int(measured['region_middle_candidates'])}")
    print(f"high region:   {int(measured['region_high_candidates'])}")
    print("exact region combinations")
    print(
        f"low only={int(measured['low_only'])} "
        f"middle only={int(measured['middle_only'])} "
        f"high only={int(measured['high_only'])}"
    )
    print(
        f"low+middle={int(measured['low_middle'])} "
        f"low+high={int(measured['low_high'])} "
        f"middle+high={int(measured['middle_high'])} "
        f"all three={int(measured['low_middle_high'])}"
    )


_original_print_state_stats = v7.print_state_stats


def print_state_stats(brain: DormantSurfaceFlowBrain, label: str) -> None:
    _original_print_state_stats(brain, label)
    if isinstance(brain, RegionDistributionMeasuredRecoveryBrain):
        print_region_distribution(brain)
        print()


v7.print_state_stats = print_state_stats


def main() -> None:
    print("SphereBrain candidate region-reach distribution experiment v8")
    print("behavior is unchanged from v7; measurement only")
    print("task: y = 2x")
    print(
        f"pretrain={v7.PRETRAIN_EPOCHS} epochs, lesion={v7.LESION_FRACTION:.0%}, "
        f"recovery={v7.RECOVERY_EPOCHS} epochs"
    )
    print("candidate threshold=max(0.04, pre-lesion mean + 0.02)")
    print("reports exact 1/2/3-region reach and low/middle/high combinations\n")

    v7.run_condition("ordinary reinforcement", v7.build_ordinary_brain())
    v7.run_condition(
        "candidate region-reach distribution v8",
        build_dormant_brain(),
    )


if __name__ == "__main__":
    main()
