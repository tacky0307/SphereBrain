from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment_v13 as v13
from dormant_surface_flow_v14 import PromotedCooccurrenceTrackingBrain


PRETRAIN_EPOCH_OPTIONS = (20, 50, 100, 200)
STABLE_PAIR_DISTINCT_INPUTS = 3
TOP_PAIR_ROWS = 20
_LAST_BRAIN: PromotedCooccurrenceTrackingBrain | None = None


def build_brain() -> PromotedCooccurrenceTrackingBrain:
    global _LAST_BRAIN
    _LAST_BRAIN = PromotedCooccurrenceTrackingBrain(
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
        candidate_required_experiences=v13.DISTINCT_INPUTS_REQUIRED,
        max_candidates_per_experience=v13.ACTUAL_CANDIDATE_WIDTH,
        max_promotions_per_epoch=v13.MAX_PROMOTIONS_PER_EPOCH,
        max_promotions_total=v13.MAX_PROMOTIONS_TOTAL,
        distinct_inputs_required=v13.DISTINCT_INPUTS_REQUIRED,
        stable_pair_distinct_inputs=STABLE_PAIR_DISTINCT_INPUTS,
    )
    return _LAST_BRAIN


def edge_label(edge: tuple[int, int]) -> str:
    return f"{edge[0]:03d}->{edge[1]:03d}"


def print_cooccurrence(brain: PromotedCooccurrenceTrackingBrain) -> dict[str, float]:
    stats = brain.cooccurrence_stats()
    print("promoted-path co-occurrence network")
    print(f"co-occurrence events:             {int(stats['cooccurrence_events_total'])}")
    print(f"unique promoted-path pairs:       {int(stats['cooccurrence_unique_pairs'])}")
    print(
        f"stable pairs (>= {STABLE_PAIR_DISTINCT_INPUTS} distinct inputs): "
        f"{int(stats['stable_pairs'])}"
    )
    print(f"stable-pair mean stability:       {stats['stable_pair_mean_stability']:.3f}")
    print(
        "stable-pair mean joint contribution: "
        f"{stats['stable_pair_mean_joint_contribution']:.5f}"
    )
    print(
        "stable-pair mean distinct inputs:    "
        f"{stats['stable_pair_mean_distinct_inputs']:.2f}"
    )
    print(f"stable co-occurrence clusters:    {int(stats['stable_clusters'])}")
    print(f"largest stable cluster:           {int(stats['largest_stable_cluster'])}")
    print(f"duplicate events ignored:         {int(stats['duplicate_events_ignored'])}")

    rows = brain.cooccurrence_rows()
    if not rows:
        print("co-occurrence detail: no promoted-path pairs\n")
        return stats

    print("top promoted-path pairs")
    print(
        "path A    | path B    | events | region | inputs | stability | "
        "joint mean | stable"
    )
    region_names = {0: "low", 1: "middle", 2: "high", -1: "none"}
    for row in rows[:TOP_PAIR_ROWS]:
        region = region_names.get(int(row['best_region']), str(row['best_region']))
        print(
            f"{edge_label(row['edge_a'])} | {edge_label(row['edge_b'])} | "
            f"{int(row['events']):6d} | {region:6s} | "
            f"{int(row['distinct_inputs']):6d} | "
            f"{float(row['stability']):9.3f} | "
            f"{float(row['joint_mean']):10.5f} | "
            f"{'YES' if int(row['stable']) else 'no'}"
        )
    print()
    return stats


def run_condition(epochs: int) -> dict[str, float]:
    result = v13.run_pretrain_condition(epochs)
    if _LAST_BRAIN is None:
        raise RuntimeError("v14 brain was not created")
    stats = print_cooccurrence(_LAST_BRAIN)
    result.update(stats)
    return result


def print_comparison(results: list[dict[str, float]]) -> None:
    print("=" * 150)
    print("v14a promoted-path co-occurrence network comparison")
    print("=" * 150)
    print(
        "epochs | learned | damaged | recovered | promoted | pair events | "
        "unique pairs | stable pairs | stability | clusters | largest | joint mean"
    )
    for row in results:
        print(
            f"{int(row['epochs']):6d} | "
            f"{row['pre_mae']:7.4f} | "
            f"{row['damaged_mae']:7.4f} | "
            f"{row['recovered_mae']:9.4f} | "
            f"{int(row['promoted']):8d} | "
            f"{int(row['cooccurrence_events_total']):11d} | "
            f"{int(row['cooccurrence_unique_pairs']):12d} | "
            f"{int(row['stable_pairs']):12d} | "
            f"{row['stable_pair_mean_stability']:9.3f} | "
            f"{int(row['stable_clusters']):8d} | "
            f"{int(row['largest_stable_cluster']):7d} | "
            f"{row['stable_pair_mean_joint_contribution']:10.5f}"
        )


def main() -> None:
    v13.build_brain = build_brain
    print("SphereBrain promoted-path co-occurrence network experiment v14a")
    print("task: y = 2x")
    print("pretraining comparison: 20, 50, 100, 200 epochs")
    print(f"lesion fixed at {v13.LESION_FRACTION:.0%}; recovery={v13.RECOVERY_EPOCHS} epochs")
    print("v14a is measurement-only: no bridge or propagation bias is added")
    print("co-occurrence counts promoted paths used in the same recovery observation")
    print("same epoch + region + input + path pair counts once")
    print(
        f"stable pair requires {STABLE_PAIR_DISTINCT_INPUTS} distinct input values "
        "inside one region\n"
    )

    results = [run_condition(epochs) for epochs in PRETRAIN_EPOCH_OPTIONS]
    print_comparison(results)


if __name__ == "__main__":
    main()
