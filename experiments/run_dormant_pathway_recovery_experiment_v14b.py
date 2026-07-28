from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment_v13 as v13
from dormant_surface_flow_v14b import ExperienceClusterTransitionBrain

PRETRAIN_EPOCH_OPTIONS = (20, 50, 100, 200)
CLUSTER_SIMILARITY_THRESHOLD = 0.985
CLUSTER_MIN_SHARED_INPUTS = 3
TRANSITION_MIN_DISTINCT_INPUT_PAIRS = 3
TOP_TRANSITIONS = 20
_LAST_BRAIN: ExperienceClusterTransitionBrain | None = None


def build_brain() -> ExperienceClusterTransitionBrain:
    global _LAST_BRAIN
    _LAST_BRAIN = ExperienceClusterTransitionBrain(
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
        stable_pair_distinct_inputs=3,
        cluster_similarity_threshold=CLUSTER_SIMILARITY_THRESHOLD,
        cluster_min_shared_inputs=CLUSTER_MIN_SHARED_INPUTS,
        transition_min_distinct_input_pairs=TRANSITION_MIN_DISTINCT_INPUT_PAIRS,
    )
    return _LAST_BRAIN


def edge_label(edge: tuple[int, int]) -> str:
    return f"{edge[0]:03d}->{edge[1]:03d}"


def print_cluster_analysis(brain: ExperienceClusterTransitionBrain) -> dict[str, float]:
    stats = brain.cluster_transition_stats()
    print("experience-group separation")
    print(f"activity snapshots:              {int(stats['activity_snapshots'])}")
    print(f"experience clusters:             {int(stats['experience_clusters'])}")
    print(f"largest experience cluster:      {int(stats['largest_experience_cluster'])}")
    print(f"singleton clusters:              {int(stats['singleton_clusters'])}")

    region_names = {0: "low", 1: "middle", 2: "high"}
    for profile in brain.cluster_profiles():
        dominant = ", ".join(
            f"{region_names.get(region, str(region))}:{input_key:.3f}={value:.4f}"
            for region, input_key, value in profile["dominant_inputs"]
        ) or "none"
        edges = ", ".join(edge_label(edge) for edge in profile["edges"][:8])
        if int(profile["size"]) > 8:
            edges += ", ..."
        print(
            f"cluster {int(profile['cluster_id']):02d}: size={int(profile['size']):2d} "
            f"dominant=[{dominant}] edges=[{edges}]"
        )

    print("\ndirected experience-group transitions")
    print(f"observed directed transitions:   {int(stats['directed_transitions'])}")
    print(f"candidate cluster bridges:       {int(stats['candidate_cluster_bridges'])}")
    print(f"candidate mean transition lift:  {stats['candidate_mean_lift']:.3f}")
    rows = brain.transition_rows()
    if not rows:
        print("transition detail: no transitions between separate clusters\n")
        return stats

    print("from | to | events | input pairs | epochs | P(to|from) | P(to) | lift | target strength | candidate")
    for row in rows[:TOP_TRANSITIONS]:
        print(
            f"{int(row['source_cluster']):4d} | {int(row['target_cluster']):2d} | "
            f"{int(row['events']):6d} | {int(row['distinct_input_pairs']):11d} | "
            f"{int(row['distinct_epochs']):6d} | "
            f"{float(row['conditional_probability']):10.3f} | "
            f"{float(row['baseline_probability']):5.3f} | "
            f"{float(row['transition_lift']):4.2f} | "
            f"{float(row['target_strength_mean']):15.5f} | "
            f"{'YES' if int(row['candidate']) else 'no'}"
        )
    print()
    return stats


def run_condition(epochs: int) -> dict[str, float]:
    result = v13.run_pretrain_condition(epochs)
    if _LAST_BRAIN is None:
        raise RuntimeError("v14b brain was not created")
    result.update(print_cluster_analysis(_LAST_BRAIN))
    return result


def print_comparison(results: list[dict[str, float]]) -> None:
    print("=" * 136)
    print("v14b experience-group separation and directed-transition comparison")
    print("=" * 136)
    print(
        "epochs | learned | damaged | recovered | promoted | snapshots | clusters | largest | singletons | transitions | bridge candidates | mean lift"
    )
    for row in results:
        print(
            f"{int(row['epochs']):6d} | {row['pre_mae']:7.4f} | "
            f"{row['damaged_mae']:7.4f} | {row['recovered_mae']:9.4f} | "
            f"{int(row['promoted']):8d} | {int(row['activity_snapshots']):9d} | "
            f"{int(row['experience_clusters']):8d} | "
            f"{int(row['largest_experience_cluster']):7d} | "
            f"{int(row['singleton_clusters']):10d} | "
            f"{int(row['directed_transitions']):11d} | "
            f"{int(row['candidate_cluster_bridges']):17d} | "
            f"{row['candidate_mean_lift']:9.3f}"
        )


def main() -> None:
    v13.build_brain = build_brain
    print("SphereBrain experience-group and transition experiment v14b")
    print("task: y = 2x")
    print("pretraining comparison: 20, 50, 100, 200 epochs")
    print(f"lesion fixed at {v13.LESION_FRACTION:.0%}; recovery={v13.RECOVERY_EPOCHS} epochs")
    print("v14b is observation-only: no cluster bridge, propagation bias, or new edge")
    print("groups are separated by promoted-path contribution profiles across inputs")
    print(f"cluster cosine threshold: {CLUSTER_SIMILARITY_THRESHOLD:.3f}")
    print(f"minimum shared active inputs: {CLUSTER_MIN_SHARED_INPUTS}")
    print("directed transitions are measured between consecutive recovery observations")
    print(
        f"bridge candidate requires >= {TRANSITION_MIN_DISTINCT_INPUT_PAIRS} distinct input pairs, "
        "at least 2 epochs, and transition lift > 1.0\n"
    )
    results = [run_condition(epochs) for epochs in PRETRAIN_EPOCH_OPTIONS]
    print_comparison(results)


if __name__ == "__main__":
    main()
