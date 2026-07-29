from __future__ import annotations

import copy
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_multi_experience_experiment_v15a as v15a
from dormant_surface_flow_v16c import CachedPromisingFrontierPlayBrain
from experiments.run_activity_frontier_experiment_v16a import recovery_epoch

PRETRAIN_EPOCHS = 50
WARMUP_RECOVERY_EPOCHS = 5
PLAY_EPOCHS = 20

CONDITIONS = (
    ("off", 0.0, 0.0),
    ("soft", 0.20, 0.50),
    ("medium", 0.20, 1.50),
    ("strong", 0.20, 4.00),
)


def build_brain() -> CachedPromisingFrontierPlayBrain:
    return CachedPromisingFrontierPlayBrain(
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
        candidate_required_experiences=2,
        max_candidates_per_experience=50,
        max_promotions_per_epoch=10,
        max_promotions_total=100,
        distinct_inputs_required=2,
        stable_pair_distinct_inputs=3,
        cluster_similarity_threshold=0.985,
        cluster_min_shared_inputs=3,
        transition_min_distinct_input_pairs=3,
        bridge_facilitation=0.02,
        bridge_mode="learned",
        touch_probability=0.20,
        touch_facilitation=0.50,
        touch_edge_limit=4,
        reproduction_threshold=2,
    )


def prepare_shared_state():
    started = perf_counter()
    brain = build_brain()
    input_encoder = v15a.ContextualInputEncoder(brain)
    output_encoder = v15a.build_output_encoder(brain)
    training = v15a.build_examples(v15a.TRAIN_X)
    testing = v15a.build_examples(v15a.TEST_X)

    print("=" * 118)
    print("v16c shared preparation — performed once for every condition")
    print("=" * 118)

    v15a.pretrain(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    v15a.collect_baseline(brain, input_encoder, training + testing)
    pre_mae, _, pre_edges = v15a.evaluate(
        "before lesion", brain, input_encoder, output_encoder, testing
    )
    disabled = brain.lesion_most_used_edges(
        fraction=v15a.LESION_FRACTION, bidirectional=True
    )
    print(f"lesion: disabled {len(disabled)} directed edges")
    damaged_mae, _, _ = v15a.evaluate(
        "after lesion", brain, input_encoder, output_encoder, testing
    )

    brain.set_recovery_mode(True)
    brain.set_contribution_phase("recovery")
    for epoch in range(WARMUP_RECOVERY_EPOCHS):
        recovery_epoch(
            brain,
            input_encoder,
            output_encoder,
            training,
            1000 + epoch,
            False,
            False,
        )
        multi = brain.multi_experience_stats()
        print(
            f"warmup {epoch + 1:02d}: clusters={int(multi['experience_clusters'])} "
            f"bridge-candidates={int(multi['cross_experience_bridge_candidates'])}"
        )

    candidates = int(
        brain.multi_experience_stats()["cross_experience_bridge_candidates"]
    )
    brain.freeze_experience_bridges()
    bridge = brain.bridge_stats()
    print(
        f"freeze: candidates={candidates} clusters={int(bridge['frozen_clusters'])} "
        f"bridges={int(bridge['frozen_bridges'])}"
    )
    elapsed = perf_counter() - started
    print(f"shared preparation time: {elapsed:.2f}s\n")

    # Copy the whole bundle together so encoder-to-brain references remain intact.
    bundle = (brain, input_encoder, output_encoder)
    return bundle, training, testing, pre_mae, damaged_mae, pre_edges, elapsed


def run_condition(
    shared_bundle,
    training,
    testing,
    pre_mae: float,
    damaged_mae: float,
    pre_edges,
    name: str,
    probability: float,
    facilitation: float,
) -> dict[str, float]:
    started = perf_counter()
    brain, input_encoder, output_encoder = copy.deepcopy(shared_bundle)
    brain.exploration_probability = float(probability)
    brain.exploration_facilitation = float(facilitation)

    print("=" * 118)
    print(
        f"v16c cached promising-frontier play: {name} "
        f"probability={probability:.0%} touch={facilitation:.0%}"
    )
    print("=" * 118)

    active = probability > 0.0 and facilitation > 0.0
    for epoch in range(PLAY_EPOCHS):
        recovery_epoch(
            brain,
            input_encoder,
            output_encoder,
            training,
            2000 + epoch,
            True,
            active,
        )
        flow = brain.frontier_stats()
        if epoch == 0 or (epoch + 1) % 2 == 0:
            print(
                f"play {epoch + 1:02d}: eligible={int(flow['eligible_opportunities'])} "
                f"triggered={int(flow['exploration_triggered'])} "
                f"selected={int(flow['selected_edges'])} "
                f"traversed={int(flow['traversed_edges'])} "
                f"conversion={flow['selection_conversion']:.1%} "
                f"reproduced={int(flow['reproduced_edges'])} "
                f"emergent={int(flow['emergent_candidates'])}"
            )

    brain.set_frontier_runtime(False)
    brain.set_bridge_runtime(False)
    brain.set_multi_experience(None, None, None, None)
    brain.set_contribution_phase("idle")
    brain.set_recovery_mode(False)
    recovered_mae, _, recovered_edges = v15a.evaluate(
        "after recovery (all transient assistance off)",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )
    flow = brain.frontier_stats()
    measured = brain.recovery_measurement_stats()
    elapsed = perf_counter() - started
    print(
        f"summary: {name} recovered={recovered_mae:.4f} "
        f"events={int(flow['events'])} unique={int(flow['unique_edges'])} "
        f"reproduced={int(flow['reproduced_edges'])} "
        f"emergent={int(flow['emergent_candidates'])} "
        f"new-routes={len(recovered_edges - pre_edges)} "
        f"teacher-wakes={int(measured['teacher_direct_reactivations_total'])} "
        f"condition-time={elapsed:.2f}s\n"
    )
    return {
        "name": name,
        "probability": probability,
        "facilitation": facilitation,
        "pre_mae": pre_mae,
        "damaged_mae": damaged_mae,
        "recovered_mae": recovered_mae,
        "new_routes": float(len(recovered_edges - pre_edges)),
        "elapsed_seconds": elapsed,
        **flow,
    }


def main() -> None:
    total_started = perf_counter()
    print("SphereBrain v16c — Cached Promising Frontier Play")
    print("The brain and learning rules are unchanged.")
    print("Pretraining, lesion, warmup, and bridge freezing are performed once.")
    print("Every condition starts from a deep copy of exactly the same frozen state.")
    print("Adjacency and frozen bridge neighbourhood lookups are cached.\n")

    (
        shared_bundle,
        training,
        testing,
        pre_mae,
        damaged_mae,
        pre_edges,
        preparation_time,
    ) = prepare_shared_state()

    results = [
        run_condition(
            shared_bundle,
            training,
            testing,
            pre_mae,
            damaged_mae,
            pre_edges,
            *condition,
        )
        for condition in CONDITIONS
    ]

    total_elapsed = perf_counter() - total_started
    condition_elapsed = sum(row["elapsed_seconds"] for row in results)
    print("=" * 184)
    print("v16c comparison")
    print("=" * 184)
    print(
        "name   | touch | recovered | eligible | trigger | selected | traversed | "
        "convert | unique | reproduced | emergent | new routes | seconds"
    )
    for row in results:
        print(
            f"{row['name']:6s} | {row['facilitation']:5.0%} | "
            f"{row['recovered_mae']:9.4f} | "
            f"{int(row['eligible_opportunities']):8d} | "
            f"{int(row['exploration_triggered']):7d} | "
            f"{int(row['selected_edges']):8d} | "
            f"{int(row['traversed_edges']):9d} | "
            f"{row['selection_conversion']:7.1%} | "
            f"{int(row['unique_edges']):6d} | "
            f"{int(row['reproduced_edges']):10d} | "
            f"{int(row['emergent_candidates']):8d} | "
            f"{int(row['new_routes']):10d} | "
            f"{row['elapsed_seconds']:7.2f}"
        )

    old_equivalent_preparations = preparation_time * len(CONDITIONS)
    avoided = old_equivalent_preparations - preparation_time
    estimated_old_total = old_equivalent_preparations + condition_elapsed
    estimated_speedup = estimated_old_total / total_elapsed if total_elapsed else 0.0
    print(
        f"\ntiming: shared-preparation={preparation_time:.2f}s "
        f"conditions={condition_elapsed:.2f}s total={total_elapsed:.2f}s"
    )
    print(
        f"avoided repeated preparation: approximately {avoided:.2f}s; "
        f"estimated experiment speedup={estimated_speedup:.2f}x"
    )


if __name__ == "__main__":
    main()
