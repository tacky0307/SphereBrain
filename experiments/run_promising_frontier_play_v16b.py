from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_multi_experience_experiment_v15a as v15a
from dormant_surface_flow_v16b import PromisingFrontierPlayBrain
from experiments.run_activity_frontier_experiment_v16a import recovery_epoch

PRETRAIN_EPOCHS = 50
WARMUP_RECOVERY_EPOCHS = 5
PLAY_EPOCHS = 20

# We deliberately try several strengths. The goal of this run is first to find
# whether a live frontier can be crossed at all, not to defend one parameter.
CONDITIONS = (
    ("off", 0.0, 0.0),
    ("soft", 0.20, 0.50),
    ("medium", 0.20, 1.50),
    ("strong", 0.20, 4.00),
)


def build_brain(probability: float, facilitation: float) -> PromisingFrontierPlayBrain:
    return PromisingFrontierPlayBrain(
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
        touch_probability=probability,
        touch_facilitation=facilitation,
        touch_edge_limit=4,
        reproduction_threshold=2,
    )


def run_condition(name: str, probability: float, facilitation: float) -> dict[str, float]:
    brain = build_brain(probability, facilitation)
    input_encoder = v15a.ContextualInputEncoder(brain)
    output_encoder = v15a.build_output_encoder(brain)
    training = v15a.build_examples(v15a.TRAIN_X)
    testing = v15a.build_examples(v15a.TEST_X)

    print("=" * 118)
    print(f"v16b promising-frontier play: {name} probability={probability:.0%} touch={facilitation:.0%}")
    print("=" * 118)

    v15a.pretrain(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    v15a.collect_baseline(brain, input_encoder, training + testing)
    pre_mae, _, pre_edges = v15a.evaluate("before lesion", brain, input_encoder, output_encoder, testing)
    disabled = brain.lesion_most_used_edges(fraction=v15a.LESION_FRACTION, bidirectional=True)
    print(f"lesion: disabled {len(disabled)} directed edges")
    damaged_mae, _, _ = v15a.evaluate("after lesion", brain, input_encoder, output_encoder, testing)

    brain.set_recovery_mode(True)
    brain.set_contribution_phase("recovery")
    for epoch in range(WARMUP_RECOVERY_EPOCHS):
        recovery_epoch(brain, input_encoder, output_encoder, training, 1000 + epoch, False, False)
        multi = brain.multi_experience_stats()
        print(f"warmup {epoch + 1:02d}: clusters={int(multi['experience_clusters'])} bridge-candidates={int(multi['cross_experience_bridge_candidates'])}")

    candidates = int(brain.multi_experience_stats()["cross_experience_bridge_candidates"])
    brain.freeze_experience_bridges()
    bridge = brain.bridge_stats()
    print(f"freeze: candidates={candidates} clusters={int(bridge['frozen_clusters'])} bridges={int(bridge['frozen_bridges'])}")

    active = probability > 0.0 and facilitation > 0.0
    for epoch in range(PLAY_EPOCHS):
        recovery_epoch(brain, input_encoder, output_encoder, training, 2000 + epoch, True, active)
        flow = brain.frontier_stats()
        if epoch == 0 or (epoch + 1) % 2 == 0:
            print(
                f"play {epoch + 1:02d}: eligible={int(flow['eligible_opportunities'])} "
                f"triggered={int(flow['exploration_triggered'])} selected={int(flow['selected_edges'])} "
                f"traversed={int(flow['traversed_edges'])} conversion={flow['selection_conversion']:.1%} "
                f"reproduced={int(flow['reproduced_edges'])} emergent={int(flow['emergent_candidates'])}"
            )

    brain.set_frontier_runtime(False)
    brain.set_bridge_runtime(False)
    brain.set_multi_experience(None, None, None, None)
    brain.set_contribution_phase("idle")
    brain.set_recovery_mode(False)
    recovered_mae, _, recovered_edges = v15a.evaluate(
        "after recovery (all transient assistance off)", brain, input_encoder, output_encoder, testing
    )
    flow = brain.frontier_stats()
    measured = brain.recovery_measurement_stats()
    print(
        f"summary: {name} recovered={recovered_mae:.4f} events={int(flow['events'])} "
        f"unique={int(flow['unique_edges'])} reproduced={int(flow['reproduced_edges'])} "
        f"emergent={int(flow['emergent_candidates'])} new-routes={len(recovered_edges - pre_edges)} "
        f"teacher-wakes={int(measured['teacher_direct_reactivations_total'])}\n"
    )
    return {
        "name": name,
        "probability": probability,
        "facilitation": facilitation,
        "pre_mae": pre_mae,
        "damaged_mae": damaged_mae,
        "recovered_mae": recovered_mae,
        "new_routes": float(len(recovered_edges - pre_edges)),
        **flow,
    }


def main() -> None:
    print("SphereBrain v16b — Promising Frontier Play")
    print("Touch what is already alive, strong, recurrent, and near experience-made bridges.")
    print("This run tries several transient touch strengths instead of assuming one is correct.")
    print("No physical edge, stored-weight edit, dormant wake, or teacher assistance.\n")

    results = [run_condition(*condition) for condition in CONDITIONS]
    print("=" * 170)
    print("v16b comparison")
    print("=" * 170)
    print("name   | touch | recovered | eligible | trigger | selected | traversed | convert | unique | reproduced | emergent | new routes")
    for row in results:
        print(
            f"{row['name']:6s} | {row['facilitation']:5.0%} | {row['recovered_mae']:9.4f} | "
            f"{int(row['eligible_opportunities']):8d} | {int(row['exploration_triggered']):7d} | "
            f"{int(row['selected_edges']):8d} | {int(row['traversed_edges']):9d} | "
            f"{row['selection_conversion']:7.1%} | {int(row['unique_edges']):6d} | "
            f"{int(row['reproduced_edges']):10d} | {int(row['emergent_candidates']):8d} | {int(row['new_routes']):10d}"
        )


if __name__ == "__main__":
    main()
