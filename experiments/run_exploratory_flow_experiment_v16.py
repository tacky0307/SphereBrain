from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_multi_experience_experiment_v15a as v15a
from dormant_surface_flow_v16 import ExploratoryFlowBrain
from experiments.run_dormant_pathway_recovery_experiment import input_region


PRETRAIN_EPOCHS = 50
WARMUP_RECOVERY_EPOCHS = 5
EXPLORATION_EPOCHS = 20
EXPLORATION_PROBABILITY = 0.01
EXPLORATION_FACILITATION = 0.08
MODES = ("off", "guided", "random")


def build_brain(mode: str) -> ExploratoryFlowBrain:
    return ExploratoryFlowBrain(
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
        exploration_mode=mode,
        exploration_probability=EXPLORATION_PROBABILITY,
        exploration_facilitation=EXPLORATION_FACILITATION,
        exploration_edge_limit=8,
        exploration_reproduction_threshold=2,
    )


def recovery_epoch(
    brain,
    input_encoder,
    output_encoder,
    examples,
    epoch_key: int,
    bridge_on: bool,
    exploration_on: bool,
):
    minimum, maximum = float(min(v15a.TRAIN_X)), float(max(v15a.TRAIN_X))
    brain.begin_recovery_epoch()
    for example in v15a.shuffled_epoch(examples, epoch_key):
        region = input_region(example.x, minimum, maximum)
        brain.set_multi_experience(region, example.task_id, example.task_name, example.x)
        pattern = input_encoder.encode(example.task_id, example.x)

        # Sensory observation may use learned recall and sparse exploration.
        brain.set_bridge_runtime(bridge_on)
        brain.set_exploration_runtime(exploration_on, epoch=epoch_key)
        v15a.observe(brain, pattern)

        # Teacher propagation receives neither bridge nor exploratory assistance.
        brain.set_exploration_runtime(False)
        brain.set_bridge_runtime(False)
        brain.experience(pattern, output_encoder.encode(example.y))


def run_condition(mode: str) -> dict[str, float]:
    brain = build_brain(mode)
    input_encoder = v15a.ContextualInputEncoder(brain)
    output_encoder = v15a.build_output_encoder(brain)
    training = v15a.build_examples(v15a.TRAIN_X)
    testing = v15a.build_examples(v15a.TEST_X)

    print("=" * 112)
    print(f"v16 exploratory-flow condition: {mode}")
    print("=" * 112)

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
            bridge_on=False,
            exploration_on=False,
        )
        multi = brain.multi_experience_stats()
        print(
            f"warmup {epoch + 1:02d}: clusters={int(multi['experience_clusters'])} "
            f"bridge-candidates={int(multi['cross_experience_bridge_candidates'])}"
        )

    candidates_before_freeze = int(
        brain.multi_experience_stats()["cross_experience_bridge_candidates"]
    )
    brain.freeze_experience_bridges()
    bridge = brain.bridge_stats()
    print(
        f"freeze: candidates={candidates_before_freeze} "
        f"clusters={int(bridge['frozen_clusters'])} "
        f"bridges={int(bridge['frozen_bridges'])}"
    )

    for epoch in range(EXPLORATION_EPOCHS):
        recovery_epoch(
            brain,
            input_encoder,
            output_encoder,
            training,
            2000 + epoch,
            bridge_on=True,
            exploration_on=(mode != "off"),
        )
        flow = brain.exploratory_flow_stats()
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(
                f"explore {epoch + 1:02d}: attempts={int(flow['exploration_attempts'])} "
                f"triggered={int(flow['exploration_triggered'])} "
                f"events={int(flow['exploration_events'])} "
                f"traversed={int(flow['exploration_traversed_edges'])} "
                f"reproduced={int(flow['exploration_reproduced_edges'])} "
                f"emergent={int(flow['exploration_emergent_candidates'])}"
            )

    brain.set_exploration_runtime(False)
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
    flow = brain.exploratory_flow_stats()
    measured = brain.recovery_measurement_stats()
    print(
        f"summary: mode={mode} recovered={recovered_mae:.4f} "
        f"triggered={int(flow['exploration_triggered'])} "
        f"events={int(flow['exploration_events'])} "
        f"unique={int(flow['exploration_unique_edges'])} "
        f"reproduced={int(flow['exploration_reproduced_edges'])} "
        f"emergent={int(flow['exploration_emergent_candidates'])} "
        f"vanished={int(flow['exploration_vanished_edges'])} "
        f"new-routes={len(recovered_edges - pre_edges)} "
        f"teacher-wakes={int(measured['teacher_direct_reactivations_total'])}\n"
    )
    return {
        "mode": mode,
        "pre_mae": pre_mae,
        "damaged_mae": damaged_mae,
        "recovered_mae": recovered_mae,
        "new_routes": float(len(recovered_edges - pre_edges)),
        **flow,
    }


def main() -> None:
    print("SphereBrain v16 — Exploratory Flow: Experience Creates Experience")
    print("comparison: no exploration / bridge-guided exploration / random exploration")
    print(
        f"pretraining={PRETRAIN_EPOCHS}, warmup={WARMUP_RECOVERY_EPOCHS}, "
        f"exploration={EXPLORATION_EPOCHS} epochs"
    )
    print(
        f"exploration probability={EXPLORATION_PROBABILITY:.1%}, "
        f"transient facilitation={EXPLORATION_FACILITATION:.1%}"
    )
    print("guided targets are underused enabled paths 1–2 graph steps beyond a learned bridge")
    print("no physical edge, no stored-weight change, no dormant wake, no teacher assistance")
    print("an exploratory edge is reproduced only after appearing in at least two epochs")
    print("an emergent candidate must also appear across at least two experience tasks\n")

    results = [run_condition(mode) for mode in MODES]

    print("=" * 160)
    print("v16 Exploratory Flow comparison")
    print("=" * 160)
    print(
        "mode   | before | damaged | recovered | attempts | triggered | events | traversed | unique | reproduced | emergent | vanished | new routes"
    )
    for row in results:
        print(
            f"{row['mode']:6s} | {row['pre_mae']:6.4f} | {row['damaged_mae']:7.4f} | "
            f"{row['recovered_mae']:9.4f} | {int(row['exploration_attempts']):8d} | "
            f"{int(row['exploration_triggered']):9d} | {int(row['exploration_events']):6d} | "
            f"{int(row['exploration_traversed_edges']):9d} | {int(row['exploration_unique_edges']):6d} | "
            f"{int(row['exploration_reproduced_edges']):10d} | "
            f"{int(row['exploration_emergent_candidates']):8d} | "
            f"{int(row['exploration_vanished_edges']):8d} | {int(row['new_routes']):10d}"
        )


if __name__ == "__main__":
    main()
