from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_multi_experience_experiment_v15a as v15a
from dormant_surface_flow_v15b import ExperienceBridgeFacilitationBrain
from experiments.run_dormant_pathway_recovery_experiment import input_region


PRETRAIN_EPOCHS = 50
WARMUP_RECOVERY_EPOCHS = 5
BRIDGE_RECOVERY_EPOCHS = 5
BRIDGE_FACILITATION = 0.02
MODES = ("off", "learned", "shuffled")


def build_brain(mode: str) -> ExperienceBridgeFacilitationBrain:
    return ExperienceBridgeFacilitationBrain(
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
        bridge_facilitation=BRIDGE_FACILITATION,
        bridge_mode=mode,
    )


def recovery_epoch(brain, input_encoder, output_encoder, examples, epoch_key: int, bridge_on: bool):
    minimum, maximum = float(min(v15a.TRAIN_X)), float(max(v15a.TRAIN_X))
    brain.begin_recovery_epoch()
    for example in v15a.shuffled_epoch(examples, epoch_key):
        region = input_region(example.x, minimum, maximum)
        brain.set_multi_experience(region, example.task_id, example.task_name, example.x)
        pattern = input_encoder.encode(example.task_id, example.x)
        brain.set_bridge_runtime(bridge_on)
        v15a.observe(brain, pattern)
        # The transient bridge is sensory recall only; teacher propagation gets no help.
        brain.set_bridge_runtime(False)
        brain.experience(pattern, output_encoder.encode(example.y))


def evaluate_sequence(label, brain, input_encoder, output_encoder, examples):
    ordered = v15a.shuffled_epoch(examples, 9001)
    minimum, maximum = float(min(v15a.TEST_X)), float(max(v15a.TEST_X))
    errors: list[float] = []
    traversed: set[tuple[int, int]] = set()
    brain.previous_frozen_cluster = None
    brain.previous_task = None
    for example in ordered:
        region = input_region(example.x, minimum, maximum)
        brain.set_multi_experience(region, example.task_id, example.task_name, example.x)
        brain.set_bridge_runtime(True)
        prediction, _, edges = v15a.predict(
            brain, example, input_encoder, output_encoder
        )
        errors.append(abs(prediction - example.y))
        traversed.update(edges)
    brain.set_bridge_runtime(False)
    brain.set_multi_experience(None, None, None, None)
    mae = float(np.mean(errors))
    print(f"{label}: sequence_mae={mae:.4f} routes={len(traversed)}")
    return mae, traversed


def run_condition(mode: str) -> dict[str, float]:
    brain = build_brain(mode)
    input_encoder = v15a.ContextualInputEncoder(brain)
    output_encoder = v15a.build_output_encoder(brain)
    training = v15a.build_examples(v15a.TRAIN_X)
    testing = v15a.build_examples(v15a.TEST_X)

    print("=" * 108)
    print(f"v15b condition: {mode}")
    print("=" * 108)
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
            brain, input_encoder, output_encoder, training, 1000 + epoch, False
        )
        measured = brain.recovery_measurement_stats()
        multi = brain.multi_experience_stats()
        print(
            f"warmup {epoch + 1:02d}: promotions={int(measured['promotions_this_epoch'])} "
            f"total={int(measured['selective_promotions_total'])} "
            f"clusters={int(multi['experience_clusters'])} "
            f"candidates={int(multi['cross_experience_bridge_candidates'])}"
        )

    candidates_before_freeze = int(
        brain.multi_experience_stats()["cross_experience_bridge_candidates"]
    )
    brain.freeze_experience_bridges()
    frozen = brain.bridge_stats()
    print(
        f"bridge freeze: candidates={candidates_before_freeze} "
        f"frozen_clusters={int(frozen['frozen_clusters'])} "
        f"frozen_bridges={int(frozen['frozen_bridges'])} mode={mode}"
    )

    for epoch in range(BRIDGE_RECOVERY_EPOCHS):
        recovery_epoch(
            brain,
            input_encoder,
            output_encoder,
            training,
            1000 + WARMUP_RECOVERY_EPOCHS + epoch,
            mode != "off",
        )
        bridge = brain.bridge_stats()
        measured = brain.recovery_measurement_stats()
        print(
            f"bridge {epoch + 1:02d}: promotions={int(measured['promotions_this_epoch'])} "
            f"total={int(measured['selective_promotions_total'])} "
            f"applications={int(bridge['bridge_application_events'])} "
            f"target-traversals={int(bridge['bridge_target_edges_traversed'])}"
        )

    brain.set_bridge_runtime(False)
    brain.set_multi_experience(None, None, None, None)
    brain.set_contribution_phase("idle")
    brain.set_recovery_mode(False)

    recovered_mae, _, recovered_edges = v15a.evaluate(
        "after recovery (bridges off / isolated)",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )
    sequence_mae, sequence_edges = evaluate_sequence(
        "after recovery (bridges active / sequence)",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )
    bridge = brain.bridge_stats()
    measured = brain.recovery_measurement_stats()
    multi = brain.multi_experience_stats()
    print(
        f"summary: mode={mode} isolated={recovered_mae:.4f} sequence={sequence_mae:.4f} "
        f"bridges={int(bridge['frozen_bridges'])} "
        f"applications={int(bridge['bridge_application_events'])} "
        f"target-traversals={int(bridge['bridge_target_edges_traversed'])} "
        f"dormant-skipped={int(bridge['bridge_dormant_edges_skipped'])} "
        f"new-isolated-routes={len(recovered_edges - pre_edges)} "
        f"new-sequence-routes={len(sequence_edges - pre_edges)} "
        f"teacher-wakes={int(measured['teacher_direct_reactivations_total'])}\n"
    )
    return {
        "mode": mode,
        "pre_mae": pre_mae,
        "damaged_mae": damaged_mae,
        "isolated_mae": recovered_mae,
        "sequence_mae": sequence_mae,
        "new_isolated_routes": float(len(recovered_edges - pre_edges)),
        "new_sequence_routes": float(len(sequence_edges - pre_edges)),
        **bridge,
        **multi,
    }


def main() -> None:
    print("SphereBrain experience-cluster bridge experiment v15b")
    print("comparison: no bridge / learned bridge / shuffled false bridge")
    print(f"pretraining={PRETRAIN_EPOCHS}, warmup={WARMUP_RECOVERY_EPOCHS}, assisted={BRIDGE_RECOVERY_EPOCHS}")
    print(f"maximum transient facilitation={BRIDGE_FACILITATION:.1%}")
    print("no physical edge, no stored-weight change, no dormant-edge boost")
    print("bridge runtime is disabled during teacher propagation\n")

    results = [run_condition(mode) for mode in MODES]
    print("=" * 150)
    print("v15b causal bridge comparison")
    print("=" * 150)
    print(
        "mode     | before | damaged | isolated | sequence | bridges | applications | target traversals | new isolated | new sequence"
    )
    for row in results:
        print(
            f"{row['mode']:8s} | {row['pre_mae']:6.4f} | {row['damaged_mae']:7.4f} | "
            f"{row['isolated_mae']:8.4f} | {row['sequence_mae']:8.4f} | "
            f"{int(row['frozen_bridges']):7d} | {int(row['bridge_application_events']):12d} | "
            f"{int(row['bridge_target_edges_traversed']):17d} | "
            f"{int(row['new_isolated_routes']):12d} | {int(row['new_sequence_routes']):12d}"
        )


if __name__ == "__main__":
    main()
