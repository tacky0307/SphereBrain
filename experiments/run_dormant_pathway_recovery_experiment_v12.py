from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment as base
from dormant_surface_flow_v11 import Top50SharedExperienceRecoveryBrain
from experiments.run_pathway_recovery_experiment import (
    CANDIDATE_Y,
    DECODER_POWER,
    PRETRAIN_EPOCHS,
    RECOVERY_EPOCHS,
    TEST_X,
    TRAIN_X,
    Example,
    build_encoders,
    candidate_score,
    observe,
    output_node_energy,
    train,
)

LESION_FRACTIONS = (0.10, 0.15, 0.20)
MAX_PROMOTIONS_PER_EPOCH = 10
MAX_PROMOTIONS_TOTAL = 100
DISTINCT_INPUTS_REQUIRED = 2
ACTUAL_CANDIDATE_WIDTH = 50


def build_brain() -> Top50SharedExperienceRecoveryBrain:
    return Top50SharedExperienceRecoveryBrain(
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
        max_promotions_per_epoch=MAX_PROMOTIONS_PER_EPOCH,
        max_promotions_total=MAX_PROMOTIONS_TOTAL,
        distinct_inputs_required=DISTINCT_INPUTS_REQUIRED,
    )


def predict(brain, x, input_encoder, output_encoder):
    result = observe(brain, input_encoder.encode(x))
    energy = output_node_energy(brain, result)
    energy_by_node = {
        node: float(energy[index])
        for index, node in enumerate(brain.output_nodes)
    }
    scored = np.array(
        [
            candidate_score(energy_by_node, output_encoder.encode(float(y)))
            for y in CANDIDATE_Y
        ],
        dtype=float,
    )
    weights = np.clip(scored, 0.0, None) ** DECODER_POWER
    total = float(np.sum(weights))
    if total <= 1e-12:
        prediction, confidence = 0.5, 0.0
    else:
        prediction = float(np.sum(CANDIDATE_Y * weights) / total)
        confidence = float(np.max(weights) / total)
    return prediction, confidence, set(result.traversed_edges)


def evaluate(label, brain, input_encoder, output_encoder, examples, promoted_edges=None):
    errors: list[float] = []
    traversed: set[tuple[int, int]] = set()
    promoted_edges = set() if promoted_edges is None else set(promoted_edges)
    promoted_unique: set[tuple[int, int]] = set()
    promoted_traversal_events = 0
    inputs_using_promoted = 0

    print(label)
    for example in examples:
        prediction, confidence, edges = predict(
            brain, example.x, input_encoder, output_encoder
        )
        error = abs(prediction - example.y)
        errors.append(error)
        traversed.update(edges)

        used_promoted = edges & promoted_edges
        promoted_unique.update(used_promoted)
        promoted_traversal_events += len(used_promoted)
        if used_promoted:
            inputs_using_promoted += 1

        print(
            f"x={example.x:.3f} expected={example.y:.3f} "
            f"predicted={prediction:.3f} error={error:.3f} "
            f"confidence={confidence:.4f} promoted_edges_used={len(used_promoted)}"
        )

    mae = float(np.mean(errors))
    print(f"MAE: {mae:.4f} | traversed unique edges: {len(traversed)}")
    if promoted_edges:
        print(
            f"promoted traversal: unique={len(promoted_unique)}/{len(promoted_edges)} "
            f"events={promoted_traversal_events} "
            f"inputs_using_promoted={inputs_using_promoted}/{len(examples)}"
        )
    print()
    return {
        "mae": mae,
        "edges": traversed,
        "promoted_unique": promoted_unique,
        "promoted_events": promoted_traversal_events,
        "inputs_using_promoted": inputs_using_promoted,
    }


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    minimum = min(example.x for example in examples)
    maximum = max(example.x for example in examples)
    recovery_promoted_unique: set[tuple[int, int]] = set()
    recovery_promoted_events = 0
    recovery_inputs_using_promoted = 0

    for epoch in range(epochs):
        brain.begin_recovery_epoch()
        epoch_unique: set[tuple[int, int]] = set()
        epoch_events = 0
        epoch_inputs = 0

        for example in examples:
            region = base.input_region(example.x, minimum, maximum)
            brain.set_experience(region, example.x)

            result = observe(brain, input_encoder.encode(example.x))
            edges = set(result.traversed_edges)
            promoted = set(brain.selective_promoted_edges)
            used_promoted = edges & promoted
            epoch_unique.update(used_promoted)
            epoch_events += len(used_promoted)
            if used_promoted:
                epoch_inputs += 1

            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )

        recovery_promoted_unique.update(epoch_unique)
        recovery_promoted_events += epoch_events
        recovery_inputs_using_promoted += epoch_inputs

        measured = brain.recovery_measurement_stats()
        print(
            f"epoch {epoch + 1:02d}: "
            f"promotions={int(measured['promotions_this_epoch'])} "
            f"total={int(measured['selective_promotions_total'])} "
            f"shared2+={int(measured['candidate_paths_two_or_more_distinct_inputs'])} "
            f"promoted_used_unique={len(epoch_unique)} "
            f"promoted_use_events={epoch_events} "
            f"inputs_using_promoted={epoch_inputs}/{len(examples)}"
        )

    brain.set_experience(None, None)
    print(
        "recovery-training promoted traversal totals: "
        f"unique={len(recovery_promoted_unique)}/"
        f"{len(brain.selective_promoted_edges)} "
        f"events={recovery_promoted_events} "
        f"inputs_using_promoted={recovery_inputs_using_promoted}/"
        f"{len(examples) * epochs}\n"
    )
    return {
        "unique": recovery_promoted_unique,
        "events": recovery_promoted_events,
        "inputs": recovery_inputs_using_promoted,
    }


def run_fraction(lesion_fraction: float) -> dict[str, float]:
    brain = build_brain()
    input_encoder, output_encoder = build_encoders(brain)
    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    print("=" * 76)
    print(f"lesion condition: {lesion_fraction:.0%}")
    print("=" * 76)

    train(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    base.collect_prelesion_baseline(brain, input_encoder, training + testing)
    before = evaluate(
        "--- before lesion ---", brain, input_encoder, output_encoder, testing
    )

    reactivations_before = int(brain.pathway_state_stats()["reactivations"])
    disabled = brain.lesion_most_used_edges(
        fraction=lesion_fraction, bidirectional=True
    )
    stats = brain.pathway_stats()
    print(
        f"lesion: disabled {len(disabled)} directed edges "
        f"({lesion_fraction:.0%} of used-edge ranking, reverse directions included)"
    )
    print(
        f"pathways now: enabled={int(stats['enabled_edges'])} "
        f"disabled={int(stats['disabled_edges'])} "
        f"mean_enabled_weight={stats['mean_enabled_weight']:.4f}\n"
    )

    damaged = evaluate(
        "--- immediately after lesion ---", brain, input_encoder, output_encoder, testing
    )

    brain.set_recovery_mode(True)
    print(
        "recovery mode: ON (top50; 2 distinct inputs in same region; "
        "max 10 promotions/epoch; max 100 total)\n"
    )
    recovery_usage = recovery_train(
        brain, input_encoder, output_encoder, training, RECOVERY_EPOCHS
    )
    brain.set_recovery_mode(False)

    promoted = set(brain.selective_promoted_edges)
    recovered = evaluate(
        f"--- after {RECOVERY_EPOCHS} recovery epochs ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
        promoted_edges=promoted,
    )

    measured = brain.recovery_measurement_stats()
    reactivations_after = int(brain.pathway_state_stats()["reactivations"])
    damage = damaged["mae"] - before["mae"]
    recovered_amount = damaged["mae"] - recovered["mae"]
    recovery_ratio = 0.0 if damage <= 1e-12 else recovered_amount / damage

    print("condition summary")
    print(f"before lesion MAE:                 {before['mae']:.4f}")
    print(f"immediately damaged MAE:           {damaged['mae']:.4f}")
    print(f"recovered MAE:                     {recovered['mae']:.4f}")
    print(f"damage increase:                   {damage:+.4f}")
    print(f"recovered amount:                  {recovered_amount:+.4f}")
    print(f"recovery ratio:                    {recovery_ratio:.1%}")
    print(f"promoted pathways:                 {len(promoted)}")
    print(
        f"promoted used during recovery:     "
        f"{len(recovery_usage['unique'])}/{len(promoted)} unique, "
        f"{recovery_usage['events']} events"
    )
    print(
        f"promoted used in final evaluation: "
        f"{len(recovered['promoted_unique'])}/{len(promoted)} unique, "
        f"{recovered['promoted_events']} events"
    )
    print(
        f"test inputs using promoted paths:  "
        f"{recovered['inputs_using_promoted']}/{len(testing)}"
    )
    print(f"newly traversed evaluation routes: {len(recovered['edges'] - before['edges'])}")
    print(f"teacher-direct reactivations:      {int(measured['teacher_direct_reactivations_total'])}")
    print(f"teacher attempts blocked dormant:  {int(measured['teacher_blocked_dormant_total'])}")
    print(f"all dormant reactivations:         {reactivations_after - reactivations_before}\n")

    return {
        "lesion": lesion_fraction,
        "before": before["mae"],
        "damaged": damaged["mae"],
        "recovered": recovered["mae"],
        "damage": damage,
        "recovered_amount": recovered_amount,
        "recovery_ratio": recovery_ratio,
        "promoted": len(promoted),
        "recovery_used_unique": len(recovery_usage["unique"]),
        "recovery_use_events": recovery_usage["events"],
        "eval_used_unique": len(recovered["promoted_unique"]),
        "eval_use_events": recovered["promoted_events"],
        "eval_inputs": recovered["inputs_using_promoted"],
        "new_routes": len(recovered["edges"] - before["edges"]),
        "teacher_direct": int(measured["teacher_direct_reactivations_total"]),
    }


def print_comparison(rows: list[dict[str, float]]) -> None:
    print("=" * 112)
    print("v12 lesion severity comparison")
    print("=" * 112)
    print(
        "lesion | before | damaged | recovered | damage | recovered amt | ratio | "
        "promoted | used train | used eval | eval events | new routes"
    )
    for row in rows:
        print(
            f"{row['lesion']:>6.0%} | "
            f"{row['before']:.4f} | {row['damaged']:.4f} | {row['recovered']:.4f} | "
            f"{row['damage']:+.4f} | {row['recovered_amount']:+.4f} | "
            f"{row['recovery_ratio']:>6.1%} | "
            f"{int(row['promoted']):>8} | "
            f"{int(row['recovery_used_unique']):>4}/{int(row['promoted']):<3} | "
            f"{int(row['eval_used_unique']):>4}/{int(row['promoted']):<3} | "
            f"{int(row['eval_use_events']):>11} | "
            f"{int(row['new_routes']):>10}"
        )
    print()


def main() -> None:
    print("SphereBrain lesion severity and promoted-path traversal experiment v12")
    print("task: y = 2x")
    print(f"pretrain={PRETRAIN_EPOCHS} epochs, recovery={RECOVERY_EPOCHS} epochs")
    print("lesion comparison: 10%, 15%, 20%")
    print("actual recovery candidate width: top 50")
    print("promotion: 2 distinct input values inside the same region")
    print("tracks promoted-path traversal during recovery and final evaluation")
    print("teacher direct wake of dormant pathways remains blocked\n")

    rows = [run_fraction(fraction) for fraction in LESION_FRACTIONS]
    print_comparison(rows)


if __name__ == "__main__":
    main()
