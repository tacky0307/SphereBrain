from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_dormant_pathway_recovery_experiment as base
from dormant_surface_flow_v13 import PromotedContributionTrackingBrain
from experiments.run_pathway_recovery_experiment import (
    CANDIDATE_Y,
    DECODER_POWER,
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


PRETRAIN_EPOCH_OPTIONS = (20, 50, 100, 200)
LESION_FRACTION = 0.10
ACTUAL_CANDIDATE_WIDTH = 50
DISTINCT_INPUTS_REQUIRED = 2
MAX_PROMOTIONS_PER_EPOCH = 10
MAX_PROMOTIONS_TOTAL = 100


def build_brain() -> PromotedContributionTrackingBrain:
    return PromotedContributionTrackingBrain(
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
        max_candidates_per_experience=ACTUAL_CANDIDATE_WIDTH,
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
        return 0.5, 0.0, set(result.traversed_edges)
    prediction = float(np.sum(CANDIDATE_Y * weights) / total)
    confidence = float(np.max(weights) / total)
    return prediction, confidence, set(result.traversed_edges)


def evaluate(label, brain, input_encoder, output_encoder, examples, verbose=True):
    errors: list[float] = []
    traversed: set[tuple[int, int]] = set()
    if verbose:
        print(label)
    for example in examples:
        prediction, confidence, edges = predict(
            brain, example.x, input_encoder, output_encoder
        )
        error = abs(prediction - example.y)
        errors.append(error)
        traversed.update(edges)
        if verbose:
            promoted_used = len(edges & brain.selective_promoted_edges)
            print(
                f"x={example.x:.3f} expected={example.y:.3f} "
                f"predicted={prediction:.3f} error={error:.3f} "
                f"confidence={confidence:.4f} promoted_edges_used={promoted_used}"
            )
    mae = float(np.mean(errors))
    if verbose:
        print(f"MAE: {mae:.4f} | traversed unique edges: {len(traversed)}\n")
    return mae, traversed


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    minimum = min(example.x for example in examples)
    maximum = max(example.x for example in examples)
    brain.set_contribution_phase("recovery")

    for epoch in range(epochs):
        brain.begin_recovery_epoch()
        for example in examples:
            region = base.input_region(example.x, minimum, maximum)
            brain.set_experience(region, example.x)
            observe(brain, input_encoder.encode(example.x))
            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )
        measured = brain.recovery_measurement_stats()
        contribution = brain.promoted_contribution_stats()
        print(
            f"epoch {epoch + 1:02d}: "
            f"promotions={int(measured['promotions_this_epoch'])} "
            f"total={int(measured['selective_promotions_total'])} "
            f"shared2+={int(measured['candidate_paths_two_or_more_distinct_inputs'])} "
            f"candidate_mean={contribution['candidate_mean']:.5f} "
            f"first_post_mean={contribution['first_post_mean']:.5f} "
            f"recovery_mean={contribution['recovery_mean']:.5f}"
        )

    brain.set_experience(None, None)
    brain.set_contribution_phase("idle")
    print()


def measurement_pass(brain, phase, input_encoder, examples):
    brain.set_contribution_phase(phase)
    for example in examples:
        observe(brain, input_encoder.encode(example.x))
    brain.set_contribution_phase("idle")


def print_edge_rows(brain):
    rows = brain.promoted_edge_contribution_rows()
    if not rows:
        print("promoted contribution detail: no promoted pathways")
        return
    print("promoted contribution detail")
    print(
        "edge       | candidate | first post | recovery avg | "
        "recovery-end | final eval | final n"
    )
    for row in rows:
        print(
            f"{int(row['source']):03d}->{int(row['target']):03d} | "
            f"{float(row['candidate']):9.5f} | "
            f"{float(row['first_post']):10.5f} | "
            f"{float(row['recovery_mean']):12.5f} | "
            f"{float(row['recovery_end_mean']):12.5f} | "
            f"{float(row['final_eval_mean']):10.5f} | "
            f"{int(row['final_samples']):7d}"
        )


def run_pretrain_condition(pretrain_epochs: int) -> dict[str, float]:
    brain = build_brain()
    input_encoder, output_encoder = build_encoders(brain)
    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    print("=" * 92)
    print(f"pretraining condition: {pretrain_epochs} epochs")
    print("=" * 92)

    train(brain, input_encoder, output_encoder, training, pretrain_epochs)
    base.collect_prelesion_baseline(brain, input_encoder, training + testing)
    pre_mae, pre_edges = evaluate(
        "--- learned state before lesion ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )

    disabled = brain.lesion_most_used_edges(
        fraction=LESION_FRACTION, bidirectional=True
    )
    stats = brain.pathway_stats()
    print(
        f"lesion: disabled {len(disabled)} directed edges "
        f"({LESION_FRACTION:.0%} of used-edge ranking, reverse directions included)"
    )
    print(
        f"pathways now: enabled={int(stats['enabled_edges'])} "
        f"disabled={int(stats['disabled_edges'])} "
        f"mean_enabled_weight={stats['mean_enabled_weight']:.4f}\n"
    )

    damaged_mae, _ = evaluate(
        "--- immediately after lesion ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )

    brain.set_recovery_mode(True)
    print(
        "recovery mode: ON "
        "(top50; 2 distinct inputs in same region; max 10/epoch; max 100 total)\n"
    )
    recovery_train(
        brain, input_encoder, output_encoder, training, RECOVERY_EPOCHS
    )
    brain.set_recovery_mode(False)

    # A dedicated observation pass measures the promoted paths after training,
    # without adding teacher reinforcement.
    measurement_pass(brain, "recovery_end", input_encoder, training + testing)

    brain.set_contribution_phase("final_eval")
    recovered_mae, recovered_edges = evaluate(
        f"--- after {RECOVERY_EPOCHS} recovery epochs ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )
    brain.set_contribution_phase("idle")

    measured = brain.recovery_measurement_stats()
    contribution = brain.promoted_contribution_stats()
    damage = damaged_mae - pre_mae
    recovered_amount = damaged_mae - recovered_mae
    recovery_ratio = 0.0 if damage <= 1e-12 else recovered_amount / damage

    print("condition summary")
    print(f"pretraining epochs:                 {pretrain_epochs}")
    print(f"learned-state MAE:                  {pre_mae:.4f}")
    print(f"immediately damaged MAE:            {damaged_mae:.4f}")
    print(f"recovered MAE:                      {recovered_mae:.4f}")
    print(f"damage increase:                    {damage:+.4f}")
    print(f"recovered amount:                   {recovered_amount:+.4f}")
    print(f"recovery ratio:                     {recovery_ratio:.1%}")
    print(f"promoted pathways:                  {int(contribution['promoted_edges'])}")
    print(f"candidate contribution mean:        {contribution['candidate_mean']:.5f}")
    print(f"first post-promotion mean:           {contribution['first_post_mean']:.5f}")
    print(f"recovery contribution mean:         {contribution['recovery_mean']:.5f}")
    print(f"recovery-end contribution mean:     {contribution['recovery_end_mean']:.5f}")
    print(f"final-evaluation contribution mean: {contribution['final_eval_mean']:.5f}")
    print(
        "promoted seen at recovery end:      "
        f"{int(contribution['promoted_seen_recovery_end'])}/"
        f"{int(contribution['promoted_edges'])}"
    )
    print(
        "promoted seen in final evaluation:  "
        f"{int(contribution['promoted_seen_final_eval'])}/"
        f"{int(contribution['promoted_edges'])}"
    )
    print(
        f"teacher-direct reactivations:       "
        f"{int(measured['teacher_direct_reactivations_total'])}"
    )
    print(
        f"teacher attempts blocked dormant:  "
        f"{int(measured['teacher_blocked_dormant_total'])}"
    )
    print(f"newly traversed evaluation routes:  {len(recovered_edges - pre_edges)}")
    print_edge_rows(brain)
    print()

    return {
        "epochs": float(pretrain_epochs),
        "pre_mae": pre_mae,
        "damaged_mae": damaged_mae,
        "recovered_mae": recovered_mae,
        "damage": damage,
        "recovered_amount": recovered_amount,
        "recovery_ratio": recovery_ratio,
        "promoted": contribution["promoted_edges"],
        "candidate_mean": contribution["candidate_mean"],
        "first_post_mean": contribution["first_post_mean"],
        "recovery_mean": contribution["recovery_mean"],
        "recovery_end_mean": contribution["recovery_end_mean"],
        "final_eval_mean": contribution["final_eval_mean"],
    }


def print_comparison(results):
    print("=" * 142)
    print("v13 learning-duration and promoted-contribution comparison")
    print("=" * 142)
    print(
        "epochs | learned MAE | damaged | recovered | damage | recovery | ratio | "
        "promoted | candidate | first-post | recovery | recovery-end | final-eval"
    )
    for row in results:
        print(
            f"{int(row['epochs']):6d} | "
            f"{row['pre_mae']:11.4f} | "
            f"{row['damaged_mae']:7.4f} | "
            f"{row['recovered_mae']:9.4f} | "
            f"{row['damage']:+.4f} | "
            f"{row['recovered_amount']:+.4f} | "
            f"{row['recovery_ratio']:6.1%} | "
            f"{int(row['promoted']):8d} | "
            f"{row['candidate_mean']:9.5f} | "
            f"{row['first_post_mean']:10.5f} | "
            f"{row['recovery_mean']:8.5f} | "
            f"{row['recovery_end_mean']:12.5f} | "
            f"{row['final_eval_mean']:10.5f}"
        )


def main() -> None:
    print("SphereBrain learning-duration and promoted-contribution experiment v13")
    print("task: y = 2x")
    print("pretraining comparison: 20, 50, 100, 200 epochs")
    print(f"lesion fixed at {LESION_FRACTION:.0%}; recovery={RECOVERY_EPOCHS} epochs")
    print("actual recovery candidate width: top 50")
    print("promotion: 2 distinct input values inside the same region")
    print("contribution stages: candidate, first-post, recovery, recovery-end, final-eval")
    print("teacher direct wake of dormant pathways remains blocked\n")

    results = [run_pretrain_condition(epochs) for epochs in PRETRAIN_EPOCH_OPTIONS]
    print_comparison(results)


if __name__ == "__main__":
    main()
