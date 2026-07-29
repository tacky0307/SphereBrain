from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dormant_surface_flow import DormantSurfaceFlowBrain
from dormant_surface_flow_v7 import DiverseRegionCappedRecoveryBrain
from surface_flow import SurfaceFlowBrain
from experiments.run_pathway_recovery_experiment import (
    CANDIDATE_Y,
    DECODER_POWER,
    LESION_FRACTION,
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


INPUT_REGION_COUNT = 3
MAX_PROMOTIONS_PER_EPOCH = 10
MAX_PROMOTIONS_TOTAL = 100


def build_ordinary_brain() -> SurfaceFlowBrain:
    return SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        bidirectional_plasticity_enabled=False,
    )


def build_dormant_brain() -> DiverseRegionCappedRecoveryBrain:
    return DiverseRegionCappedRecoveryBrain(
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
        max_promotions_per_epoch=MAX_PROMOTIONS_PER_EPOCH,
        max_promotions_total=MAX_PROMOTIONS_TOTAL,
    )


def input_region(x: float, minimum: float, maximum: float) -> int:
    if maximum <= minimum:
        return 0
    normalized = float(np.clip((x - minimum) / (maximum - minimum), 0.0, 1.0))
    return min(INPUT_REGION_COUNT - 1, int(normalized * INPUT_REGION_COUNT))


def predict_absolute(brain, x, input_encoder, output_encoder):
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


def evaluate_absolute(label, brain, input_encoder, output_encoder, examples):
    errors: list[float] = []
    traversed: set[tuple[int, int]] = set()
    print(label)
    for example in examples:
        prediction, confidence, edges = predict_absolute(
            brain, example.x, input_encoder, output_encoder
        )
        error = abs(prediction - example.y)
        errors.append(error)
        traversed.update(edges)
        print(
            f"x={example.x:.3f} expected={example.y:.3f} "
            f"predicted={prediction:.3f} error={error:.3f} "
            f"confidence={confidence:.4f}"
        )
    mae = float(np.mean(errors))
    print(f"MAE: {mae:.4f} | traversed unique edges: {len(traversed)}\n")
    return mae, traversed


def collect_prelesion_baseline(brain, input_encoder, examples) -> None:
    if not isinstance(brain, DiverseRegionCappedRecoveryBrain):
        return
    brain.begin_prelesion_baseline_collection()
    for example in examples:
        observe(brain, input_encoder.encode(example.x))
    brain.end_prelesion_baseline_collection()


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    minimum = min(example.x for example in examples)
    maximum = max(example.x for example in examples)
    for epoch in range(epochs):
        if isinstance(brain, DiverseRegionCappedRecoveryBrain):
            brain.begin_recovery_epoch()
        for example in examples:
            if isinstance(brain, DiverseRegionCappedRecoveryBrain):
                region = input_region(example.x, minimum, maximum)
                brain.set_experience_region(region)
                # Candidate selection happens before teacher reinforcement.
                observe(brain, input_encoder.encode(example.x))
            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )
        if isinstance(brain, DiverseRegionCappedRecoveryBrain):
            measured = brain.recovery_measurement_stats()
            print(
                f"epoch {epoch + 1:02d}: promotions={int(measured['promotions_this_epoch'])} "
                f"total={int(measured['selective_promotions_total'])} "
                f"pending={int(measured['pending_candidate_edges'])}"
            )
    if isinstance(brain, DiverseRegionCappedRecoveryBrain):
        brain.set_experience_region(None)
    print()


def print_state_stats(brain: DormantSurfaceFlowBrain, label: str) -> None:
    stats = brain.pathway_state_stats()
    print(label)
    print(
        f"normal={int(stats['normal'])} protected={int(stats['protected'])} "
        f"dormant={int(stats['dormant'])} reactivations={int(stats['reactivations'])} "
        f"recovery_mode={'ON' if stats['recovery_mode'] else 'OFF'}"
    )
    if isinstance(brain, DiverseRegionCappedRecoveryBrain):
        measured = brain.recovery_measurement_stats()
        print(
            f"candidate_events={int(measured['candidate_selection_events_total'])} "
            f"unique_candidates={int(measured['candidate_unique_edges_total'])} "
            f"promotions={int(measured['selective_promotions_total'])} "
            f"pending={int(measured['pending_candidate_edges'])}"
        )
        print(
            f"duplicate_same_region_ignored={int(measured['duplicate_region_selections_ignored'])} "
            f"epoch_cap_blocks={int(measured['epoch_promotion_cap_blocks'])} "
            f"total_cap_blocks={int(measured['total_promotion_cap_blocks'])}"
        )
        print(
            f"teacher_direct_wakes={int(measured['teacher_direct_reactivations_total'])} "
            f"teacher_blocked_dormant={int(measured['teacher_blocked_dormant_total'])} "
            f"baseline_mean_active_edges={int(measured['baseline_mean_active_edges'])} "
            f"baseline_mean_contribution={measured['baseline_mean_contribution']:.4f}"
        )
    print()


def run_condition(name: str, brain: SurfaceFlowBrain) -> None:
    input_encoder, output_encoder = build_encoders(brain)
    training = [Example(float(x), float(2.0 * x)) for x in TRAIN_X]
    testing = [Example(float(x), float(2.0 * x)) for x in TEST_X]

    print("=" * 76)
    print(f"condition: {name}")
    print("=" * 76)

    train(brain, input_encoder, output_encoder, training, PRETRAIN_EPOCHS)
    collect_prelesion_baseline(brain, input_encoder, training + testing)
    pre_mae, pre_edges = evaluate_absolute(
        "--- before lesion ---", brain, input_encoder, output_encoder, testing
    )
    if isinstance(brain, DormantSurfaceFlowBrain):
        print_state_stats(brain, "pathway states before lesion")
        reactivations_before = int(brain.pathway_state_stats()["reactivations"])
    else:
        reactivations_before = 0

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

    damaged_mae, _ = evaluate_absolute(
        "--- immediately after lesion ---", brain, input_encoder, output_encoder, testing
    )

    if isinstance(brain, DormantSurfaceFlowBrain):
        brain.set_recovery_mode(True)
        print(
            "recovery mode: ON (3 distinct input regions; max 10 promotions/epoch; max 100 total)\n"
        )

    recovery_train(brain, input_encoder, output_encoder, training, RECOVERY_EPOCHS)

    if isinstance(brain, DormantSurfaceFlowBrain):
        brain.set_recovery_mode(False)

    recovered_mae, recovered_edges = evaluate_absolute(
        f"--- after {RECOVERY_EPOCHS} recovery epochs ---",
        brain,
        input_encoder,
        output_encoder,
        testing,
    )

    if isinstance(brain, DormantSurfaceFlowBrain):
        print_state_stats(brain, "pathway states after recovery")
        reactivations_after = int(brain.pathway_state_stats()["reactivations"])
    else:
        reactivations_after = 0

    damage = damaged_mae - pre_mae
    recovered_amount = damaged_mae - recovered_mae
    recovery_ratio = 0.0 if damage <= 1e-12 else recovered_amount / damage

    print("summary")
    print(f"before lesion MAE:       {pre_mae:.4f}")
    print(f"immediately damaged MAE: {damaged_mae:.4f}")
    print(f"recovered MAE:           {recovered_mae:.4f}")
    print(f"damage increase:         {damage:+.4f}")
    print(f"recovered amount:        {recovered_amount:+.4f}")
    print(f"recovery ratio:          {recovery_ratio:.1%}")
    print(f"retained pre-lesion routes: {len(pre_edges & recovered_edges)}")
    print(f"newly traversed routes:     {len(recovered_edges - pre_edges)}")

    if isinstance(brain, DiverseRegionCappedRecoveryBrain):
        measured = brain.recovery_measurement_stats()
        print("recovery pathway measurements")
        print(f"candidate selection events:       {int(measured['candidate_selection_events_total'])}")
        print(f"unique candidate pathways:        {int(measured['candidate_unique_edges_total'])}")
        print(f"promoted from 3 distinct regions: {int(measured['selective_promotions_total'])}")
        print(f"same-region repeats ignored:      {int(measured['duplicate_region_selections_ignored'])}")
        print(f"epoch-cap promotion blocks:       {int(measured['epoch_promotion_cap_blocks'])}")
        print(f"total-cap promotion blocks:       {int(measured['total_promotion_cap_blocks'])}")
        print(f"teacher-direct reactivations:     {int(measured['teacher_direct_reactivations_total'])}")
        print(f"teacher attempts blocked dormant: {int(measured['teacher_blocked_dormant_total'])}")
        print(f"candidates still pending:         {int(measured['pending_candidate_edges'])}")
        print(f"all dormant reactivations:        {reactivations_after - reactivations_before}")
    print()


def main() -> None:
    print("SphereBrain diverse-region capped dormant recovery experiment v7")
    print("task: y = 2x")
    print(
        f"pretrain={PRETRAIN_EPOCHS} epochs, lesion={LESION_FRACTION:.0%}, "
        f"recovery={RECOVERY_EPOCHS} epochs"
    )
    print("candidate threshold=max(0.04, pre-lesion mean + 0.02)")
    print("top 20 candidates per experience")
    print("promotion requires 3 distinct input regions; same-region repeats count once")
    print("promotion caps: max 10 per epoch, max 100 total")
    print("teacher reinforcement only after promotion; dormant direct wake is blocked\n")

    run_condition("ordinary reinforcement", build_ordinary_brain())
    run_condition("diverse-region capped dormant recovery v7", build_dormant_brain())


if __name__ == "__main__":
    main()
