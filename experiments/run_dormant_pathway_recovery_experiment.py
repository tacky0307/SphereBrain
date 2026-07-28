from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dormant_surface_flow import DormantSurfaceFlowBrain
from dormant_surface_flow_v6 import MeanBaselineStagedRecoveryBrain
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


def build_ordinary_brain() -> SurfaceFlowBrain:
    return SurfaceFlowBrain(
        node_count=600,
        neighbors_per_node=8,
        seed=42,
        bidirectional_plasticity_enabled=False,
    )


def build_dormant_brain() -> MeanBaselineStagedRecoveryBrain:
    return MeanBaselineStagedRecoveryBrain(
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
    )


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
    if not isinstance(brain, MeanBaselineStagedRecoveryBrain):
        return
    brain.begin_prelesion_baseline_collection()
    for example in examples:
        observe(brain, input_encoder.encode(example.x))
    brain.end_prelesion_baseline_collection()


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    for _ in range(epochs):
        for example in examples:
            # Stage 1: signal propagation selects candidates and may promote an
            # edge after its third selection.
            if isinstance(brain, MeanBaselineStagedRecoveryBrain):
                observe(brain, input_encoder.encode(example.x))
            # Stage 2: teacher reinforcement runs afterwards. The v6 brain blocks
            # dormant edges here; only already-promoted protected edges can learn.
            brain.experience(
                input_encoder.encode(example.x),
                output_encoder.encode(example.y),
            )


def print_state_stats(brain: DormantSurfaceFlowBrain, label: str) -> None:
    stats = brain.pathway_state_stats()
    print(label)
    print(
        f"normal={int(stats['normal'])} protected={int(stats['protected'])} "
        f"dormant={int(stats['dormant'])} reactivations={int(stats['reactivations'])} "
        f"recovery_mode={'ON' if stats['recovery_mode'] else 'OFF'}"
    )
    if isinstance(brain, MeanBaselineStagedRecoveryBrain):
        measured = brain.recovery_measurement_stats()
        print(
            f"candidate_events={int(measured['candidate_selection_events_total'])} "
            f"unique_candidates={int(measured['candidate_unique_edges_total'])} "
            f"promotions={int(measured['selective_promotions_total'])} "
            f"pending={int(measured['pending_candidate_edges'])}"
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
            "recovery mode: ON (mean+margin candidates first; teacher cannot wake dormant edges)\n"
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

    if isinstance(brain, MeanBaselineStagedRecoveryBrain):
        measured = brain.recovery_measurement_stats()
        print("recovery pathway measurements")
        print(
            "candidate selection events:       "
            f"{int(measured['candidate_selection_events_total'])}"
        )
        print(
            "unique candidate pathways:        "
            f"{int(measured['candidate_unique_edges_total'])}"
        )
        print(
            "promoted after 3 selections:       "
            f"{int(measured['selective_promotions_total'])}"
        )
        print(
            "teacher-direct reactivations:      "
            f"{int(measured['teacher_direct_reactivations_total'])}"
        )
        print(
            "teacher attempts blocked dormant: "
            f"{int(measured['teacher_blocked_dormant_total'])}"
        )
        print(
            "candidates still pending:          "
            f"{int(measured['pending_candidate_edges'])}"
        )
        print(
            "all dormant reactivations:         "
            f"{reactivations_after - reactivations_before}"
        )
    print()


def main() -> None:
    print("SphereBrain mean-baseline staged dormant recovery experiment v6")
    print("task: y = 2x")
    print(
        f"pretrain={PRETRAIN_EPOCHS} epochs, lesion={LESION_FRACTION:.0%}, "
        f"recovery={RECOVERY_EPOCHS} epochs"
    )
    print("candidate threshold=max(0.04, pre-lesion mean + 0.02)")
    print("top 20 candidates per experience; promote after 3 selections")
    print("strict order: dormant -> candidate -> protected -> teacher reinforcement")
    print("teacher reinforcement cannot directly wake dormant pathways\n")

    run_condition("ordinary reinforcement", build_ordinary_brain())
    run_condition("mean-baseline staged dormant recovery v6", build_dormant_brain())


if __name__ == "__main__":
    main()
