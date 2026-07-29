from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dormant_surface_flow_v15 import MultiExperienceTransitionBrain
from experiments.run_dormant_pathway_recovery_experiment import input_region
from experiments.run_pathway_recovery_experiment import (
    CANDIDATE_Y,
    DECODER_POWER,
    PATTERN_WIDTH,
    candidate_score,
    observe,
    output_node_energy,
)
from surface_encoders import ScalarSurfaceEncoder, ordered_surface_nodes


@dataclass(frozen=True)
class MultiExample:
    task_id: int
    task_name: str
    x: float
    y: float


TASKS = (
    (0, "double", lambda x: 2.0 * x),
    (1, "square", lambda x: 4.0 * x * x),
    (2, "inverse", lambda x: 1.0 - 2.0 * x),
)
TRAIN_X = np.arange(0.00, 0.5001, 0.05)
TEST_X = np.arange(0.025, 0.5000, 0.05)
PRETRAIN_EPOCH_OPTIONS = (20, 50, 100)
RECOVERY_EPOCHS = 10
LESION_FRACTION = 0.10
RANDOM_SEED = 314159


class ContextualInputEncoder:
    """Encode value and environmental context on separate input-node bands."""

    def __init__(self, brain: MultiExperienceTransitionBrain) -> None:
        nodes = ordered_surface_nodes(brain.positions, brain.input_nodes)
        split = max(PATTERN_WIDTH + 2, int(len(nodes) * 0.72))
        value_nodes = nodes[:split]
        context_nodes = nodes[split:]
        if len(context_nodes) < PATTERN_WIDTH:
            raise ValueError("not enough input nodes for contextual encoding")
        self.value_encoder = ScalarSurfaceEncoder(value_nodes, width=PATTERN_WIDTH)
        self.context_encoder = ScalarSurfaceEncoder(context_nodes, width=PATTERN_WIDTH)
        self.context_values = {0: 0.0, 1: 0.5, 2: 1.0}

    def encode(self, task_id: int, x: float) -> dict[int, float]:
        pattern = dict(self.value_encoder.encode(float(x)))
        for node, activity in self.context_encoder.encode(self.context_values[task_id]).items():
            pattern[node] = pattern.get(node, 0.0) + 0.85 * float(activity)
        return pattern


def build_brain() -> MultiExperienceTransitionBrain:
    return MultiExperienceTransitionBrain(
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
    )


def build_examples(xs: np.ndarray) -> list[MultiExample]:
    examples: list[MultiExample] = []
    for task_id, task_name, function in TASKS:
        for x in xs:
            examples.append(
                MultiExample(task_id, task_name, float(x), float(np.clip(function(float(x)), 0.0, 1.0)))
            )
    return examples


def build_output_encoder(brain: MultiExperienceTransitionBrain) -> ScalarSurfaceEncoder:
    return ScalarSurfaceEncoder(
        ordered_surface_nodes(brain.positions, brain.output_nodes),
        width=PATTERN_WIDTH,
    )


def predict(brain, example, input_encoder, output_encoder):
    result = observe(brain, input_encoder.encode(example.task_id, example.x))
    energy = output_node_energy(brain, result)
    energy_by_node = {
        node: float(energy[index]) for index, node in enumerate(brain.output_nodes)
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
    return (
        float(np.sum(CANDIDATE_Y * weights) / total),
        float(np.max(weights) / total),
        set(result.traversed_edges),
    )


def evaluate(label, brain, input_encoder, output_encoder, examples, verbose=False):
    by_task: dict[str, list[float]] = {name: [] for _, name, _ in TASKS}
    traversed: set[tuple[int, int]] = set()
    if verbose:
        print(label)
    for example in examples:
        prediction, confidence, edges = predict(brain, example, input_encoder, output_encoder)
        error = abs(prediction - example.y)
        by_task[example.task_name].append(error)
        traversed.update(edges)
        if verbose:
            print(
                f"task={example.task_name:7s} x={example.x:.3f} expected={example.y:.3f} "
                f"predicted={prediction:.3f} error={error:.3f} confidence={confidence:.4f}"
            )
    task_mae = {name: float(np.mean(values)) for name, values in by_task.items()}
    overall = float(np.mean([value for values in by_task.values() for value in values]))
    print(
        f"{label}: overall={overall:.4f} "
        + " ".join(f"{name}={task_mae[name]:.4f}" for _, name, _ in TASKS)
        + f" routes={len(traversed)}"
    )
    return overall, task_mae, traversed


def shuffled_epoch(examples: list[MultiExample], epoch: int) -> list[MultiExample]:
    ordered = list(examples)
    random.Random(RANDOM_SEED + epoch).shuffle(ordered)
    return ordered


def pretrain(brain, input_encoder, output_encoder, examples, epochs):
    for epoch in range(epochs):
        for example in shuffled_epoch(examples, epoch):
            brain.experience(
                input_encoder.encode(example.task_id, example.x),
                output_encoder.encode(example.y),
            )


def collect_baseline(brain, input_encoder, examples):
    brain.begin_prelesion_baseline_collection()
    for example in examples:
        observe(brain, input_encoder.encode(example.task_id, example.x))
    brain.end_prelesion_baseline_collection()


def recovery_train(brain, input_encoder, output_encoder, examples, epochs):
    brain.set_contribution_phase("recovery")
    minimum, maximum = float(min(TRAIN_X)), float(max(TRAIN_X))
    for epoch in range(epochs):
        brain.begin_recovery_epoch()
        for example in shuffled_epoch(examples, 1000 + epoch):
            region = input_region(example.x, minimum, maximum)
            brain.set_multi_experience(
                region, example.task_id, example.task_name, example.x
            )
            pattern = input_encoder.encode(example.task_id, example.x)
            observe(brain, pattern)
            brain.experience(pattern, output_encoder.encode(example.y))
        measured = brain.recovery_measurement_stats()
        multi = brain.multi_experience_stats()
        print(
            f"epoch {epoch + 1:02d}: promotions={int(measured['promotions_this_epoch'])} "
            f"total={int(measured['selective_promotions_total'])} "
            f"clusters={int(multi['experience_clusters'])} "
            f"specialized={int(multi['specialized_clusters'])} "
            f"cross={int(multi['cross_experience_transitions'])} "
            f"candidates={int(multi['cross_experience_bridge_candidates'])}"
        )
    brain.set_multi_experience(None, None, None, None)
    brain.set_contribution_phase("idle")


def edge_label(edge):
    return f"{edge[0]:03d}->{edge[1]:03d}"


def print_profiles(brain):
    print("experience-cluster task specialization")
    for row in brain.cluster_experience_profiles():
        shares = row["task_shares"]
        share_text = ", ".join(
            f"{name}={float(shares.get(name, 0.0)):.2f}" for _, name, _ in TASKS
        )
        edges = row["edges"]
        preview = ", ".join(edge_label(edge) for edge in edges[:6])
        if len(edges) > 6:
            preview += ", ..."
        print(
            f"cluster {int(row['cluster_id']):02d}: size={int(row['size']):2d} "
            f"dominant={row['dominant_task']} share={float(row['dominant_share']):.3f} "
            f"specialization={float(row['specialization']):.3f} "
            f"[{share_text}] edges=[{preview}]"
        )


def print_transitions(brain):
    rows = brain.cross_experience_transition_rows()
    print("cross-experience directed transitions")
    if not rows:
        print("no cross-experience transitions")
        return
    print("from | to | task transition | events | pairs | epochs | lift | target | candidate")
    for row in rows[:25]:
        print(
            f"{int(row['source_cluster']):4d} | {int(row['target_cluster']):2d} | "
            f"{row['source_task']:7s}->{row['target_task']:7s} | "
            f"{int(row['events']):6d} | {int(row['distinct_input_pairs']):5d} | "
            f"{int(row['distinct_epochs']):6d} | {float(row['transition_lift']):5.2f} | "
            f"{float(row['target_strength_mean']):6.3f} | "
            f"{'YES' if int(row['candidate']) else 'no'}"
        )


def run_condition(pretrain_epochs: int) -> dict[str, float]:
    brain = build_brain()
    input_encoder = ContextualInputEncoder(brain)
    output_encoder = build_output_encoder(brain)
    training = build_examples(TRAIN_X)
    testing = build_examples(TEST_X)

    print("=" * 104)
    print(f"multi-experience pretraining: {pretrain_epochs} epochs")
    print("=" * 104)
    pretrain(brain, input_encoder, output_encoder, training, pretrain_epochs)
    collect_baseline(brain, input_encoder, training + testing)
    pre_mae, _, pre_edges = evaluate("before lesion", brain, input_encoder, output_encoder, testing)

    disabled = brain.lesion_most_used_edges(fraction=LESION_FRACTION, bidirectional=True)
    print(f"lesion: disabled {len(disabled)} directed edges")
    damaged_mae, _, _ = evaluate("after lesion", brain, input_encoder, output_encoder, testing)

    brain.set_recovery_mode(True)
    recovery_train(brain, input_encoder, output_encoder, training, RECOVERY_EPOCHS)
    brain.set_recovery_mode(False)
    recovered_mae, _, recovered_edges = evaluate("after recovery", brain, input_encoder, output_encoder, testing)

    stats = brain.multi_experience_stats()
    measured = brain.recovery_measurement_stats()
    print_profiles(brain)
    print_transitions(brain)
    print(
        f"summary: promoted={int(measured['selective_promotions_total'])} "
        f"clusters={int(stats['experience_clusters'])} "
        f"specialized={int(stats['specialized_clusters'])} "
        f"dominant-share={stats['mean_dominant_task_share']:.3f} "
        f"cross-transitions={int(stats['cross_experience_transitions'])} "
        f"bridge-candidates={int(stats['cross_experience_bridge_candidates'])} "
        f"new-eval-routes={len(recovered_edges - pre_edges)} "
        f"teacher-wakes={int(measured['teacher_direct_reactivations_total'])}\n"
    )
    return {
        "epochs": float(pretrain_epochs),
        "pre_mae": pre_mae,
        "damaged_mae": damaged_mae,
        "recovered_mae": recovered_mae,
        **stats,
    }


def main() -> None:
    print("SphereBrain multi-experience evolution experiment v15a")
    print("experiences: double, square, inverse")
    print("value input and environmental context cue use separate input-node bands")
    print("context is sensory input, not teacher wake or answer information")
    print("training order is deterministically shuffled each epoch")
    print("v15a is observation-only: no virtual bridge or new physical edge\n")
    results = [run_condition(epochs) for epochs in PRETRAIN_EPOCH_OPTIONS]
    print("=" * 132)
    print("v15a comparison")
    print("=" * 132)
    print("epochs | before | damaged | recovered | clusters | specialized | dominant share | cross | candidates | mean lift")
    for row in results:
        print(
            f"{int(row['epochs']):6d} | {row['pre_mae']:6.4f} | "
            f"{row['damaged_mae']:7.4f} | {row['recovered_mae']:9.4f} | "
            f"{int(row['experience_clusters']):8d} | "
            f"{int(row['specialized_clusters']):11d} | "
            f"{row['mean_dominant_task_share']:14.3f} | "
            f"{int(row['cross_experience_transitions']):5d} | "
            f"{int(row['cross_experience_bridge_candidates']):10d} | "
            f"{row['candidate_mean_lift']:9.3f}"
        )


if __name__ == "__main__":
    main()
