from __future__ import annotations

from surface_encoders import TextSurfaceEncoder
from surface_flow import SurfaceFlowBrain

CUE = "空が"
TARGET = "青い"
DISTRACTORS = ["広い", "暗い", "雨"]
CHECKPOINTS = {0, 1, 2, 5, 10, 20, 50, 75, 100}


def observe_scores(brain, cue_pattern, output_encoder):
    observed = brain.propagate(cue_pattern, noise=0.0, steps=40, threshold=0.04)
    labels = [TARGET, *DISTRACTORS]
    scores = {
        label: brain.target_score(observed, output_encoder.encode(label))
        for label in labels
    }
    return observed, scores


def print_checkpoint(brain, experience_no, cue_pattern, output_encoder, baseline_scores):
    observed, raw_scores = observe_scores(brain, cue_pattern, output_encoder)
    gains = {label: raw_scores[label] - baseline_scores[label] for label in raw_scores}
    ranked = sorted(gains.items(), key=lambda item: item[1], reverse=True)

    print(f"--- experience {experience_no:>3} ---")
    print("observed output nodes:", observed.output_nodes)
    print("learned gain over experience 0:")
    for rank, (label, gain) in enumerate(ranked, start=1):
        marker = " <- target" if label == TARGET else ""
        print(
            f"  {rank}. {label}: gain={gain:+.4f} "
            f"raw={raw_scores[label]:.4f} baseline={baseline_scores[label]:.4f}{marker}"
        )
    best_label, best_gain = ranked[0]
    print("recalled word:", best_label if best_gain > 1e-6 else "none")
    print()


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    input_encoder = TextSurfaceEncoder(brain.input_nodes)
    output_encoder = TextSurfaceEncoder(brain.output_nodes)

    cue_pattern = input_encoder.encode(CUE)
    target_pattern = output_encoder.encode(TARGET)

    print(f"training pair: {CUE} -> {TARGET}")
    print("cue input pattern:", cue_pattern)
    print("target output pattern:", target_pattern)
    for label in DISTRACTORS:
        print(f"distractor '{label}' output pattern:", output_encoder.encode(label))
    print()

    _, baseline_scores = observe_scores(brain, cue_pattern, output_encoder)
    print_checkpoint(brain, 0, cue_pattern, output_encoder, baseline_scores)

    for experience_no in range(1, 101):
        reinforced = brain.experience(cue_pattern, target_pattern)
        if experience_no == 1:
            print("teacher-guided route edges:", len(reinforced))
            print()
        if experience_no in CHECKPOINTS:
            print_checkpoint(brain, experience_no, cue_pattern, output_encoder, baseline_scores)


if __name__ == "__main__":
    main()
