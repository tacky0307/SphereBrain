from __future__ import annotations

from surface_flow import SurfaceFlowBrain

CUE = "空が"
TARGET = "青い"
DISTRACTORS = ["広い", "暗い", "雨"]
CHECKPOINTS = {0, 1, 2, 5, 10, 20, 50, 75, 100}


def score_label(brain: SurfaceFlowBrain, result, label: str) -> float:
    return brain.target_score(result, brain.concept_to_outputs(label))


def print_checkpoint(brain: SurfaceFlowBrain, experience_no: int, cue_nodes: list[int]) -> None:
    observed = brain.propagate(cue_nodes, learn=False, noise=0.0)
    labels = [TARGET, *DISTRACTORS]
    scores = {label: score_label(brain, observed, label) for label in labels}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    print(f"--- experience {experience_no:>3} ---")
    print("observed output nodes:", observed.output_nodes)
    for rank, (label, score) in enumerate(ranked, start=1):
        marker = " <- target" if label == TARGET else ""
        print(f"  {rank}. {label}: {score:.4f}{marker}")
    print("recalled word:", ranked[0][0] if ranked[0][1] > 0 else "none")
    print()


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    cue_nodes = brain.stimulus_to_inputs(CUE)
    target_nodes = brain.concept_to_outputs(TARGET)

    print(f"training pair: {CUE} -> {TARGET}")
    print("cue input nodes:", cue_nodes)
    print("target output nodes:", target_nodes)
    for label in DISTRACTORS:
        print(f"distractor '{label}' output nodes:", brain.concept_to_outputs(label))
    print()

    print_checkpoint(brain, 0, cue_nodes)

    for experience_no in range(1, 101):
        brain.propagate(
            cue_nodes,
            learn=True,
            target_output_nodes=target_nodes,
        )
        if experience_no in CHECKPOINTS:
            print_checkpoint(brain, experience_no, cue_nodes)


if __name__ == "__main__":
    main()
