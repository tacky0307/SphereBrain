from surface_flow import SurfaceFlowBrain


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    stimuli = {"A": "空は青い", "B": "今日は雨です"}
    latest = {}

    for label, text in stimuli.items():
        sources = brain.stimulus_to_inputs(text)
        for _ in range(20):
            latest[label] = brain.propagate(sources, learn=True)

    a_again = brain.propagate(brain.stimulus_to_inputs(stimuli["A"]), learn=False)
    b_again = brain.propagate(brain.stimulus_to_inputs(stimuli["B"]), learn=False)

    print("input surface nodes:", len(brain.input_nodes))
    print("output surface nodes:", len(brain.output_nodes))
    print("A output nodes:", a_again.output_nodes)
    print("B output nodes:", b_again.output_nodes)
    print("A/A similarity:", round(brain.output_similarity(latest["A"], a_again), 4))
    print("B/B similarity:", round(brain.output_similarity(latest["B"], b_again), 4))
    print("A/B similarity:", round(brain.output_similarity(a_again, b_again), 4))


if __name__ == "__main__":
    main()
