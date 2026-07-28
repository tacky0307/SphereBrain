from __future__ import annotations

import csv
from pathlib import Path

from surface_flow import SurfaceFlowBrain, SurfaceFlowResult


CHECKPOINTS = {0, 1, 2, 5, 10, 20}
STIMULUS = "空は青い"


def summarize(result: SurfaceFlowResult) -> dict[str, object]:
    visible_steps = [
        step_no
        for step_no, step in enumerate(result.output_history, start=1)
        if step
    ]
    values = [value for step in result.output_history for value in step.values()]
    unique_edges = set(result.traversed_edges)

    return {
        "first_output_step": visible_steps[0] if visible_steps else None,
        "last_output_step": visible_steps[-1] if visible_steps else None,
        "output_node_count": len(result.output_nodes),
        "output_nodes": " ".join(map(str, result.output_nodes)),
        "output_energy": sum(values),
        "peak_output": max(values, default=0.0),
        "traversed_edge_count": len(unique_edges),
        "active_node_count": len({node for step in result.activation_history for node in step}),
    }


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    sources = brain.stimulus_to_inputs(STIMULUS)
    snapshots: dict[int, SurfaceFlowResult] = {}

    # Learning before any experience: the sphere's initial response.
    snapshots[0] = brain.propagate(sources, learn=False)

    for experience_no in range(1, 21):
        learned = brain.propagate(sources, learn=True)
        if experience_no in CHECKPOINTS:
            # Observe without further learning so measurement does not alter the brain.
            snapshots[experience_no] = brain.propagate(sources, learn=False)

    final_result = snapshots[20]
    rows: list[dict[str, object]] = []

    print(f'stimulus: {STIMULUS}')
    print(f'input nodes: {sources}')
    print()

    for checkpoint in sorted(snapshots):
        result = snapshots[checkpoint]
        summary = summarize(result)
        summary["experience_no"] = checkpoint
        summary["similarity_to_initial"] = brain.output_similarity(snapshots[0], result)
        summary["similarity_to_final"] = brain.output_similarity(result, final_result)
        rows.append(summary)

        print(f'--- experience {checkpoint:>2} ---')
        print('output nodes:', result.output_nodes)
        print('first output step:', summary['first_output_step'])
        print('output energy:', round(float(summary['output_energy']), 4))
        print('peak output:', round(float(summary['peak_output']), 4))
        print('traversed edges:', summary['traversed_edge_count'])
        print('similarity to initial:', round(float(summary['similarity_to_initial']), 4))
        print('similarity to final:', round(float(summary['similarity_to_final']), 4))
        print()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "sky_blue_learning_trace.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print('CSV:', csv_path)


if __name__ == "__main__":
    main()
