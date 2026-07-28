from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from surface_flow import SurfaceFlowBrain, SurfaceFlowResult


CHECKPOINTS = {0, 1, 2, 5, 10, 20, 30, 50, 75, 100}
STIMULUS = "空は青い"
TOP_PATHS = 12


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


def strongest_paths(brain: SurfaceFlowBrain, limit: int = TOP_PATHS) -> list[tuple[int, int, int, float]]:
    used = np.argwhere(brain.usage > 0)
    paths = [
        (
            int(source),
            int(target),
            int(brain.usage[source, target]),
            float(brain.weights[source, target]),
        )
        for source, target in used
    ]
    paths.sort(key=lambda item: (item[2], item[3]), reverse=True)
    return paths[:limit]


def main() -> None:
    brain = SurfaceFlowBrain(node_count=600, neighbors_per_node=8, seed=42)
    sources = brain.stimulus_to_inputs(STIMULUS)
    snapshots: dict[int, SurfaceFlowResult] = {}
    path_snapshots: dict[int, list[tuple[int, int, int, float]]] = {}

    snapshots[0] = brain.propagate(sources, learn=False)
    path_snapshots[0] = strongest_paths(brain)

    for experience_no in range(1, 101):
        brain.propagate(sources, learn=True)
        if experience_no in CHECKPOINTS:
            snapshots[experience_no] = brain.propagate(sources, learn=False)
            path_snapshots[experience_no] = strongest_paths(brain)

    final_result = snapshots[100]
    rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []

    print(f"stimulus: {STIMULUS}")
    print(f"input nodes: {sources}")
    print()

    for checkpoint in sorted(snapshots):
        result = snapshots[checkpoint]
        summary = summarize(result)
        summary["experience_no"] = checkpoint
        summary["similarity_to_initial"] = brain.output_similarity(snapshots[0], result)
        summary["similarity_to_final"] = brain.output_similarity(result, final_result)
        rows.append(summary)

        print(f"--- experience {checkpoint:>3} ---")
        print("output nodes:", result.output_nodes)
        print("first output step:", summary["first_output_step"])
        print("output energy:", round(float(summary["output_energy"]), 4))
        print("peak output:", round(float(summary["peak_output"]), 4))
        print("traversed edges:", summary["traversed_edge_count"])
        print("active nodes:", summary["active_node_count"])
        print("similarity to initial:", round(float(summary["similarity_to_initial"]), 4))
        print("similarity to final:", round(float(summary["similarity_to_final"]), 4))
        print("strongest learned paths:")

        paths = path_snapshots[checkpoint]
        if not paths:
            print("  none")
        for rank, (source, target, usage, weight) in enumerate(paths, start=1):
            print(f"  {rank:>2}. {source} -> {target}  usage={usage:>3}  weight={weight:.4f}")
            path_rows.append(
                {
                    "experience_no": checkpoint,
                    "rank": rank,
                    "source": source,
                    "target": target,
                    "usage": usage,
                    "weight": weight,
                }
            )
        print()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    summary_csv = output_dir / "sky_blue_learning_trace_100.csv"
    paths_csv = output_dir / "sky_blue_strongest_paths_100.csv"

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with paths_csv.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["experience_no", "rank", "source", "target", "usage", "weight"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(path_rows)

    print("Summary CSV:", summary_csv)
    print("Paths CSV:", paths_csv)


if __name__ == "__main__":
    main()
