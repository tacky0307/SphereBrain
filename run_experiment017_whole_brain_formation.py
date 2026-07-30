from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from brain import SphereBrain


OUTPUT_DIR = Path("results") / "experiment017_whole_brain_formation"
STIMULUS = "空は青い"
NODE_COUNT = 80
REFLECTIONS_PER_EXPERIENCE = 2


def write_snapshot(brain: SphereBrain, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "node",
                "activity",
                "previous_activity",
                "peak_activity",
                "node_usage",
                "degree",
                "mean_connected_weight",
                "max_connected_weight",
            ]
        )

        for node in range(brain.node_count):
            connected = brain.adjacency[node]
            weights = brain.weights[node][connected]
            writer.writerow(
                [
                    node,
                    float(brain.activity[node]),
                    float(brain.previous_activity[node]),
                    float(brain.peak_activity[node]),
                    int(brain.node_usage[node]),
                    int(np.count_nonzero(connected)),
                    float(weights.mean()) if weights.size else 0.0,
                    float(weights.max()) if weights.size else 0.0,
                ]
            )


def write_trace_summary(brain: SphereBrain, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_index",
                "source",
                "total_activity",
                "active_node_count",
                "stimulus_total",
                "metadata",
            ]
        )
        for frame in brain.trace:
            writer.writerow(
                [
                    frame.time_index,
                    frame.source,
                    frame.total_activity,
                    int(np.count_nonzero(frame.activity > 0.0)),
                    float(frame.stimulus.sum()),
                    repr(frame.metadata),
                ]
            )


def growth_row(brain: SphereBrain, stage: str, frame_index: int | None) -> list[object]:
    upper = np.triu_indices(brain.node_count, k=1)
    connected_weights = brain.weights[upper][brain.adjacency[upper]]
    used_edges = brain.usage[upper][brain.usage[upper] > 0]

    return [
        stage,
        frame_index if frame_index is not None else "",
        len(brain.trace),
        float(brain.peak_activity.sum()),
        int(np.count_nonzero(brain.peak_activity > 0.0)),
        int(np.count_nonzero(brain.node_usage > 0)),
        int(brain.node_usage.sum()),
        int(np.count_nonzero(used_edges)),
        int(used_edges.sum()) if used_edges.size else 0,
        float(connected_weights.mean()) if connected_weights.size else 0.0,
        float(connected_weights.max()) if connected_weights.size else 0.0,
    ]


def write_growth(rows: list[list[object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "trace_frame_index",
                "trace_count",
                "peak_activity_total",
                "peak_active_node_count",
                "used_node_count",
                "node_usage_total",
                "used_edge_count",
                "edge_usage_total",
                "mean_connected_weight",
                "max_connected_weight",
            ]
        )
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    brain = SphereBrain(
        node_count=NODE_COUNT,
        reflections_per_experience=REFLECTIONS_PER_EXPERIENCE,
    )

    growth_rows: list[list[object]] = []
    growth_rows.append(growth_row(brain, "initial", None))

    _, experience_frame = brain.experience(STIMULUS, noise=0.0)
    growth_rows.append(
        growth_row(brain, "experience", experience_frame.time_index)
    )

    reflections = brain.complete_reflections(noise=0.0)
    for index, (_, frame, _) in enumerate(reflections, start=1):
        growth_rows.append(
            growth_row(brain, f"reflection_{index}", frame.time_index)
        )

    write_snapshot(brain, OUTPUT_DIR / "whole_brain_snapshot.csv")
    write_trace_summary(brain, OUTPUT_DIR / "whole_brain_trace.csv")
    write_growth(growth_rows, OUTPUT_DIR / "brain_growth.csv")

    brain.save(OUTPUT_DIR / "brain_state.json")

    print("Experiment 017 complete")
    print(f"stimulus: {STIMULUS}")
    print(f"trace frames: {len(brain.trace)}")
    print(f"output: {OUTPUT_DIR.resolve()}")
    for frame in brain.trace:
        print(
            frame.time_index,
            frame.source,
            round(frame.total_activity, 6),
        )


if __name__ == "__main__":
    main()
