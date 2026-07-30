from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from brain import SphereBrain


EXPERIMENT_NAME = "Experiment017 Whole Brain Formation"
INPUT_TEXT = "空は青い"
CYCLES = 100
OUTPUT_CSV = Path("experiment017_whole_brain_formation.csv")


CSV_FIELDS = [
    "cycle",
    "trace_count",
    "experience_activity",
    "reflection1_activity",
    "reflection2_activity",
    "average_weight",
    "maximum_weight",
    "active_edges",
    "average_node_usage",
    "maximum_node_usage",
]


def connected_weights(brain: SphereBrain) -> np.ndarray:
    """Return one value per undirected connected edge."""

    upper_triangle = np.triu(brain.adjacency, k=1)
    return brain.weights[upper_triangle]


def active_edge_count(brain: SphereBrain) -> int:
    """Count undirected edges that have been used at least once."""

    upper_triangle = np.triu(brain.adjacency, k=1)
    return int(np.count_nonzero(brain.usage[upper_triangle] > 0))


def run_experiment() -> None:
    brain = SphereBrain(
        node_count=240,
        reflections_per_experience=2,
    )

    rows: list[dict[str, int | float]] = []

    print(f"=== {EXPERIMENT_NAME} ===")
    print(f"input: {INPUT_TEXT}")
    print(f"cycles: {CYCLES}")

    for cycle in range(1, CYCLES + 1):
        _, experience_frame = brain.experience(INPUT_TEXT)
        reflections = brain.complete_reflections()

        if len(reflections) != 2:
            raise RuntimeError(
                "Experiment017 requires exactly two reflections per experience; "
                f"received {len(reflections)}"
            )

        reflection1_frame = reflections[0][1]
        reflection2_frame = reflections[1][1]
        weights = connected_weights(brain)

        row = {
            "cycle": cycle,
            "trace_count": len(brain.trace),
            "experience_activity": float(experience_frame.total_activity),
            "reflection1_activity": float(reflection1_frame.total_activity),
            "reflection2_activity": float(reflection2_frame.total_activity),
            "average_weight": float(weights.mean()) if weights.size else 0.0,
            "maximum_weight": float(weights.max()) if weights.size else 0.0,
            "active_edges": active_edge_count(brain),
            "average_node_usage": float(brain.node_usage.mean()),
            "maximum_node_usage": int(brain.node_usage.max()),
        }
        rows.append(row)

        if cycle == 1 or cycle % 10 == 0 or cycle == CYCLES:
            print(
                f"cycle={cycle:03d} "
                f"activity=({row['experience_activity']:.6f}, "
                f"{row['reflection1_activity']:.6f}, "
                f"{row['reflection2_activity']:.6f}) "
                f"avg_weight={row['average_weight']:.6f} "
                f"active_edges={row['active_edges']}"
            )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    final_weights = connected_weights(brain)

    print("\n=== Experiment017 Summary ===")
    print(f"Trace count: {len(brain.trace)}")
    print(
        "Average weight: "
        f"{float(final_weights.mean()) if final_weights.size else 0.0:.6f}"
    )
    print(
        "Maximum weight: "
        f"{float(final_weights.max()) if final_weights.size else 0.0:.6f}"
    )
    print(f"Maximum node usage: {int(brain.node_usage.max())}")
    print(f"CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    run_experiment()
