from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_wave_core_v0 import run_probe, trace_metrics
from wave_core import SphereWaveCore, WaveConfig


CONDITIONS = (
    {"name": "reference_A_then_B_delay3", "order": "AB", "delay": 3, "a_strength": 1.0, "b_strength": 1.0},
    {"name": "A_then_B_delay0", "order": "AB", "delay": 0, "a_strength": 1.0, "b_strength": 1.0},
    {"name": "A_then_B_delay1", "order": "AB", "delay": 1, "a_strength": 1.0, "b_strength": 1.0},
    {"name": "A_then_B_delay6", "order": "AB", "delay": 6, "a_strength": 1.0, "b_strength": 1.0},
    {"name": "A_then_B_delay12", "order": "AB", "delay": 12, "a_strength": 1.0, "b_strength": 1.0},
    {"name": "Aweak_then_B", "order": "AB", "delay": 3, "a_strength": 0.5, "b_strength": 1.0},
    {"name": "A_then_Bweak", "order": "AB", "delay": 3, "a_strength": 1.0, "b_strength": 0.5},
    {"name": "B_then_A_delay3", "order": "BA", "delay": 3, "a_strength": 1.0, "b_strength": 1.0},
)


def experience(
    core: SphereWaveCore,
    a_region: tuple[int, ...],
    b_region: tuple[int, ...],
    condition: dict[str, Any],
    repetitions: int,
) -> set[tuple[int, int]]:
    changed_edges: set[tuple[int, int]] = set()
    first_region, second_region = (
        (a_region, b_region) if condition["order"] == "AB" else (b_region, a_region)
    )
    first_strength, second_strength = (
        (condition["a_strength"], condition["b_strength"])
        if condition["order"] == "AB"
        else (condition["b_strength"], condition["a_strength"])
    )

    for repetition in range(repetitions):
        core.reset_activity()
        core.stimulate(first_region, strength=first_strength)
        if condition["delay"] > 0:
            first_trace = core.advance(
                int(condition["delay"]),
                learn=True,
                name=f"{condition['name']}_{repetition}_first",
            )
            changed_edges.update((a, b) for a, b, _ in first_trace.changed_edges)
        core.stimulate(second_region, strength=second_strength)
        second_trace = core.run_until_quiet(
            name=f"{condition['name']}_{repetition}_second",
            learn=True,
        )
        changed_edges.update((a, b) for a, b, _ in second_trace.changed_edges)

    return changed_edges


def jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def run_condition(condition: dict[str, Any], repetitions: int) -> dict[str, Any]:
    core = SphereWaveCore(WaveConfig(seed=27))
    a_region = core.stimulus_region(anchor=24, radius=3)
    b_region = core.stimulus_region(anchor=142, radius=3)

    baseline_trace = run_probe(core, a_region, name=f"baseline_{condition['name']}")
    baseline = trace_metrics(baseline_trace, core, b_region)
    terrain_before = core.conductivity.copy()

    edge_set = experience(core, a_region, b_region, condition, repetitions)

    trained_trace = run_probe(core, a_region, name=f"trained_{condition['name']}")
    trained = trace_metrics(trained_trace, core, b_region)
    terrain_delta = core.conductivity - terrain_before

    change = {
        "b_region_peak": trained["b_region_peak"] - baseline["b_region_peak"],
        "b_region_activity_integral": trained["b_region_activity_integral"] - baseline["b_region_activity_integral"],
        "center_distance_to_b": trained["closest_center_distance_to_b"] - baseline["closest_center_distance_to_b"],
        "activity_integral": trained["activity_integral"] - baseline["activity_integral"],
        "center_path_length": trained["center_path_length"] - baseline["center_path_length"],
        "changed_directed_edges": int(np.count_nonzero(terrain_delta > 1e-10)),
        "total_conductivity_change": float(np.sum(terrain_delta)),
    }

    return {
        "condition": condition,
        "baseline": baseline,
        "trained": trained,
        "change": change,
        "edge_set": sorted([list(edge) for edge in edge_set]),
    }


def save_csv(path: Path, results: list[dict[str, Any]], reference_edges: set[tuple[int, int]]) -> None:
    fields = [
        "condition", "order", "delay", "a_strength", "b_strength",
        "b_region_peak", "b_region_activity_integral", "center_distance_to_b",
        "activity_integral", "center_path_length", "changed_directed_edges",
        "total_conductivity_change", "edge_jaccard_vs_reference",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            condition = result["condition"]
            edges = {tuple(edge) for edge in result["edge_set"]}
            writer.writerow({
                "condition": condition["name"],
                "order": condition["order"],
                "delay": condition["delay"],
                "a_strength": condition["a_strength"],
                "b_strength": condition["b_strength"],
                **result["change"],
                "edge_jaccard_vs_reference": jaccard(reference_edges, edges),
            })


def main() -> None:
    repetitions = 20
    output_dir = ROOT / "data" / "wave_core_condition_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [run_condition(dict(condition), repetitions) for condition in CONDITIONS]
    reference_edges = {tuple(edge) for edge in results[0]["edge_set"]}

    for result in results:
        edges = {tuple(edge) for edge in result["edge_set"]}
        result["comparison_to_reference"] = {
            "edge_jaccard": jaccard(reference_edges, edges),
            "same_changed_edge_set": edges == reference_edges,
            "unique_edges_vs_reference": len(edges - reference_edges),
            "missing_edges_vs_reference": len(reference_edges - edges),
        }

    payload = {
        "experiment": "SphereBrain Wave Core v0 / Experiment 004",
        "purpose": "Observe which experience conditions cause a fixed brain to form a different learned route.",
        "fixed": {
            "network": "same node positions, adjacency, conductivity and seed",
            "seed": 27,
            "repetitions": repetitions,
        },
        "results": results,
        "interpretation_rule": "A lower edge Jaccard value means the condition changed which directed edges were strengthened. No condition is labeled correct or incorrect.",
    }

    (output_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(output_dir / "condition_comparison.csv", results, reference_edges)

    compact = {
        result["condition"]["name"]: {
            **result["change"],
            **result["comparison_to_reference"],
        }
        for result in results
    }
    print(json.dumps({"experiment": payload["experiment"], "conditions": compact}, ensure_ascii=False, indent=2))
    print(f"\nObservation files: {output_dir}")
    print(f"Comparison CSV: {output_dir / 'condition_comparison.csv'}")


if __name__ == "__main__":
    main()
