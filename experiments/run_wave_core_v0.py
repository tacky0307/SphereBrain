from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wave_core import SphereWaveCore, WaveConfig


def region_activity(trace, region: tuple[int, ...]) -> float:
    return trace.max_activity_for(region)


def center_distance_to_region(trace, core: SphereWaveCore, region: tuple[int, ...]) -> float:
    if not trace.snapshots:
        return float("inf")
    target = np.mean(core.positions[np.asarray(region, dtype=int)], axis=0)
    centers = np.asarray([item.center for item in trace.snapshots], dtype=float)
    distances = np.linalg.norm(centers - target[None, :], axis=1)
    return float(np.min(distances))


def run_probe(core: SphereWaveCore, a_region: tuple[int, ...], name: str):
    core.reset_activity()
    core.stimulate(a_region, strength=1.0)
    return core.run_until_quiet(name=name, learn=False)


def experience_a_then_b(
    core: SphereWaveCore,
    a_region: tuple[int, ...],
    b_region: tuple[int, ...],
    repetitions: int,
    delay_steps: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []

    for repetition in range(1, repetitions + 1):
        core.reset_activity()
        core.stimulate(a_region, strength=1.0)
        first = core.advance(delay_steps, learn=True, name=f"experience_{repetition}_a")
        core.stimulate(b_region, strength=1.0)
        second = core.run_until_quiet(
            name=f"experience_{repetition}_b",
            learn=True,
        )
        records.append(
            {
                "repetition": repetition,
                "steps": first.steps + second.steps,
                "changed_edges": len(first.changed_edges) + len(second.changed_edges),
                "peak_total_activity": max(
                    first.peak_total_activity,
                    second.peak_total_activity,
                ),
            }
        )

    return records


def save_trace_csv(path: Path, trace, b_region: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b_ids = np.asarray(b_region, dtype=int)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "total_activity",
                "b_region_mean_activity",
                "b_region_max_activity",
                "center_x",
                "center_y",
                "center_z",
                "fired_nodes",
            ]
        )
        for snapshot in trace.snapshots:
            values = snapshot.activity[b_ids]
            writer.writerow(
                [
                    snapshot.step,
                    snapshot.total_activity,
                    float(np.mean(values)),
                    float(np.max(values)),
                    *snapshot.center,
                    " ".join(str(value) for value in snapshot.fired_nodes),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SphereBrain Wave Core v0: observe terrain change after A→B experience",
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "wave_core_v0")
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.delay_steps < 0:
        raise SystemExit("--delay-steps must be zero or greater")

    core = SphereWaveCore(WaveConfig())

    # The labels A and B exist only outside the core.  Internally they are
    # distributed stimulus regions with no symbolic meaning.
    a_region = core.stimulus_region(anchor=24, radius=3)
    b_region = core.stimulus_region(anchor=142, radius=3)

    baseline = run_probe(core, a_region, name="baseline_a")
    baseline_b_peak = region_activity(baseline, b_region)
    baseline_b_distance = center_distance_to_region(baseline, core, b_region)
    terrain_before = core.conductivity.copy()

    experience_records = experience_a_then_b(
        core,
        a_region,
        b_region,
        repetitions=args.repetitions,
        delay_steps=args.delay_steps,
    )

    trained = run_probe(core, a_region, name="trained_a")
    trained_b_peak = region_activity(trained, b_region)
    trained_b_distance = center_distance_to_region(trained, core, b_region)
    terrain_delta = core.conductivity - terrain_before

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    save_trace_csv(output / "baseline_a.csv", baseline, b_region)
    save_trace_csv(output / "trained_a.csv", trained, b_region)

    with (output / "experience_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(experience_records[0].keys()))
        writer.writeheader()
        writer.writerows(experience_records)

    changed = np.argwhere(terrain_delta > 1e-10)
    terrain_rows = sorted(
        (
            (int(a), int(b), float(terrain_delta[a, b]))
            for a, b in changed
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    with (output / "terrain_changes.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_node", "target_node", "conductivity_change"])
        writer.writerows(terrain_rows)

    result = {
        "experiment": "SphereBrain Wave Core v0",
        "a_region": a_region,
        "b_region": b_region,
        "repetitions": args.repetitions,
        "delay_steps": args.delay_steps,
        "baseline": {
            "steps": baseline.steps,
            "b_region_peak": baseline_b_peak,
            "closest_center_distance_to_b": baseline_b_distance,
        },
        "trained": {
            "steps": trained.steps,
            "b_region_peak": trained_b_peak,
            "closest_center_distance_to_b": trained_b_distance,
        },
        "change": {
            "b_region_peak": trained_b_peak - baseline_b_peak,
            "center_distance_to_b": trained_b_distance - baseline_b_distance,
            "changed_directed_edges": len(terrain_rows),
            "total_conductivity_change": float(np.sum(terrain_delta)),
        },
        "interpretation": (
            "This run records a tendency, not a right/wrong answer. "
            "A positive B-region peak change or a negative distance change "
            "suggests that A→B experience altered the terrain toward B."
        ),
    }

    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nObservation files: {output}")


if __name__ == "__main__":
    main()
