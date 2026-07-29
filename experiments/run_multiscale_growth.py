from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wave_core import SphereWaveCore, WaveConfig
from wave_core.multiscale import MultiScaleConfig, MultiScaleExperienceReflector
from experiments.run_experience_growth import perturb, probe, regions, terrain

NAME = "Experiment 007: Multi-scale Growth"


def path_metrics(
    core: SphereWaveCore,
    source_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
) -> dict[str, float | int]:
    targets = set(target_ids)
    distances = np.full(core.config.node_count, np.inf)
    previous = np.full(core.config.node_count, -1, dtype=int)
    queue: list[tuple[float, int]] = []

    for node_id in source_ids:
        distances[node_id] = 0.0
        heapq.heappush(queue, (0.0, node_id))

    destination = -1
    while queue:
        distance, node_id = heapq.heappop(queue)
        if distance != distances[node_id]:
            continue
        if node_id in targets:
            destination = node_id
            break
        for neighbor in np.flatnonzero(core.adjacency[node_id]):
            conductivity = max(float(core.conductivity[node_id, neighbor]), 1e-12)
            candidate = distance + 1.0 / conductivity
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node_id
                heapq.heappush(queue, (candidate, int(neighbor)))

    if destination < 0:
        return {"ac_path_cost": float("inf"), "ac_path_mean": 0.0, "ac_path_edges": 0}

    values = []
    current = destination
    while previous[current] >= 0:
        parent = int(previous[current])
        values.append(float(core.conductivity[parent, current]))
        current = parent
    return {
        "ac_path_cost": float(distances[destination]),
        "ac_path_mean": float(np.mean(values)) if values else 0.0,
        "ac_path_edges": len(values),
    }


def train_experience(
    core: SphereWaveCore,
    area: dict[str, tuple[int, ...]],
    repetition: int,
    delay: int,
    a_strength: float,
    c_strength: float,
    reflector: MultiScaleExperienceReflector | None,
) -> dict:
    core.reset_activity()
    core.stimulate(area["A"], a_strength)
    first = core.advance(delay, learn=True, name=f"multi_{repetition}_a")
    core.stimulate(area["C"], c_strength)
    second = core.run_until_quiet(name=f"multi_{repetition}_c", learn=True)
    snapshots = first.snapshots + second.snapshots

    reflection_change = 0.0
    reflection_edges = 0
    intervals = 0
    corridor_mass = 0.0
    if reflector is not None and snapshots:
        history = np.stack([snapshot.activity for snapshot in snapshots], axis=0)
        reflection = reflector.reflect(core, history)
        reflection_change = reflection.total_change
        reflection_edges = len(reflection.changed_edges)
        intervals = reflection.interval_count
        corridor_mass = reflection.corridor_mass

    return {
        "repetition": repetition,
        "steps": len(snapshots),
        "local_edges": len(first.changed_edges) + len(second.changed_edges),
        "reflection_edges": reflection_edges,
        "reflection_change": reflection_change,
        "intervals": intervals,
        "corridor_mass": corridor_mass,
    }


def observation_row(
    brain: int,
    checkpoint: int,
    mode: str,
    seen: dict,
    baseline: dict,
    land: dict,
    path: dict,
    baseline_path: dict,
) -> dict:
    c_value = seen["scores"]["C"]["integral"]
    return {
        "brain": brain,
        "checkpoint": checkpoint,
        "mode": mode,
        "winner": seen["winner"],
        "c_integral": c_value,
        "c_peak": seen["scores"]["C"]["peak"],
        "c_first_step": seen["scores"]["C"]["first_step"],
        "c_selectivity": seen["c_selectivity"],
        "c_ratio": seen["c_ratio"],
        "c_change_from_baseline": c_value - baseline["scores"]["C"]["integral"],
        "terrain_total_change": land["total_change"],
        "terrain_changed_edges": land["changed_edges"],
        **path,
        "ac_path_cost_change": float(path["ac_path_cost"]) - float(baseline_path["ac_path_cost"]),
        "ac_path_mean_change": float(path["ac_path_mean"]) - float(baseline_path["ac_path_mean"]),
    }


def run_brain(args, brain: int, checkpoints: list[int], output: Path) -> dict:
    seed = args.seed + brain
    config = WaveConfig(seed=seed)
    multiscale = SphereWaveCore(config)
    control = SphereWaveCore(config)
    rng = np.random.default_rng(seed)
    area = regions(multiscale, args.region_radius)
    perturb(multiscale, rng, args.initial_terrain_noise)
    control.conductivity = multiscale.conductivity.copy()
    initial = multiscale.conductivity.copy()

    multi_config = MultiScaleConfig(
        learning_rate=args.multiscale_learning_rate,
        window_steps=args.window_steps,
        bridge_passes=args.bridge_passes,
        diffusion_decay=args.diffusion_decay,
        interval_span=args.interval_span,
    )
    reflector = MultiScaleExperienceReflector(multi_config)

    bases = {}
    base_paths = {}
    rows = []
    for mode, core in (("control", control), ("multiscale", multiscale)):
        bases[mode] = probe(core, area, f"{mode}_0")
        base_paths[mode] = path_metrics(core, area["A"], area["C"])
        rows.append(observation_row(
            brain, 0, mode, bases[mode], bases[mode],
            terrain(initial, core.conductivity, core.adjacency),
            base_paths[mode], base_paths[mode],
        ))

    training = []
    for repetition in range(1, args.repetitions + 1):
        delay = max(0, args.delay + int(rng.integers(-args.delay_jitter, args.delay_jitter + 1)))
        a_strength = max(0.05, 1.0 + float(rng.normal(0.0, args.strength_jitter)))
        c_strength = max(0.05, 1.0 + float(rng.normal(0.0, args.strength_jitter)))
        train_experience(control, area, repetition, delay, a_strength, c_strength, None)
        training.append(train_experience(
            multiscale, area, repetition, delay, a_strength, c_strength, reflector
        ))

        if repetition in checkpoints:
            for mode, core in (("control", control), ("multiscale", multiscale)):
                seen = probe(core, area, f"{mode}_{repetition}")
                rows.append(observation_row(
                    brain, repetition, mode, seen, bases[mode],
                    terrain(initial, core.conductivity, core.adjacency),
                    path_metrics(core, area["A"], area["C"]), base_paths[mode],
                ))

    brain_dir = output / f"brain_{brain:02d}"
    brain_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("growth_curve.csv", rows), ("training.csv", training)):
        with (brain_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    result = {
        "brain": brain,
        "seed": seed,
        "regions": {key: list(value) for key, value in area.items()},
        "multiscale_config": asdict(multi_config),
        "growth": rows,
    }
    (brain_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def arguments():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--brains", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--checkpoints", default="1,5,10,25,50,100,200")
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--delay-jitter", type=int, default=2)
    parser.add_argument("--strength-jitter", type=float, default=0.08)
    parser.add_argument("--initial-terrain-noise", type=float, default=0.012)
    parser.add_argument("--region-radius", type=int, default=2)
    parser.add_argument("--multiscale-learning-rate", type=float, default=0.00028)
    parser.add_argument("--window-steps", type=int, default=4)
    parser.add_argument("--bridge-passes", type=int, default=24)
    parser.add_argument("--diffusion-decay", type=float, default=0.84)
    parser.add_argument("--interval-span", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2800)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_007"))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    checkpoints = sorted({
        int(value) for value in args.checkpoints.split(",")
        if value.strip() and 0 < int(value) <= args.repetitions
    })
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_brain(args, brain, checkpoints, args.output) for brain in range(1, args.brains + 1)]
    rows = [row for result in results for row in result["growth"]]
    with (args.output / "growth_curve.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_multi = [row for row in rows if row["checkpoint"] == args.repetitions and row["mode"] == "multiscale"]
    final_control = [row for row in rows if row["checkpoint"] == args.repetitions and row["mode"] == "control"]
    summary = {
        "experiment": NAME,
        "brains": args.brains,
        "repetitions": args.repetitions,
        "multiscale_mean_c_change": float(np.mean([row["c_change_from_baseline"] for row in final_multi])),
        "control_mean_c_change": float(np.mean([row["c_change_from_baseline"] for row in final_control])),
        "multiscale_mean_path_cost_change": float(np.mean([row["ac_path_cost_change"] for row in final_multi])),
        "control_mean_path_cost_change": float(np.mean([row["ac_path_cost_change"] for row in final_control])),
        "note": "First inspect path growth. C does not need to receive activity or win yet.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(NAME)
    print(f"brains: {args.brains}")
    print(f"repetitions: {args.repetitions}")
    print(f"multiscale mean C change: {summary['multiscale_mean_c_change']:.12g}")
    print(f"control mean C change: {summary['control_mean_c_change']:.12g}")
    print(f"multiscale mean path cost change: {summary['multiscale_mean_path_cost_change']:.12g}")
    print(f"control mean path cost change: {summary['control_mean_path_cost_change']:.12g}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
