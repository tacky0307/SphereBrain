from __future__ import annotations

import argparse
import csv
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
from wave_core.recall import RecallConfig, RecallPathDiagnostics
from experiments.run_experience_growth import perturb, regions, terrain
from experiments.run_multiscale_growth import train_experience

NAME = "Experiment 008: Recall Path Diagnostics"


def probe_with_diagnostics(
    core: SphereWaveCore,
    area: dict[str, tuple[int, ...]],
    diagnostics: RecallPathDiagnostics,
    name: str,
) -> tuple[dict, list[dict]]:
    core.reset_activity()
    core.stimulate(area["A"], strength=1.0)
    trace = core.run_until_quiet(name=name, learn=False)
    return diagnostics.analyze(core, trace, area["A"], area["C"])


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

    reflector_config = MultiScaleConfig(
        learning_rate=args.multiscale_learning_rate,
        window_steps=args.window_steps,
        bridge_passes=args.bridge_passes,
        diffusion_decay=args.diffusion_decay,
        interval_span=args.interval_span,
    )
    reflector = MultiScaleExperienceReflector(reflector_config)
    recall_config = RecallConfig(
        active_threshold=args.active_threshold,
        meaningful_threshold=args.meaningful_threshold,
    )
    diagnostics = RecallPathDiagnostics(recall_config)

    rows: list[dict] = []
    step_exports: dict[str, list[dict]] = {}

    def observe(checkpoint: int) -> None:
        for mode, core in (("control", control), ("multiscale", multiscale)):
            summary, steps = probe_with_diagnostics(
                core, area, diagnostics, f"recall_{mode}_{checkpoint}"
            )
            land = terrain(initial, core.conductivity, core.adjacency)
            rows.append({
                "brain": brain,
                "checkpoint": checkpoint,
                "mode": mode,
                **summary,
                "terrain_total_change": land["total_change"],
                "terrain_changed_edges": land["changed_edges"],
            })
            if checkpoint == args.repetitions:
                step_exports[mode] = steps

    observe(0)
    training: list[dict] = []
    for repetition in range(1, args.repetitions + 1):
        delay = max(0, args.delay + int(rng.integers(-args.delay_jitter, args.delay_jitter + 1)))
        a_strength = max(0.05, 1.0 + float(rng.normal(0.0, args.strength_jitter)))
        c_strength = max(0.05, 1.0 + float(rng.normal(0.0, args.strength_jitter)))
        train_experience(control, area, repetition, delay, a_strength, c_strength, None)
        training.append(train_experience(
            multiscale, area, repetition, delay, a_strength, c_strength, reflector
        ))
        if repetition in checkpoints:
            observe(repetition)

    brain_dir = output / f"brain_{brain:02d}"
    brain_dir.mkdir(parents=True, exist_ok=True)
    with (brain_dir / "diagnostic_curve.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (brain_dir / "training.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training[0].keys()))
        writer.writeheader()
        writer.writerows(training)
    for mode, steps in step_exports.items():
        with (brain_dir / f"final_{mode}_steps.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(steps[0].keys()))
            writer.writeheader()
            writer.writerows(steps)

    result = {
        "brain": brain,
        "seed": seed,
        "regions": {key: list(value) for key, value in area.items()},
        "multiscale_config": asdict(reflector_config),
        "recall_config": asdict(recall_config),
        "diagnostics": rows,
    }
    (brain_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def arguments():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--checkpoints", default="100,200,300,400,500")
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
    parser.add_argument("--active-threshold", type=float, default=1e-8)
    parser.add_argument("--meaningful-threshold", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2900)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_008"))
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
    rows = [row for result in results for row in result["diagnostics"]]
    with (args.output / "diagnostic_curve.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_multi = [row for row in rows if row["checkpoint"] == args.repetitions and row["mode"] == "multiscale"]
    final_control = [row for row in rows if row["checkpoint"] == args.repetitions and row["mode"] == "control"]
    summary = {
        "experiment": NAME,
        "brains": args.brains,
        "repetitions": args.repetitions,
        "multiscale_mean_path_progress": float(np.mean([row["path_progress"] for row in final_multi])),
        "control_mean_path_progress": float(np.mean([row["path_progress"] for row in final_control])),
        "multiscale_mean_closest_meaningful_distance": float(np.mean([row["closest_meaningful_distance_to_target"] for row in final_multi])),
        "control_mean_closest_meaningful_distance": float(np.mean([row["closest_meaningful_distance_to_target"] for row in final_control])),
        "multiscale_mean_path_integral": float(np.mean([row["path_integral"] for row in final_multi])),
        "control_mean_path_integral": float(np.mean([row["path_integral"] for row in final_control])),
        "note": "This experiment observes recall only. It does not add prediction or force C activity.",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(NAME)
    print(f"brains: {args.brains}")
    print(f"repetitions: {args.repetitions}")
    print(f"multiscale mean path progress: {summary['multiscale_mean_path_progress']:.12g}")
    print(f"control mean path progress: {summary['control_mean_path_progress']:.12g}")
    print(f"multiscale mean closest meaningful distance: {summary['multiscale_mean_closest_meaningful_distance']:.12g}")
    print(f"control mean closest meaningful distance: {summary['control_mean_closest_meaningful_distance']:.12g}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
