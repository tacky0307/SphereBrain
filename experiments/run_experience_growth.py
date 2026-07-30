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

from wave_core import (
    ExperienceBuffer,
    ExperienceConfig,
    ExperienceReflector,
    SphereWaveCore,
    WaveConfig,
)

NAME = "Experiment 006: Experience Growth"


def regions(core: SphereWaveCore, radius: int) -> dict[str, tuple[int, ...]]:
    targets = {
        "A": np.array([1.0, 0.0, 0.0]),
        "B": np.array([0.0, 1.0, 0.0]),
        "C": np.array([-1.0, 0.0, 0.0]),
        "D": np.array([0.0, -1.0, 0.0]),
    }
    result = {}
    used = set()
    for name, target in targets.items():
        distances = np.linalg.norm(core.positions - target[None, :], axis=1)
        anchor = next(int(value) for value in np.argsort(distances) if int(value) not in used)
        used.add(anchor)
        result[name] = core.stimulus_region(anchor, radius=radius)
    return result


def perturb(core: SphereWaveCore, rng: np.random.Generator, amount: float) -> None:
    noise = rng.normal(0.0, amount, core.conductivity.shape) * core.adjacency
    core.conductivity += noise
    np.clip(
        core.conductivity,
        core.config.conductivity_min,
        core.config.conductivity_max,
        out=core.conductivity,
    )
    core.conductivity[~core.adjacency] = 0.0


def probe(core: SphereWaveCore, area: dict[str, tuple[int, ...]], name: str) -> dict:
    core.reset_activity()
    core.stimulate(area["A"], strength=1.0)
    trace = core.run_until_quiet(name=name, learn=False)
    scores = {}
    for label in ("B", "C", "D"):
        ids = np.asarray(area[label], dtype=int)
        values = [float(np.mean(item.activity[ids])) for item in trace.snapshots]
        scores[label] = {
            "integral": float(sum(values)),
            "peak": max(values, default=0.0),
            "first_step": next((i + 1 for i, value in enumerate(values) if value > 1e-8), -1),
        }
    winner = max(scores, key=lambda key: (scores[key]["integral"], scores[key]["peak"]))
    other = (scores["B"]["integral"] + scores["D"]["integral"]) / 2.0
    return {
        "winner": winner,
        "scores": scores,
        "c_selectivity": scores["C"]["integral"] - other,
        "c_ratio": scores["C"]["integral"] / max(other, 1e-15),
    }


def experience(
    core: SphereWaveCore,
    area: dict[str, tuple[int, ...]],
    repetition: int,
    delay: int,
    a_strength: float,
    c_strength: float,
    reflector: ExperienceReflector | None,
) -> dict:
    memory = ExperienceBuffer(f"experience_{repetition}", core.config.node_count)
    core.reset_activity()
    memory.mark_stimulus("A", area["A"], a_strength)
    core.stimulate(area["A"], a_strength)
    first = core.advance(delay, learn=True, name=f"experience_{repetition}_a")
    memory.extend(first.snapshots)
    memory.mark_stimulus("C", area["C"], c_strength)
    core.stimulate(area["C"], c_strength)
    second = core.run_until_quiet(name=f"experience_{repetition}_c", learn=True)
    memory.extend(second.snapshots)

    reflected_change = 0.0
    reflected_edges = 0
    if reflector is not None:
        reflection = reflector.reflect(core, memory.summarize(reflector.config))
        reflected_change = reflection.total_change
        reflected_edges = len(reflection.changed_edges)
    return {
        "repetition": repetition,
        "delay": delay,
        "a_strength": a_strength,
        "c_strength": c_strength,
        "steps": first.steps + second.steps,
        "local_edges": len(first.changed_edges) + len(second.changed_edges),
        "reflection_edges": reflected_edges,
        "reflection_change": reflected_change,
    }


def terrain(initial: np.ndarray, current: np.ndarray, adjacency: np.ndarray) -> dict:
    values = (current - initial)[adjacency]
    return {
        "changed_edges": int(np.count_nonzero(np.abs(values) > 1e-10)),
        "total_change": float(np.sum(np.abs(values))),
    }


def row(brain: int, checkpoint: int, mode: str, seen: dict, baseline: dict, land: dict) -> dict:
    c_value = seen["scores"]["C"]["integral"]
    return {
        "brain": brain,
        "checkpoint": checkpoint,
        "mode": mode,
        "winner": seen["winner"],
        "c_integral": c_value,
        "c_peak": seen["scores"]["C"]["peak"],
        "c_first_step": seen["scores"]["C"]["first_step"],
        "b_integral": seen["scores"]["B"]["integral"],
        "d_integral": seen["scores"]["D"]["integral"],
        "c_selectivity": seen["c_selectivity"],
        "c_ratio": seen["c_ratio"],
        "c_change_from_baseline": c_value - baseline["scores"]["C"]["integral"],
        "terrain_total_change": land["total_change"],
        "terrain_changed_edges": land["changed_edges"],
    }


def run_brain(args, brain: int, checkpoints: list[int], output: Path) -> dict:
    seed = args.seed + brain
    config = WaveConfig(seed=seed)
    learned = SphereWaveCore(config)
    control = SphereWaveCore(config)
    rng = np.random.default_rng(seed)
    area = regions(learned, args.region_radius)
    perturb(learned, rng, args.initial_terrain_noise)
    control.conductivity = learned.conductivity.copy()
    initial = learned.conductivity.copy()

    reflection_config = ExperienceConfig(
        learning_rate=args.experience_learning_rate,
        temporal_decay=args.temporal_decay,
        spatial_passes=args.spatial_passes,
        spatial_decay=args.spatial_decay,
    )
    reflector = ExperienceReflector(reflection_config)
    learned_base = probe(learned, area, "learned_0")
    control_base = probe(control, area, "control_0")
    rows = [
        row(brain, 0, "reflected", learned_base, learned_base, terrain(initial, learned.conductivity, learned.adjacency)),
        row(brain, 0, "control", control_base, control_base, terrain(initial, control.conductivity, control.adjacency)),
    ]
    records = []

    for repetition in range(1, args.repetitions + 1):
        delay = max(0, args.delay + int(rng.integers(-args.delay_jitter, args.delay_jitter + 1)))
        a_strength = max(0.05, 1.0 + float(rng.normal(0.0, args.strength_jitter)))
        c_strength = max(0.05, 1.0 + float(rng.normal(0.0, args.strength_jitter)))
        experience(control, area, repetition, delay, a_strength, c_strength, None)
        records.append(experience(learned, area, repetition, delay, a_strength, c_strength, reflector))

        if repetition in checkpoints:
            for mode, core, baseline in (
                ("control", control, control_base),
                ("reflected", learned, learned_base),
            ):
                seen = probe(core, area, f"{mode}_{repetition}")
                rows.append(row(brain, repetition, mode, seen, baseline, terrain(initial, core.conductivity, core.adjacency)))

    brain_dir = output / f"brain_{brain:02d}"
    brain_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("growth_curve.csv", rows), ("training.csv", records)):
        with (brain_dir / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    result = {
        "brain": brain,
        "seed": seed,
        "regions": {key: list(value) for key, value in area.items()},
        "experience_config": asdict(reflection_config),
        "growth": rows,
    }
    (brain_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def arguments():
    parser = argparse.ArgumentParser(description=NAME)
    parser.add_argument("--brains", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--checkpoints", default="10,25,50,100,200")
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--delay-jitter", type=int, default=2)
    parser.add_argument("--strength-jitter", type=float, default=0.08)
    parser.add_argument("--initial-terrain-noise", type=float, default=0.012)
    parser.add_argument("--region-radius", type=int, default=2)
    parser.add_argument("--experience-learning-rate", type=float, default=0.00035)
    parser.add_argument("--temporal-decay", type=float, default=0.94)
    parser.add_argument("--spatial-passes", type=int, default=8)
    parser.add_argument("--spatial-decay", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=2800)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_006"))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    checkpoints = sorted({int(value) for value in args.checkpoints.split(",") if value.strip() and 0 < int(value) <= args.repetitions})
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_brain(args, brain, checkpoints, args.output) for brain in range(1, args.brains + 1)]
    rows = [item for result in results for item in result["growth"]]
    with (args.output / "growth_curve.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_reflected = [item for item in rows if item["checkpoint"] == args.repetitions and item["mode"] == "reflected"]
    final_control = [item for item in rows if item["checkpoint"] == args.repetitions and item["mode"] == "control"]
    summary = {
        "experiment": NAME,
        "brains": args.brains,
        "repetitions": args.repetitions,
        "checkpoints": checkpoints,
        "reflected_mean_c_change": float(np.mean([item["c_change_from_baseline"] for item in final_reflected])),
        "control_mean_c_change": float(np.mean([item["c_change_from_baseline"] for item in final_control])),
        "reflected_c_winners": sum(item["winner"] == "C" for item in final_reflected),
        "control_c_winners": sum(item["winner"] == "C" for item in final_control),
        "note": "C does not need to win. Compare gradual C response against the local-only control.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(NAME)
    print(f"brains: {args.brains}")
    print(f"repetitions: {args.repetitions}")
    print(f"reflected mean C change: {summary['reflected_mean_c_change']:.12g}")
    print(f"control mean C change: {summary['control_mean_c_change']:.12g}")
    print(f"reflected C winners: {summary['reflected_c_winners']}/{args.brains}")
    print(f"control C winners: {summary['control_c_winners']}/{args.brains}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
