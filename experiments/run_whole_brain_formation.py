from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wave_core import SphereWaveCore, WaveConfig


EXPERIMENT_NAME = "Experiment 005: Whole Brain Formation"


def choose_anchor(core: SphereWaveCore, target: np.ndarray, used: set[int]) -> int:
    distances = np.linalg.norm(core.positions - target[None, :], axis=1)
    for node_id in np.argsort(distances):
        value = int(node_id)
        if value not in used:
            return value
    raise RuntimeError("could not choose a unique anchor")


def build_regions(core: SphereWaveCore, radius: int) -> dict[str, tuple[int, ...]]:
    targets = {
        "A": np.asarray([1.0, 0.0, 0.0]),
        "B": np.asarray([0.0, 1.0, 0.0]),
        "C": np.asarray([-1.0, 0.0, 0.0]),
        "D": np.asarray([0.0, -1.0, 0.0]),
    }
    used: set[int] = set()
    result: dict[str, tuple[int, ...]] = {}
    for name, target in targets.items():
        anchor = choose_anchor(core, target, used)
        used.add(anchor)
        result[name] = core.stimulus_region(anchor, radius=radius)
    return result


def perturb_initial_terrain(
    core: SphereWaveCore,
    rng: np.random.Generator,
    amount: float,
) -> None:
    if amount <= 0:
        return
    noise = rng.normal(0.0, amount, size=core.conductivity.shape)
    noise *= core.adjacency
    core.conductivity += noise
    np.clip(
        core.conductivity,
        core.config.conductivity_min,
        core.config.conductivity_max,
        out=core.conductivity,
    )
    core.conductivity[~core.adjacency] = 0.0


def train_a_to_c(
    core: SphereWaveCore,
    regions: dict[str, tuple[int, ...]],
    repetitions: int,
    base_delay: int,
    rng: np.random.Generator,
    delay_jitter: int,
    strength_jitter: float,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for repetition in range(1, repetitions + 1):
        delay = max(0, base_delay + int(rng.integers(-delay_jitter, delay_jitter + 1)))
        a_strength = max(0.05, 1.0 + float(rng.normal(0.0, strength_jitter)))
        c_strength = max(0.05, 1.0 + float(rng.normal(0.0, strength_jitter)))

        core.reset_activity()
        core.stimulate(regions["A"], strength=a_strength)
        first = core.advance(delay, learn=True, name=f"train_{repetition}_a")
        core.stimulate(regions["C"], strength=c_strength)
        second = core.run_until_quiet(name=f"train_{repetition}_c", learn=True)

        records.append(
            {
                "repetition": repetition,
                "delay": delay,
                "a_strength": a_strength,
                "c_strength": c_strength,
                "steps": first.steps + second.steps,
                "changed_edges": len(first.changed_edges) + len(second.changed_edges),
            }
        )
    return records


def probe(
    core: SphereWaveCore,
    regions: dict[str, tuple[int, ...]],
    name: str,
) -> tuple[Any, dict[str, dict[str, float]], str]:
    core.reset_activity()
    core.stimulate(regions["A"], strength=1.0)
    trace = core.run_until_quiet(name=name, learn=False)

    scores: dict[str, dict[str, float]] = {}
    for region_name in ("B", "C", "D"):
        ids = np.asarray(regions[region_name], dtype=int)
        per_step = [float(np.mean(snapshot.activity[ids])) for snapshot in trace.snapshots]
        scores[region_name] = {
            "peak": max(per_step, default=0.0),
            "integral": float(sum(per_step)),
        }

    winner = max(scores, key=lambda key: (scores[key]["integral"], scores[key]["peak"]))
    return trace, scores, winner


def terrain_metrics(before: np.ndarray, after: np.ndarray, adjacency: np.ndarray) -> dict[str, float | int]:
    delta = after - before
    values = after[adjacency]
    changed = np.abs(delta) > 1e-10
    positive = delta > 1e-10
    return {
        "directed_edges": int(np.count_nonzero(adjacency)),
        "changed_directed_edges": int(np.count_nonzero(changed)),
        "strengthened_directed_edges": int(np.count_nonzero(positive)),
        "total_absolute_change": float(np.sum(np.abs(delta))),
        "mean_conductivity": float(np.mean(values)),
        "std_conductivity": float(np.std(values)),
        "max_conductivity": float(np.max(values)),
        "effective_edge_count": int(np.count_nonzero(values > float(np.mean(values) + np.std(values)))),
    }


def lesion_strongest_edges(
    core: SphereWaveCore,
    fraction: float,
) -> list[tuple[int, int, float]]:
    if fraction <= 0:
        return []
    edge_ids = np.argwhere(core.adjacency)
    strengths = np.asarray([core.conductivity[a, b] for a, b in edge_ids], dtype=float)
    count = min(len(edge_ids), max(1, int(round(len(edge_ids) * fraction))))
    selected = np.argsort(strengths)[-count:]
    lesions: list[tuple[int, int, float]] = []
    for index in selected:
        a, b = edge_ids[index]
        old = float(core.conductivity[a, b])
        core.conductivity[a, b] = core.config.conductivity_min
        lesions.append((int(a), int(b), old))
    return lesions


def save_trace_csv(path: Path, trace, regions: dict[str, tuple[int, ...]]) -> None:
    arrays = {name: np.asarray(ids, dtype=int) for name, ids in regions.items()}
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "total_activity",
                "active_nodes",
                "center_x",
                "center_y",
                "center_z",
                "A_mean",
                "B_mean",
                "C_mean",
                "D_mean",
                "fired_nodes",
            ]
        )
        for snapshot in trace.snapshots:
            writer.writerow(
                [
                    snapshot.step,
                    snapshot.total_activity,
                    int(np.count_nonzero(snapshot.activity >= 0.015)),
                    *snapshot.center,
                    *[
                        float(np.mean(snapshot.activity[arrays[name]]))
                        for name in ("A", "B", "C", "D")
                    ],
                    " ".join(str(value) for value in snapshot.fired_nodes),
                ]
            )


def save_terrain_csv(
    path: Path,
    core: SphereWaveCore,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target", "before", "after", "delta"])
        for source, target in np.argwhere(core.adjacency):
            writer.writerow(
                [
                    int(source),
                    int(target),
                    float(before[source, target]),
                    float(after[source, target]),
                    float(after[source, target] - before[source, target]),
                ]
            )


def run_brain(args: argparse.Namespace, brain_index: int, output_dir: Path) -> dict[str, Any]:
    seed = args.seed + brain_index
    config = WaveConfig(seed=seed)
    core = SphereWaveCore(config)
    rng = np.random.default_rng(seed)
    regions = build_regions(core, radius=args.region_radius)

    perturb_initial_terrain(core, rng, args.initial_terrain_noise)
    initial = core.conductivity.copy()
    baseline_trace, baseline_scores, baseline_winner = probe(core, regions, "baseline_a")

    training_records = train_a_to_c(
        core,
        regions,
        repetitions=args.repetitions,
        base_delay=args.delay,
        rng=rng,
        delay_jitter=args.delay_jitter,
        strength_jitter=args.strength_jitter,
    )
    trained = core.conductivity.copy()
    trained_trace, trained_scores, trained_winner = probe(core, regions, "trained_a")

    before_lesion = core.conductivity.copy()
    lesions = lesion_strongest_edges(core, args.lesion_fraction)
    lesion_trace, lesion_scores, lesion_winner = probe(core, regions, "lesioned_a")
    core.conductivity = before_lesion

    brain_dir = output_dir / f"brain_{brain_index:02d}"
    brain_dir.mkdir(parents=True, exist_ok=True)
    save_trace_csv(brain_dir / "baseline_probe.csv", baseline_trace, regions)
    save_trace_csv(brain_dir / "trained_probe.csv", trained_trace, regions)
    save_trace_csv(brain_dir / "lesioned_probe.csv", lesion_trace, regions)
    save_terrain_csv(brain_dir / "terrain.csv", core, initial, trained)

    with (brain_dir / "training.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_records[0].keys()))
        writer.writeheader()
        writer.writerows(training_records)

    result = {
        "brain": brain_index,
        "seed": seed,
        "regions": {name: list(ids) for name, ids in regions.items()},
        "baseline": {
            "winner": baseline_winner,
            "correct": baseline_winner == "C",
            "scores": baseline_scores,
        },
        "trained": {
            "winner": trained_winner,
            "correct": trained_winner == "C",
            "scores": trained_scores,
        },
        "lesioned": {
            "winner": lesion_winner,
            "correct": lesion_winner == "C",
            "scores": lesion_scores,
            "removed_directed_edges": len(lesions),
        },
        "terrain": terrain_metrics(initial, trained, core.adjacency),
    }
    (brain_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def write_summary_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "brain",
                "seed",
                "baseline_winner",
                "trained_winner",
                "trained_correct",
                "lesioned_winner",
                "lesioned_correct",
                "C_integral",
                "B_integral",
                "D_integral",
                "changed_edges",
                "conductivity_std",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item["brain"],
                    item["seed"],
                    item["baseline"]["winner"],
                    item["trained"]["winner"],
                    int(item["trained"]["correct"]),
                    item["lesioned"]["winner"],
                    int(item["lesioned"]["correct"]),
                    item["trained"]["scores"]["C"]["integral"],
                    item["trained"]["scores"]["B"]["integral"],
                    item["trained"]["scores"]["D"]["integral"],
                    item["terrain"]["changed_directed_edges"],
                    item["terrain"]["std_conductivity"],
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--brains", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--delay-jitter", type=int, default=2)
    parser.add_argument("--strength-jitter", type=float, default=0.08)
    parser.add_argument("--initial-terrain-noise", type=float, default=0.012)
    parser.add_argument("--lesion-fraction", type=float, default=0.03)
    parser.add_argument("--region-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_005"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.brains < 1 or args.repetitions < 1:
        raise ValueError("brains and repetitions must be positive")
    if not 0.0 <= args.lesion_fraction < 1.0:
        raise ValueError("lesion-fraction must be in [0, 1)")

    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_brain(args, index, args.output) for index in range(1, args.brains + 1)]

    trained_correct = sum(int(item["trained"]["correct"]) for item in results)
    lesioned_correct = sum(int(item["lesioned"]["correct"]) for item in results)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "hypothesis": (
            "Different learned terrains can form one functional class: "
            "A stimulation produces C without prescribing an internal route."
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "trained_accuracy": trained_correct / len(results),
        "lesioned_accuracy": lesioned_correct / len(results),
        "brains": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(args.output / "summary.csv", results)

    print(EXPERIMENT_NAME)
    print(f"brains: {len(results)}")
    print(f"trained A -> C accuracy: {trained_correct}/{len(results)}")
    print(f"after lesion accuracy: {lesioned_correct}/{len(results)}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
