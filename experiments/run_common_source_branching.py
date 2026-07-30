from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wave_core.attractor import AttractorConfig, AttractorSphereCore, AttractorTrace


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(left, right) / denominator)


def region_mean(pattern: np.ndarray, region: tuple[int, ...]) -> float:
    if not region:
        return 0.0
    return float(np.mean(pattern[np.asarray(region, dtype=int)]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_pair_experience(
    core: AttractorSphereCore,
    source_region: tuple[int, ...],
    destination_region: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Apply source then destination and retain phase summaries."""
    core.reset_activity()
    phases: dict[str, list[np.ndarray]] = {"source": [], "destination": [], "settled": []}

    core.stimulate(source_region, strength=1.0)
    for _ in range(5):
        phases["source"].append(core.step(learn=True).activity.copy())

    core.stimulate(destination_region, strength=1.0)
    for _ in range(7):
        phases["destination"].append(core.step(learn=True).activity.copy())

    for _ in range(8):
        phases["settled"].append(core.step(learn=True).activity.copy())

    return {
        name: np.mean(np.stack(patterns, axis=0), axis=0)
        for name, patterns in phases.items()
    }


def trajectory_metrics(
    trace: AttractorTrace,
    named_regions: dict[str, tuple[int, ...]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, region in named_regions.items():
        values = np.asarray(
            [region_mean(snapshot.activity, region) for snapshot in trace.snapshots],
            dtype=float,
        )
        if values.size == 0:
            metrics[f"{name}_peak_activity"] = 0.0
            metrics[f"{name}_cumulative_activity"] = 0.0
            metrics[f"{name}_peak_step"] = 0.0
            metrics[f"{name}_final_activity"] = 0.0
            continue
        peak_index = int(np.argmax(values))
        metrics[f"{name}_peak_activity"] = float(values[peak_index])
        metrics[f"{name}_cumulative_activity"] = float(np.sum(values))
        metrics[f"{name}_peak_step"] = float(trace.snapshots[peak_index].step)
        metrics[f"{name}_final_activity"] = float(values[-1])
    return metrics


def record_trace(
    path: Path,
    trace: AttractorTrace,
    named_regions: dict[str, tuple[int, ...]],
) -> None:
    rows: list[dict[str, object]] = []
    for snapshot in trace.snapshots:
        row: dict[str, object] = {
            "step": snapshot.step,
            "total_activity": snapshot.total_activity,
            "active_count": snapshot.active_count,
            "mean_fatigue": float(np.mean(snapshot.fatigue)),
        }
        for name, region in named_regions.items():
            row[f"region_{name}_activity"] = region_mean(snapshot.activity, region)
        rows.append(row)
    write_csv(path, rows)


def run_recall(
    core: AttractorSphereCore,
    cue_regions: tuple[tuple[tuple[int, ...], float], ...],
    label: str,
    output_dir: Path,
    trials: int,
    named_regions: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, float]]:
    patterns: list[np.ndarray] = []
    metric_lists: dict[str, list[float]] = {
        "mean_lifetime": [],
        "mean_total_activity": [],
    }
    for name in named_regions:
        for suffix in ("peak_activity", "cumulative_activity", "peak_step", "final_activity"):
            metric_lists[f"{name}_{suffix}"] = []

    for trial_index in range(1, trials + 1):
        core.reset_activity()
        for region, strength in cue_regions:
            core.stimulate(region, strength=strength)
        trace = core.run_until_settled(name=f"{label}_trial_{trial_index}", learn=False)
        patterns.append(trace.final_pattern)
        metric_lists["mean_lifetime"].append(float(trace.lifetime))
        metric_lists["mean_total_activity"].append(float(np.sum(trace.final_pattern)))
        for name, value in trajectory_metrics(trace, named_regions).items():
            metric_lists[name].append(value)

        if trial_index == 1:
            record_trace(output_dir / f"{label}_trial_01_steps.csv", trace, named_regions)
            np.savetxt(
                output_dir / f"{label}_trial_01_final_pattern.csv",
                trace.final_pattern,
                delimiter=",",
                header="activity",
                comments="",
            )

    mean_pattern = np.mean(np.stack(patterns, axis=0), axis=0)
    pairwise: list[float] = []
    for left_index in range(len(patterns)):
        for right_index in range(left_index + 1, len(patterns)):
            pairwise.append(cosine_similarity(patterns[left_index], patterns[right_index]))

    metrics = {name: float(np.mean(values)) for name, values in metric_lists.items()}
    metrics["trial_similarity"] = float(np.mean(pairwise)) if pairwise else 1.0
    metrics["A_peak_selectivity"] = metrics["A_peak_activity"] - metrics["F_peak_activity"]
    metrics["A_cumulative_selectivity"] = (
        metrics["A_cumulative_activity"] - metrics["F_cumulative_activity"]
    )
    metrics["A_final_selectivity"] = metrics["A_final_activity"] - metrics["F_final_activity"]
    return mean_pattern, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 013: Common Source Branching")
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoints", default="0,100,200,300,400,500")
    parser.add_argument("--context-strength", type=float, default=0.15)
    args = parser.parse_args()

    checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    checkpoints = [value for value in checkpoints if 0 <= value <= args.repetitions]
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)

    output_root = Path("results") / "experiment_013"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print("Experiment 013: Common Source Branching")
    print("shared source: P -> A and P -> F")
    print("disconnected comparison: Q -> A and R -> F")
    print(f"brains: {args.brains}")
    print(f"repetitions per experience: {args.repetitions}")
    print(f"trials: {args.trials}")
    print(f"weak context strength: {args.context_strength}")

    for brain_index in range(1, args.brains + 1):
        config = AttractorConfig(seed=131 + brain_index - 1)
        shared = AttractorSphereCore(config)
        disconnected = shared.clone()
        untrained = shared.clone()

        anchors = {
            "A": 12,
            "P": config.node_count // 2,
            "F": config.node_count // 3 + 7,
            "Q": config.node_count // 4 + 19,
            "R": (3 * config.node_count) // 4 - 17,
        }
        regions = {
            "A": shared.stimulus_region(anchors["A"], radius=2),
            "F": shared.stimulus_region(anchors["F"], radius=2),
            "P": shared.stimulus_region(anchors["P"], radius=3),
            "Q": shared.stimulus_region(anchors["Q"], radius=3),
            "R": shared.stimulus_region(anchors["R"], radius=3),
        }
        named_regions = {name: regions[name] for name in ("A", "F", "P", "Q", "R")}
        brain_dir = output_root / f"brain_{brain_index:02d}"
        brain_dir.mkdir(parents=True, exist_ok=True)

        latest_pa = {name: np.zeros(config.node_count) for name in ("source", "destination", "settled")}
        latest_pf = {name: np.zeros(config.node_count) for name in ("source", "destination", "settled")}

        cue_specs = {
            "P": ((regions["P"], 1.0),),
            "P_weak_A": ((regions["P"], 1.0), (regions["A"], args.context_strength)),
            "P_weak_F": ((regions["P"], 1.0), (regions["F"], args.context_strength)),
            "A": ((regions["A"], 1.0),),
            "F": ((regions["F"], 1.0),),
        }

        for repetition in range(args.repetitions + 1):
            if repetition in checkpoints:
                results: dict[tuple[str, str], tuple[np.ndarray, dict[str, float]]] = {}
                for mode, core in (("untrained", untrained), ("disconnected", disconnected), ("shared", shared)):
                    for cue, cue_regions in cue_specs.items():
                        results[(mode, cue)] = run_recall(
                            core,
                            cue_regions,
                            f"checkpoint_{repetition:04d}_{mode}_{cue}",
                            brain_dir,
                            args.trials,
                            named_regions,
                        )

                for (mode, cue), (pattern, metrics) in results.items():
                    if mode == "shared":
                        similarity_to_a_experience = cosine_similarity(pattern, latest_pa["destination"])
                        similarity_to_f_experience = cosine_similarity(pattern, latest_pf["destination"])
                    else:
                        similarity_to_a_experience = 0.0
                        similarity_to_f_experience = 0.0

                    context_selectivity = 0.0
                    context_pattern_separation = 0.0
                    if mode == "shared":
                        weak_a = results[("shared", "P_weak_A")][1]
                        weak_f = results[("shared", "P_weak_F")][1]
                        context_selectivity = (
                            weak_a["A_peak_selectivity"] - weak_f["A_peak_selectivity"]
                        )
                        context_pattern_separation = 1.0 - cosine_similarity(
                            results[("shared", "P_weak_A")][0],
                            results[("shared", "P_weak_F")][0],
                        )

                    summary_rows.append(
                        {
                            "brain": brain_index,
                            "checkpoint": repetition,
                            "mode": mode,
                            "cue": cue,
                            **metrics,
                            "similarity_to_A_experience": similarity_to_a_experience,
                            "similarity_to_F_experience": similarity_to_f_experience,
                            "experience_preference": similarity_to_a_experience - similarity_to_f_experience,
                            "context_selectivity": context_selectivity,
                            "context_pattern_separation": context_pattern_separation,
                        }
                    )

            if repetition < args.repetitions:
                if repetition % 2 == 0:
                    latest_pa = run_pair_experience(shared, regions["P"], regions["A"])
                    latest_pf = run_pair_experience(shared, regions["P"], regions["F"])
                    run_pair_experience(disconnected, regions["Q"], regions["A"])
                    run_pair_experience(disconnected, regions["R"], regions["F"])
                else:
                    latest_pf = run_pair_experience(shared, regions["P"], regions["F"])
                    latest_pa = run_pair_experience(shared, regions["P"], regions["A"])
                    run_pair_experience(disconnected, regions["R"], regions["F"])
                    run_pair_experience(disconnected, regions["Q"], regions["A"])

    write_csv(output_root / "common_source_branching_curve.csv", summary_rows)

    final_rows = [
        row for row in summary_rows
        if row["checkpoint"] == args.repetitions and row["mode"] == "shared"
    ]
    print("final shared-source results:")
    for row in final_rows:
        print(
            f"cue {row['cue']}: A peak={float(row['A_peak_activity']):.12g}, "
            f"F peak={float(row['F_peak_activity']):.12g}, "
            f"A peak selectivity={float(row['A_peak_selectivity']):.12g}, "
            f"trial similarity={float(row['trial_similarity']):.12g}"
        )
    if final_rows:
        print(f"context selectivity: {float(final_rows[0]['context_selectivity']):.12g}")
        print(f"context pattern separation: {float(final_rows[0]['context_pattern_separation']):.12g}")
    print(f"output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
