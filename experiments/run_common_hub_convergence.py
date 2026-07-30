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


def pattern_concentration(pattern: np.ndarray) -> float:
    total = float(np.sum(pattern))
    if total <= 1e-15:
        return 0.0
    probabilities = pattern / total
    return float(np.sum(probabilities * probabilities))


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
    """Apply source then destination and return phase-level experience patterns."""
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


def trajectory_metrics(trace: AttractorTrace, region: tuple[int, ...]) -> dict[str, float]:
    values = np.asarray([region_mean(snapshot.activity, region) for snapshot in trace.snapshots], dtype=float)
    if values.size == 0:
        return {
            "P_peak_activity": 0.0,
            "P_cumulative_activity": 0.0,
            "P_mean_activity": 0.0,
            "P_peak_step": 0.0,
            "P_final_activity": 0.0,
        }
    peak_index = int(np.argmax(values))
    return {
        "P_peak_activity": float(values[peak_index]),
        "P_cumulative_activity": float(np.sum(values)),
        "P_mean_activity": float(np.mean(values)),
        "P_peak_step": float(trace.snapshots[peak_index].step),
        "P_final_activity": float(values[-1]),
    }


def record_trace(path: Path, trace: AttractorTrace, named_regions: dict[str, tuple[int, ...]]) -> None:
    rows: list[dict[str, object]] = []
    for snapshot in trace.snapshots:
        row: dict[str, object] = {
            "step": snapshot.step,
            "total_activity": snapshot.total_activity,
            "active_count": snapshot.active_count,
            "active_fraction": snapshot.active_count / len(snapshot.activity),
            "pattern_concentration": pattern_concentration(snapshot.activity),
            "center_x": snapshot.center[0],
            "center_y": snapshot.center[1],
            "center_z": snapshot.center[2],
            "mean_fatigue": float(np.mean(snapshot.fatigue)),
        }
        for name, region in named_regions.items():
            row[f"region_{name}_activity"] = region_mean(snapshot.activity, region)
        rows.append(row)
    write_csv(path, rows)


def run_recall(
    core: AttractorSphereCore,
    stimulus_region: tuple[int, ...],
    p_region: tuple[int, ...],
    label: str,
    output_dir: Path,
    trials: int,
    named_regions: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, float]]:
    patterns: list[np.ndarray] = []
    metric_lists: dict[str, list[float]] = {
        "mean_lifetime": [],
        "mean_active_fraction": [],
        "mean_concentration": [],
        "P_peak_activity": [],
        "P_cumulative_activity": [],
        "P_mean_activity": [],
        "P_peak_step": [],
        "P_final_activity": [],
    }

    for trial_index in range(1, trials + 1):
        core.reset_activity()
        core.stimulate(stimulus_region, strength=1.0)
        trace = core.run_until_settled(name=f"{label}_trial_{trial_index}", learn=False)
        final_pattern = trace.final_pattern
        patterns.append(final_pattern)

        metric_lists["mean_lifetime"].append(float(trace.lifetime))
        metric_lists["mean_active_fraction"].append(
            float(np.mean([s.active_count / core.config.node_count for s in trace.snapshots[-5:]]))
        )
        metric_lists["mean_concentration"].append(pattern_concentration(final_pattern))
        for name, value in trajectory_metrics(trace, p_region).items():
            metric_lists[name].append(value)

        if trial_index == 1:
            record_trace(output_dir / f"{label}_trial_01_steps.csv", trace, named_regions)
            np.savetxt(
                output_dir / f"{label}_trial_01_final_pattern.csv",
                final_pattern,
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
    metrics["mean_total_activity"] = float(np.sum(mean_pattern))
    return mean_pattern, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 012: Common Hub Convergence")
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoints", default="0,100,200,300,400,500")
    args = parser.parse_args()

    checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    checkpoints = [value for value in checkpoints if 0 <= value <= args.repetitions]
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)

    output_root = Path("results") / "experiment_012"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print("Experiment 012: Common Hub Convergence")
    print("shared destination: A -> P and F -> P")
    print("disconnected comparison: A -> Q and F -> R")
    print(f"brains: {args.brains}")
    print(f"repetitions per experience: {args.repetitions}")
    print(f"trials: {args.trials}")

    for brain_index in range(1, args.brains + 1):
        config = AttractorConfig(seed=121 + brain_index - 1)
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
            # P is intentionally broader: the shared concept is treated as a region, not one point.
            "P": shared.stimulus_region(anchors["P"], radius=3),
            "Q": shared.stimulus_region(anchors["Q"], radius=3),
            "R": shared.stimulus_region(anchors["R"], radius=3),
        }

        brain_dir = output_root / f"brain_{brain_index:02d}"
        brain_dir.mkdir(parents=True, exist_ok=True)

        latest_ap = {name: np.zeros(config.node_count) for name in ("source", "destination", "settled")}
        latest_fp = {name: np.zeros(config.node_count) for name in ("source", "destination", "settled")}

        for repetition in range(args.repetitions + 1):
            if repetition in checkpoints:
                results: dict[tuple[str, str], tuple[np.ndarray, dict[str, float]]] = {}
                for mode, core in (("untrained", untrained), ("disconnected", disconnected), ("shared", shared)):
                    for cue in ("A", "F", "P"):
                        results[(mode, cue)] = run_recall(
                            core,
                            regions[cue],
                            regions["P"],
                            f"checkpoint_{repetition:04d}_{mode}_{cue}",
                            brain_dir,
                            args.trials,
                            {name: regions[name] for name in ("A", "F", "P", "Q", "R")},
                        )

                similarities = {
                    mode: cosine_similarity(results[(mode, "A")][0], results[(mode, "F")][0])
                    for mode in ("untrained", "disconnected", "shared")
                }
                shared_p_pattern = results[("shared", "P")][0]

                for (mode, cue), (pattern, metrics) in results.items():
                    if mode == "shared" and cue == "A":
                        own_destination = latest_ap["destination"]
                        own_settled = latest_ap["settled"]
                    elif mode == "shared" and cue == "F":
                        own_destination = latest_fp["destination"]
                        own_settled = latest_fp["settled"]
                    elif mode == "shared" and cue == "P":
                        own_destination = np.mean(
                            np.stack([latest_ap["destination"], latest_fp["destination"]]), axis=0
                        )
                        own_settled = np.mean(np.stack([latest_ap["settled"], latest_fp["settled"]]), axis=0)
                    else:
                        own_destination = np.zeros(config.node_count)
                        own_settled = np.zeros(config.node_count)

                    comparison_metrics = results[("disconnected", cue)][1]
                    summary_rows.append(
                        {
                            "brain": brain_index,
                            "checkpoint": repetition,
                            "mode": mode,
                            "cue": cue,
                            **metrics,
                            "recall_to_P_cue_similarity": cosine_similarity(pattern, shared_p_pattern)
                            if mode == "shared"
                            else 0.0,
                            "destination_experience_similarity": cosine_similarity(pattern, own_destination),
                            "settled_experience_similarity": cosine_similarity(pattern, own_settled),
                            "A_vs_F_pattern_similarity": similarities[mode],
                            "convergence_gain_vs_disconnected": (
                                similarities["shared"] - similarities["disconnected"]
                                if mode == "shared"
                                else 0.0
                            ),
                            "P_peak_gain_vs_disconnected": (
                                metrics["P_peak_activity"] - comparison_metrics["P_peak_activity"]
                                if mode == "shared"
                                else 0.0
                            ),
                            "P_cumulative_gain_vs_disconnected": (
                                metrics["P_cumulative_activity"] - comparison_metrics["P_cumulative_activity"]
                                if mode == "shared"
                                else 0.0
                            ),
                            "P_final_gain_vs_disconnected": (
                                metrics["P_final_activity"] - comparison_metrics["P_final_activity"]
                                if mode == "shared"
                                else 0.0
                            ),
                        }
                    )

            if repetition < args.repetitions:
                if repetition % 2 == 0:
                    latest_ap = run_pair_experience(shared, regions["A"], regions["P"])
                    latest_fp = run_pair_experience(shared, regions["F"], regions["P"])
                    run_pair_experience(disconnected, regions["A"], regions["Q"])
                    run_pair_experience(disconnected, regions["F"], regions["R"])
                else:
                    latest_fp = run_pair_experience(shared, regions["F"], regions["P"])
                    latest_ap = run_pair_experience(shared, regions["A"], regions["P"])
                    run_pair_experience(disconnected, regions["F"], regions["R"])
                    run_pair_experience(disconnected, regions["A"], regions["Q"])

    write_csv(output_root / "common_hub_convergence_curve.csv", summary_rows)

    final_rows = [
        row for row in summary_rows
        if row["checkpoint"] == args.repetitions and row["mode"] == "shared"
    ]
    print("final shared-destination results:")
    for row in final_rows:
        print(
            f"cue {row['cue']}: P peak={float(row['P_peak_activity']):.12g}, "
            f"P cumulative={float(row['P_cumulative_activity']):.12g}, "
            f"P final={float(row['P_final_activity']):.12g}, "
            f"P-cue similarity={float(row['recall_to_P_cue_similarity']):.12g}"
        )
    if final_rows:
        print(f"A-vs-F similarity: {float(final_rows[0]['A_vs_F_pattern_similarity']):.12g}")
        print(f"convergence gain vs disconnected: {float(final_rows[0]['convergence_gain_vs_disconnected']):.12g}")
    print(f"output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
