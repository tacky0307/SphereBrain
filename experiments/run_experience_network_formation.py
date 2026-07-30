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


def pattern_concentration(pattern: np.ndarray) -> float:
    total = float(np.sum(pattern))
    if total <= 1e-15:
        return 0.0
    probabilities = pattern / total
    return float(np.sum(probabilities * probabilities))


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


def record_trace(path: Path, trace: AttractorTrace) -> None:
    rows: list[dict[str, object]] = []
    for snapshot in trace.snapshots:
        rows.append(
            {
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
        )
    write_csv(path, rows)


def run_sequence(
    core: AttractorSphereCore,
    regions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> dict[str, np.ndarray]:
    """Expose the sphere to a three-stage experience and return phase summaries."""
    core.reset_activity()
    phase_patterns: dict[str, list[np.ndarray]] = {"early": [], "middle": [], "late": [], "settled": []}

    for phase_name, region, steps in (
        ("early", regions[0], 5),
        ("middle", regions[1], 7),
        ("late", regions[2], 7),
    ):
        core.stimulate(region, strength=1.0)
        for _ in range(steps):
            phase_patterns[phase_name].append(core.step(learn=True).activity.copy())

    for _ in range(8):
        phase_patterns["settled"].append(core.step(learn=True).activity.copy())

    return {
        name: np.mean(np.stack(patterns, axis=0), axis=0)
        for name, patterns in phase_patterns.items()
    }


def run_recall(
    core: AttractorSphereCore,
    stimulus_region: tuple[int, ...],
    label: str,
    output_dir: Path,
    trials: int,
    named_regions: dict[str, tuple[int, ...]],
) -> tuple[np.ndarray, dict[str, float]]:
    patterns: list[np.ndarray] = []
    lifetimes: list[float] = []
    active_fractions: list[float] = []
    concentrations: list[float] = []
    region_values: dict[str, list[float]] = {name: [] for name in named_regions}

    for trial_index in range(1, trials + 1):
        core.reset_activity()
        core.stimulate(stimulus_region, strength=1.0)
        trace = core.run_until_settled(name=f"{label}_trial_{trial_index}", learn=False)
        final_pattern = trace.final_pattern
        patterns.append(final_pattern)
        lifetimes.append(float(trace.lifetime))
        active_fractions.append(
            float(np.mean([s.active_count / core.config.node_count for s in trace.snapshots[-5:]]))
        )
        concentrations.append(pattern_concentration(final_pattern))
        for name, region in named_regions.items():
            region_values[name].append(region_mean(final_pattern, region))

        if trial_index == 1:
            record_trace(output_dir / f"{label}_trial_01_steps.csv", trace)
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

    metrics: dict[str, float] = {
        "mean_lifetime": float(np.mean(lifetimes)),
        "mean_active_fraction": float(np.mean(active_fractions)),
        "mean_concentration": float(np.mean(concentrations)),
        "trial_similarity": float(np.mean(pairwise)) if pairwise else 1.0,
        "mean_total_activity": float(np.sum(mean_pattern)),
    }
    for name, values in region_values.items():
        metrics[f"region_{name}_activity"] = float(np.mean(values))
    return mean_pattern, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 011: Experience Network Formation")
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoints", default="0,100,200,300,400,500")
    args = parser.parse_args()

    checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    checkpoints = [value for value in checkpoints if 0 <= value <= args.repetitions]
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)

    output_root = Path("results") / "experiment_011"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print("Experiment 011: Experience Network Formation")
    print("shared network: A -> P -> C and F -> P -> M")
    print("disconnected comparison: A -> Q -> C and F -> R -> M")
    print(f"brains: {args.brains}")
    print(f"repetitions per experience: {args.repetitions}")
    print(f"trials: {args.trials}")

    for brain_index in range(1, args.brains + 1):
        config = AttractorConfig(seed=101 + brain_index - 1)
        shared = AttractorSphereCore(config)
        disconnected = shared.clone()
        untrained = shared.clone()

        anchors = {
            "A": 12,
            "P": config.node_count // 2,
            "C": config.node_count // 2 + 31,
            "F": config.node_count // 3 + 7,
            "M": config.node_count - 29,
            "Q": config.node_count // 4 + 19,
            "R": (3 * config.node_count) // 4 - 17,
        }
        regions = {
            name: shared.stimulus_region(anchor, radius=2)
            for name, anchor in anchors.items()
        }

        brain_dir = output_root / f"brain_{brain_index:02d}"
        brain_dir.mkdir(parents=True, exist_ok=True)

        latest_shared_ac = {name: np.zeros(config.node_count) for name in ("early", "middle", "late", "settled")}
        latest_shared_fm = {name: np.zeros(config.node_count) for name in ("early", "middle", "late", "settled")}

        for repetition in range(args.repetitions + 1):
            if repetition in checkpoints:
                recall_results: dict[tuple[str, str], tuple[np.ndarray, dict[str, float]]] = {}
                for mode, core in (("untrained", untrained), ("disconnected", disconnected), ("shared", shared)):
                    for cue in ("A", "F", "P"):
                        recall_results[(mode, cue)] = run_recall(
                            core,
                            regions[cue],
                            f"checkpoint_{repetition:04d}_{mode}_{cue}",
                            brain_dir,
                            args.trials,
                            {name: regions[name] for name in ("A", "P", "C", "F", "M")},
                        )

                shared_a = recall_results[("shared", "A")][0]
                shared_f = recall_results[("shared", "F")][0]
                disconnected_a = recall_results[("disconnected", "A")][0]
                disconnected_f = recall_results[("disconnected", "F")][0]
                untrained_a = recall_results[("untrained", "A")][0]
                untrained_f = recall_results[("untrained", "F")][0]

                similarities = {
                    "shared": cosine_similarity(shared_a, shared_f),
                    "disconnected": cosine_similarity(disconnected_a, disconnected_f),
                    "untrained": cosine_similarity(untrained_a, untrained_f),
                }

                for (mode, cue), (pattern, metrics) in recall_results.items():
                    if mode == "shared":
                        own = latest_shared_ac["settled"] if cue == "A" else latest_shared_fm["settled"]
                        other = latest_shared_fm["settled"] if cue == "A" else latest_shared_ac["settled"]
                    else:
                        own = np.zeros(config.node_count)
                        other = np.zeros(config.node_count)

                    branch_selectivity = 0.0
                    if cue == "A":
                        branch_selectivity = metrics["region_C_activity"] - metrics["region_M_activity"]
                    elif cue == "F":
                        branch_selectivity = metrics["region_M_activity"] - metrics["region_C_activity"]

                    summary_rows.append(
                        {
                            "brain": brain_index,
                            "checkpoint": repetition,
                            "mode": mode,
                            "cue": cue,
                            **metrics,
                            "own_settled_similarity": cosine_similarity(pattern, own),
                            "other_settled_similarity": cosine_similarity(pattern, other),
                            "experience_selectivity": cosine_similarity(pattern, own) - cosine_similarity(pattern, other),
                            "branch_selectivity": branch_selectivity,
                            "A_vs_F_pattern_similarity": similarities[mode],
                            "shared_hub_gain_vs_disconnected": (
                                metrics["region_P_activity"]
                                - recall_results[("disconnected", cue)][1]["region_P_activity"]
                                if mode == "shared"
                                else 0.0
                            ),
                            "shared_separation_change_vs_disconnected": (
                                similarities["disconnected"] - similarities["shared"]
                                if mode == "shared"
                                else 0.0
                            ),
                        }
                    )

            if repetition < args.repetitions:
                if repetition % 2 == 0:
                    latest_shared_ac = run_sequence(shared, (regions["A"], regions["P"], regions["C"]))
                    latest_shared_fm = run_sequence(shared, (regions["F"], regions["P"], regions["M"]))
                    run_sequence(disconnected, (regions["A"], regions["Q"], regions["C"]))
                    run_sequence(disconnected, (regions["F"], regions["R"], regions["M"]))
                else:
                    latest_shared_fm = run_sequence(shared, (regions["F"], regions["P"], regions["M"]))
                    latest_shared_ac = run_sequence(shared, (regions["A"], regions["P"], regions["C"]))
                    run_sequence(disconnected, (regions["F"], regions["R"], regions["M"]))
                    run_sequence(disconnected, (regions["A"], regions["Q"], regions["C"]))

    write_csv(output_root / "experience_network_curve.csv", summary_rows)

    final_rows = [
        row for row in summary_rows
        if row["checkpoint"] == args.repetitions and row["mode"] == "shared"
    ]
    print("final shared-network results:")
    for row in final_rows:
        print(
            f"cue {row['cue']}: P activity={float(row['region_P_activity']):.12g}, "
            f"hub gain={float(row['shared_hub_gain_vs_disconnected']):.12g}, "
            f"branch selectivity={float(row['branch_selectivity']):.12g}, "
            f"experience selectivity={float(row['experience_selectivity']):.12g}"
        )
    if final_rows:
        print(f"A-vs-F similarity: {float(final_rows[0]['A_vs_F_pattern_similarity']):.12g}")
        print(
            "separation change vs disconnected: "
            f"{float(final_rows[0]['shared_separation_change_vs_disconnected']):.12g}"
        )
    print(f"output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
