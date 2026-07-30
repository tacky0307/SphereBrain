from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wave_core.attractor import AttractorConfig, AttractorSphereCore, AttractorTrace
from wave_core.experience_field import ExperienceFieldAttractorCore, ExperienceFieldConfig


def region_mean(pattern: np.ndarray, region: tuple[int, ...]) -> float:
    return float(np.mean(pattern[np.asarray(region, dtype=int)])) if region else 0.0


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-15 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_pair(
    core: AttractorSphereCore,
    source: tuple[int, ...],
    destination: tuple[int, ...],
    use_field: bool,
) -> np.ndarray:
    core.reset_activity()
    if use_field:
        assert isinstance(core, ExperienceFieldAttractorCore)
        core.begin_experience()

    history: list[np.ndarray] = []
    core.stimulate(source, 1.0)
    for _ in range(5):
        history.append(core.step(learn=True).activity.copy())

    core.stimulate(destination, 1.0)
    for _ in range(7):
        history.append(core.step(learn=True).activity.copy())

    for _ in range(8):
        history.append(core.step(learn=True).activity.copy())

    if use_field:
        core.end_experience()
    return np.mean(np.stack(history, axis=0), axis=0)


def recall(
    core: AttractorSphereCore,
    cue: tuple[int, ...],
    p_region: tuple[int, ...],
    trials: int,
    name: str,
    output: Path,
) -> tuple[np.ndarray, dict[str, float]]:
    patterns: list[np.ndarray] = []
    peaks: list[float] = []
    cumulatives: list[float] = []
    finals: list[float] = []

    for trial in range(1, trials + 1):
        core.reset_activity()
        core.stimulate(cue, 1.0)
        trace = core.run_until_settled(f"{name}_{trial}", learn=False)
        patterns.append(trace.final_pattern)
        values = [region_mean(item.activity, p_region) for item in trace.snapshots]
        peaks.append(max(values, default=0.0))
        cumulatives.append(float(sum(values)))
        finals.append(values[-1] if values else 0.0)
        if trial == 1:
            write_csv(
                output / f"{name}_trial_01_steps.csv",
                [
                    {
                        "step": item.step,
                        "total_activity": item.total_activity,
                        "P_activity": region_mean(item.activity, p_region),
                    }
                    for item in trace.snapshots
                ],
            )

    mean_pattern = np.mean(np.stack(patterns, axis=0), axis=0)
    pairwise = [
        cosine_similarity(patterns[i], patterns[j])
        for i in range(len(patterns))
        for j in range(i + 1, len(patterns))
    ]
    return mean_pattern, {
        "P_peak_activity": float(np.mean(peaks)),
        "P_cumulative_activity": float(np.mean(cumulatives)),
        "P_final_activity": float(np.mean(finals)),
        "trial_similarity": float(np.mean(pairwise)) if pairwise else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 014: Experience Field Association")
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoints", default="0,100,200,300,400,500")
    parser.add_argument("--trace-decay", type=float, default=0.92)
    parser.add_argument("--field-direction-rate", type=float, default=0.0010)
    parser.add_argument("--field-capacity-rate", type=float, default=0.0006)
    args = parser.parse_args()

    checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    checkpoints = [value for value in checkpoints if 0 <= value <= args.repetitions]
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)

    output = Path("results") / "experiment_014"
    rows: list[dict[str, object]] = []

    print("Experiment 014: Experience Field Association")
    print("training: A -> P and F -> P")
    print("comparison: local plasticity vs transient experience field")

    for brain in range(1, args.brains + 1):
        attractor_config = AttractorConfig(seed=141 + brain - 1)
        field_config = ExperienceFieldConfig(
            trace_decay=args.trace_decay,
            directional_learning_rate=args.field_direction_rate,
            capacity_learning_rate=args.field_capacity_rate,
        )
        field = ExperienceFieldAttractorCore(attractor_config, field_config)
        local = AttractorSphereCore(attractor_config)
        untrained = AttractorSphereCore(attractor_config)

        anchors = {
            "A": 12,
            "P": attractor_config.node_count // 2,
            "F": attractor_config.node_count // 3 + 7,
        }
        regions = {
            "A": field.stimulus_region(anchors["A"], radius=2),
            "F": field.stimulus_region(anchors["F"], radius=2),
            "P": field.stimulus_region(anchors["P"], radius=3),
        }
        brain_dir = output / f"brain_{brain:02d}"
        latest = {
            "field_A": np.zeros(attractor_config.node_count),
            "field_F": np.zeros(attractor_config.node_count),
            "local_A": np.zeros(attractor_config.node_count),
            "local_F": np.zeros(attractor_config.node_count),
        }

        for repetition in range(args.repetitions + 1):
            if repetition in checkpoints:
                results: dict[tuple[str, str], tuple[np.ndarray, dict[str, float]]] = {}
                for mode, core in (("untrained", untrained), ("local", local), ("field", field)):
                    for cue in ("A", "F", "P"):
                        results[(mode, cue)] = recall(
                            core,
                            regions[cue],
                            regions["P"],
                            args.trials,
                            f"checkpoint_{repetition:04d}_{mode}_{cue}",
                            brain_dir,
                        )

                for (mode, cue), (pattern, metrics) in results.items():
                    comparison = results[("local", cue)][1]
                    experience_pattern = (
                        latest[f"{mode}_{cue}"]
                        if mode in ("field", "local") and cue in ("A", "F")
                        else np.zeros(attractor_config.node_count)
                    )
                    rows.append(
                        {
                            "brain": brain,
                            "checkpoint": repetition,
                            "mode": mode,
                            "cue": cue,
                            **metrics,
                            "experience_similarity": cosine_similarity(pattern, experience_pattern),
                            "P_peak_gain_vs_local": metrics["P_peak_activity"] - comparison["P_peak_activity"]
                            if mode == "field"
                            else 0.0,
                            "P_cumulative_gain_vs_local": metrics["P_cumulative_activity"]
                            - comparison["P_cumulative_activity"]
                            if mode == "field"
                            else 0.0,
                        }
                    )

            if repetition < args.repetitions:
                order = ("A", "F") if repetition % 2 == 0 else ("F", "A")
                for cue in order:
                    latest[f"local_{cue}"] = train_pair(local, regions[cue], regions["P"], False)
                    latest[f"field_{cue}"] = train_pair(field, regions[cue], regions["P"], True)

    write_csv(output / "experience_field_association_curve.csv", rows)
    final = [
        row for row in rows
        if row["checkpoint"] == args.repetitions and row["mode"] == "field"
    ]
    print("final experience-field results:")
    for row in final:
        print(
            f"cue {row['cue']}: P peak={float(row['P_peak_activity']):.12g}, "
            f"P cumulative={float(row['P_cumulative_activity']):.12g}, "
            f"peak gain vs local={float(row['P_peak_gain_vs_local']):.12g}, "
            f"cumulative gain vs local={float(row['P_cumulative_gain_vs_local']):.12g}"
        )
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
