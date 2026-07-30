from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from wave_core.attractor import AttractorConfig, AttractorSphereCore, AttractorTrace


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / denominator)


def mean_pairwise_similarity(patterns: list[np.ndarray]) -> float:
    values: list[float] = []
    for left in range(len(patterns)):
        for right in range(left + 1, len(patterns)):
            values.append(cosine_similarity(patterns[left], patterns[right]))
    return float(np.mean(values)) if values else 1.0


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
                "center_x": snapshot.center[0],
                "center_y": snapshot.center[1],
                "center_z": snapshot.center[2],
                "mean_excitation": float(np.mean(snapshot.excitation)),
                "mean_inhibition": float(np.mean(snapshot.inhibition)),
                "mean_fatigue": float(np.mean(snapshot.fatigue)),
            }
        )
    write_csv(path, rows)


def run_experience(core: AttractorSphereCore, a_region: tuple[int, ...], c_region: tuple[int, ...]) -> np.ndarray:
    core.reset_activity()
    patterns: list[np.ndarray] = []

    core.stimulate(a_region, strength=1.0)
    for _ in range(5):
        patterns.append(core.step(learn=True).activity)

    core.stimulate(c_region, strength=1.0)
    for _ in range(7):
        patterns.append(core.step(learn=True).activity)

    for _ in range(8):
        patterns.append(core.step(learn=True).activity)

    return np.mean(np.stack(patterns, axis=0), axis=0)


def recall_trials(
    core: AttractorSphereCore,
    a_region: tuple[int, ...],
    trials: int,
    experience_pattern: np.ndarray,
    output_dir: Path,
    label: str,
) -> dict[str, object]:
    patterns: list[np.ndarray] = []
    lifetimes: list[int] = []
    active_counts: list[float] = []
    experience_similarities: list[float] = []

    for trial in range(1, trials + 1):
        core.reset_activity()
        core.stimulate(a_region, strength=1.0)
        trace = core.run_until_settled(name=f"{label}_trial_{trial}", learn=False)
        pattern = trace.final_pattern
        patterns.append(pattern)
        lifetimes.append(trace.lifetime)
        active_counts.append(float(np.mean([item.active_count for item in trace.snapshots[-5:]])))
        experience_similarities.append(cosine_similarity(pattern, experience_pattern))

        if trial == 1:
            record_trace(output_dir / f"{label}_trial_01_steps.csv", trace)
            np.savetxt(
                output_dir / f"{label}_trial_01_final_pattern.csv",
                pattern,
                delimiter=",",
                header="activity",
                comments="",
            )

    return {
        "mode": label,
        "mean_lifetime": float(np.mean(lifetimes)),
        "mean_final_total_activity": float(np.mean([np.sum(pattern) for pattern in patterns])),
        "mean_active_cluster_size": float(np.mean(active_counts)),
        "trial_to_trial_similarity": mean_pairwise_similarity(patterns),
        "experience_pattern_similarity": float(np.mean(experience_similarities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 009: Spontaneous State Formation")
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoints", default="0,100,200,300,400,500")
    args = parser.parse_args()

    checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    checkpoints = [value for value in checkpoints if 0 <= value <= args.repetitions]
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)

    output_root = Path("results") / "experiment_009"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print("Experiment 009: Spontaneous State Formation")
    print(f"brains: {args.brains}")
    print(f"repetitions: {args.repetitions}")
    print(f"trials: {args.trials}")

    for brain_index in range(1, args.brains + 1):
        config = AttractorConfig(seed=27 + brain_index - 1)
        trained = AttractorSphereCore(config)
        control = trained.clone()

        a_anchor = 12
        c_anchor = config.node_count // 2 + 31
        a_region = trained.stimulus_region(a_anchor, radius=2)
        c_region = trained.stimulus_region(c_anchor, radius=2)

        brain_dir = output_root / f"brain_{brain_index:02d}"
        brain_dir.mkdir(parents=True, exist_ok=True)

        blank_pattern = np.zeros(config.node_count, dtype=float)
        latest_experience_pattern = blank_pattern

        for repetition in range(args.repetitions + 1):
            if repetition in checkpoints:
                trained_metrics = recall_trials(
                    trained,
                    a_region,
                    args.trials,
                    latest_experience_pattern,
                    brain_dir,
                    f"checkpoint_{repetition:04d}_trained",
                )
                control_metrics = recall_trials(
                    control,
                    a_region,
                    args.trials,
                    blank_pattern,
                    brain_dir,
                    f"checkpoint_{repetition:04d}_control",
                )
                for metrics in (control_metrics, trained_metrics):
                    summary_rows.append(
                        {
                            "brain": brain_index,
                            "checkpoint": repetition,
                            **metrics,
                            "mean_direction": float(np.mean(trained.direction[trained.adjacency]))
                            if metrics["mode"].endswith("trained")
                            else float(np.mean(control.direction[control.adjacency])),
                            "mean_capacity": float(np.mean(trained.capacity[trained.adjacency]))
                            if metrics["mode"].endswith("trained")
                            else float(np.mean(control.capacity[control.adjacency])),
                        }
                    )

            if repetition < args.repetitions:
                latest_experience_pattern = run_experience(trained, a_region, c_region)

    write_csv(output_root / "state_formation_curve.csv", summary_rows)

    final_checkpoint = args.repetitions
    final_rows = [row for row in summary_rows if row["checkpoint"] == final_checkpoint]
    for mode in ("control", "trained"):
        selected = [row for row in final_rows if str(row["mode"]).endswith(mode)]
        if selected:
            print(
                f"{mode} mean lifetime: "
                f"{np.mean([float(row['mean_lifetime']) for row in selected]):.12g}"
            )
            print(
                f"{mode} trial similarity: "
                f"{np.mean([float(row['trial_to_trial_similarity']) for row in selected]):.12g}"
            )
            print(
                f"{mode} experience similarity: "
                f"{np.mean([float(row['experience_pattern_similarity']) for row in selected]):.12g}"
            )

    print(f"output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
