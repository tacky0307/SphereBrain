from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow both direct execution and ``python -m`` execution from the repository.
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
                "mean_excitation": float(np.mean(snapshot.excitation)),
                "mean_inhibition": float(np.mean(snapshot.inhibition)),
                "mean_fatigue": float(np.mean(snapshot.fatigue)),
            }
        )
    write_csv(path, rows)


def run_experience(
    core: AttractorSphereCore,
    source_region: tuple[int, ...],
    destination_region: tuple[int, ...],
) -> np.ndarray:
    """Apply one experience without providing a target during later recall."""
    core.reset_activity()
    patterns: list[np.ndarray] = []

    core.stimulate(source_region, strength=1.0)
    for _ in range(5):
        patterns.append(core.step(learn=True).activity.copy())

    core.stimulate(destination_region, strength=1.0)
    for _ in range(7):
        patterns.append(core.step(learn=True).activity.copy())

    for _ in range(8):
        patterns.append(core.step(learn=True).activity.copy())

    return np.mean(np.stack(patterns, axis=0), axis=0)


def run_recall(
    core: AttractorSphereCore,
    stimulus_region: tuple[int, ...],
    label: str,
    output_dir: Path,
    trials: int,
) -> tuple[np.ndarray, dict[str, float]]:
    patterns: list[np.ndarray] = []
    lifetimes: list[float] = []
    active_fractions: list[float] = []
    concentrations: list[float] = []

    for trial_index in range(1, trials + 1):
        core.reset_activity()
        core.stimulate(stimulus_region, strength=1.0)
        trace = core.run_until_settled(name=f"{label}_trial_{trial_index}", learn=False)
        final_pattern = trace.final_pattern
        patterns.append(final_pattern)
        lifetimes.append(float(trace.lifetime))
        active_fractions.append(
            float(
                np.mean(
                    [snapshot.active_count / core.config.node_count for snapshot in trace.snapshots[-5:]]
                )
            )
        )
        concentrations.append(pattern_concentration(final_pattern))

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
    trial_similarities: list[float] = []
    for left_index in range(len(patterns)):
        for right_index in range(left_index + 1, len(patterns)):
            trial_similarities.append(cosine_similarity(patterns[left_index], patterns[right_index]))

    metrics = {
        "mean_lifetime": float(np.mean(lifetimes)),
        "mean_active_fraction": float(np.mean(active_fractions)),
        "mean_concentration": float(np.mean(concentrations)),
        "trial_similarity": float(np.mean(trial_similarities)) if trial_similarities else 1.0,
        "mean_total_activity": float(np.sum(mean_pattern)),
    }
    return mean_pattern, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 010: Multiple Attractor Differentiation")
    parser.add_argument("--brains", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoints", default="0,100,200,300,400,500")
    args = parser.parse_args()

    checkpoints = sorted({int(value) for value in args.checkpoints.split(",")})
    checkpoints = [value for value in checkpoints if 0 <= value <= args.repetitions]
    if args.repetitions not in checkpoints:
        checkpoints.append(args.repetitions)

    output_root = Path("results") / "experiment_010"
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    print("Experiment 010: Multiple Attractor Differentiation")
    print(f"brains: {args.brains}")
    print(f"repetitions per experience: {args.repetitions}")
    print(f"trials: {args.trials}")

    for brain_index in range(1, args.brains + 1):
        config = AttractorConfig(seed=81 + brain_index - 1)
        trained = AttractorSphereCore(config)
        control = trained.clone()

        # Four spatially separated anchors. No destination is used during recall.
        a_anchor = 12
        c_anchor = config.node_count // 2 + 31
        f_anchor = config.node_count // 3 + 7
        m_anchor = config.node_count - 29

        a_region = trained.stimulus_region(a_anchor, radius=2)
        c_region = trained.stimulus_region(c_anchor, radius=2)
        f_region = trained.stimulus_region(f_anchor, radius=2)
        m_region = trained.stimulus_region(m_anchor, radius=2)

        brain_dir = output_root / f"brain_{brain_index:02d}"
        brain_dir.mkdir(parents=True, exist_ok=True)

        latest_pattern_ac = np.zeros(config.node_count, dtype=float)
        latest_pattern_fm = np.zeros(config.node_count, dtype=float)

        for repetition in range(args.repetitions + 1):
            if repetition in checkpoints:
                trained_a, trained_a_metrics = run_recall(
                    trained,
                    a_region,
                    f"checkpoint_{repetition:04d}_trained_A",
                    brain_dir,
                    args.trials,
                )
                trained_f, trained_f_metrics = run_recall(
                    trained,
                    f_region,
                    f"checkpoint_{repetition:04d}_trained_F",
                    brain_dir,
                    args.trials,
                )
                control_a, control_a_metrics = run_recall(
                    control,
                    a_region,
                    f"checkpoint_{repetition:04d}_control_A",
                    brain_dir,
                    args.trials,
                )
                control_f, control_f_metrics = run_recall(
                    control,
                    f_region,
                    f"checkpoint_{repetition:04d}_control_F",
                    brain_dir,
                    args.trials,
                )

                trained_cross_similarity = cosine_similarity(trained_a, trained_f)
                control_cross_similarity = cosine_similarity(control_a, control_f)
                separation_gain = control_cross_similarity - trained_cross_similarity

                for mode, cue, pattern, metrics, own_exp, other_exp in (
                    ("control", "A", control_a, control_a_metrics, np.zeros(config.node_count), np.zeros(config.node_count)),
                    ("control", "F", control_f, control_f_metrics, np.zeros(config.node_count), np.zeros(config.node_count)),
                    ("trained", "A", trained_a, trained_a_metrics, latest_pattern_ac, latest_pattern_fm),
                    ("trained", "F", trained_f, trained_f_metrics, latest_pattern_fm, latest_pattern_ac),
                ):
                    summary_rows.append(
                        {
                            "brain": brain_index,
                            "checkpoint": repetition,
                            "mode": mode,
                            "cue": cue,
                            **metrics,
                            "own_experience_similarity": cosine_similarity(pattern, own_exp),
                            "other_experience_similarity": cosine_similarity(pattern, other_exp),
                            "cue_selectivity": cosine_similarity(pattern, own_exp) - cosine_similarity(pattern, other_exp),
                            "A_vs_F_pattern_similarity": trained_cross_similarity if mode == "trained" else control_cross_similarity,
                            "separation_gain_over_control": separation_gain if mode == "trained" else 0.0,
                            "mean_direction": float(np.mean(trained.direction[trained.adjacency]))
                            if mode == "trained"
                            else float(np.mean(control.direction[control.adjacency])),
                            "mean_capacity": float(np.mean(trained.capacity[trained.adjacency]))
                            if mode == "trained"
                            else float(np.mean(control.capacity[control.adjacency])),
                        }
                    )

            if repetition < args.repetitions:
                # Alternate the two experiences so neither receives a systematic
                # recency advantage. Each uses the same local learning rule.
                if repetition % 2 == 0:
                    latest_pattern_ac = run_experience(trained, a_region, c_region)
                    latest_pattern_fm = run_experience(trained, f_region, m_region)
                else:
                    latest_pattern_fm = run_experience(trained, f_region, m_region)
                    latest_pattern_ac = run_experience(trained, a_region, c_region)

    write_csv(output_root / "multiple_attractor_curve.csv", summary_rows)

    final_rows = [row for row in summary_rows if row["checkpoint"] == args.repetitions]
    trained_rows = [row for row in final_rows if row["mode"] == "trained"]
    if trained_rows:
        print(
            "trained A-vs-F final pattern similarity: "
            f"{float(trained_rows[0]['A_vs_F_pattern_similarity']):.12g}"
        )
        print(
            "separation gain over control: "
            f"{float(trained_rows[0]['separation_gain_over_control']):.12g}"
        )
        for row in trained_rows:
            print(
                f"trained cue {row['cue']} selectivity: "
                f"{float(row['cue_selectivity']):.12g}"
            )

    print(f"output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
