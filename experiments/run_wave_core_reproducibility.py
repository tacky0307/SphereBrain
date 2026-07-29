from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_wave_core_v0 import experience_a_then_b, run_probe, trace_metrics
from wave_core import SphereWaveCore, WaveConfig


CHANGE_METRICS = (
    "b_region_peak",
    "b_region_activity_integral",
    "center_distance_to_b",
    "activity_integral",
    "center_path_length",
    "changed_directed_edges",
    "total_conductivity_change",
)


def run_trial(seed: int, repetitions: int, delay_steps: int) -> dict[str, Any]:
    config = replace(WaveConfig(), seed=seed)
    core = SphereWaveCore(config)

    a_region = core.stimulus_region(anchor=24, radius=3)
    b_region = core.stimulus_region(anchor=142, radius=3)

    baseline_trace = run_probe(core, a_region, name=f"baseline_seed_{seed}")
    baseline = trace_metrics(baseline_trace, core, b_region)
    terrain_before = core.conductivity.copy()

    experience_a_then_b(
        core,
        a_region,
        b_region,
        repetitions=repetitions,
        delay_steps=delay_steps,
    )

    trained_trace = run_probe(core, a_region, name=f"trained_seed_{seed}")
    trained = trace_metrics(trained_trace, core, b_region)
    terrain_delta = core.conductivity - terrain_before

    change = {
        "b_region_peak": trained["b_region_peak"] - baseline["b_region_peak"],
        "b_region_activity_integral": (
            trained["b_region_activity_integral"]
            - baseline["b_region_activity_integral"]
        ),
        "center_distance_to_b": (
            trained["closest_center_distance_to_b"]
            - baseline["closest_center_distance_to_b"]
        ),
        "activity_integral": trained["activity_integral"] - baseline["activity_integral"],
        "center_path_length": (
            trained["center_path_length"] - baseline["center_path_length"]
        ),
        "changed_directed_edges": int(np.count_nonzero(terrain_delta > 1e-10)),
        "total_conductivity_change": float(np.sum(terrain_delta)),
    }

    return {
        "seed": seed,
        "a_region": a_region,
        "b_region": b_region,
        "baseline": baseline,
        "trained": trained,
        "change": change,
    }


def sign_counts(values: np.ndarray, tolerance: float = 1e-15) -> dict[str, int]:
    return {
        "positive": int(np.count_nonzero(values > tolerance)),
        "negative": int(np.count_nonzero(values < -tolerance)),
        "zero": int(np.count_nonzero(np.abs(values) <= tolerance)),
    }


def summarize_metric(values: np.ndarray, seeds: np.ndarray) -> dict[str, Any]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    median = float(np.median(values))

    if std > 0.0:
        z_scores = (values - mean) / std
        outlier_indexes = np.flatnonzero(np.abs(z_scores) >= 2.0)
    else:
        z_scores = np.zeros_like(values)
        outlier_indexes = np.asarray([], dtype=int)

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sign_counts": sign_counts(values),
        "outliers_abs_z_ge_2": [
            {
                "seed": int(seeds[index]),
                "value": float(values[index]),
                "z_score": float(z_scores[index]),
            }
            for index in outlier_indexes
        ],
    }


def build_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = np.asarray([trial["seed"] for trial in trials], dtype=int)
    metrics: dict[str, Any] = {}

    for metric in CHANGE_METRICS:
        values = np.asarray([trial["change"][metric] for trial in trials], dtype=float)
        metrics[metric] = summarize_metric(values, seeds)

    toward_b = np.asarray(
        [trial["change"]["center_distance_to_b"] < 0.0 for trial in trials],
        dtype=bool,
    )
    b_peak_up = np.asarray(
        [trial["change"]["b_region_peak"] > 0.0 for trial in trials],
        dtype=bool,
    )

    return {
        "trial_count": len(trials),
        "metrics": metrics,
        "observed_pattern_counts": {
            "center_moved_toward_b": int(np.count_nonzero(toward_b)),
            "center_moved_away_from_b": int(np.count_nonzero(~toward_b)),
            "b_region_peak_increased": int(np.count_nonzero(b_peak_up)),
            "b_region_peak_not_increased": int(np.count_nonzero(~b_peak_up)),
            "both_toward_b_and_peak_up": int(np.count_nonzero(toward_b & b_peak_up)),
        },
    }


def save_trials_csv(path: Path, trials: list[dict[str, Any]]) -> None:
    fields = ["seed"] + list(CHANGE_METRICS)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow({"seed": trial["seed"], **trial["change"]})


def save_metric_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "mean",
                "std",
                "median",
                "min",
                "max",
                "positive",
                "negative",
                "zero",
                "outlier_count",
            ]
        )
        for metric, item in summary["metrics"].items():
            writer.writerow(
                [
                    metric,
                    item["mean"],
                    item["std"],
                    item["median"],
                    item["min"],
                    item["max"],
                    item["sign_counts"]["positive"],
                    item["sign_counts"]["negative"],
                    item["sign_counts"]["zero"],
                    len(item["outliers_abs_z_ge_2"]),
                ]
            )


def render_report(path: Path, result: dict[str, Any]) -> None:
    payload = json.dumps(result, ensure_ascii=False)
    document = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Experiment 003</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #111827; color: #e5e7eb; }
main { max-width: 1200px; margin: auto; padding: 24px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; }
.card { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; }
.muted { color: #9ca3af; }
.value { font-size: 1.6rem; font-weight: 700; margin-top: 8px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { border-bottom: 1px solid #374151; padding: 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
</style>
</head>
<body>
<main>
<h1>Experiment 003：再現性観測</h1>
<p class="muted">seedだけを変え、同一条件で繰り返した結果。期待との一致・不一致を同じ重さで記録します。</p>
<div id="cards" class="grid"></div>
<div class="card" style="margin-top:12px">
<h2>変化量の統計</h2>
<table id="metrics"></table>
</div>
<div class="card" style="margin-top:12px">
<h2>観測パターン</h2>
<table id="patterns"></table>
</div>
<p class="muted">詳細データ: trials.csv / metric_summary.csv / result.json</p>
</main>
<script>
const result = __PAYLOAD__;
const summary = result.summary;
const fmt = value => Number(value).toPrecision(6);
const distance = summary.metrics.center_distance_to_b;
const peak = summary.metrics.b_region_peak;
document.getElementById("cards").innerHTML = [
  ["Trials", summary.trial_count],
  ["Mean distance change", fmt(distance.mean)],
  ["Toward B", summary.observed_pattern_counts.center_moved_toward_b],
  ["Mean B peak change", fmt(peak.mean)],
  ["B peak increased", summary.observed_pattern_counts.b_region_peak_increased]
].map(([name,value]) => `<div class="card"><div class="muted">${name}</div><div class="value">${value}</div></div>`).join("");

document.getElementById("metrics").innerHTML =
  `<tr><th>Metric</th><th>Mean</th><th>Std</th><th>Median</th><th>Positive</th><th>Negative</th><th>Outliers</th></tr>` +
  Object.entries(summary.metrics).map(([name,item]) =>
    `<tr><td>${name}</td><td>${fmt(item.mean)}</td><td>${fmt(item.std)}</td><td>${fmt(item.median)}</td><td>${item.sign_counts.positive}</td><td>${item.sign_counts.negative}</td><td>${item.outliers_abs_z_ge_2.length}</td></tr>`
  ).join("");

document.getElementById("patterns").innerHTML =
  `<tr><th>Pattern</th><th>Count</th></tr>` +
  Object.entries(summary.observed_pattern_counts).map(([name,value]) =>
    `<tr><td>${name}</td><td>${value}</td></tr>`
  ).join("");
</script>
</body>
</html>""".replace("__PAYLOAD__", payload)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SphereBrain Wave Core Experiment 003: reproducibility across seeds",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=27)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "wave_core_reproducibility",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.delay_steps < 0:
        raise SystemExit("--delay-steps must be zero or greater")

    trials = [
        run_trial(
            seed=args.seed_start + index,
            repetitions=args.repetitions,
            delay_steps=args.delay_steps,
        )
        for index in range(args.trials)
    ]
    summary = build_summary(trials)

    result = {
        "experiment": "SphereBrain Wave Core v0 / Experiment 003",
        "trial_count": args.trials,
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.trials - 1,
        "repetitions": args.repetitions,
        "delay_steps": args.delay_steps,
        "base_config": asdict(WaveConfig()),
        "summary": summary,
        "interpretation": (
            "This experiment measures reproducibility across network seeds. "
            "Positive and negative outcomes are retained without classifying either as failure."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    save_trials_csv(args.output / "trials.csv", trials)
    save_metric_summary_csv(args.output / "metric_summary.csv", summary)
    (args.output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_report(args.output / "reproducibility_report.html", result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nObservation files: {args.output}")
    print(f"Open report: {args.output / 'reproducibility_report.html'}")


if __name__ == "__main__":
    main()
