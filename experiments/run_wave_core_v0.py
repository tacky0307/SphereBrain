from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wave_core import SphereWaveCore, WaveConfig


def region_activity(trace, region: tuple[int, ...]) -> float:
    return trace.max_activity_for(region)


def center_distance_to_region(trace, core: SphereWaveCore, region: tuple[int, ...]) -> float:
    if not trace.snapshots:
        return float("inf")
    target = np.mean(core.positions[np.asarray(region, dtype=int)], axis=0)
    centers = np.asarray([item.center for item in trace.snapshots], dtype=float)
    distances = np.linalg.norm(centers - target[None, :], axis=1)
    return float(np.min(distances))


def trace_metrics(trace, core: SphereWaveCore, b_region: tuple[int, ...]) -> dict[str, Any]:
    if not trace.snapshots:
        return {
            "steps": 0,
            "peak_total_activity": 0.0,
            "final_total_activity": 0.0,
            "activity_integral": 0.0,
            "peak_active_nodes": 0,
            "total_firing_events": 0,
            "center_path_length": 0.0,
            "closest_center_distance_to_b": float("inf"),
            "b_region_peak": 0.0,
            "b_region_activity_integral": 0.0,
        }

    b_ids = np.asarray(b_region, dtype=int)
    target = np.mean(core.positions[b_ids], axis=0)
    centers = np.asarray([item.center for item in trace.snapshots], dtype=float)
    path_length = (
        float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
        if len(centers) > 1
        else 0.0
    )
    center_distances = np.linalg.norm(centers - target[None, :], axis=1)
    totals = np.asarray([item.total_activity for item in trace.snapshots], dtype=float)
    b_means = np.asarray(
        [float(np.mean(item.activity[b_ids])) for item in trace.snapshots],
        dtype=float,
    )
    active_counts = [
        int(np.count_nonzero(item.activity >= core.config.quiet_threshold))
        for item in trace.snapshots
    ]

    return {
        "steps": trace.steps,
        "peak_total_activity": float(np.max(totals)),
        "final_total_activity": float(totals[-1]),
        "activity_integral": float(np.sum(totals)),
        "peak_active_nodes": max(active_counts, default=0),
        "total_firing_events": int(
            sum(len(item.fired_nodes) for item in trace.snapshots)
        ),
        "center_path_length": path_length,
        "closest_center_distance_to_b": float(np.min(center_distances)),
        "b_region_peak": region_activity(trace, b_region),
        "b_region_activity_integral": float(np.sum(b_means)),
    }


def run_probe(core: SphereWaveCore, a_region: tuple[int, ...], name: str):
    core.reset_activity()
    core.stimulate(a_region, strength=1.0)
    return core.run_until_quiet(name=name, learn=False)


def experience_a_then_b(
    core: SphereWaveCore,
    a_region: tuple[int, ...],
    b_region: tuple[int, ...],
    repetitions: int,
    delay_steps: int,
) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []

    for repetition in range(1, repetitions + 1):
        core.reset_activity()
        core.stimulate(a_region, strength=1.0)
        first = core.advance(delay_steps, learn=True, name=f"experience_{repetition}_a")
        core.stimulate(b_region, strength=1.0)
        second = core.run_until_quiet(
            name=f"experience_{repetition}_b",
            learn=True,
        )
        records.append(
            {
                "repetition": repetition,
                "steps": first.steps + second.steps,
                "changed_edges": len(first.changed_edges) + len(second.changed_edges),
                "peak_total_activity": max(
                    first.peak_total_activity,
                    second.peak_total_activity,
                ),
                "activity_integral": float(
                    sum(item.total_activity for item in first.snapshots)
                    + sum(item.total_activity for item in second.snapshots)
                ),
                "firing_events": int(
                    sum(len(item.fired_nodes) for item in first.snapshots)
                    + sum(len(item.fired_nodes) for item in second.snapshots)
                ),
            }
        )

    return records


def save_trace_csv(
    path: Path,
    trace,
    core: SphereWaveCore,
    b_region: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    b_ids = np.asarray(b_region, dtype=int)
    target = np.mean(core.positions[b_ids], axis=0)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "total_activity",
                "total_fatigue",
                "active_nodes",
                "b_region_mean_activity",
                "b_region_max_activity",
                "center_x",
                "center_y",
                "center_z",
                "center_distance_to_b",
                "fired_node_count",
                "fired_nodes",
            ]
        )
        for snapshot in trace.snapshots:
            values = snapshot.activity[b_ids]
            center = np.asarray(snapshot.center, dtype=float)
            writer.writerow(
                [
                    snapshot.step,
                    snapshot.total_activity,
                    float(np.sum(snapshot.fatigue)),
                    int(np.count_nonzero(snapshot.activity >= core.config.quiet_threshold)),
                    float(np.mean(values)),
                    float(np.max(values)),
                    *snapshot.center,
                    float(np.linalg.norm(center - target)),
                    len(snapshot.fired_nodes),
                    " ".join(str(value) for value in snapshot.fired_nodes),
                ]
            )


def save_node_activity_csv(path: Path, trace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "node_id", "activity", "fatigue", "fired"])
        for snapshot in trace.snapshots:
            fired = set(snapshot.fired_nodes)
            for node_id, (activity, fatigue) in enumerate(
                zip(snapshot.activity, snapshot.fatigue)
            ):
                writer.writerow(
                    [
                        snapshot.step,
                        node_id,
                        float(activity),
                        float(fatigue),
                        int(node_id in fired),
                    ]
                )


def save_positions_csv(
    path: Path,
    core: SphereWaveCore,
    a_region: tuple[int, ...],
    b_region: tuple[int, ...],
) -> None:
    a_ids = set(a_region)
    b_ids = set(b_region)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["node_id", "x", "y", "z", "region"])
        for node_id, position in enumerate(core.positions):
            region = "A" if node_id in a_ids else "B" if node_id in b_ids else ""
            writer.writerow([node_id, *[float(value) for value in position], region])


def render_observation_report(
    path: Path,
    result: dict[str, Any],
    baseline_trace,
    trained_trace,
) -> None:
    baseline_series = [
        {
            "step": item.step,
            "activity": item.total_activity,
            "center": list(item.center),
        }
        for item in baseline_trace.snapshots
    ]
    trained_series = [
        {
            "step": item.step,
            "activity": item.total_activity,
            "center": list(item.center),
        }
        for item in trained_trace.snapshots
    ]
    payload = json.dumps(
        {
            "result": result,
            "baselineSeries": baseline_series,
            "trainedSeries": trained_series,
        },
        ensure_ascii=False,
    )
    title = html.escape(str(result["experiment"]))
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} Observation Report</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: #111827; color: #e5e7eb; }}
main {{ max-width: 1100px; margin: auto; padding: 24px; }}
h1 {{ margin-bottom: 4px; }}
.muted {{ color: #9ca3af; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap: 12px; margin: 20px 0; }}
.card {{ background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 16px; }}
.value {{ font-size: 1.7rem; font-weight: 700; margin-top: 8px; }}
canvas {{ width: 100%; height: 280px; background: #0b1220; border-radius: 10px; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 9px; border-bottom: 1px solid #374151; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.positive {{ color: #86efac; }}
.negative {{ color: #fca5a5; }}
</style>
</head>
<body>
<main>
<h1>{title}</h1>
<div class="muted">Experiment 002 observation output — no answer label is used inside the core.</div>
<div id="cards" class="grid"></div>
<div class="card">
<h2>総活動量の推移</h2>
<canvas id="activityChart" width="1000" height="280"></canvas>
</div>
<div class="card" style="margin-top:12px">
<h2>Baseline / Trained 比較</h2>
<table id="comparison"></table>
</div>
<p class="muted">詳細データ: baseline_a.csv / trained_a.csv / *_node_activity.csv / terrain_changes.csv</p>
</main>
<script>
const data = {payload};
const result = data.result;
const fmt = value => Number.isFinite(value) ? Number(value).toPrecision(6) : String(value);
const cards = [
  ["Changed directed edges", result.change.changed_directed_edges],
  ["Total conductivity change", fmt(result.change.total_conductivity_change)],
  ["B-region peak change", fmt(result.change.b_region_peak)],
  ["Distance change to B", fmt(result.change.center_distance_to_b)],
  ["Activity integral change", fmt(result.change.activity_integral)],
  ["Center path change", fmt(result.change.center_path_length)]
];
document.getElementById("cards").innerHTML = cards.map(([name,value]) =>
  `<div class="card"><div class="muted">${name}</div><div class="value">${value}</div></div>`
).join("");

const keys = [
  ["steps","Steps"],
  ["peak_total_activity","Peak total activity"],
  ["final_total_activity","Final total activity"],
  ["activity_integral","Activity integral"],
  ["peak_active_nodes","Peak active nodes"],
  ["total_firing_events","Firing events"],
  ["center_path_length","Center path length"],
  ["closest_center_distance_to_b","Closest distance to B"],
  ["b_region_peak","B-region peak"],
  ["b_region_activity_integral","B-region integral"]
];
document.getElementById("comparison").innerHTML =
  `<tr><th>Metric</th><th>Baseline</th><th>Trained</th><th>Change</th></tr>` +
  keys.map(([key,label]) => {
    const before = result.baseline[key], after = result.trained[key];
    const change = Number(after) - Number(before);
    const cls = change > 0 ? "positive" : change < 0 ? "negative" : "";
    return `<tr><td>${label}</td><td>${fmt(before)}</td><td>${fmt(after)}</td><td class="${cls}">${fmt(change)}</td></tr>`;
  }).join("");

function drawSeries(canvas, seriesList) {
  const ctx = canvas.getContext("2d");
  const all = seriesList.flatMap(item => item.series.map(point => point.activity));
  const maxY = Math.max(...all, 1e-9);
  const maxX = Math.max(...seriesList.flatMap(item => item.series.map(point => point.step)), 1);
  const pad = 35, w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = "#374151";
  ctx.beginPath(); ctx.moveTo(pad,10); ctx.lineTo(pad,h-pad); ctx.lineTo(w-10,h-pad); ctx.stroke();
  const colors = ["#60a5fa","#f59e0b"];
  seriesList.forEach((item,index) => {
    ctx.strokeStyle = colors[index];
    ctx.lineWidth = 3;
    ctx.beginPath();
    item.series.forEach((point,i) => {
      const x = pad + (point.step / maxX) * (w-pad-15);
      const y = h-pad - (point.activity / maxY) * (h-pad-20);
      i ? ctx.lineTo(x,y) : ctx.moveTo(x,y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[index];
    ctx.fillText(item.name, pad + index * 100, 18);
  });
}
drawSeries(document.getElementById("activityChart"), [
  {name:"Baseline", series:data.baselineSeries},
  {name:"Trained", series:data.trainedSeries}
]);
</script>
</body>
</html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SphereBrain Wave Core v0: observe terrain change after A→B experience",
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--delay-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "wave_core_v0")
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.delay_steps < 0:
        raise SystemExit("--delay-steps must be zero or greater")

    core = SphereWaveCore(WaveConfig())

    # The labels A and B exist only outside the core. Internally they are
    # distributed stimulus regions with no symbolic meaning.
    a_region = core.stimulus_region(anchor=24, radius=3)
    b_region = core.stimulus_region(anchor=142, radius=3)

    baseline = run_probe(core, a_region, name="baseline_a")
    baseline_metrics = trace_metrics(baseline, core, b_region)
    terrain_before = core.conductivity.copy()

    experience_records = experience_a_then_b(
        core,
        a_region,
        b_region,
        repetitions=args.repetitions,
        delay_steps=args.delay_steps,
    )

    trained = run_probe(core, a_region, name="trained_a")
    trained_metrics = trace_metrics(trained, core, b_region)
    terrain_delta = core.conductivity - terrain_before

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    save_trace_csv(output / "baseline_a.csv", baseline, core, b_region)
    save_trace_csv(output / "trained_a.csv", trained, core, b_region)
    save_node_activity_csv(output / "baseline_node_activity.csv", baseline)
    save_node_activity_csv(output / "trained_node_activity.csv", trained)
    save_positions_csv(output / "node_positions.csv", core, a_region, b_region)

    with (output / "experience_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(experience_records[0].keys()))
        writer.writeheader()
        writer.writerows(experience_records)

    changed = np.argwhere(np.abs(terrain_delta) > 1e-10)
    terrain_rows = sorted(
        (
            (
                int(a),
                int(b),
                float(terrain_before[a, b]),
                float(core.conductivity[a, b]),
                float(terrain_delta[a, b]),
            )
            for a, b in changed
        ),
        key=lambda item: abs(item[4]),
        reverse=True,
    )
    with (output / "terrain_changes.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_node",
                "target_node",
                "conductivity_before",
                "conductivity_after",
                "conductivity_change",
            ]
        )
        writer.writerows(terrain_rows)

    result = {
        "experiment": "SphereBrain Wave Core v0 / Experiment 002",
        "a_region": a_region,
        "b_region": b_region,
        "repetitions": args.repetitions,
        "delay_steps": args.delay_steps,
        "config": {
            key: value
            for key, value in vars(core.config).items()
        },
        "baseline": baseline_metrics,
        "trained": trained_metrics,
        "change": {
            "b_region_peak": (
                trained_metrics["b_region_peak"] - baseline_metrics["b_region_peak"]
            ),
            "b_region_activity_integral": (
                trained_metrics["b_region_activity_integral"]
                - baseline_metrics["b_region_activity_integral"]
            ),
            "center_distance_to_b": (
                trained_metrics["closest_center_distance_to_b"]
                - baseline_metrics["closest_center_distance_to_b"]
            ),
            "activity_integral": (
                trained_metrics["activity_integral"]
                - baseline_metrics["activity_integral"]
            ),
            "center_path_length": (
                trained_metrics["center_path_length"]
                - baseline_metrics["center_path_length"]
            ),
            "changed_directed_edges": len(terrain_rows),
            "total_conductivity_change": float(np.sum(terrain_delta)),
            "absolute_conductivity_change": float(np.sum(np.abs(terrain_delta))),
        },
        "interpretation": (
            "This run records observations, including changes that do not match "
            "the A→B expectation. No single metric is treated as a right/wrong answer."
        ),
    }

    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_observation_report(
        output / "observation_report.html",
        result,
        baseline,
        trained,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nObservation files: {output}")
    print(f"Open report: {output / 'observation_report.html'}")


if __name__ == "__main__":
    main()
