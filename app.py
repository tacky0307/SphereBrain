from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from datetime import datetime
import json
import shutil
import time
import webbrowser

from flask import Flask, request, redirect, url_for, render_template_string, send_file
from waitress import serve

from brain import SphereBrain, SignalResult
from memory_store import MemoryStore
from research_store import ResearchStore
from visualization import build_html
from audio_listener import SystemAudioListener


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BACKUPS = DATA / "backups"
DATA.mkdir(exist_ok=True)
BACKUPS.mkdir(exist_ok=True)

BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"
RESEARCH_DB_FILE = DATA / "research.db"
VIEW_FILE = DATA / "brain_view.html"
CONFIG_FILE = DATA / "config.json"

brain = SphereBrain.load(BRAIN_FILE) if BRAIN_FILE.exists() else SphereBrain(
    node_count=600,
    neighbors_per_node=8,
)
memory = MemoryStore(DB_FILE)
research = ResearchStore(RESEARCH_DB_FILE)
brain_lock = Lock()
research_lock = Lock()

last_result: SignalResult | None = None
last_input = ""
last_activity = datetime.now().isoformat(timespec="seconds")
running = True
trial_sequence = 0

app = Flask(__name__)


def load_config() -> dict:
    default = {
        "audio_model": "small",
        "audio_chunk_seconds": 18,
        "auto_start_audio": False,
        "idle_cycle_seconds": 45,
        "backup_hours": 6,
        "structure_version": "sphere-600-v1",
        "config_version": "default-v1",
    }
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(
            json.dumps(default, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return default
    try:
        saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        default.update(saved)
    except Exception:
        pass
    return default


config = load_config()
experiment_id = research.create_experiment(
    name="Sphere Brain continuous observation",
    purpose="Record how repeated inputs and internal cycles alter paths in the spherical network.",
    hypothesis="Repeated and context-linked activity will create persistent preferred paths.",
    protocol_version="0.4",
    metadata={"node_count": brain.node_count, "neighbors_per_node": brain.neighbors_per_node},
)
session_id = research.start_session(
    experiment_id=experiment_id,
    structure_version=str(config["structure_version"]),
    config_version=str(config["config_version"]),
    random_seed=brain.seed,
    metadata={"started_by": "app.py"},
)


PAGE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>Sphere Brain v0.4</title>
<style>
body { font-family: system-ui,sans-serif; margin:0; background:#f5f7fb; color:#1f2937; }
header { background:#172554; color:white; padding:18px 24px; }
main { max-width:1180px; margin:24px auto; padding:0 16px 40px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:white; border-radius:14px; padding:18px; box-shadow:0 4px 18px rgba(0,0,0,.07); }
textarea { width:100%; min-height:100px; box-sizing:border-box; padding:12px; font-size:16px; }
button { background:#ea580c; color:white; border:0; padding:11px 18px; border-radius:9px; font-size:15px; cursor:pointer; }
button.secondary { background:#334155; }
button.stop { background:#b91c1c; }
iframe { width:100%; height:670px; border:0; background:white; }
small,.muted { color:#6b7280; }
table { width:100%; border-collapse:collapse; }
td,th { padding:8px; border-bottom:1px solid #e5e7eb; text-align:left; vertical-align:top; }
.badge { display:inline-block; background:#dcfce7; padding:4px 8px; border-radius:999px; }
.badge.off { background:#e5e7eb; }
.badge.error { background:#fee2e2; }
.actions { display:flex; gap:10px; flex-wrap:wrap; }
code { background:#f1f5f9; padding:2px 5px; }
.metrics { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; }
.metric { background:#f8fafc; border-radius:10px; padding:10px; }
.metric strong { display:block; font-size:22px; }
@media(max-width:800px){ .grid{grid-template-columns:1fr;} iframe{height:520px;} }
</style>
</head>
<body>
<header>
<h1>Sphere Brain v0.4</h1>
<div>経路記憶・研究ログ・内部活動・システム音声聴取</div>
</header>
<main>
<div class="grid">
<section class="card">
<h2>文字入力</h2>
<form method="post" action="/input">
<textarea name="text" placeholder="言葉や文章を入力してください"></textarea>
<p><button type="submit">球体へ入力する</button></p>
</form>
<small>直近の記憶が文脈として加わり、経路と重み変化を研究DBへ保存します。</small>
</section>

<section class="card">
<h2>現在の状態</h2>
<p><span class="badge">球体：稼働中</span></p>
<p>記憶件数：{{ memory_count }}</p>
<p>最終入力：{{ last_input or "まだありません" }}</p>
<p>最終活動：{{ last_activity }}</p>
<p>ノード数：{{ brain.node_count }}</p>
</section>

<section class="card">
<h2>研究データ</h2>
<div class="metrics">
<div class="metric"><span>試行</span><strong>{{ research_summary.trials }}</strong></div>
<div class="metric"><span>経路ステップ</span><strong>{{ research_summary.path_steps }}</strong></div>
<div class="metric"><span>スナップショット</span><strong>{{ research_summary.snapshots }}</strong></div>
<div class="metric"><span>分析指標</span><strong>{{ research_summary.metrics }}</strong></div>
</div>
<p class="muted"><code>data/research.db</code> に追記保存。過去の観測は上書きしません。</p>
</section>

<section class="card">
<h2>PC音声の聴取</h2>
{% if audio.running %}
<p><span class="badge">聴取中</span> {{ audio.state }}</p>
{% elif audio.last_error %}
<p><span class="badge error">エラー</span> {{ audio.last_error }}</p>
{% else %}
<p><span class="badge off">停止中</span></p>
{% endif %}
<p>文字化した区間：{{ audio.chunks_processed }}</p>
<p>無音として省略：{{ audio.chunks_skipped }}</p>
<p>最新の文字：{{ audio.last_text or "まだありません" }}</p>
<div class="actions">
<form method="post" action="/audio/start"><button type="submit">音声聴取を開始</button></form>
<form method="post" action="/audio/stop"><button class="stop" type="submit">音声聴取を停止</button></form>
</div>
<p class="muted">既定スピーカーから流れる音を約{{ chunk_seconds }}秒ごとにローカルで文字化します。生の音声は保存しません。</p>
</section>

<section class="card">
<h2>保存とバックアップ</h2>
<p>脳：<code>data/brain.json</code></p>
<p>記憶：<code>data/memory.db</code></p>
<p>研究：<code>data/research.db</code></p>
<p>自動バックアップ：{{ backup_hours }}時間ごと</p>
<div class="actions">
<form method="post" action="/save"><button class="secondary" type="submit">今すぐ保存</button></form>
<form method="post" action="/backup"><button class="secondary" type="submit">今すぐバックアップ</button></form>
</div>
</section>
</div>

<section class="card" style="margin-top:16px">
<h2>球体内部</h2>
<iframe src="/brain-view?ts={{ timestamp }}"></iframe>
</section>

<section class="card" style="margin-top:16px">
<h2>最近の記憶</h2>
<table>
<tr><th>時刻</th><th>種類</th><th>入力</th><th>活動</th></tr>
{% for item in memories %}
<tr>
<td>{{ item.created_at }}</td>
<td>{{ item.kind }}</td>
<td>{{ (item.input_text or "内部活動")[:180] }}</td>
<td>{{ item.activated_nodes|length }}ノード / {{ item.traversed_edges|length }}経路</td>
</tr>
{% endfor %}
</table>
</section>
</main>
</body>
</html>
"""


def record_research_trial(text: str, kind: str, result: SignalResult) -> None:
    global trial_sequence
    with research_lock:
        trial_sequence += 1
        input_id = research.add_input(
            raw_value=text,
            input_type="text" if kind != "idle" else "internal",
            source=kind,
            metadata={"kind": kind},
        )
        trial = research.start_trial(
            session_id=session_id,
            sequence_no=trial_sequence,
            input_id=input_id,
            source_nodes=result.source_nodes,
        )
        try:
            research.add_snapshot(
                trial.trial_id,
                step_no=0,
                snapshot_type="initial",
                state={"source_nodes": result.source_nodes, "activation": result.activation_history[0]},
            )
            for step in result.path_steps:
                research.add_path_step(
                    trial_id=trial.trial_id,
                    step_no=step.step_no,
                    from_node=step.from_node,
                    to_node=step.to_node,
                    activation=step.activation,
                    weight_before=step.weight_before,
                    weight_after=step.weight_after,
                    selected_by=step.selected_by,
                )
            research.add_snapshot(
                trial.trial_id,
                step_no=max(0, len(result.activation_history) - 1),
                snapshot_type="final",
                state={
                    "activated_nodes": result.activated_nodes,
                    "final_activation": result.final_activation.tolist(),
                },
            )
            research.add_output(
                trial.trial_id,
                value={
                    "activated_nodes": result.activated_nodes,
                    "traversed_edges": result.traversed_edges,
                },
                output_type="activation_result",
                decoder_version="raw-0.4",
            )
            research.add_metric(trial.trial_id, "activated_node_count", len(result.activated_nodes))
            research.add_metric(trial.trial_id, "unique_edge_count", len(result.traversed_edges))
            research.add_metric(trial.trial_id, "path_step_count", len(result.path_steps))
            research.add_metric(
                trial.trial_id,
                "mean_final_activation",
                float(result.final_activation.mean()),
            )
            research.finish_trial(trial.trial_id, result.activated_nodes)
        except Exception as exc:
            research.finish_trial(trial.trial_id, [], status="error", error_text=str(exc))
            raise


def ingest_text(text: str, kind: str = "input", importance: float = 1.0) -> None:
    global last_result, last_input, last_activity

    text = " ".join(text.strip().split())
    if not text:
        return

    chunks = [text[i:i + 600] for i in range(0, len(text), 600)]

    for chunk in chunks:
        sources = brain.text_to_sources(chunk, count=4)
        context = memory.recent_context_nodes(memory_limit=7, node_limit=18)

        with brain_lock:
            result = brain.propagate(
                source_nodes=sources,
                context_nodes=context,
                steps=20,
                learn=True,
            )

        memory.add_memory(
            kind=kind,
            input_text=chunk,
            source_nodes=result.source_nodes,
            activated_nodes=result.activated_nodes,
            traversed_edges=result.traversed_edges,
            importance=importance,
        )
        record_research_trial(chunk, kind, result)

        last_result = result
        last_input = chunk
        last_activity = datetime.now().isoformat(timespec="seconds")

    save_all()


audio_listener = SystemAudioListener(
    on_text=lambda text: ingest_text(text, kind="audio", importance=0.72),
    model_size=str(config["audio_model"]),
    chunk_seconds=int(config["audio_chunk_seconds"]),
)


def save_all() -> None:
    global last_result
    with brain_lock:
        brain.save(BRAIN_FILE)
        if last_result is None:
            strong = brain.strongest_edges(55)
            edges = [(e["a"], e["b"]) for e in strong]
            nodes = sorted({n for edge in edges for n in edge})
            build_html(brain, VIEW_FILE, edges, nodes, "Sphere Brain：強い経路")
        else:
            build_html(
                brain,
                VIEW_FILE,
                last_result.traversed_edges,
                last_result.activated_nodes,
                f"Sphere Brain：{last_input or '内部活動'}",
            )


def make_backup() -> None:
    save_all()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BACKUPS / stamp
    target.mkdir(exist_ok=True)
    for path in (BRAIN_FILE, DB_FILE, RESEARCH_DB_FILE, CONFIG_FILE):
        if path.exists():
            shutil.copy2(path, target / path.name)

    backups = sorted([p for p in BACKUPS.iterdir() if p.is_dir()])
    for old in backups[:-28]:
        shutil.rmtree(old, ignore_errors=True)


def background_loop() -> None:
    global last_result, last_activity, last_input
    idle_seconds = max(20, int(config["idle_cycle_seconds"]))
    backup_seconds = max(3600, int(config["backup_hours"]) * 3600)
    last_backup = time.monotonic()

    while running:
        time.sleep(idle_seconds)
        try:
            context = memory.recent_context_nodes(memory_limit=10, node_limit=30)
            if context:
                with brain_lock:
                    result = brain.idle_cycle(context)
                if result:
                    memory.add_memory(
                        kind="idle",
                        input_text="",
                        source_nodes=result.source_nodes,
                        activated_nodes=result.activated_nodes,
                        traversed_edges=result.traversed_edges,
                        importance=0.18,
                    )
                    record_research_trial("internal idle cycle", "idle", result)
                    last_result = result
                    last_input = "内部活動"
                    last_activity = datetime.now().isoformat(timespec="seconds")
            save_all()

            if time.monotonic() - last_backup >= backup_seconds:
                make_backup()
                last_backup = time.monotonic()
        except Exception as exc:
            print("background error:", exc)


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        memory_count=memory.count(),
        memories=memory.recent(20),
        research_summary=research.summary(),
        last_input=last_input,
        last_activity=last_activity,
        brain=brain,
        audio=audio_listener.status,
        timestamp=int(time.time()),
        chunk_seconds=config["audio_chunk_seconds"],
        backup_hours=config["backup_hours"],
    )


@app.post("/input")
def input_text():
    text = request.form.get("text", "")
    ingest_text(text, kind="input", importance=1.0)
    return redirect(url_for("index"))


@app.post("/audio/start")
def audio_start():
    audio_listener.start()
    return redirect(url_for("index"))


@app.post("/audio/stop")
def audio_stop():
    audio_listener.stop()
    return redirect(url_for("index"))


@app.post("/save")
def save_now():
    save_all()
    return redirect(url_for("index"))


@app.post("/backup")
def backup_now():
    make_backup()
    return redirect(url_for("index"))


@app.route("/brain-view")
def brain_view():
    if not VIEW_FILE.exists():
        save_all()
    return send_file(VIEW_FILE)


if __name__ == "__main__":
    Thread(target=background_loop, daemon=True).start()
    save_all()

    if bool(config.get("auto_start_audio")):
        audio_listener.start()

    webbrowser.open("http://127.0.0.1:5050")
    print("Sphere Brain v0.4: http://127.0.0.1:5050")
    try:
        serve(app, host="127.0.0.1", port=5050, threads=6)
    finally:
        research.finish_session(session_id)
