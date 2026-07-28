from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import time
import webbrowser

from flask import Flask, request, render_template_string, send_file
from waitress import serve

from brain import SphereBrain
from visualization import build_html

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
BRAIN_FILE = DATA / "brain.json"
DB_FILE = DATA / "memory.db"
VIEW_FILE = DATA / "recall_view.html"

if not BRAIN_FILE.exists():
    raise FileNotFoundError("data/brain.json が見つかりません。先にSphere Brain本体を起動してください。")
if not DB_FILE.exists():
    raise FileNotFoundError("data/memory.db が見つかりません。先に記憶データを作成してください。")

brain = SphereBrain.load(BRAIN_FILE)
app = Flask(__name__)

last_query = ""
last_result = None
last_matches: list[dict] = []

PAGE = """
<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sphere Brain 想起モード</title>
<style>
body { font-family:system-ui,sans-serif; margin:0; background:#f5f7fb; color:#1f2937; }
header { background:#312e81; color:white; padding:18px 24px; }
main { max-width:1180px; margin:24px auto; padding:0 16px 40px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:white; border-radius:14px; padding:18px; box-shadow:0 4px 18px rgba(0,0,0,.07); }
textarea { width:100%; min-height:100px; box-sizing:border-box; padding:12px; font-size:16px; }
button { background:#7c3aed; color:white; border:0; padding:11px 18px; border-radius:9px; font-size:15px; cursor:pointer; }
.badge { display:inline-block; background:#ede9fe; color:#5b21b6; padding:4px 9px; border-radius:999px; }
.notice { background:#ecfdf5; border:1px solid #86efac; padding:10px 12px; border-radius:10px; }
.metric { background:#f8fafc; border-radius:10px; padding:10px; margin-bottom:8px; }
.metric strong { font-size:22px; }
table { width:100%; border-collapse:collapse; }
th,td { padding:9px; border-bottom:1px solid #e5e7eb; text-align:left; vertical-align:top; }
iframe { width:100%; height:650px; border:0; background:white; }
small,.muted { color:#64748b; }
@media(max-width:800px){ .grid{grid-template-columns:1fr;} iframe{height:500px;} }
</style>
</head>
<body>
<header>
<h1>Sphere Brain 想起モード</h1>
<div>学習・保存を行わず、現在の脳から記憶をたどります</div>
</header>
<main>
<p class="notice"><strong>安全モード：</strong>想起を実行しても、記憶件数・接続重み・使用回数は変更されません。</p>
<div class="grid">
<section class="card">
<h2>【想起】</h2>
<form method="post" action="/recall">
<textarea name="text" placeholder="例：今日は　／　空は　／　雨"></textarea>
<p><button type="submit">想起する</button></p>
</form>
<small>入力を4つの起点ノードへ変換し、ノイズなし・学習なしで経路を伝播します。</small>
</section>
<section class="card">
<h2>想起結果</h2>
{% if result %}
<p><span class="badge">{{ query }}</span></p>
<div class="metric">起点ノード：<strong>{{ result.source_nodes|join(', ') }}</strong></div>
<div class="metric">活性化ノード：<strong>{{ result.activated_nodes|length }}</strong></div>
<div class="metric">通過経路：<strong>{{ result.traversed_edges|length }}</strong></div>
{% else %}
<p class="muted">まだ想起していません。</p>
{% endif %}
</section>
</div>

{% if matches %}
<section class="card" style="margin-top:16px">
<h2>近い記憶</h2>
<table>
<tr><th>近さ</th><th>時刻</th><th>種類</th><th>記憶内容</th><th>共通ノード</th></tr>
{% for item in matches %}
<tr>
<td>{{ '%.1f'|format(item.score * 100) }}%</td>
<td>{{ item.created_at }}</td>
<td>{{ item.kind }}</td>
<td>{{ item.input_text or '内部活動' }}</td>
<td>{{ item.shared_nodes }}</td>
</tr>
{% endfor %}
</table>
<p class="muted">近さは、想起時と記憶時の活性化ノードおよび起点ノードの重なりから算出しています。文章生成による回答ではありません。</p>
</section>
{% endif %}

<section class="card" style="margin-top:16px">
<h2>想起した経路</h2>
{% if result %}<iframe src="/recall-view?ts={{ timestamp }}"></iframe>{% else %}<p class="muted">想起すると球体内の経路を表示します。</p>{% endif %}
</section>
</main>
</body>
</html>
"""


def load_memories() -> list[dict]:
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, kind, input_text, source_nodes, activated_nodes, traversed_edges, importance "
            "FROM memories ORDER BY id DESC"
        ).fetchall()
    memories = []
    for row in rows:
        item = dict(row)
        item["source_nodes"] = json.loads(item["source_nodes"])
        item["activated_nodes"] = json.loads(item["activated_nodes"])
        item["traversed_edges"] = json.loads(item["traversed_edges"])
        memories.append(item)
    return memories


def find_similar(result, limit: int = 8) -> list[dict]:
    recalled_nodes = set(result.activated_nodes)
    recalled_sources = set(result.source_nodes)
    ranked = []
    for item in load_memories():
        memory_nodes = set(item["activated_nodes"])
        memory_sources = set(item["source_nodes"])
        union = recalled_nodes | memory_nodes
        node_score = len(recalled_nodes & memory_nodes) / len(union) if union else 0.0
        source_union = recalled_sources | memory_sources
        source_score = len(recalled_sources & memory_sources) / len(source_union) if source_union else 0.0
        score = node_score * 0.8 + source_score * 0.2
        ranked.append({
            **item,
            "score": score,
            "shared_nodes": len(recalled_nodes & memory_nodes),
        })
    ranked.sort(key=lambda item: (item["score"], item["importance"], item["id"]), reverse=True)
    return [item for item in ranked[:limit] if item["score"] > 0]


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        query=last_query,
        result=last_result,
        matches=last_matches,
        timestamp=int(time.time()),
    )


@app.post("/recall")
def recall():
    global last_query, last_result, last_matches
    text = " ".join(request.form.get("text", "").strip().split())
    if text:
        sources = brain.text_to_sources(text, count=4)
        # learn=False、noise=0.0。脳・記憶DB・研究DBには書き込まない。
        result = brain.propagate(
            source_nodes=sources,
            context_nodes=None,
            steps=20,
            threshold=0.15,
            noise=0.0,
            learn=False,
        )
        last_query = text
        last_result = result
        last_matches = find_similar(result)
        build_html(
            brain,
            VIEW_FILE,
            result.traversed_edges,
            result.activated_nodes,
            f"想起：{text}",
        )
    return index()


@app.route("/recall-view")
def recall_view():
    if not VIEW_FILE.exists():
        return "まだ想起結果がありません。", 404
    return send_file(VIEW_FILE)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5051")
    print("Sphere Brain 想起モード: http://127.0.0.1:5051")
    serve(app, host="127.0.0.1", port=5051, threads=4)
