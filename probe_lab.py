from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

from brain import SphereBrain

BASE = Path(__file__).resolve().parent
BRAIN_FILE = BASE / "data" / "brain.json"
DB_FILE = BASE / "data" / "memory.db"
app = Flask(__name__)


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_experiences(limit: int = 300) -> list[dict]:
    if not DB_FILE.exists():
        return []
    with sqlite3.connect(f"file:{DB_FILE.as_posix()}?mode=ro", uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id,input_text,activated_nodes,traversed_edges FROM memories "
            "WHERE kind='input' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    items = []
    for row in rows:
        try:
            nodes = {int(n) for n in json.loads(row["activated_nodes"] or "[]")}
            edges = {normalize_edge(e) for e in json.loads(row["traversed_edges"] or "[]")}
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        items.append({"id": int(row["id"]), "text": row["input_text"] or "", "nodes": nodes, "edges": edges})
    return items


def run_natural_probe(text: str, steps: int) -> dict:
    if not BRAIN_FILE.exists():
        raise FileNotFoundError("data/brain.json がありません。先に通常のSphereBrainを起動してください。")

    brain = SphereBrain.load(BRAIN_FILE)
    result = brain.propagate(
        brain.text_to_sources(text),
        steps=steps,
        threshold=0.15,
        noise=0.0,
        learn=False,
        context_nodes=None,
    )

    probe_nodes = {int(n) for n in result.activated_nodes}
    probe_edges = {normalize_edge(e) for e in result.traversed_edges}
    matches = []
    for memory in load_experiences():
        node_score = jaccard(probe_nodes, memory["nodes"])
        edge_score = jaccard(probe_edges, memory["edges"])
        score = 0.35 * node_score + 0.65 * edge_score if probe_edges else node_score
        if score <= 0:
            continue
        matches.append({
            "text": memory["text"],
            "score": round(score * 100, 1),
            "nodes": round(node_score * 100, 1),
            "edges": round(edge_score * 100, 1),
        })
    matches.sort(key=lambda x: (-x["score"], x["text"]))

    history = []
    for index, nodes in enumerate(result.activation_history):
        history.append({"step": index, "nodes": [int(n) for n in nodes], "count": len(nodes)})

    return {
        "source_nodes": [int(n) for n in result.source_nodes],
        "active_nodes": len(probe_nodes),
        "active_edges": len(probe_edges),
        "history": history,
        "edges": sorted(probe_edges),
        "matches": matches[:12],
    }


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Natural Route Probe v0.2</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--orange:#ff9d52;--green:#8ce3a9}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(80,145,210,.17),transparent 35%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 220px;gap:14px}input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:12px;font-size:16px}button{background:linear-gradient(135deg,#ec6f35,#ff9d52);border:0;color:white;border-radius:10px;padding:12px 18px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.muted{color:var(--muted)}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:28px;font-weight:800}.chain{display:flex;gap:7px;flex-wrap:wrap;align-items:center}.node{background:#071522;border:1px solid var(--line);border-radius:999px;padding:6px 9px;color:var(--cyan)}.arrow{color:var(--muted)}.step{padding:13px 0;border-bottom:1px solid var(--line)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid var(--line)}th{color:var(--muted)}.score{font-size:21px;font-weight:800}@media(max-width:700px){.grid,.stats{grid-template-columns:1fr}th:nth-child(n+4),td:nth-child(n+4){display:none}}
</style></head><body><main class="wrap">
<div class="card"><div class="eyebrow">Natural Route Probe v0.2</div><h1>SphereBrainが自然に向かった経路を見る</h1><p class="muted">続きの言葉は与えません。途中入力だけを非学習でCoreへ流し、活動がどこへ伝播したかを観測します。</p></div>
<div class="card"><form method="post"><div class="grid"><div><div class="eyebrow">Partial Input</div><h2>途中まで入力</h2><input name="text" value="{{text}}" placeholder="犬は" required></div><div><div class="eyebrow">Propagation Steps</div><h2>伝播段数</h2><input name="steps" type="number" min="1" max="60" value="{{steps}}"></div></div><p><button type="submit">自然経路を走らせる</button></p></form><p class="note">候補選択・文章生成・学習・保存・経路強化は行いません。brain.jsonとmemory.dbは読み取り専用です。</p></div>
{% if error %}<div class="card"><strong>{{error}}</strong></div>{% endif %}
{% if result %}
<div class="card"><div class="eyebrow">Natural Activity</div><h2>「{{text}}」から生じた活動</h2><div class="stats"><div class="stat"><div class="muted">入口ノード</div><div class="value">{{result.source_nodes|length}}</div><div>{{result.source_nodes}}</div></div><div class="stat"><div class="muted">活動ノード</div><div class="value">{{result.active_nodes}}</div></div><div class="stat"><div class="muted">通過経路</div><div class="value">{{result.active_edges}}</div></div></div></div>
<div class="card"><div class="eyebrow">Propagation History</div><h2>時間ごとの活動地点</h2>{% for h in result.history %}<div class="step"><strong>Step {{h.step}}</strong> <span class="muted">{{h.count}} nodes</span><div class="chain">{% for n in h.nodes %}<span class="node">{{n}}</span>{% else %}<span class="muted">活動なし</span>{% endfor %}</div></div>{% endfor %}</div>
<div class="card"><div class="eyebrow">Traversed Routes</div><h2>実際に通った経路</h2><div class="chain">{% for e in result.edges %}<span class="node">{{e[0]}} → {{e[1]}}</span>{% else %}<span class="muted">入口から先へ伝播しませんでした。これは現在のCore状態または伝播条件そのものの観測結果です。</span>{% endfor %}</div></div>
<div class="card"><div class="eyebrow">Observer Interpretation</div><h2>この自然経路に近かった過去経験</h2><p class="note">以下はSphereBrainが言葉を選んだ結果ではありません。自然に生じた経路を、Observerが過去経験へ照合した後付けの説明です。</p><table><thead><tr><th>過去経験</th><th>経路全体の近さ</th><th>ノード一致</th><th>エッジ一致</th></tr></thead><tbody>{% for m in result.matches %}<tr><td><strong>{{m.text}}</strong></td><td class="score">{{m.score}}%</td><td>{{m.nodes}}%</td><td>{{m.edges}}%</td></tr>{% else %}<tr><td colspan="4" class="muted">近い過去経験は見つかりませんでした。</td></tr>{% endfor %}</tbody></table></div>
{% endif %}
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text = "犬は"
    steps = 24
    result = None
    error = ""
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        try:
            steps = max(1, min(60, int(request.form.get("steps", "24"))))
            result = run_natural_probe(text, steps)
        except Exception as exc:
            error = str(exc)
    return render_template_string(PAGE, text=text, steps=steps, result=result, error=error)


def main() -> None:
    url = "http://127.0.0.1:5077"
    webbrowser.open(url)
    serve(app, host="127.0.0.1", port=5077, threads=4)


if __name__ == "__main__":
    main()
