from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import webbrowser

import numpy as np
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
        items.append({"text": row["input_text"] or "", "nodes": nodes, "edges": edges})
    return items


def edge_score(brain: SphereBrain, source: int, target: int, visited: set[int]) -> tuple[float, float, int, int]:
    weight = float(brain.weights[source, target])
    usage = int(brain.usage[source, target])
    target_usage = int(brain.node_usage[target])
    learned = usage / (usage + 4.0)
    familiar_target = target_usage / (target_usage + 12.0)
    revisit_penalty = 0.35 if target in visited else 1.0
    score = (0.55 * weight + 0.35 * learned + 0.10 * familiar_target) * revisit_penalty
    return score, weight, usage, target_usage


def ranked_choices(brain: SphereBrain, source: int, previous: int | None, visited: set[int]) -> list[dict]:
    neighbors = [int(n) for n in np.flatnonzero(brain.adjacency[source])]
    candidates = []
    for target in neighbors:
        if previous is not None and target == previous and len(neighbors) > 1:
            continue
        score, weight, usage, target_usage = edge_score(brain, source, target, visited)
        candidates.append({
            "target": target,
            "score": score,
            "weight": weight,
            "usage": usage,
            "target_usage": target_usage,
            "revisited": target in visited,
        })
    candidates.sort(key=lambda item: (-item["score"], -item["usage"], -item["weight"], item["target"]))
    return candidates


def choose_start(brain: SphereBrain, sources: list[int]) -> tuple[int, list[dict]]:
    starts = []
    for source in sources:
        choices = ranked_choices(brain, source, None, {source})
        best = choices[0] if choices else None
        starts.append({
            "source": source,
            "best_score": best["score"] if best else -1.0,
            "best_target": best["target"] if best else None,
            "best_usage": best["usage"] if best else 0,
            "best_weight": best["weight"] if best else 0.0,
        })
    starts.sort(key=lambda item: (-item["best_score"], -item["best_usage"], -item["best_weight"], item["source"]))
    return starts[0]["source"], starts


def run_forced_probe(text: str, steps: int) -> dict:
    if not BRAIN_FILE.exists():
        raise FileNotFoundError("data/brain.json がありません。先に通常のSphereBrainを起動してください。")

    brain = SphereBrain.load(BRAIN_FILE)
    source_nodes = [int(n) for n in brain.text_to_sources(text)]
    current, start_options = choose_start(brain, source_nodes)
    path = [current]
    visited = {current}
    decisions = []
    previous = None

    for step in range(steps):
        choices = ranked_choices(brain, current, previous, visited)
        if not choices:
            break

        unvisited = [choice for choice in choices if not choice["revisited"]]
        selected = unvisited[0] if unvisited else choices[0]
        alternatives = []
        for rank, choice in enumerate(choices[:5], 1):
            alternatives.append({
                "rank": rank,
                "target": choice["target"],
                "score": round(choice["score"] * 100, 1),
                "weight": round(choice["weight"], 4),
                "usage": choice["usage"],
                "revisited": choice["revisited"],
                "selected": choice["target"] == selected["target"],
            })

        next_node = int(selected["target"])
        decisions.append({
            "step": step + 1,
            "source": current,
            "target": next_node,
            "score": round(selected["score"] * 100, 1),
            "weight": round(selected["weight"], 4),
            "usage": selected["usage"],
            "alternatives": alternatives,
        })
        previous, current = current, next_node
        path.append(current)
        if current in visited:
            break
        visited.add(current)

    path_edges = {normalize_edge((path[i], path[i + 1])) for i in range(len(path) - 1)}
    path_nodes = set(path)
    matches = []
    for memory in load_experiences():
        node_score = jaccard(path_nodes, memory["nodes"])
        edge_score_value = jaccard(path_edges, memory["edges"])
        score = 0.25 * node_score + 0.75 * edge_score_value if path_edges else node_score
        if score <= 0:
            continue
        matches.append({
            "text": memory["text"],
            "score": round(score * 100, 1),
            "nodes": round(node_score * 100, 1),
            "edges": round(edge_score_value * 100, 1),
        })
    matches.sort(key=lambda item: (-item["score"], item["text"]))

    return {
        "source_nodes": source_nodes,
        "chosen_start": path[0],
        "start_options": start_options,
        "path": path,
        "edges": sorted(path_edges),
        "decisions": decisions,
        "matches": matches[:12],
    }


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Forced Route Choice Probe v0.3</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--orange:#ff9d52;--green:#8ce3a9;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(80,145,210,.17),transparent 35%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1240px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 220px;gap:14px}input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:12px;font-size:16px}button{background:linear-gradient(135deg,#ec6f35,#ff9d52);border:0;color:white;border-radius:10px;padding:12px 18px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.muted{color:var(--muted)}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stat,.decision{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:28px;font-weight:800}.chain{display:flex;gap:7px;flex-wrap:wrap;align-items:center}.node{background:#071522;border:1px solid var(--line);border-radius:999px;padding:7px 10px;color:var(--cyan)}.chosen{border-color:var(--green);color:var(--green)}.arrow{color:var(--muted)}.decision{margin:12px 0}.alternatives{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px}.alternative{border:1px solid var(--line);border-radius:10px;padding:9px}.alternative.selected{border-color:var(--green);background:rgba(140,227,169,.08)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid var(--line)}th{color:var(--muted)}.score{font-size:21px;font-weight:800}@media(max-width:800px){.grid,.stats,.alternatives{grid-template-columns:1fr}th:nth-child(n+4),td:nth-child(n+4){display:none}}
</style></head><body><main class="wrap">
<div class="card"><div class="eyebrow">Forced Route Choice Probe v0.3</div><h1>SphereBrainに経路を選ばせる</h1><p class="muted">途中入力から伝播を待つのではなく、各分岐で保存済みCoreの重みと使用履歴が最も強い経路を必ず一本選ばせます。</p></div>
<div class="card"><form method="post"><div class="grid"><div><div class="eyebrow">Partial Input</div><h2>途中まで入力</h2><input name="text" value="{{text}}" placeholder="犬は" required></div><div><div class="eyebrow">Choice Steps</div><h2>選択段数</h2><input name="steps" type="number" min="1" max="60" value="{{steps}}"></div></div><p><button type="submit">経路を選ばせる</button></p></form><p class="note">閾値で止めません。言葉の候補も与えません。各地点から選べる経路を順位付けし、最上位を強制的に選択します。学習・保存・経路強化は行いません。</p></div>
{% if error %}<div class="card"><strong>{{error}}</strong></div>{% endif %}
{% if result %}
<div class="card"><div class="eyebrow">Selected Route</div><h2>「{{text}}」からSphereBrainが選んだ一本</h2><div class="stats"><div class="stat"><div class="muted">入口候補</div><div class="value">{{result.source_nodes|length}}</div><div>{{result.source_nodes}}</div></div><div class="stat"><div class="muted">選択した入口</div><div class="value">{{result.chosen_start}}</div></div><div class="stat"><div class="muted">選択した経路数</div><div class="value">{{result.edges|length}}</div></div></div><div class="chain" style="margin-top:18px">{% for n in result.path %}<span class="node {% if loop.first %}chosen{% endif %}">{{n}}</span>{% if not loop.last %}<span class="arrow">→</span>{% endif %}{% endfor %}</div></div>
<div class="card"><div class="eyebrow">Choice Log</div><h2>各分岐で何を選んだか</h2>{% for d in result.decisions %}<div class="decision"><strong>Step {{d.step}}: {{d.source}} → {{d.target}}</strong><span class="muted">　選択度 {{d.score}}% / weight {{d.weight}} / usage {{d.usage}}</span><div class="alternatives">{% for a in d.alternatives %}<div class="alternative {% if a.selected %}selected{% endif %}"><strong>#{{a.rank}} → {{a.target}}</strong><div>{{a.score}}%</div><div class="muted">w {{a.weight}} / use {{a.usage}}</div>{% if a.selected %}<div style="color:var(--green)">選択</div>{% endif %}</div>{% endfor %}</div></div>{% else %}<p class="muted">入口から選べる接続がありませんでした。</p>{% endfor %}</div>
<div class="card"><div class="eyebrow">Observer Interpretation</div><h2>選ばれた経路に近かった過去経験</h2><p class="note">ここに出る言葉は回答ではありません。SphereBrainが選んだ経路を、Observerが過去経験へ後から照合した説明です。</p><table><thead><tr><th>過去経験</th><th>経路全体の近さ</th><th>ノード一致</th><th>エッジ一致</th></tr></thead><tbody>{% for m in result.matches %}<tr><td><strong>{{m.text}}</strong></td><td class="score">{{m.score}}%</td><td>{{m.nodes}}%</td><td>{{m.edges}}%</td></tr>{% else %}<tr><td colspan="4" class="muted">近い過去経験は見つかりませんでした。</td></tr>{% endfor %}</tbody></table></div>
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
            result = run_forced_probe(text, steps)
        except Exception as exc:
            error = str(exc)
    return render_template_string(PAGE, text=text, steps=steps, result=result, error=error)


def main() -> None:
    url = "http://127.0.0.1:5077"
    webbrowser.open(url)
    serve(app, host="127.0.0.1", port=5077, threads=4)


if __name__ == "__main__":
    main()
