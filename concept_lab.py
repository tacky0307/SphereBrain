from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

BASE = Path(__file__).resolve().parent
DB_FILE = BASE / "data" / "memory.db"
app = Flask(__name__)

DEFAULT_A = "空"
DEFAULT_B = "青,青い,青く"


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def edge_set(memory: dict) -> set[tuple[int, int]]:
    return {normalize_edge(edge) for edge in memory["traversed_edges"]}


def load_memories(limit: int = 2000) -> list[dict]:
    if not DB_FILE.exists():
        return []
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, kind, input_text, activated_nodes, traversed_edges
            FROM memories
            WHERE COALESCE(input_text, '') <> '' AND kind IN ('trainer', 'input')
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        item = dict(row)
        item["activated_nodes"] = json.loads(item["activated_nodes"])
        item["traversed_edges"] = json.loads(item["traversed_edges"])
        result.append(item)
    return result


def parse_terms(raw: str) -> list[str]:
    return [term.strip() for term in raw.replace("\n", ",").split(",") if term.strip()]


def select_group(memories: list[dict], terms: list[str]) -> list[dict]:
    if not terms:
        return []
    return [m for m in memories if any(term in (m.get("input_text") or "") for term in terms)]


def jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def summarize(group: list[dict]) -> dict:
    sets = [edge_set(m) for m in group]
    pair_scores = [jaccard(a, b) for a, b in combinations(sets, 2)]
    freq: Counter[tuple[int, int]] = Counter()
    node_freq: Counter[int] = Counter()
    for memory, edges in zip(group, sets):
        freq.update(edges)
        node_freq.update(set(int(n) for n in memory["activated_nodes"]))
    threshold = max(2, round(len(group) * 0.30)) if group else 0
    stable = {edge for edge, count in freq.items() if count >= threshold}
    top_edges = [
        {"a": edge[0], "b": edge[1], "count": count, "rate": round(count / len(group) * 100, 1)}
        for edge, count in freq.most_common(12)
    ] if group else []
    top_nodes = [
        {"node": node, "count": count, "rate": round(count / len(group) * 100, 1)}
        for node, count in node_freq.most_common(12)
    ] if group else []
    return {
        "count": len(group),
        "avg_similarity": round((sum(pair_scores) / len(pair_scores) * 100), 1) if pair_scores else 0.0,
        "stable_edges": stable,
        "stable_count": len(stable),
        "threshold": threshold,
        "top_edges": top_edges,
        "top_nodes": top_nodes,
        "examples": [m["input_text"] for m in group[:8]],
    }


def compare(a: dict, b: dict) -> dict:
    common = a["stable_edges"] & b["stable_edges"]
    union = a["stable_edges"] | b["stable_edges"]
    return {
        "common": len(common),
        "overlap": round(len(common) / len(union) * 100, 1) if union else 0.0,
        "a_only": len(a["stable_edges"] - b["stable_edges"]),
        "b_only": len(b["stable_edges"] - a["stable_edges"]),
        "common_edges": [{"a": x, "b": y} for x, y in sorted(common)[:30]],
    }


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Concept Lab v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1320px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.formgrid{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end}.edge{display:flex;justify-content:space-between;padding:9px 11px;background:#071522;border:1px solid rgba(36,70,109,.7);border-radius:10px;margin:7px 0}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.bar{height:16px;background:#071522;border-radius:999px;overflow:hidden;border:1px solid var(--line)}.fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--cyan))}.warning{color:#ffd0a9;background:rgba(255,157,82,.09);border:1px solid rgba(255,157,82,.35);padding:12px;border-radius:12px}@media(max-width:900px){.grid,.stats,.formgrid{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Concept Lab v0.1</h1><p>Coreの活動記録から、概念候補の共通経路と分離を観測する</p></div></header>
<main class="wrap">
<div class="warning">これは概念の断定装置ではなく、経路から「概念候補」を比較する初期観測装置です。入力文字は実験グループの選別だけに使い、数値はCoreの実活動経路から計算します。</div>
<section class="card"><form method="get"><div class="formgrid"><div><div class="eyebrow">Candidate A</div><input name="a" value="{{ raw_a }}" placeholder="空"></div><div><div class="eyebrow">Candidate B</div><input name="b" value="{{ raw_b }}" placeholder="青,青い,青く"></div><button>比較する</button></div></form><p class="muted">カンマ区切りで表記ゆれを指定できます。例：青,青い,青く</p></section>
<section class="stats card"><div class="stat"><div class="eyebrow">Loaded</div><div class="value">{{ total }}</div><div class="muted">分析対象の経験</div></div><div class="stat"><div class="eyebrow">A Experiences</div><div class="value">{{ sa.count }}</div></div><div class="stat"><div class="eyebrow">B Experiences</div><div class="value">{{ sb.count }}</div></div><div class="stat"><div class="eyebrow">Stable Overlap</div><div class="value">{{ comp.overlap }}%</div><div class="muted">安定経路のJaccard交差率</div></div></section>
<section class="grid">
{% for label,s in [('A',sa),('B',sb)] %}<article class="card"><div class="eyebrow">Concept Candidate {{ label }}</div><h2>{{ terms_a|join(' / ') if label=='A' else terms_b|join(' / ') }}</h2><div class="stats" style="grid-template-columns:repeat(3,1fr)"><div class="stat"><b>{{ s.count }}</b><div class="muted">経験数</div></div><div class="stat"><b>{{ s.avg_similarity }}%</b><div class="muted">群内平均一致率</div></div><div class="stat"><b>{{ s.stable_count }}</b><div class="muted">安定経路</div></div></div><p class="muted">安定経路は、この群の30%以上（最低2経験）に現れた経路。</p><h3>頻出経路</h3>{% for e in s.top_edges %}<div class="edge"><code>{{e.a}} → {{e.b}}</code><b>{{e.count}}回 / {{e.rate}}%</b></div>{% else %}<p class="muted">該当データがありません。</p>{% endfor %}<h3>代表文章</h3>{% for x in s.examples %}<span class="pill">{{x}}</span>{% endfor %}</article>{% endfor %}
</section>
<section class="card"><div class="eyebrow">Relation</div><h2>候補Aと候補Bの関係</h2><div class="stats"><div class="stat"><div class="value">{{comp.common}}</div><div class="muted">共通安定経路</div></div><div class="stat"><div class="value">{{comp.a_only}}</div><div class="muted">Aだけの安定経路</div></div><div class="stat"><div class="value">{{comp.b_only}}</div><div class="muted">Bだけの安定経路</div></div><div class="stat"><div class="value">{{comp.overlap}}%</div><div class="muted">交差率</div></div></div><h3>共通経路</h3>{% for e in comp.common_edges %}<span class="pill">{{e.a}} → {{e.b}}</span>{% else %}<p class="muted">まだ共通安定経路はありません。</p>{% endfor %}</section>
</main></body></html>
"""


@app.route("/")
def index():
    raw_a = request.args.get("a", DEFAULT_A)
    raw_b = request.args.get("b", DEFAULT_B)
    memories = load_memories()
    terms_a, terms_b = parse_terms(raw_a), parse_terms(raw_b)
    sa = summarize(select_group(memories, terms_a))
    sb = summarize(select_group(memories, terms_b))
    return render_template_string(PAGE, raw_a=raw_a, raw_b=raw_b, terms_a=terms_a, terms_b=terms_b, total=len(memories), sa=sa, sb=sb, comp=compare(sa, sb))


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5052")
    print("SphereBrain Concept Lab v0.1: http://127.0.0.1:5052")
    serve(app, host="127.0.0.1", port=5052, threads=4)
