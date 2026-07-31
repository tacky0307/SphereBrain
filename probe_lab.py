from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProbeSettings:
    steps: int = 18
    context_nodes: int = 16


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def jaccard(left: set, right: set) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_reference_routes(prefix: str, candidate: str) -> tuple[set[tuple[int, int]], int, str]:
    if not DB_FILE.exists():
        return set(), 0, "no database"

    exact = f"{prefix}{candidate}".strip()
    with sqlite3.connect(f"file:{DB_FILE.as_posix()}?mode=ro", uri=True, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT input_text, traversed_edges FROM memories "
            "WHERE kind='input' AND input_text=? ORDER BY id DESC LIMIT 100",
            (exact,),
        ).fetchall()
        source = "exact experience"
        if not rows:
            rows = conn.execute(
                "SELECT input_text, traversed_edges FROM memories "
                "WHERE kind='input' AND input_text LIKE ? ORDER BY id DESC LIMIT 100",
                (f"%{candidate.strip()}",),
            ).fetchall()
            source = "candidate-ending experiences"

    edges: set[tuple[int, int]] = set()
    for row in rows:
        try:
            edges.update(normalize_edge(edge) for edge in json.loads(row["traversed_edges"] or "[]"))
        except (TypeError, json.JSONDecodeError):
            continue
    return edges, len(rows), source


def pick_context(result, limit: int) -> list[int]:
    final = result.final_activation
    ranked = sorted(result.activated_nodes, key=lambda node: float(final[node]), reverse=True)
    return ranked[:limit]


def classify(score: float) -> str:
    if score >= 0.60:
        return "○"
    if score >= 0.35:
        return "△"
    return "×"


def run_probe(prefix: str, candidates: list[str], settings: ProbeSettings) -> list[dict]:
    if not BRAIN_FILE.exists():
        raise FileNotFoundError("data/brain.json がありません。先に通常のSphereBrainを起動してください。")

    base_brain = SphereBrain.load(BRAIN_FILE)
    prefix_result = base_brain.propagate(
        base_brain.text_to_sources(prefix),
        steps=settings.steps,
        noise=0.0,
        learn=False,
    )
    prefix_edges = {normalize_edge(edge) for edge in prefix_result.traversed_edges}
    context = pick_context(prefix_result, settings.context_nodes)

    results = []
    for candidate in candidates:
        # Each candidate starts from the same persisted Core. No learning or saving occurs.
        brain = SphereBrain.load(BRAIN_FILE)
        continuation = brain.propagate(
            brain.text_to_sources(candidate),
            steps=settings.steps,
            noise=0.0,
            learn=False,
            context_nodes=context,
        )
        continuation_edges = {normalize_edge(edge) for edge in continuation.traversed_edges}
        reference_edges, reference_count, reference_source = load_reference_routes(prefix, candidate)

        continuity = jaccard(prefix_edges, continuation_edges)
        learned_resonance = jaccard(continuation_edges, reference_edges) if reference_edges else 0.0
        # Reference resonance is primary; continuity keeps the partial input involved.
        score = 0.72 * learned_resonance + 0.28 * continuity
        results.append({
            "candidate": candidate,
            "mark": classify(score),
            "score": round(score * 100, 1),
            "learned": round(learned_resonance * 100, 1),
            "continuity": round(continuity * 100, 1),
            "reference_count": reference_count,
            "reference_source": reference_source,
            "active_nodes": len(continuation.activated_nodes),
            "active_edges": len(continuation_edges),
        })

    return sorted(results, key=lambda item: (-item["score"], item["candidate"]))


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Probe Resonance Lab v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--orange:#ff9d52;--green:#8ce3a9;--yellow:#ffd166;--red:#ff8585}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(80,145,210,.17),transparent 35%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1100px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}textarea,input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:12px;font-size:16px}textarea{min-height:150px}button{background:linear-gradient(135deg,#ec6f35,#ff9d52);border:0;color:white;border-radius:10px;padding:12px 18px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.muted{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px 10px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:13px}.mark{font-size:28px;font-weight:900}.score{font-size:24px;font-weight:800}.ok{color:var(--green)}.maybe{color:var(--yellow)}.no{color:var(--red)}.note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}@media(max-width:700px){.grid{grid-template-columns:1fr}th:nth-child(n+5),td:nth-child(n+5){display:none}}
</style></head><body><main class="wrap">
<div class="card"><div class="eyebrow">Probe Resonance Lab v0.1</div><h1>SphereBrainに続きを選ばせる</h1><p class="muted">途中の刺激と候補をCoreへ非学習で流し、過去の経験経路との共鳴を順位付けします。</p></div>
<div class="card"><form method="post"><div class="grid"><div><div class="eyebrow">Partial Input</div><h2>途中まで入力</h2><input name="prefix" value="{{prefix}}" placeholder="犬は" required></div><div><div class="eyebrow">Candidates</div><h2>候補（1行に1つ）</h2><textarea name="candidates" required>{{candidate_text}}</textarea></div></div><p><button type="submit">続きをProbeする</button></p></form><p class="note">brain.jsonとmemory.dbは読み取り専用です。Probeによる学習・保存・経路強化は行いません。</p></div>
{% if error %}<div class="card"><strong class="no">{{error}}</strong></div>{% endif %}
{% if results %}<div class="card"><div class="eyebrow">Resonance Ranking</div><h2>「{{prefix}}」の続き候補</h2><table><thead><tr><th>判定</th><th>候補</th><th>総合</th><th>経験共鳴</th><th>途中状態の連続性</th><th>参照経験</th></tr></thead><tbody>{% for r in results %}<tr><td class="mark {% if r.mark=='○' %}ok{% elif r.mark=='△' %}maybe{% else %}no{% endif %}">{{r.mark}}</td><td><strong>{{r.candidate}}</strong><div class="muted">{{r.active_nodes}} nodes / {{r.active_edges}} edges</div></td><td class="score">{{r.score}}%</td><td>{{r.learned}}%</td><td>{{r.continuity}}%</td><td>{{r.reference_count}}件<div class="muted">{{r.reference_source}}</div></td></tr>{% endfor %}</tbody></table></div>
<div class="card"><div class="eyebrow">Important</div><p class="note">これは文章生成ではなく、外部から与えた候補の共鳴試験です。また現在のEncoderは文章全体をハッシュで刺激へ変換するため、「犬は」と「犬は走る」が自動的に同じ入口を共有する保証はありません。低い結果も失敗ではなく、次のEncoder設計を決める観測結果になります。</p></div>{% endif %}
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prefix = "犬は"
    candidate_text = "走る\n食べる\n眠る\n飛ぶ"
    results = []
    error = ""
    if request.method == "POST":
        prefix = request.form.get("prefix", "").strip()
        candidate_text = request.form.get("candidates", "")
        candidates = [line.strip() for line in candidate_text.splitlines() if line.strip()]
        try:
            results = run_probe(prefix, candidates, ProbeSettings())
        except Exception as exc:
            error = str(exc)
    return render_template_string(PAGE, prefix=prefix, candidate_text=candidate_text, results=results, error=error)


def main() -> None:
    url = "http://127.0.0.1:5077"
    webbrowser.open(url)
    serve(app, host="127.0.0.1", port=5077, threads=4)


if __name__ == "__main__":
    main()
