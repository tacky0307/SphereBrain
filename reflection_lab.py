from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3
import webbrowser

from flask import Flask, render_template_string, request
from waitress import serve

BASE = Path(__file__).resolve().parent
DB_FILE = BASE / "data" / "memory.db"
PATTERN_DB_FILE = BASE / "data" / "pattern_candidates.db"
app = Flask(__name__)


@dataclass(frozen=True)
class PatternKey:
    kind: str
    values: tuple

    @property
    def label(self) -> str:
        if self.kind == "edge":
            return f"{self.values[0]} → {self.values[1]}"
        if self.kind == "path3":
            return " → ".join(str(v) for v in self.values)
        return "{" + ", ".join(str(v) for v in self.values) + "}"


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def load_memories(limit: int = 4000) -> list[dict]:
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
            ORDER BY id ASC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    result = []
    for row in rows:
        item = dict(row)
        item["activated_nodes"] = [int(v) for v in json.loads(item["activated_nodes"])]
        item["traversed_edges"] = [normalize_edge(v) for v in json.loads(item["traversed_edges"])]
        result.append(item)
    return result


def extract_patterns(memory: dict) -> set[PatternKey]:
    patterns: set[PatternKey] = set()
    edges = list(dict.fromkeys(memory["traversed_edges"]))
    nodes = list(dict.fromkeys(memory["activated_nodes"]))

    for edge in edges:
        patterns.add(PatternKey("edge", edge))

    for index in range(len(nodes) - 2):
        patterns.add(PatternKey("path3", tuple(nodes[index:index + 3])))

    # Small co-activation signatures. The modulo bucket keeps the first version bounded.
    bucketed = sorted({node % 64 for node in nodes})
    for index in range(len(bucketed) - 2):
        patterns.add(PatternKey("nodes3", tuple(bucketed[index:index + 3])))

    return patterns


def initialize_pattern_db() -> None:
    PATTERN_DB_FILE.parent.mkdir(exist_ok=True)
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pattern_candidates (
                pattern_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                pattern_json TEXT NOT NULL,
                experience_count INTEGER NOT NULL,
                total_experiences INTEGER NOT NULL,
                occurrence_rate REAL NOT NULL,
                distinct_text_count INTEGER NOT NULL,
                stability REAL NOT NULL,
                score REAL NOT NULL,
                status TEXT NOT NULL,
                example_texts TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def pattern_id(pattern: PatternKey) -> str:
    return pattern.kind + ":" + ",".join(str(v) for v in pattern.values)


def analyze(memories: list[dict], min_experiences: int = 2, min_texts: int = 2) -> dict:
    total = len(memories)
    pattern_experiences: Counter[PatternKey] = Counter()
    pattern_texts: dict[PatternKey, set[str]] = defaultdict(set)
    pattern_examples: dict[PatternKey, list[str]] = defaultdict(list)
    first_half: Counter[PatternKey] = Counter()
    second_half: Counter[PatternKey] = Counter()
    midpoint = max(1, total // 2)

    for index, memory in enumerate(memories):
        text = memory.get("input_text") or ""
        patterns = extract_patterns(memory)
        for pattern in patterns:
            pattern_experiences[pattern] += 1
            pattern_texts[pattern].add(text)
            if text not in pattern_examples[pattern] and len(pattern_examples[pattern]) < 8:
                pattern_examples[pattern].append(text)
            if index < midpoint:
                first_half[pattern] += 1
            else:
                second_half[pattern] += 1

    candidates = []
    first_size = midpoint
    second_size = max(1, total - midpoint)
    for pattern, count in pattern_experiences.items():
        distinct_text_count = len(pattern_texts[pattern])
        if count < min_experiences or distinct_text_count < min_texts:
            continue

        occurrence_rate = count / total if total else 0.0
        rate_first = first_half[pattern] / first_size if first_size else 0.0
        rate_second = second_half[pattern] / second_size if second_size else 0.0
        stability = max(0.0, 1.0 - abs(rate_first - rate_second))
        diversity = min(1.0, distinct_text_count / max(2, min_texts * 2))
        score = occurrence_rate * 0.45 + stability * 0.30 + diversity * 0.25

        status = "activity pattern"
        if distinct_text_count >= 3 and stability >= 0.75 and occurrence_rate >= 0.10:
            status = "concept candidate"
        elif distinct_text_count >= 2 and stability >= 0.60:
            status = "stable candidate"

        candidates.append({
            "pattern": pattern,
            "id": pattern_id(pattern),
            "label": pattern.label,
            "kind": pattern.kind,
            "count": count,
            "rate": round(occurrence_rate * 100, 1),
            "distinct_text_count": distinct_text_count,
            "stability": round(stability * 100, 1),
            "score": round(score * 100, 1),
            "status": status,
            "examples": pattern_examples[pattern],
        })

    candidates.sort(key=lambda item: (-item["score"], -item["distinct_text_count"], -item["count"], item["id"]))
    return {
        "total": total,
        "candidate_count": len(candidates),
        "concept_count": sum(1 for item in candidates if item["status"] == "concept candidate"),
        "stable_count": sum(1 for item in candidates if item["status"] == "stable candidate"),
        "candidates": candidates[:120],
    }


def save_candidates(result: dict) -> None:
    initialize_pattern_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        for item in result["candidates"]:
            conn.execute(
                """
                INSERT INTO pattern_candidates (
                    pattern_id, kind, pattern_json, experience_count, total_experiences,
                    occurrence_rate, distinct_text_count, stability, score, status, example_texts, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(pattern_id) DO UPDATE SET
                    experience_count=excluded.experience_count,
                    total_experiences=excluded.total_experiences,
                    occurrence_rate=excluded.occurrence_rate,
                    distinct_text_count=excluded.distinct_text_count,
                    stability=excluded.stability,
                    score=excluded.score,
                    status=excluded.status,
                    example_texts=excluded.example_texts,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    item["id"], item["kind"], json.dumps(item["pattern"].values), item["count"],
                    result["total"], item["rate"], item["distinct_text_count"], item["stability"],
                    item["score"], item["status"], json.dumps(item["examples"], ensure_ascii=False),
                ),
            )


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Reflection Lab v0.1</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff;--yellow:#ffd166}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1380px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px;align-items:end}input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.pattern{padding:14px;background:#071522;border:1px solid rgba(36,70,109,.7);border-radius:12px;margin:10px 0}.head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}.metrics{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-top:9px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.status{font-size:12px;border-radius:999px;padding:5px 9px;border:1px solid var(--line)}.concept{color:var(--green)}.stable{color:var(--yellow)}@media(max-width:900px){.stats,.controls{grid-template-columns:1fr}.head{display:block}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Reflection Lab v0.1</h1><p>言葉ではなく、Traceに繰り返し現れる活動経路を観測する</p></div></header>
<main class="wrap">
<section class="card"><form method="get"><div class="controls"><div><div class="eyebrow">Minimum Experiences</div><input type="number" min="2" max="1000" name="min_exp" value="{{min_exp}}"></div><div><div class="eyebrow">Minimum Distinct Texts</div><input type="number" min="2" max="100" name="min_texts" value="{{min_texts}}"></div><button>共通活動を抽出する</button></div></form><p class="muted">同じ文章の反復だけでは概念候補にしません。複数の異なる経験で再現され、時間的に安定した活動を候補化します。</p></section>
<section class="stats card"><div class="stat"><div class="eyebrow">Experiences</div><div class="value">{{x.total}}</div></div><div class="stat"><div class="eyebrow">Patterns</div><div class="value">{{x.candidate_count}}</div></div><div class="stat"><div class="eyebrow">Stable</div><div class="value">{{x.stable_count}}</div></div><div class="stat"><div class="eyebrow">Concept Candidates</div><div class="value">{{x.concept_count}}</div></div></section>
<section class="card"><div class="eyebrow">Activity Patterns</div><h2>共通活動候補</h2>{% for item in x.candidates %}<div class="pattern"><div class="head"><div><code>{{item.label}}</code><div class="metrics"><span>{{item.kind}}</span><span>{{item.count}}経験</span><span>{{item.distinct_text_count}}種類の入力</span><span>出現率 {{item.rate}}%</span><span>安定度 {{item.stability}}%</span><span>総合 {{item.score}}</span></div></div><span class="status {% if item.status == 'concept candidate' %}concept{% elif item.status == 'stable candidate' %}stable{% endif %}">{{item.status}}</span></div><div>{% for text in item.examples %}<span class="pill">{{text}}</span>{% endfor %}</div></div>{% else %}<p class="muted">条件を満たす活動パターンはまだありません。</p>{% endfor %}</section>
</main></body></html>
"""


@app.route("/")
def index():
    min_exp = max(2, int(request.args.get("min_exp", 2)))
    min_texts = max(2, int(request.args.get("min_texts", 2)))
    memories = load_memories()
    result = analyze(memories, min_experiences=min_exp, min_texts=min_texts)
    save_candidates(result)
    return render_template_string(PAGE, x=result, min_exp=min_exp, min_texts=min_texts)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5053")
    print("SphereBrain Reflection Lab v0.1: http://127.0.0.1:5053")
    serve(app, host="127.0.0.1", port=5053, threads=4)
