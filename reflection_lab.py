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
    values: tuple[int, ...]

    @property
    def label(self) -> str:
        if self.kind in {"edge", "path3"}:
            return " → ".join(str(v) for v in self.values)
        return "{" + ", ".join(str(v) for v in self.values) + "}"


def normalize_edge(edge) -> tuple[int, int]:
    a, b = int(edge[0]), int(edge[1])
    return (a, b) if a <= b else (b, a)


def load_memories(limit: int = 8000) -> list[dict]:
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
        item["input_text"] = str(item.get("input_text") or "").strip()
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

    # Co-activation signatures are deliberately language-blind. Modulo bucketing
    # bounds the candidate space while retaining a coarse spatial fingerprint.
    bucketed = sorted({node % 64 for node in nodes})
    for index in range(len(bucketed) - 2):
        patterns.add(PatternKey("nodes3", tuple(bucketed[index:index + 3])))

    return patterns


def pattern_id(pattern: PatternKey) -> str:
    return pattern.kind + ":" + ",".join(str(v) for v in pattern.values)


def initialize_pattern_db() -> None:
    PATTERN_DB_FILE.parent.mkdir(exist_ok=True)
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS reflection_patterns_v2 (
                pattern_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                pattern_json TEXT NOT NULL,
                classification TEXT NOT NULL,
                experience_count INTEGER NOT NULL,
                total_experiences INTEGER NOT NULL,
                global_rate REAL NOT NULL,
                distinct_text_count INTEGER NOT NULL,
                target_text_count INTEGER NOT NULL,
                target_rate REAL NOT NULL,
                other_rate REAL NOT NULL,
                selectivity REAL NOT NULL,
                stability REAL NOT NULL,
                score REAL NOT NULL,
                target_texts TEXT NOT NULL,
                other_texts TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def _safe_rate(value: int, total: int) -> float:
    return value / total if total else 0.0


def analyze(
    memories: list[dict],
    min_experiences: int = 2,
    min_texts: int = 2,
    common_threshold: float = 0.80,
    selectivity_threshold: float = 0.20,
) -> dict:
    total = len(memories)
    text_totals: Counter[str] = Counter(memory["input_text"] for memory in memories)
    text_count = len(text_totals)

    pattern_count: Counter[PatternKey] = Counter()
    pattern_by_text: dict[PatternKey, Counter[str]] = defaultdict(Counter)
    patterns_per_memory: list[set[PatternKey]] = []

    for memory in memories:
        patterns = extract_patterns(memory)
        patterns_per_memory.append(patterns)
        text = memory["input_text"]
        for pattern in patterns:
            pattern_count[pattern] += 1
            pattern_by_text[pattern][text] += 1

    # Per-input temporal halves. This avoids mistaking staged learning order for
    # instability: each exact experience is compared against its own history.
    text_seen: Counter[str] = Counter()
    first_hits: dict[PatternKey, Counter[str]] = defaultdict(Counter)
    second_hits: dict[PatternKey, Counter[str]] = defaultdict(Counter)
    first_sizes: Counter[str] = Counter()
    second_sizes: Counter[str] = Counter()
    for memory, patterns in zip(memories, patterns_per_memory):
        text = memory["input_text"]
        position = text_seen[text]
        midpoint = max(1, text_totals[text] // 2)
        if position < midpoint:
            first_sizes[text] += 1
            target = first_hits
        else:
            second_sizes[text] += 1
            target = second_hits
        for pattern in patterns:
            target[pattern][text] += 1
        text_seen[text] += 1

    groups: dict[str, list[dict]] = {
        "common skeleton": [],
        "concept candidate": [],
        "individual memory": [],
        "selective pattern": [],
        "unclassified": [],
    }

    for pattern, count in pattern_count.items():
        if count < min_experiences:
            continue

        rates = {
            text: _safe_rate(pattern_by_text[pattern][text], occurrences)
            for text, occurrences in text_totals.items()
        }
        present_texts = [text for text, rate in rates.items() if rate > 0.0]
        distinct_text_count = len(present_texts)
        global_rate = _safe_rate(count, total)

        # A target group is inferred from activity only: no tokenization or word
        # matching is used. Inputs whose occurrence rate is clearly above the
        # pattern's global baseline form the pattern's selective experience set.
        target_cutoff = max(0.50, global_rate + 0.15)
        target_texts = [text for text, rate in rates.items() if rate >= target_cutoff]
        if not target_texts and rates:
            peak = max(rates.values())
            target_texts = [text for text, rate in rates.items() if rate >= peak - 0.05 and rate >= 0.35]

        target_experiences = sum(text_totals[text] for text in target_texts)
        target_hits = sum(pattern_by_text[pattern][text] for text in target_texts)
        other_texts_all = [text for text in text_totals if text not in target_texts]
        other_experiences = sum(text_totals[text] for text in other_texts_all)
        other_hits = sum(pattern_by_text[pattern][text] for text in other_texts_all)
        target_rate = _safe_rate(target_hits, target_experiences)
        other_rate = _safe_rate(other_hits, other_experiences)
        selectivity = target_rate - other_rate

        stability_values = []
        for text in target_texts:
            first_rate = _safe_rate(first_hits[pattern][text], first_sizes[text])
            second_rate = _safe_rate(second_hits[pattern][text], second_sizes[text])
            stability_values.append(max(0.0, 1.0 - abs(first_rate - second_rate)))
        stability = sum(stability_values) / len(stability_values) if stability_values else 0.0

        classification = "unclassified"
        if global_rate >= common_threshold and distinct_text_count >= max(2, int(text_count * 0.8)):
            classification = "common skeleton"
        elif distinct_text_count == 1:
            classification = "individual memory"
        elif (
            len(target_texts) >= min_texts
            and target_rate >= 0.60
            and selectivity >= selectivity_threshold
            and stability >= 0.60
            and global_rate < common_threshold
        ):
            classification = "concept candidate"
        elif len(target_texts) >= min_texts and selectivity >= 0.10:
            classification = "selective pattern"

        diversity = min(1.0, len(target_texts) / max(2, min_texts * 2))
        score = (
            max(0.0, selectivity) * 0.45
            + target_rate * 0.25
            + stability * 0.20
            + diversity * 0.10
        )
        if classification == "common skeleton":
            score = global_rate * 0.70 + stability * 0.30
        elif classification == "individual memory":
            score = max(rates.values(), default=0.0) * 0.70 + stability * 0.30

        ranked_targets = sorted(target_texts, key=lambda text: (-rates[text], text))[:8]
        ranked_others = sorted(other_texts_all, key=lambda text: (rates[text], text))[:8]
        item = {
            "pattern": pattern,
            "id": pattern_id(pattern),
            "label": pattern.label,
            "kind": pattern.kind,
            "classification": classification,
            "count": count,
            "global_rate": round(global_rate * 100, 1),
            "distinct_text_count": distinct_text_count,
            "target_text_count": len(target_texts),
            "target_rate": round(target_rate * 100, 1),
            "other_rate": round(other_rate * 100, 1),
            "selectivity": round(selectivity * 100, 1),
            "stability": round(stability * 100, 1),
            "score": round(score * 100, 1),
            "target_texts": ranked_targets,
            "other_texts": ranked_others,
        }
        groups[classification].append(item)

    for items in groups.values():
        items.sort(key=lambda item: (-item["score"], -item["selectivity"], -item["count"], item["id"]))

    displayed = {name: items[:80] for name, items in groups.items()}
    return {
        "total": total,
        "text_count": text_count,
        "pattern_count": sum(len(items) for items in groups.values()),
        "common_count": len(groups["common skeleton"]),
        "concept_count": len(groups["concept candidate"]),
        "individual_count": len(groups["individual memory"]),
        "selective_count": len(groups["selective pattern"]),
        "groups": displayed,
    }


def save_candidates(result: dict) -> None:
    initialize_pattern_db()
    with sqlite3.connect(PATTERN_DB_FILE, timeout=30) as conn:
        conn.execute("DELETE FROM reflection_patterns_v2")
        for items in result["groups"].values():
            for item in items:
                conn.execute(
                    """
                    INSERT INTO reflection_patterns_v2 (
                        pattern_id, kind, pattern_json, classification, experience_count,
                        total_experiences, global_rate, distinct_text_count, target_text_count,
                        target_rate, other_rate, selectivity, stability, score,
                        target_texts, other_texts, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        item["id"], item["kind"], json.dumps(item["pattern"].values),
                        item["classification"], item["count"], result["total"],
                        item["global_rate"], item["distinct_text_count"], item["target_text_count"],
                        item["target_rate"], item["other_rate"], item["selectivity"],
                        item["stability"], item["score"],
                        json.dumps(item["target_texts"], ensure_ascii=False),
                        json.dumps(item["other_texts"], ensure_ascii=False),
                    ),
                )


PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Reflection Lab v0.2</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#24466d;--text:#e8f0fb;--muted:#91a8c3;--cyan:#65d9ff;--green:#69e09a;--orange:#ff9d52;--purple:#b89cff;--yellow:#ffd166;--red:#ff8b8b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,rgba(65,132,190,.18),transparent 34%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1380px;margin:auto;padding:22px}header{border-bottom:1px solid var(--line)}h1{margin:0}h2{margin:8px 0 14px}header p,.muted{color:var(--muted)}.card{background:linear-gradient(180deg,#112742,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:15px}.value{font-size:30px;font-weight:800;margin-top:5px}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}.controls{display:grid;grid-template-columns:repeat(4,1fr) auto;gap:12px;align-items:end}input{width:100%;background:#071522;border:1px solid #31567f;color:var(--text);padding:11px;border-radius:10px;font-size:15px}button{background:linear-gradient(135deg,#ee6b2f,#ff9d52);color:white;border:0;border-radius:10px;padding:11px 18px;font-weight:700;cursor:pointer}.pattern{padding:14px;background:#071522;border:1px solid rgba(36,70,109,.7);border-radius:12px;margin:10px 0}.head{display:flex;justify-content:space-between;gap:14px;align-items:flex-start}.metrics{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px;margin-top:9px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}.pill.other{color:var(--muted)}.status{font-size:12px;border-radius:999px;padding:5px 9px;border:1px solid var(--line);white-space:nowrap}.concept{color:var(--green)}.skeleton{color:var(--cyan)}.individual{color:var(--yellow)}.selective{color:var(--purple)}.section-note{border-left:3px solid var(--cyan);padding-left:12px;color:var(--muted)}.split{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}.label{font-size:12px;color:var(--muted);margin-bottom:4px}@media(max-width:1000px){.stats{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:1fr 1fr}.split{grid-template-columns:1fr}}@media(max-width:650px){.stats,.controls{grid-template-columns:1fr}.head{display:block}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Reflection Lab v0.2</h1><p>全体共通の骨格を除き、経験群に選択的な活動を観測する</p></div></header>
<main class="wrap">
<section class="card"><form method="get"><div class="controls">
<div><div class="eyebrow">Minimum Experiences</div><input type="number" min="2" max="8000" name="min_exp" value="{{min_exp}}"></div>
<div><div class="eyebrow">Minimum Target Texts</div><input type="number" min="2" max="100" name="min_texts" value="{{min_texts}}"></div>
<div><div class="eyebrow">Common Skeleton %</div><input type="number" min="50" max="100" name="common" value="{{common}}"></div>
<div><div class="eyebrow">Minimum Selectivity pt</div><input type="number" min="5" max="90" name="selectivity" value="{{selectivity}}"></div>
<button>活動を分類する</button></div></form>
<p class="muted">判定はTraceの出現分布だけを使います。入力文は活動が現れた経験を人間に示すラベルとしてのみ表示し、単語の一致や文章解析には使いません。</p></section>
<section class="stats card">
<div class="stat"><div class="eyebrow">Experiences</div><div class="value">{{x.total}}</div><div class="muted">{{x.text_count}}種類</div></div>
<div class="stat"><div class="eyebrow">Common Skeleton</div><div class="value">{{x.common_count}}</div></div>
<div class="stat"><div class="eyebrow">Concept Candidates</div><div class="value">{{x.concept_count}}</div></div>
<div class="stat"><div class="eyebrow">Individual Memory</div><div class="value">{{x.individual_count}}</div></div>
<div class="stat"><div class="eyebrow">Selective Patterns</div><div class="value">{{x.selective_count}}</div></div>
</section>
{% set sections = [
('concept candidate','概念候補','複数の異なる経験に高頻度で現れ、その他の経験では弱い活動。言葉ではなく活動分布から選ばれます。','concept'),
('common skeleton','全経験の共通骨格','ほぼすべての経験で使われる基幹経路。概念候補から除外します。','skeleton'),
('individual memory','個別記憶','一種類の入力経験だけに現れる活動。経験の固有記憶として扱います。','individual'),
('selective pattern','選択的活動（保留）','複数経験に偏っていますが、概念候補の強さや安定性にはまだ届かない活動です。','selective')
] %}
{% for key,title,note,css in sections %}
<section class="card"><div class="eyebrow">Activity Classification</div><h2>{{title}} <span class="muted">{{x.groups[key]|length}}件表示</span></h2><p class="section-note">{{note}}</p>
{% for item in x.groups[key] %}<div class="pattern"><div class="head"><div><code>{{item.label}}</code><div class="metrics"><span>{{item.kind}}</span><span>{{item.count}}経験</span><span>全体 {{item.global_rate}}%</span><span>対象群 {{item.target_rate}}%</span><span>その他 {{item.other_rate}}%</span><span>差分 {{'%+.1f'|format(item.selectivity)}} pt</span><span>安定度 {{item.stability}}%</span><span>総合 {{item.score}}</span></div></div><span class="status {{css}}">{{item.classification}}</span></div>
<div class="split"><div><div class="label">主に現れた経験</div>{% for text in item.target_texts %}<span class="pill">{{text}}</span>{% else %}<span class="muted">対象群なし</span>{% endfor %}</div><div><div class="label">弱かった／現れなかった経験</div>{% for text in item.other_texts %}<span class="pill other">{{text}}</span>{% else %}<span class="muted">比較対象なし</span>{% endfor %}</div></div></div>{% else %}<p class="muted">現在の条件を満たす活動はありません。</p>{% endfor %}</section>
{% endfor %}
</main></body></html>
"""


@app.route("/")
def index():
    min_exp = max(2, int(request.args.get("min_exp", 2)))
    min_texts = max(2, int(request.args.get("min_texts", 2)))
    common = min(100, max(50, int(request.args.get("common", 80))))
    selectivity = min(90, max(5, int(request.args.get("selectivity", 20))))
    memories = load_memories()
    result = analyze(
        memories,
        min_experiences=min_exp,
        min_texts=min_texts,
        common_threshold=common / 100.0,
        selectivity_threshold=selectivity / 100.0,
    )
    save_candidates(result)
    return render_template_string(
        PAGE,
        x=result,
        min_exp=min_exp,
        min_texts=min_texts,
        common=common,
        selectivity=selectivity,
    )


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5053")
    print("SphereBrain Reflection Lab v0.2: http://127.0.0.1:5053")
    serve(app, host="127.0.0.1", port=5053, threads=4)
