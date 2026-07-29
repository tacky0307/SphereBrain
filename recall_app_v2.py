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

if not BRAIN_FILE.exists() or not DB_FILE.exists():
    raise FileNotFoundError("brain.json または memory.db が見つかりません。")

brain = SphereBrain.load(BRAIN_FILE)
app = Flask(__name__)
last_query = ""
last_result = None
last_matches: list[dict] = []

PAGE = """
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sphere Brain 集中想起モード</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f7fb;color:#1f2937}header{background:#312e81;color:white;padding:18px 24px}main{max-width:1180px;margin:24px auto;padding:0 16px 40px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:white;border-radius:14px;padding:18px;box-shadow:0 4px 18px rgba(0,0,0,.07)}textarea{width:100%;min-height:100px;box-sizing:border-box;padding:12px;font-size:16px}button{background:#7c3aed;color:white;border:0;padding:11px 18px;border-radius:9px;font-size:15px;cursor:pointer}.badge{display:inline-block;background:#ede9fe;color:#5b21b6;padding:4px 9px;border-radius:999px}.notice{background:#ecfdf5;border:1px solid #86efac;padding:10px 12px;border-radius:10px}.metric{background:#f8fafc;border-radius:10px;padding:10px;margin-bottom:8px}.metric strong{font-size:22px}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}iframe{width:100%;height:650px;border:0;background:white}small,.muted{color:#64748b}@media(max-width:800px){.grid{grid-template-columns:1fr}iframe{height:500px}}
</style></head><body>
<header><h1>Sphere Brain 集中想起モード v2</h1><div>伝播を狭め、入力記憶を優先して思い出します</div></header><main>
<p class="notice"><strong>安全モード：</strong>学習・保存・使用回数の変更はありません。内部活動は順位を下げます。</p>
<div class="grid"><section class="card"><h2>【想起】</h2><form method="post" action="/recall"><textarea name="text" placeholder="例：今日は　／　空は　／　雨　／　私は"></textarea><p><button type="submit">集中して想起する</button></p></form><small>8ステップ・高めの閾値で、近い範囲だけをたどります。</small></section>
<section class="card"><h2>想起結果</h2>{% if result %}<p><span class="badge">{{ query }}</span></p><div class="metric">起点ノード：<strong>{{ result.source_nodes|join(', ') }}</strong></div><div class="metric">活性化ノード：<strong>{{ result.activated_nodes|length }}</strong></div><div class="metric">通過経路：<strong>{{ result.traversed_edges|length }}</strong></div>{% else %}<p class="muted">まだ想起していません。</p>{% endif %}</section></div>
{% if matches %}<section class="card" style="margin-top:16px"><h2>近い入力記憶</h2><table><tr><th>総合点</th><th>記憶内容</th><th>種類</th><th>経路</th><th>文字</th></tr>{% for item in matches %}<tr><td>{{ '%.1f'|format(item.score*100) }}%</td><td>{{ item.input_text or '内部活動' }}</td><td>{{ item.kind }}</td><td>{{ '%.1f'|format(item.path_score*100) }}%</td><td>{{ '%.1f'|format(item.text_score*100) }}%</td></tr>{% endfor %}</table><p class="muted">総合点は、狭い範囲の経路一致70％、文字の部分一致25％、起点一致5％。内部活動は減点します。</p></section>{% endif %}
<section class="card" style="margin-top:16px"><h2>想起した経路</h2>{% if result %}<iframe src="/recall-view?ts={{ timestamp }}"></iframe>{% else %}<p class="muted">想起すると表示します。</p>{% endif %}</section>
</main></body></html>
"""

def load_memories() -> list[dict]:
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM memories ORDER BY id DESC").fetchall()
    result=[]
    for row in rows:
        item=dict(row)
        for key in ("source_nodes","activated_nodes","traversed_edges"):
            item[key]=json.loads(item[key])
        result.append(item)
    return result

def character_ngrams(text: str) -> set[str]:
    clean="".join(text.split())
    if not clean:
        return set()
    if len(clean)==1:
        return {clean}
    grams={clean[i:i+2] for i in range(len(clean)-1)}
    grams.update(clean)
    return grams

def text_similarity(query: str, memory_text: str) -> float:
    q="".join(query.split())
    m="".join((memory_text or "").split())
    if not q or not m:
        return 0.0
    if q in m:
        return min(1.0, 0.75 + 0.25 * len(q)/len(m))
    qg=character_ngrams(q); mg=character_ngrams(m)
    union=qg|mg
    return len(qg&mg)/len(union) if union else 0.0

def find_similar(query: str, result, limit: int=8) -> list[dict]:
    recalled=set(result.activated_nodes); sources=set(result.source_nodes)
    ranked=[]
    for item in load_memories():
        nodes=set(item["activated_nodes"]); memory_sources=set(item["source_nodes"])
        union=recalled|nodes
        path_score=len(recalled&nodes)/len(union) if union else 0.0
        source_union=sources|memory_sources
        source_score=len(sources&memory_sources)/len(source_union) if source_union else 0.0
        text_score=text_similarity(query,item.get("input_text") or "")
        kind_factor=0.58 if item["kind"]=="idle" else 1.0
        score=(path_score*0.70+text_score*0.25+source_score*0.05)*kind_factor
        ranked.append({**item,"score":score,"path_score":path_score,"text_score":text_score})
    ranked.sort(key=lambda x:(x["score"],x["importance"],x["id"]),reverse=True)
    return [x for x in ranked[:limit] if x["score"]>0]

@app.route("/")
def index():
    return render_template_string(PAGE,query=last_query,result=last_result,matches=last_matches,timestamp=int(time.time()))

@app.post("/recall")
def recall():
    global last_query,last_result,last_matches
    text=" ".join(request.form.get("text","").strip().split())
    if text:
        result=brain.propagate(source_nodes=brain.text_to_sources(text,count=4),context_nodes=None,steps=8,threshold=0.22,noise=0.0,learn=False)
        last_query=text; last_result=result; last_matches=find_similar(text,result)
        build_html(brain,VIEW_FILE,result.traversed_edges,result.activated_nodes,f"集中想起：{text}")
    return index()

@app.route("/recall-view")
def recall_view():
    if not VIEW_FILE.exists(): return "まだ想起結果がありません。",404
    return send_file(VIEW_FILE)

if __name__=="__main__":
    webbrowser.open("http://127.0.0.1:5051")
    print("Sphere Brain 集中想起モード v2: http://127.0.0.1:5051")
    serve(app,host="127.0.0.1",port=5051,threads=4)
