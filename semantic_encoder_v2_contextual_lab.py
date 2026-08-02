from __future__ import annotations

from flask import Flask, render_template_string, request

from semantic_encoder_v2 import StructuredInput, encode_and_experience, load_brain
from semantic_encoder_v2_contextual import (
    encode_and_experience_contextual,
    load_contextual_brain,
    train_contextual,
)

app = Flask(__name__)


def _jaccard(left, right) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _weighted(left, right) -> float:
    a = {int(i): float(v) for i, v in enumerate(left) if float(v) > 0}
    b = {int(i): float(v) for i, v in enumerate(right) if float(v) > 0}
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys) / sum(
        max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys
    )


def _stage(name, left, right) -> dict:
    return {
        "name": name,
        "final_similarity": _weighted(left.final_activation, right.final_activation),
        "node_similarity": _jaccard(left.activated_nodes, right.activated_nodes),
        "edge_similarity": _jaccard(left.traversed_edges, right.traversed_edges),
        "left_nodes": len(left.activated_nodes),
        "right_nodes": len(right.activated_nodes),
        "left_edges": len(left.traversed_edges),
        "right_edges": len(right.traversed_edges),
        "left_only_edges": len(set(left.traversed_edges) - set(right.traversed_edges)),
        "right_only_edges": len(set(right.traversed_edges) - set(left.traversed_edges)),
    }


def compare(left_item: StructuredInput, right_item: StructuredInput) -> dict:
    legacy_brain = load_brain()
    legacy_left = encode_and_experience(legacy_brain, left_item, learn=False)
    legacy_right = encode_and_experience(legacy_brain, right_item, learn=False)

    contextual_brain = load_contextual_brain()
    contextual_left = encode_and_experience_contextual(contextual_brain, left_item, learn=False)
    contextual_right = encode_and_experience_contextual(contextual_brain, right_item, learn=False)

    return {
        "legacy": [
            _stage("主体", legacy_left.subject_result, legacy_right.subject_result),
            _stage("関係", legacy_left.relation_result, legacy_right.relation_result),
            _stage("内容", legacy_left.content_result, legacy_right.content_result),
        ],
        "contextual": [
            _stage("主体", contextual_left.subject_result, contextual_right.subject_result),
            _stage("関係", contextual_left.relation_result, contextual_right.relation_result),
            _stage("内容", contextual_left.content_result, contextual_right.content_result),
        ],
    }


TEMPLATE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Semantic Encoder v2.1 Contextual</title><style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#ef8b4b;--green:#67e59a;--red:#ff8295}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}.wrap{max-width:1400px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}label{display:block;color:var(--cyan);margin:8px 0 5px}input{width:100%;padding:11px;border:1px solid #41658b;border-radius:9px;background:#071522;color:var(--text)}button{margin-top:15px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left}.score{font-size:21px;font-weight:800}.good{color:var(--green)}.bad{color:var(--red)}p{color:var(--muted)}@media(max-width:900px){.grid,.three{grid-template-columns:1fr}}</style></head><body><header><div class="wrap"><h1>Semantic Encoder v2.1 Contextual</h1><p>旧v2と、文脈持続＋共鳴を使うv2.1を主体・関係・内容の各段階で比較します。</p></div></header><main class="wrap">
<section class="card"><form method="post"><input type="hidden" name="action" value="compare"><div class="grid"><div><h3>入力A</h3><div class="three"><div><label>主体</label><input name="as" value="{{a.subject}}"></div><div><label>関係</label><input name="ar" value="{{a.relation}}"></div><div><label>内容</label><input name="ac" value="{{a.content}}"></div></div></div><div><h3>入力B</h3><div class="three"><div><label>主体</label><input name="bs" value="{{b.subject}}"></div><div><label>関係</label><input name="br" value="{{b.relation}}"></div><div><label>内容</label><input name="bc" value="{{b.content}}"></div></div></div></div><button>旧v2とv2.1を比較する</button></form><p>比較はlearn=False・noise=0。CoreとDBを変更しません。</p></section>
{% if result %}{% for title,rows in [('旧 Semantic Encoder v2',result.legacy),('Semantic Encoder v2.1 Contextual',result.contextual)] %}<section class="card"><h2>{{title}}</h2><table><tr><th>段階</th><th>最終活性類似</th><th>累積Node</th><th>累積Edge</th><th>固有Edge A/B</th><th>Node数 A/B</th></tr>{% for row in rows %}<tr><td>{{row.name}}</td><td><span class="score {{'bad' if row.final_similarity>0.99 else 'good'}}">{{'%.1f'|format(row.final_similarity*100)}}%</span></td><td>{{'%.1f'|format(row.node_similarity*100)}}%</td><td>{{'%.1f'|format(row.edge_similarity*100)}}%</td><td>{{row.left_only_edges}} / {{row.right_only_edges}}</td><td>{{row.left_nodes}} / {{row.right_nodes}}</td></tr>{% endfor %}</table></section>{% endfor %}{% endif %}
<section class="card"><h2>v2.1で経験を追加</h2><form method="post"><input type="hidden" name="action" value="train"><div class="three"><div><label>主体</label><input name="ts" value="犬"></div><div><label>関係</label><input name="tr" value="種類"></div><div><label>内容</label><input name="tc" value="動物"></div></div><label>反復回数</label><input type="number" name="repeats" min="1" max="100" value="1"><button>v2.1で学習する</button></form><p>このボタンは学習済みCoreとsemantic_experiencesへ正式に保存します。</p>{% if message %}<p class="good">{{message}}</p>{% endif %}</section>{% if error %}<section class="card"><p class="bad">{{error}}</p></section>{% endif %}</main></body></html>'''


@app.route('/', methods=['GET', 'POST'])
def index():
    a = StructuredInput(request.form.get('as', '犬'), request.form.get('ar', '種類'), request.form.get('ac', '動物'))
    b = StructuredInput(request.form.get('bs', '車'), request.form.get('br', '種類'), request.form.get('bc', '人工物'))
    result = None
    message = ''
    error = ''
    if request.method == 'POST':
        try:
            if request.form.get('action') == 'train':
                repeats = max(1, int(request.form.get('repeats', '1') or 1))
                trained = train_contextual(request.form.get('ts', ''), request.form.get('tr', ''), request.form.get('tc', ''), repeats)
                message = f'{trained.input.label} をv2.1で {repeats} 回学習・保存しました。'
            else:
                result = compare(a, b)
        except Exception as exc:
            error = str(exc)
    return render_template_string(TEMPLATE, a=a, b=b, result=result, message=message, error=error)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5018, debug=False)
