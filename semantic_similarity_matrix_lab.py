from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_similarity_matrix as matrix_lab

app = Flask(__name__)

DEFAULT_ITEMS = """犬｜種類｜動物
猫｜種類｜動物
車｜種類｜人工物
電車｜種類｜人工物
犬｜動作｜走る
車｜動作｜走る
犬｜状態｜眠っている
犬｜性質｜大きい"""

TEMPLATE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Semantic Similarity Matrix</title>
<style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#ed985b;--green:#69e69d;--yellow:#ffd76a;--red:#ff8295}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1600px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}label{display:block;color:var(--cyan);margin:8px 0 5px}textarea,select{width:100%;padding:12px;border-radius:9px;border:1px solid #41658b;background:#071522;color:var(--text)}textarea{min-height:190px;font-family:Consolas,monospace}button{margin-top:16px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}.grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:18px}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}p{color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{padding:9px;border:1px solid var(--line);text-align:center;white-space:nowrap}th.sticky{position:sticky;left:0;background:#102139;z-index:2;text-align:left}.scroll{overflow:auto}.cell{font-weight:800}.legend{display:flex;gap:10px;flex-wrap:wrap}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--cyan)}.good{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}@media(max-width:950px){.grid,.two{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>Semantic Similarity Matrix</h1><p>複数の意味入力を、旧v2とv2.1 Contextualで一括比較します。</p></div></header>
<main class="wrap"><section class="card"><form method="post"><div class="grid"><div><label>入力（1行に 主体｜関係｜内容）</label><textarea name="items" required>{{items}}</textarea></div><div><label>観測段階</label><select name="stage"><option value="subject" {{'selected' if stage=='subject'}}>主体</option><option value="relation" {{'selected' if stage=='relation'}}>関係</option><option value="content" {{'selected' if stage=='content'}}>内容</option></select></div><div><label>類似度指標</label><select name="metric"><option value="activation" {{'selected' if metric=='activation'}}>最終活性</option><option value="nodes" {{'selected' if metric=='nodes'}}>累積Node</option><option value="edges" {{'selected' if metric=='edges'}}>累積Edge</option></select></div></div><button>行列を作成する</button></form><p>learn=False・noise=0。CoreとDBは変更しません。最大14件です。</p></section>
{% if result %}
<section class="card"><h2>分類別平均</h2><table><tr><th>共通要素</th><th>組数</th><th>旧v2</th><th>v2.1</th><th>差</th></tr>{% for row in result.summaries %}<tr><td>{{row.kind}}</td><td>{{row.count}}</td><td>{{'%.1f'|format(row.old*100)}}%</td><td class="cell">{{'%.1f'|format(row.new*100)}}%</td><td>{{'%+.1f'|format((row.new-row.old)*100)}}pt</td></tr>{% endfor %}</table><p>理想は、共有する構成要素が多い組ほど高く、共通要素なしが低くなることです。</p></section>
<div class="two">
{% for title,key in [('旧 Semantic Encoder v2','old_matrix'),('Semantic Encoder v2.1 Contextual','new_matrix')] %}<section class="card"><h2>{{title}}</h2><div class="scroll"><table><tr><th></th>{% for item in result.items %}<th title="{{item.label}}">{{loop.index}}</th>{% endfor %}</tr>{% for row in attribute(result,key) %}<tr><th class="sticky" title="{{result.items[loop.index0].label}}">{{loop.index}} {{result.items[loop.index0].label}}</th>{% for value in row %}<td class="cell" style="background:rgba(85,205,255,{{'%.3f'|format(value*0.55)}})">{{'%.0f'|format(value*100)}}</td>{% endfor %}</tr>{% endfor %}</table></div></section>{% endfor %}
</div>
<section class="card"><h2>v2.1で距離が近い組・遠い組</h2><table><tr><th>分類</th><th>入力A</th><th>入力B</th><th>旧v2</th><th>v2.1</th></tr>{% for row in result.pairs %}<tr><td>{{row.kind}}</td><td>{{row.left}}</td><td>{{row.right}}</td><td>{{'%.1f'|format(row.old*100)}}%</td><td class="cell">{{'%.1f'|format(row.new*100)}}%</td></tr>{% endfor %}</table></section>
{% endif %}{% if error %}<section class="card"><p class="bad">{{error}}</p></section>{% endif %}</main></body></html>'''

@app.route('/', methods=['GET','POST'])
def index():
    items = request.form.get('items', DEFAULT_ITEMS)
    stage = request.form.get('stage', 'content')
    metric = request.form.get('metric', 'activation')
    result = None
    error = ''
    if request.method == 'POST':
        try:
            result = matrix_lab.build_matrix(items, stage=stage, metric=metric)
        except Exception as exc:
            error = str(exc)
    return render_template_string(TEMPLATE, items=items, stage=stage, metric=metric, result=result, error=error)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5019, debug=False)
