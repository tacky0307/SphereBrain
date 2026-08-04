from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_emergent_concept_v2 as experiment

app = Flask(__name__)

TEMPLATE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SphereBrain 創発概念実験 v2</title><style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#ed985b;--green:#69e69d;--red:#ff8295}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}label{display:block;color:var(--cyan);margin:8px 0 5px}input{width:100%;padding:12px;border-radius:9px;border:1px solid #41658b;background:#071522;color:var(--text)}button{margin-top:16px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:#fff;font-weight:800;cursor:pointer}p{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{padding:10px;border:1px solid var(--line);text-align:center}.good{color:var(--green);font-weight:800}.bad{color:var(--red);font-weight:800}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><header><div class="wrap"><h1>創発概念実験 v2</h1><p>異なる新刺激「ルーク」「ネラ」が、共通の空経験から同じ内部構造へ近づくかを検証します。</p></div></header><main class="wrap">
<section class="card"><form method="post"><label>チェックポイント（カンマ区切り）</label><input name="checkpoints" value="{{checkpoints}}"><button>実験を実行する</button></form><p>専用コピーCoreのみ更新。保存済みCoreとDBは変更しません。</p></section>
{% if result %}<section class="card"><h2>学習カリキュラム</h2>{% for item in result['curriculum'] %}<span class="pill">{{item}}</span>{% endfor %}<p>{{result['note']}}</p></section>
<section class="card"><h2>概念形成の推移</h2><table><tr><th>サイクル</th><th>目標Edge類似</th><th>共有Edge</th><th>共有Edge増加</th><th>入替Edge類似</th><th>既存動作対照</th><th>特異性余白</th><th>判定</th></tr>{% for row in result['rows'] %}<tr><td>{{row['cycle']}}</td><td>{{'%.1f'|format(row['target']['edge_similarity']*100)}}%</td><td>{{row['target']['common_edges']}}</td><td>{{'%+d'|format(row['shared_gain'])}}</td><td>{{'%.1f'|format(row['swapped']['edge_similarity']*100)}}%</td><td>{{'%.1f'|format(row['legacy']['edge_similarity']*100)}}%</td><td class="{{'good' if row['specificity_margin']>0 else 'bad'}}">{{'%+.1f'|format(row['specificity_margin']*100)}}pt</td><td>{% if row['specificity_margin']>0.03 and row['shared_gain']>0 %}<span class="good">創発候補</span>{% elif row['edge_gain']>0 %}接近中{% else %}未確認{% endif %}</td></tr>{% endfor %}</table></section>
<div class="grid">{% for row in result['rows'] %}<section class="card"><h2>{{row['cycle']}}サイクル</h2><p><span class="pill">目標 共通Edge {{row['target']['common_edges']}}</span><span class="pill">目標固有 {{row['target']['left_only_edges']}} / {{row['target']['right_only_edges']}}</span></p><table><tr><th>比較</th><th>活性</th><th>Node</th><th>Edge</th></tr><tr><td>目標</td><td>{{'%.1f'|format(row['target']['activation_similarity']*100)}}%</td><td>{{'%.1f'|format(row['target']['node_similarity']*100)}}%</td><td>{{'%.1f'|format(row['target']['edge_similarity']*100)}}%</td></tr><tr><td>入替</td><td>{{'%.1f'|format(row['swapped']['activation_similarity']*100)}}%</td><td>{{'%.1f'|format(row['swapped']['node_similarity']*100)}}%</td><td>{{'%.1f'|format(row['swapped']['edge_similarity']*100)}}%</td></tr><tr><td>既存動作</td><td>{{'%.1f'|format(row['legacy']['activation_similarity']*100)}}%</td><td>{{'%.1f'|format(row['legacy']['node_similarity']*100)}}%</td><td>{{'%.1f'|format(row['legacy']['edge_similarity']*100)}}%</td></tr></table></section>{% endfor %}</div>{% endif %}{% if error %}<section class="card"><p class="bad">{{error}}</p></section>{% endif %}</main></body></html>'''


def _parse_points(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(',') if part.strip()]


@app.route('/', methods=['GET','POST'])
def index():
    checkpoints = request.form.get('checkpoints', '0,1,3,5,10')
    result = None
    error = ''
    if request.method == 'POST':
        try:
            result = experiment.run_experiment(_parse_points(checkpoints))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template_string(TEMPLATE, checkpoints=checkpoints, result=result, error=error)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5022, debug=False)
