from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_fly_concept_formation as experiment

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain 飛ぶ概念形成実験</title>
<style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#ed985b;--green:#69e69d;--yellow:#ffd76a;--red:#ff8295}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1550px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:2fr 1fr;gap:18px}label{display:block;color:var(--cyan);margin:8px 0 5px}input{width:100%;padding:11px;border-radius:9px;border:1px solid #41658b;background:#071522;color:var(--text)}button{margin-top:16px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}p{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{padding:10px;border:1px solid var(--line);text-align:center}th:first-child,td:first-child{text-align:left}.score{font-size:20px;font-weight:800}.good{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.bar{height:10px;border-radius:6px;background:#071522;overflow:hidden}.bar>span{display:block;height:100%;background:var(--cyan)}@media(max-width:950px){.grid,.two{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>飛ぶ概念形成実験</h1><p>未経験概念「飛ぶ」が、反復経験によって共通経路を獲得し、既存動作アトラクターから離れるかを追跡します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post"><div class="grid"><div><label>チェックポイント（カンマ区切り）</label><input name="checkpoints" value="{{ checkpoints }}"></div><div><label><input type="checkbox" name="support" value="1" {% if support %}checked{% endif %} style="width:auto"> 「場所｜空」の補助経験を含める</label></div></div><button>概念形成実験を実行する</button></form><p>専用コピーCoreのみを更新します。保存済みCore・DBは変更しません。最大30サイクルです。</p></section>
{% if result %}
<section class="card"><h2>学習カリキュラム</h2>{% for item in result['curriculum'] %}<span class="pill">{{ item }}</span>{% endfor %}</section>
<section class="card"><h2>概念形成の推移</h2><table><tr><th>サイクル</th><th>飛行機↔鳥 最終活性</th><th>Node類似</th><th>Edge類似</th><th>共有Edge</th><th>共有Edge増加</th><th>旧動作への平均吸着</th><th>概念余白</th></tr>
{% for snap in result['snapshots'] %}<tr><td><b>{{ snap['cycle'] }}</b></td><td>{{ '%.1f'|format(snap['pair']['activation_similarity']*100) }}%</td><td>{{ '%.1f'|format(snap['pair']['node_similarity']*100) }}%</td><td class="score">{{ '%.1f'|format(snap['pair']['edge_similarity']*100) }}%</td><td>{{ snap['pair']['common_edges'] }}</td><td class="{{ 'good' if snap['delta_common_edges']>0 else '' }}">{{ '%+d'|format(snap['delta_common_edges']) }}</td><td>{{ '%.1f'|format(snap['old_action_average']*100) }}%</td><td class="score {{ 'good' if snap['concept_margin']>0 else 'bad' }}">{{ '%+.1f'|format(snap['concept_margin']*100) }}pt</td></tr>{% endfor %}</table>
<p>概念余白 = 「飛行機の飛ぶ」と「鳥の飛ぶ」のEdge類似 − 既存の走る・止まる・歩くへの平均Edge類似。正になるほど、独立した「飛ぶ」経路が育っています。</p></section>
<div class="two">
{% for snap in result['snapshots'] %}<section class="card"><h2>{{ snap['cycle'] }}サイクル</h2><div><span class="pill">共有Edge {{ snap['pair']['common_edges'] }}</span><span class="pill">飛行機固有 {{ snap['pair']['left_only_edges'] }}</span><span class="pill">鳥固有 {{ snap['pair']['right_only_edges'] }}</span></div><h3>既存動作への吸着</h3><table><tr><th>既存経験</th><th>平均活性</th><th>平均Edge</th></tr>{% for row in snap['attractions'] %}<tr><td>{{ row['label'] }}</td><td>{{ '%.1f'|format(row['average_activation']*100) }}%</td><td>{{ '%.1f'|format(row['average_edges']*100) }}%</td></tr>{% endfor %}</table></section>{% endfor %}
</div>
{% endif %}
{% if error %}<section class="card"><p class="bad">{{ error }}</p></section>{% endif %}
</main></body></html>'''


def _parse_checkpoints(text: str) -> list[int]:
    values = []
    for token in text.replace('、', ',').split(','):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


@app.route('/', methods=['GET', 'POST'])
def index():
    checkpoints = request.form.get('checkpoints', '0,1,3,5,10')
    support = request.form.get('support', '1') == '1'
    result = None
    error = ''
    if request.method == 'POST':
        try:
            result = experiment.run_experiment(_parse_checkpoints(checkpoints), include_support=support)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template_string(TEMPLATE, checkpoints=checkpoints, support=support, result=result, error=error)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5021, debug=False)
