from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_shared_context_formation as experiment

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain 共有文脈形成実験</title>
<style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#ed985b;--green:#69e69d;--red:#ff8295}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1550px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}input{width:100%;padding:12px;border-radius:9px;border:1px solid #41658b;background:#071522;color:var(--text)}button{margin-top:15px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}p{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{padding:10px;border:1px solid var(--line);text-align:center}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.pill{display:inline-block;padding:5px 9px;border:1px solid var(--line);border-radius:999px;color:var(--cyan);margin:3px}.good{color:var(--green)}.bad{color:var(--red)}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>Shared Context Formation Experiment</h1><p>共通する自然な経験が、異なる動作経路のあいだに橋を作るかを対照Coreと比較します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post"><label>チェックポイント（カンマ区切り）</label><input name="checkpoints" value="{{ checkpoints }}"><button>共有文脈形成実験を実行する</button></form><p>専用コピーCoreのみ更新します。保存済みCore・DBは変更しません。</p></section>
{% if result %}
<section class="card"><h2>判定：<span class="{{ 'good' if result['confirmed'] else 'bad' }}">{{ result['verdict'] }}</span></h2>
<div class="grid"><div><h3>実験Core</h3>{% for item in result['experimental_curriculum'] %}<span class="pill">{{ item }}</span>{% endfor %}</div><div><h3>対照Core</h3>{% for item in result['control_curriculum'] %}<span class="pill">{{ item }}</span>{% endfor %}</div></div></section>
<section class="card"><h2>推移</h2><table><tr><th>サイクル</th><th>実験Edge類似</th><th>対照Edge類似</th><th>共有文脈効果</th><th>実験共有Edge</th><th>対照共有Edge</th><th>共有Edge差</th><th>第三主体への転移効果</th></tr>
{% for row in result['records'] %}<tr><td>{{ row['cycle'] }}</td><td>{{ '%.1f'|format(row['experimental']['target']['edge_similarity']*100) }}%</td><td>{{ '%.1f'|format(row['control']['target']['edge_similarity']*100) }}%</td><td>{{ '%+.1f'|format(row['context_effect']*100) }}pt</td><td>{{ row['experimental']['target']['shared_edges'] }}</td><td>{{ row['control']['target']['shared_edges'] }}</td><td>{{ '%+d'|format(row['shared_edge_effect']) }}</td><td>{{ '%+.1f'|format(row['transfer_effect']*100) }}pt</td></tr>{% endfor %}
</table></section>
<div class="grid">
{% for row in result['records'] %}<section class="card"><h2>{{ row['cycle'] }}サイクル</h2><h3>目標ペア：鳥の羽ばたく ↔ 飛行機の飛行する</h3><table><tr><th>Core</th><th>活性</th><th>Node</th><th>Edge</th><th>共有Edge</th><th>固有Edge A/B</th></tr>
{% for label,data in [('実験',row['experimental']['target']),('対照',row['control']['target'])] %}<tr><td>{{ label }}</td><td>{{ '%.1f'|format(data['activation_similarity']*100) }}%</td><td>{{ '%.1f'|format(data['node_similarity']*100) }}%</td><td>{{ '%.1f'|format(data['edge_similarity']*100) }}%</td><td>{{ data['shared_edges'] }}</td><td>{{ data['left_only_edges'] }} / {{ data['right_only_edges'] }}</td></tr>{% endfor %}</table>
<h3>第三主体への転移</h3><table><tr><th>Core</th><th>蝶→飛行機</th><th>ドローン→鳥</th><th>平均Edge</th></tr>{% for label,data in [('実験',row['experimental']['transfer']),('対照',row['control']['transfer'])] %}<tr><td>{{ label }}</td><td>{{ '%.1f'|format(data['butterfly_to_plane']['edge_similarity']*100) }}%</td><td>{{ '%.1f'|format(data['drone_to_bird']['edge_similarity']*100) }}%</td><td>{{ '%.1f'|format(data['average_edge_similarity']*100) }}%</td></tr>{% endfor %}</table></section>{% endfor %}
</div>
{% endif %}{% if error %}<section class="card"><p class="bad">{{ error }}</p></section>{% endif %}
</main></body></html>'''

@app.route('/', methods=['GET','POST'])
def index():
    checkpoints = request.form.get('checkpoints','0,1,3,5,10')
    result = None
    error = ''
    if request.method == 'POST':
        try:
            values = [int(v.strip()) for v in checkpoints.split(',') if v.strip()]
            result = experiment.run_experiment(values)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template_string(TEMPLATE, checkpoints=checkpoints, result=result, error=error)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5023, debug=False)
