from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_novel_integration as observer

app = Flask(__name__)

STAGE_LABELS = {
    "subject": "主体",
    "relation": "関係",
    "content": "内容",
    "all": "全体",
}

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Novel Integration Observer</title>
<style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#ed985b;--green:#69e69d;--yellow:#ffd76a;--red:#ff8295}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}label{display:block;color:var(--cyan);margin:8px 0 5px}input{width:100%;padding:12px;border-radius:9px;border:1px solid #41658b;background:#071522;color:var(--text)}button{margin-top:16px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}p{color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{padding:9px;border:1px solid var(--line);text-align:center}th{text-align:center}.left{text-align:left}.score{font-size:20px;font-weight:800}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}.good{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.scroll{overflow:auto}@media(max-width:950px){.grid,.two{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>Novel Integration Observer</h1><p>未経験入力を学習させずに流し、保存済み経路のどこへ接続されるか観測します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post"><div class="grid"><div><label>主体</label><input name="subject" value="{{ subject }}" required></div><div><label>関係</label><input name="relation" value="{{ relation }}" required></div><div><label>内容</label><input name="content" value="{{ content }}" required></div></div><button>未経験入力を観測する</button></form><p>learn=False・noise=0。Core、重み、経験DBは変更しません。</p></section>
{% if error %}<section class="card"><p class="bad">{{ error }}</p></section>{% endif %}
{% if result %}
<section class="card"><h2>{{ result['probe']['label'] }}</h2>
<span class="pill">保存済み経験 {{ result['stored_count'] }}種類</span>
<span class="pill">最初の接続 {{ stage_labels.get(result['first_connection'], '未検出') }}</span>
{% if result['exact_exists'] %}<p class="warn">この三つ組は既にDBへ保存されています。完全な未経験入力として検証する場合は、別の三つ組を入力してください。</p>{% else %}<p class="good">この三つ組と完全一致する保存経験はありません。</p>{% endif %}
<table><tr><th>段階</th><th>Probe Node</th><th>Probe Edge</th><th>最も近い保存経験</th><th>接続スコア</th><th>共通Edge</th><th>Probe固有Edge</th><th>判定</th></tr>
{% for row in result['stage_connections'] %}<tr><td>{{ stage_labels[row['stage']] }}</td><td>{{ result['probe_sizes'][row['stage']]['nodes'] }}</td><td>{{ result['probe_sizes'][row['stage']]['edges'] }}</td><td class="left">{{ row['best']['label'] }}</td><td class="score">{{ '%.1f'|format(row['best']['score']*100) }}%</td><td>{{ row['best']['common_edges'] }}</td><td>{{ row['best']['probe_only_edges'] }}</td><td class="{{ 'good' if row['connected'] else 'warn' }}">{{ '接続' if row['connected'] else '弱い' }}</td></tr>{% endfor %}</table></section>

{% for stage in ['subject','relation','content','all'] %}<section class="card"><h2>{{ stage_labels[stage] }}段階の接続候補</h2><div class="scroll"><table><tr><th>順位</th><th>保存経験</th><th>経験回数</th><th>総合</th><th>Node類似</th><th>Edge類似</th><th>Probe経路の既存Edge包含</th><th>共通Edge</th><th>Probe固有Edge</th></tr>
{% for row in result['rankings'][stage] %}<tr><td>{{ loop.index }}</td><td class="left">{{ row['label'] }}</td><td>{{ row['repetitions'] }}</td><td class="score">{{ '%.1f'|format(row['score']*100) }}%</td><td>{{ '%.1f'|format(row['node_jaccard']*100) }}%</td><td>{{ '%.1f'|format(row['edge_jaccard']*100) }}%</td><td>{{ '%.1f'|format(row['edge_containment']*100) }}%</td><td>{{ row['common_edges'] }}</td><td>{{ row['probe_only_edges'] }}</td></tr>{% endfor %}
</table></div></section>{% endfor %}
{% endif %}
</main></body></html>'''


@app.route('/', methods=['GET', 'POST'])
def index():
    subject = request.form.get('subject', 'バス')
    relation = request.form.get('relation', '種類')
    content = request.form.get('content', '人工物')
    result = None
    error = ''
    if request.method == 'POST':
        try:
            result = observer.observe_novel(subject, relation, content)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template_string(
        TEMPLATE,
        subject=subject,
        relation=relation,
        content=content,
        result=result,
        error=error,
        stage_labels=STAGE_LABELS,
    )


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5020, debug=False)
