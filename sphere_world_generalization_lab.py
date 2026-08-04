from __future__ import annotations

from flask import Flask, render_template_string, request

from sphere_world_generalization import evaluate_all_states

app = Flask(__name__)

TEMPLATE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SphereWorld 一般化実験</title><style>
:root{--bg:#07111f;--panel:#14243d;--line:#355777;--text:#eef5ff;--muted:#a9bdd4;--green:#76e3a0;--red:#ff8999;--orange:#ed985b;--cyan:#75dcff;--yellow:#ffd76a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line);background:#0c192b}h1,h2{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}button,input{padding:11px 14px;border-radius:9px;border:1px solid var(--line)}button{background:var(--orange);color:white;font-weight:800;cursor:pointer}input{background:#091624;color:white;width:120px}table{width:100%;border-collapse:collapse}th,td{border:1px solid var(--line);padding:10px;text-align:center}.ok{color:var(--green);font-weight:800}.ng{color:var(--red);font-weight:800}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 10px;margin:3px;color:var(--cyan)}details{text-align:left}.muted{color:var(--muted)}.big{font-size:1.7rem;font-weight:900}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}@media(max-width:900px){.grid{grid-template-columns:1fr}.scroll{overflow:auto}}
</style></head><body><header><div class="wrap"><h1>SphereWorld 未経験配置への一般化実験</h1><p class="muted">3状態だけ経験させ、残り6状態を経験経路から判断できるか確認します。</p></div></header><main class="wrap">
<section class="card"><form method="post"><label>反復回数 <input type="number" name="repeats" min="1" max="30" value="{{repeats}}"></label> <button>一般化実験を実行する</button></form><p class="muted">専用コピーCoreのみ更新します。保存済みCoreとDBは変更しません。</p></section>
{% if result %}<section class="card"><h2>教えた3状態</h2>{% for row in result['train_states'] %}<span class="pill">P {{row['player']}} / E {{row['enemy']}} → {{row['action']}}</span>{% endfor %}</section>
<section class="card"><div class="grid"><div><div class="muted">学習済み3状態</div><div class="big">{{'%.1f'|format(result['seen_accuracy']*100)}}%</div></div><div><div class="muted">未経験6状態</div><div class="big">{{'%.1f'|format(result['unseen_accuracy']*100)}}%</div></div><div><div class="muted">全9状態</div><div class="big">{{'%.1f'|format(result['overall_accuracy']*100)}}%</div></div></div></section>
<section class="card"><h2>9状態の判定</h2><div class="scroll"><table><tr><th>Player</th><th>Enemy</th><th>経験</th><th>期待</th><th>選択</th><th>判定</th><th>1位差</th><th>候補</th></tr>{% for row in result['rows'] %}<tr><td>{{row['player']}}</td><td>{{row['enemy']}}</td><td>{{'学習済み' if row['trained'] else '未経験'}}</td><td>{{row['expected']}}</td><td>{{row['selected']}}</td><td class="{{'ok' if row['correct'] else 'ng'}}">{{'正解' if row['correct'] else '不正解'}}</td><td>{{'%.1f'|format(row['margin']*100)}}pt</td><td><details><summary>表示</summary>{% for c in row['candidates'] %}<div>{{c['action']}}：{{'%.1f'|format(c['score']*100)}}%（Node {{'%.1f'|format(c['node_score']*100)}} / Edge {{'%.1f'|format(c['edge_score']*100)}}）</div>{% endfor %}</details></td></tr>{% endfor %}</table></div></section>{% endif %}{% if error %}<section class="card ng">{{error}}</section>{% endif %}</main></body></html>'''

@app.route('/', methods=['GET','POST'])
def index():
    repeats = int(request.form.get('repeats', 12))
    result = None
    error = ''
    if request.method == 'POST':
        try:
            result = evaluate_all_states(repeats)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template_string(TEMPLATE, repeats=repeats, result=result, error=error)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5026, debug=False)
