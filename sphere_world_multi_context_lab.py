from __future__ import annotations

from flask import Flask, render_template_string, request

from sphere_world_generalization import evaluate_all_states
from sphere_world_multi_context import evaluate_multi_context

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereWorld 0.2 Multi-Object Context</title>
<style>
:root{--bg:#07111f;--panel:#15243c;--line:#385a80;--text:#eef5ff;--muted:#a9bdd4;--cyan:#72dcff;--orange:#eda05f;--green:#78e7a6;--red:#ff8295;--yellow:#ffd76a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1650px;margin:auto;padding:28px}header{border-bottom:1px solid var(--line);background:#0b192b}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:24px;margin-top:20px}input,button{padding:12px 16px;border-radius:9px;border:1px solid var(--line);font-size:16px}input{background:#081625;color:var(--text);width:120px}button{background:var(--orange);color:white;font-weight:800;cursor:pointer}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.metric{padding:18px;border:1px solid var(--line);border-radius:12px}.value{font-size:34px;font-weight:900}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{border:1px solid var(--line);padding:10px;text-align:center;vertical-align:top;white-space:nowrap}th{background:#132039}.ok{color:var(--green);font-weight:900}.ng{color:var(--red);font-weight:900}.facts{text-align:left;white-space:normal;color:var(--muted);min-width:270px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:6px 11px;margin:3px;color:var(--cyan)}details{text-align:left}.muted{color:var(--muted)}@media(max-width:1050px){.two,.summary{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereWorld 0.2 — Multi-Object Context Encoder</h1><p class="muted">複合ラベル方式と、複数オブジェクト・相対関係を世界文脈へ統合する方式を比較します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post"><label>反復回数 <input type="number" name="repeats" min="1" max="30" value="{{ repeats }}"></label> <button>0.1と0.2を比較する</button></form><p class="muted">専用コピーCoreのみ更新します。保存済みCoreとDBは変更しません。</p></section>
{% if error %}<section class="card"><p class="ng">{{ error }}</p></section>{% endif %}
{% if old and new %}
<section class="card"><h2>未経験6状態への一般化</h2><div class="two">
<div><h3>SphereWorld 0.1 — 複合ラベル</h3><div class="summary"><div class="metric">学習済み<div class="value">{{ '%.1f'|format(old['seen_accuracy']*100) }}%</div></div><div class="metric">未経験<div class="value">{{ '%.1f'|format(old['unseen_accuracy']*100) }}%</div></div><div class="metric">全体<div class="value">{{ '%.1f'|format(old['overall_accuracy']*100) }}%</div></div></div></div>
<div><h3>SphereWorld 0.2 — 世界文脈</h3><div class="summary"><div class="metric">学習済み<div class="value">{{ '%.1f'|format(new['seen_accuracy']*100) }}%</div></div><div class="metric">未経験<div class="value">{{ '%.1f'|format(new['unseen_accuracy']*100) }}%</div></div><div class="metric">全体<div class="value">{{ '%.1f'|format(new['overall_accuracy']*100) }}%</div></div></div></div>
</div></section>
<section class="card"><h2>0.2で教えた3状態</h2>{% for row in new['train_states'] %}<div class="pill">P {{row['player']}} / E {{row['enemy']}} → {{row['action']}}</div><div class="facts">{% for fact in row['facts'] %}{{ fact }}{% if not loop.last %}<br>{% endif %}{% endfor %}</div>{% endfor %}</section>
<section class="card"><h2>9状態の比較</h2><div class="scroll"><table><tr><th>P</th><th>E</th><th>経験</th><th>期待</th><th>0.1選択</th><th>0.1</th><th>0.2選択</th><th>0.2</th><th>0.2の世界事実</th><th>0.2候補</th></tr>
{% for row in comparison %}<tr><td>{{row['player']}}</td><td>{{row['enemy']}}</td><td>{{'学習済み' if row['trained'] else '未経験'}}</td><td>{{row['expected']}}</td><td>{{row['old_selected']}}</td><td class="{{'ok' if row['old_correct'] else 'ng'}}">{{'正解' if row['old_correct'] else '不正解'}}</td><td>{{row['new_selected']}}</td><td class="{{'ok' if row['new_correct'] else 'ng'}}">{{'正解' if row['new_correct'] else '不正解'}}</td><td class="facts">{% for fact in row['facts'] %}{{fact}}{% if not loop.last %}<br>{% endif %}{% endfor %}<br><span class="muted">context nodes {{row['world_context_nodes']}}</span></td><td><details><summary>表示（1位差 {{'%.1f'|format(row['margin']*100)}}pt）</summary>{% for candidate in row['candidates'] %}<div>{{candidate['action']}}：{{'%.1f'|format(candidate['score']*100)}}%（Node {{'%.1f'|format(candidate['node_score']*100)}} / Edge {{'%.1f'|format(candidate['edge_score']*100)}}）</div>{% endfor %}</details></td></tr>{% endfor %}
</table></div></section>
{% endif %}
</main></body></html>'''


@app.route('/', methods=['GET', 'POST'])
def index():
    repeats = request.form.get('repeats', '12')
    old = new = None
    comparison = []
    error = ''
    if request.method == 'POST':
        try:
            count = max(1, min(30, int(repeats)))
            repeats = str(count)
            old = evaluate_all_states(count)
            new = evaluate_multi_context(count)
            old_map = {(r['player'], r['enemy']): r for r in old['rows']}
            for row in new['rows']:
                previous = old_map[(row['player'], row['enemy'])]
                candidates = row['candidates']
                margin = candidates[0]['score'] - candidates[1]['score'] if len(candidates) > 1 else 0.0
                comparison.append({
                    **row,
                    'old_selected': previous['selected'],
                    'old_correct': previous['correct'],
                    'new_selected': row['selected'],
                    'new_correct': row['correct'],
                    'margin': margin,
                })
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    return render_template_string(TEMPLATE, repeats=repeats, old=old, new=new, comparison=comparison, error=error)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5027, debug=False)
