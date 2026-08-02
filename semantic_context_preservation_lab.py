from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_context_preservation as observer

app = Flask(__name__)

TEMPLATE = r'''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Context Preservation Observer</title>
<style>
:root{--bg:#07111f;--panel:#12233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--cyan:#73dcff;--orange:#f08a4b;--green:#67e59a;--yellow:#ffd76a;--red:#ff8295}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1450px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}label{display:block;color:var(--cyan);margin:8px 0 5px}input{width:100%;padding:11px;border-radius:9px;border:1px solid #41658b;background:#071522;color:var(--text)}button{margin-top:16px;padding:12px 18px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}p{color:var(--muted)}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid var(--line);text-align:left}.score{font-size:22px;font-weight:800}.good{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}.pill{display:inline-block;padding:5px 9px;margin:3px;border:1px solid var(--line);border-radius:999px;color:var(--cyan)}.raw{background:#071522;border:1px solid var(--line);padding:12px;border-radius:10px;white-space:pre-wrap;word-break:break-word;font-family:Consolas,monospace;max-height:280px;overflow:auto}details{margin-top:12px}summary{cursor:pointer;color:var(--cyan);font-weight:700}@media(max-width:980px){.grid,.three{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>Context Preservation Observer</h1><p>主体の経路情報を関係段階へ渡す3方式を、学習なし・ノイズなしで比較します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post"><div class="grid"><div><label>主体A</label><input name="left_subject" value="{{left_subject}}" required></div><div><label>主体B</label><input name="right_subject" value="{{right_subject}}" required></div></div><div class="three"><div><label>関係</label><input name="relation" value="{{relation}}" required></div><div><label>主体伝播step</label><input type="number" name="subject_steps" min="1" max="20" value="{{subject_steps}}"></div><div><label>関係伝播step</label><input type="number" name="relation_steps" min="1" max="30" value="{{relation_steps}}"></div></div><button>3方式を比較する</button></form><p>既存Core・重み・経験DBは変更しません。learn=False / noise=0 相当の観測です。</p></section>
{% if result %}
<section class="card"><h2>{{result.left_subject}} ↔ {{result.right_subject}} ／ 関係「{{result.relation}}」</h2>
<table><tr><th>方式</th><th>渡す文脈の類似度</th><th>関係段階の最終類似度</th><th>最終Node</th><th>累積Node</th><th>累積Edge</th><th>固有Edge A/B</th></tr>
{% for item in result.comparisons %}<tr><td>{{item.label}}</td><td>{{'%.1f'|format(item.context_similarity*100)}}%</td><td><span class="score {{'bad' if item.final_similarity>0.99 else 'warn' if item.final_similarity>0.8 else 'good'}}">{{'%.1f'|format(item.final_similarity*100)}}%</span></td><td>{{'%.1f'|format(item.final_node_jaccard*100)}}%</td><td>{{'%.1f'|format(item.all_node_jaccard*100)}}%</td><td>{{'%.1f'|format(item.edge_jaccard*100)}}%</td><td>{{item.left_only_edges}} / {{item.right_only_edges}}</td></tr>{% endfor %}</table>
<p class="warn">最終類似度が低いほど、主体差が関係段階の最後まで残っています。ただし累積Edgeも合わせて判断します。</p></section>
{% for item in result.comparisons %}<section class="card"><h2>{{item.label}}</h2><div><span class="pill">文脈類似 {{'%.1f'|format(item.context_similarity*100)}}%</span><span class="pill">最終類似 {{'%.1f'|format(item.final_similarity*100)}}%</span><span class="pill">Edge類似 {{'%.1f'|format(item.edge_jaccard*100)}}%</span></div><div class="grid">
{% for side in [item.left,item.right] %}<div><h3>{{side.subject}}｜{{side.relation}}</h3><p>context {{side.context_nodes|length}} nodes ／ relation {{side.steps}} step ／ {{side.stop_reason}}</p><details><summary>渡した主体文脈</summary><div class="raw">{% for node,value in side.context_nodes %}{{node}} : {{'%.6f'|format(value)}}{% if not loop.last %}
{% endif %}{% endfor %}</div></details><details><summary>関係段階の最終活性</summary><div class="raw">{% for node,value in side.active_nodes %}{{node}} : {{'%.6f'|format(value)}}{% if not loop.last %}
{% endif %}{% endfor %}</div></details><details><summary>通過Edge</summary><div class="raw">{{side.traversed_edges}}</div></details></div>{% endfor %}
</div></section>{% endfor %}
{% endif %}{% if error %}<section class="card"><p class="bad">{{error}}</p></section>{% endif %}
</main></body></html>'''


@app.route('/', methods=['GET','POST'])
def index():
    left_subject = request.form.get('left_subject','犬')
    right_subject = request.form.get('right_subject','車')
    relation = request.form.get('relation','種類')
    subject_steps = int(request.form.get('subject_steps','8') or 8)
    relation_steps = int(request.form.get('relation_steps','10') or 10)
    result = None
    error = ''
    if request.method == 'POST':
        try:
            result = observer.compare_modes(left_subject,right_subject,relation,subject_steps=subject_steps,relation_steps=relation_steps)
        except Exception as exc:
            error = str(exc)
    return render_template_string(TEMPLATE,left_subject=left_subject,right_subject=right_subject,relation=relation,subject_steps=subject_steps,relation_steps=relation,result=result,error=error)


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5015,debug=False)
