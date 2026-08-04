from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_stage_observer as observer

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Stage Observer</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#294d76;--text:#eef5ff;--muted:#9bb1ca;--cyan:#67dcff;--orange:#f28b4b;--green:#67e59a;--yellow:#ffd76a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1400px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line);background:#0b1a2c}h1,h2,h3{margin-top:0}p{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-top:18px}label{display:block;color:var(--cyan);margin:10px 0 5px}input{width:100%;padding:11px;border-radius:9px;border:1px solid #355d88;background:#071522;color:var(--text)}button{margin-top:15px;padding:11px 17px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left}.score{font-size:22px;font-weight:800}.raw{font-family:ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;background:#071522;border:1px solid var(--line);padding:12px;border-radius:10px}.ok{color:var(--green)}.warn{color:var(--yellow)}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Stage Observer</h1><p>二つの入力を、A 主体直後・B 引継ぎ文脈・C 関係直後・D 自由伝播後で比較します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post">
<div class="grid">
<div><h3>入力A</h3><label>主体</label><input name="left_subject" value="{{left_subject}}" required><label>関係</label><input name="left_relation" value="{{left_relation}}" required></div>
<div><h3>入力B</h3><label>主体</label><input name="right_subject" value="{{right_subject}}" required><label>関係</label><input name="right_relation" value="{{right_relation}}" required></div>
</div>
<div class="grid"><div><label>自由伝播ステップ</label><input type="number" name="steps" min="1" max="100" value="{{steps}}"></div><div><label>適応率</label><input type="number" name="ratio" min="0.05" max="0.95" step="0.01" value="{{ratio}}"></div></div>
<button>段階ごとに比較する</button></form>
<p>観測のみ。learn=False・noise=0。Core、重み、Semantic v2経験DBは変更しません。</p></section>
{% if result %}
<section class="card"><h2>段階別の入力間類似度</h2>
<span class="pill">A {{result.left.subject}}｜{{result.left.relation}}</span><span class="pill">B {{result.right.subject}}｜{{result.right.relation}}</span>
<table><tr><th>段階</th><th>活性値込み類似度</th><th>共通Topノード</th><th>A固有</th><th>B固有</th></tr>
{% for item in result.comparisons %}<tr><td>{{item.name}}</td><td><span class="score">{{'%.1f'|format(item.similarity*100)}}%</span></td><td>{{item.common_count}}</td><td>{{item.left_only_count}}</td><td>{{item.right_only_count}}</td></tr>{% endfor %}</table>
<p class="warn">類似度が急に上がる段階が、入力差が失われる候補地点です。</p></section>
<div class="grid">
{% for side in [result.left, result.right] %}<section class="card"><h2>{{'入力A' if loop.index==1 else '入力B'}}：{{side.subject}}｜{{side.relation}}</h2>
{% for stage in side.stages %}<h3>{{stage.name}}</h3><div><span class="pill">active {{stage.active_count}}</span><span class="pill">peak {{stage.peak_node}}</span><span class="pill">value {{'%.6f'|format(stage.peak_value)}}</span></div>
<div class="raw">{% for node,value in stage.top_nodes %}{{node}} : {{'%.6f'|format(value)}}{% if not loop.last %}
{% endif %}{% endfor %}</div>{% endfor %}
</section>{% endfor %}</div>
{% endif %}
{% if error %}<section class="card"><p>{{error}}</p></section>{% endif %}
</main></body></html>'''


@app.route("/", methods=["GET", "POST"])
def index():
    left_subject = request.form.get("left_subject", "犬")
    left_relation = request.form.get("left_relation", "種類")
    right_subject = request.form.get("right_subject", "車")
    right_relation = request.form.get("right_relation", "種類")
    steps = int(request.form.get("steps", "24") or 24)
    ratio = float(request.form.get("ratio", "0.35") or 0.35)
    result = None
    error = ""
    if request.method == "POST":
        try:
            result = observer.compare_inputs(
                left_subject,
                left_relation,
                right_subject,
                right_relation,
                output_steps=steps,
                adaptive_ratio=ratio,
            )
        except Exception as exc:
            error = str(exc)
    return render_template_string(
        TEMPLATE,
        left_subject=left_subject,
        left_relation=left_relation,
        right_subject=right_subject,
        right_relation=right_relation,
        steps=steps,
        ratio=ratio,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5013, debug=False)
