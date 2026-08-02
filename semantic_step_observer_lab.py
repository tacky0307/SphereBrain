from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_step_observer as observer

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Step-by-Step Observer</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#294d76;--text:#eef5ff;--muted:#9bb1ca;--cyan:#67dcff;--orange:#f28b4b;--green:#67e59a;--yellow:#ffd76a;--red:#ff7e8a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line);background:#0b1a2c}h1,h2,h3{margin-top:0}p{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-top:18px}label{display:block;color:var(--cyan);margin:10px 0 5px}input{width:100%;padding:11px;border-radius:9px;border:1px solid #355d88;background:#071522;color:var(--text)}button{margin-top:15px;padding:11px 17px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.score{font-size:19px;font-weight:800}.raw{font-family:ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;background:#071522;border:1px solid var(--line);padding:12px;border-radius:10px;max-height:280px;overflow:auto}.ok{color:var(--green)}.warn{color:var(--yellow)}.bad{color:var(--red)}details{margin-top:10px}summary{cursor:pointer;color:var(--cyan)}@media(max-width:900px){.grid{grid-template-columns:1fr}.wide{overflow-x:auto}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Step-by-Step Observer</h1><p>主体刺激が、どのstepで同一化するかを入口から追跡します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post">
<div class="grid">
<div><label>主体A</label><input name="left_subject" value="{{left_subject}}" required></div>
<div><label>主体B</label><input name="right_subject" value="{{right_subject}}" required></div>
</div>
<div class="grid">
<div><label>主体伝播ステップ</label><input type="number" name="steps" min="0" max="50" value="{{steps}}"></div>
<div><label>固定閾値</label><input type="number" name="threshold" min="0.18" max="1" step="0.01" value="{{threshold}}"></div>
</div>
<button>stepごとに比較する</button></form>
<p>観測専用。learn=False・noise=0。保存済みCoreと経験DBは変更しません。</p></section>
{% if result %}
<section class="card"><h2>入口ノード</h2>
<div class="grid"><div><h3>{{result.left.subject}}</h3><div class="raw">role: {{result.left.role_nodes}}\nentity: {{result.left.entity_nodes}}\nall: {{result.left.source_nodes}}</div></div><div><h3>{{result.right.subject}}</h3><div class="raw">role: {{result.right.role_nodes}}\nentity: {{result.right.entity_nodes}}\nall: {{result.right.source_nodes}}</div></div></div>
<p>共通 {{result.source_overlap.common}} ／ A固有 {{result.source_overlap.left_only}} ／ B固有 {{result.source_overlap.right_only}}</p>
{% if result.first_identical_step is none %}<p class="ok"><strong>観測範囲では活性状態は完全一致していません。</strong></p>{% elif result.first_identical_step == 0 %}<p class="bad"><strong>step 0ですでに完全一致しています。</strong> Encoderの入口ノード生成を確認してください。</p>{% else %}<p class="warn"><strong>最初の完全一致：step {{result.first_identical_step}}</strong></p>{% endif %}
</section>
<section class="card wide"><h2>Step比較</h2><table><tr><th>step</th><th>活性値込み</th><th>現在Node</th><th>累積Node</th><th>累積Edge</th><th>共通/A固有/B固有</th><th>判定</th></tr>
{% for item in result.comparisons %}<tr><td>{{item.step}}</td>{% if item.comparable %}<td><span class="score">{{'%.1f'|format(item.activation_similarity*100)}}%</span></td><td>{{'%.1f'|format(item.current_node_jaccard*100)}}%</td><td>{{'%.1f'|format(item.cumulative_node_jaccard*100)}}%</td><td>{{'%.1f'|format(item.cumulative_edge_jaccard*100)}}%</td><td>{{item.common_current_nodes}} / {{item.left_only_current_nodes}} / {{item.right_only_current_nodes}}</td><td>{% if item.activation_similarity >= 0.999999 %}<span class="bad">完全一致</span>{% elif item.activation_similarity >= 0.9 %}<span class="warn">強く接近</span>{% else %}<span class="ok">差あり</span>{% endif %}</td>{% else %}<td colspan="6">片方の伝播が終了済み</td>{% endif %}</tr>{% endfor %}</table></section>
<div class="grid">
{% for side in [result.left, result.right] %}<section class="card"><h2>{{side.subject}}</h2><p>実行 {{side.executed_steps}} step ／ 停止理由 {{side.stop_reason}}</p>
{% for state in side.states %}<details {% if state.step <= 2 %}open{% endif %}><summary>step {{state.step}} — active {{state.active_count}} / new {{state.new_nodes|length}} / edge {{state.step_edges|length}}</summary><div class="raw">active:
{% for node,value in state.active_nodes %}{{node}} : {{'%.6f'|format(value)}}{% if not loop.last %}
{% endif %}{% endfor %}

new nodes: {{state.new_nodes}}
step edges: {{state.step_edges}}
cumulative nodes: {{state.cumulative_nodes|length}}
cumulative edges: {{state.cumulative_edges|length}}</div></details>{% endfor %}
</section>{% endfor %}</div>
<section class="card"><h2>入力固有の累積Edge</h2>{% for item in result.comparisons %}{% if item.comparable %}<details><summary>step {{item.step}} — A固有 {{item.left_only_edges|length}} / B固有 {{item.right_only_edges|length}}</summary><div class="grid"><div class="raw">A固有: {{item.left_only_edges}}</div><div class="raw">B固有: {{item.right_only_edges}}</div></div></details>{% endif %}{% endfor %}</section>
{% endif %}
{% if error %}<section class="card"><p class="bad">{{error}}</p></section>{% endif %}
</main></body></html>'''


@app.route("/", methods=["GET", "POST"])
def index():
    left_subject = request.form.get("left_subject", "犬")
    right_subject = request.form.get("right_subject", "車")
    steps = int(request.form.get("steps", "8") or 8)
    threshold = float(request.form.get("threshold", "0.18") or 0.18)
    result = None
    error = ""
    if request.method == "POST":
        try:
            result = observer.compare_subjects(
                left_subject,
                right_subject,
                steps=steps,
                threshold=threshold,
            )
        except Exception as exc:
            error = str(exc)
    return render_template_string(
        TEMPLATE,
        left_subject=left_subject,
        right_subject=right_subject,
        steps=steps,
        threshold=threshold,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5014, debug=False)
