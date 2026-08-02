from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_raw_output as raw_output

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Raw Output Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#294d76;--text:#eef5ff;--muted:#9bb1ca;--cyan:#67dcff;--orange:#f28b4b;--green:#67e59a;--yellow:#ffd76a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1300px;margin:auto;padding:24px}header{border-bottom:1px solid var(--line);background:#0b1a2c}h1,h2{margin-top:0}p{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px;margin-top:18px}label{display:block;color:var(--cyan);margin:10px 0 5px}input,select{width:100%;padding:11px;border-radius:9px;border:1px solid #355d88;background:#071522;color:var(--text)}button{margin-top:15px;padding:11px 17px;border:0;border-radius:9px;background:var(--orange);color:white;font-weight:800;cursor:pointer}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left}.score{font-size:20px;font-weight:800}.raw{font-family:ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;background:#071522;border:1px solid var(--line);padding:12px;border-radius:10px}.ok{color:var(--green)}.warn{color:var(--yellow)}.note{border-left:3px solid var(--cyan);padding-left:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Raw Output Lab</h1><p>Decoderより前に、主体＋関係の後でCoreが残す数値状態をそのまま観測します。</p></div></header>
<main class="wrap">
<section class="card"><form method="post">
<div class="grid"><div><label>主体</label><input name="subject" value="{{subject}}" required></div><div><label>関係</label><input name="relation" value="{{relation}}" required></div></div>
<div class="grid"><div><label>反復回数</label><input type="number" name="repeats" min="1" max="30" value="{{repeats}}"></div><div><label>自由伝播ステップ</label><input type="number" name="steps" min="1" max="100" value="{{steps}}"></div></div>
<div class="grid"><div><label>観測モード</label><select name="mode"><option value="adaptive" {% if mode=='adaptive' %}selected{% endif %}>適応閾値（推奨）</option><option value="reignite" {% if mode=='reignite' %}selected{% endif %}>再点火</option><option value="fixed" {% if mode=='fixed' %}selected{% endif %}>固定閾値（比較用）</option></select></div><div><label>適応率</label><input type="number" name="adaptive_ratio" min="0.05" max="0.95" step="0.05" value="{{adaptive_ratio}}"></div></div>
<button>Raw Outputを観測する</button></form>
<p class="note"><b>適応閾値：</b>残留活性の形は変えず、現在の強さに合わせて観測閾値だけを調整します。<br><b>再点火：</b>残留活性の比率を保ったまま最大値を1.0へ正規化します。<br><b>固定閾値：</b>前回と同じ条件で、停止比較に使います。</p>
<p>観測は learn=False・noise=0。保存済み候補はCoreへ渡さず、伝播後にDecoder参考候補として比較します。</p></section>
{% if result %}
<section class="card"><h2>反復結果</h2>
<span class="pill">入力 {{result.subject}}｜{{result.relation}}</span>
<span class="pill">モード {{result.mode}}</span>
<span class="pill">{{result.repeat_count}}回</span>
<span class="pill">出力類似度 {{'%.1f'|format(result.mean_similarity*100)}}%</span>
<table><tr><th>最終ノード</th><th>出現回数</th></tr>{% for item in result.final_node_counts %}<tr><td>{{item.node}}</td><td>{{item.count}}</td></tr>{% endfor %}</table></section>
<div class="grid">
{% for run in result.runs %}<section class="card"><h2>Run {{loop.index}}</h2>
{% set raw=run.raw_output %}
<div><span class="pill">final {{raw.final_node}}</span><span class="pill">value {{'%.5f'|format(raw.final_value)}}</span><span class="pill">step {{raw.convergence_step}}</span><span class="pill">edges {{raw.traversed_edges|length}}</span><span class="pill">{{raw.stopped_reason}}</span></div>
<p><span class="pill">開始ピーク {{'%.6f'|format(raw.initial_peak)}}</span><span class="pill">実効閾値 {{'%.6f'|format(raw.effective_threshold)}}</span></p>
{% if raw.convergence_step > 0 %}<p class="ok">自由伝播が進みました。</p>{% else %}<p class="warn">自由伝播はstep 0で停止しました。</p>{% endif %}
<h3>Raw numeric output</h3><div class="raw">{% for node,value in raw.active_nodes %}{{node}} : {{'%.6f'|format(value)}}{% if not loop.last %}
{% endif %}{% endfor %}</div>
<h3>Decoder参考候補</h3>{% if run.decoder_candidates %}<table><tr><th>内容</th><th>類似度</th><th>経験数</th></tr>{% for item in run.decoder_candidates %}<tr><td>{{item.content}}</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td>{{item.experiences}}</td></tr>{% endfor %}</table>{% else %}<p>翻訳候補なし</p>{% endif %}
</section>{% endfor %}</div>
{% endif %}
{% if error %}<section class="card"><p>{{error}}</p></section>{% endif %}
</main></body></html>'''


@app.route("/", methods=["GET", "POST"])
def index():
    subject = request.form.get("subject", "犬")
    relation = request.form.get("relation", "種類")
    repeats = int(request.form.get("repeats", "5") or 5)
    steps = int(request.form.get("steps", "24") or 24)
    mode = request.form.get("mode", "adaptive")
    adaptive_ratio = float(request.form.get("adaptive_ratio", "0.35") or 0.35)
    result = None
    error = ""
    if request.method == "POST":
        try:
            result = raw_output.observe_repeated(
                subject,
                relation,
                repeats=repeats,
                output_steps=steps,
                mode=mode,
                adaptive_ratio=adaptive_ratio,
            )
        except Exception as exc:
            error = str(exc)
    return render_template_string(
        TEMPLATE,
        subject=subject,
        relation=relation,
        repeats=repeats,
        steps=steps,
        mode=mode,
        adaptive_ratio=adaptive_ratio,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5012, debug=False)
