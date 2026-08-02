from __future__ import annotations

import webbrowser
from flask import Flask, render_template_string, request
from waitress import serve

import semantic_encoder_v2 as semantic

app = Flask(__name__)

PAGE = r"""
<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Semantic Encoder v2 Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223a;--line:#294d76;--text:#eef5ff;--muted:#9bb1ca;--cyan:#67dcff;--orange:#f28b4b;--green:#67e59a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1400px;margin:auto;padding:24px}header{background:linear-gradient(135deg,#112743,#091524);border-bottom:1px solid var(--line)}h1{margin:0 0 8px}.muted,p{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:linear-gradient(180deg,#132944,#0c1b2f);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{background:#071522;border:1px solid var(--line);border-radius:14px;padding:14px}.value{font-size:30px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em;text-transform:uppercase}label{display:block;color:var(--cyan);margin:12px 0 6px}input,select{width:100%;padding:12px;background:#071522;color:var(--text);border:1px solid #355d88;border-radius:10px;font-size:16px}button{margin-top:16px;padding:12px 18px;border:0;border-radius:10px;background:linear-gradient(135deg,#e86f36,#f7a05f);color:white;font-weight:800;cursor:pointer}.secondary{background:#173454;border:1px solid #47719c}.danger{background:#6d2630}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:4px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid var(--line);padding:10px}.score{font-size:24px;font-weight:800}.note{border-left:3px solid var(--cyan);padding-left:12px}.ok{color:var(--green)}@media(max-width:900px){.grid,.stats{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>SphereBrain Semantic Encoder v2 Lab</h1><p>文章全体を一つの記号にせず、「主体・関係・内容」を再利用可能な刺激として順番にCoreへ渡す実験です。</p></div></header>
<main class="wrap">
<section class="stats">
<div class="stat"><div class="eyebrow">Experiences</div><div class="value">{{stats.total}}</div></div>
<div class="stat"><div class="eyebrow">Subjects</div><div class="value">{{stats.subjects}}</div></div>
<div class="stat"><div class="eyebrow">Relations</div><div class="value">{{stats.relations}}</div></div>
<div class="stat"><div class="eyebrow">Contents</div><div class="value">{{stats.contents}}</div></div>
</section>
<div class="grid">
<section class="card"><div class="eyebrow">Structured Experience</div><h2>構造を持つ経験を入力</h2>
<form method="post">
<input type="hidden" name="action" value="train">
<label>主体</label><input name="subject" value="{{subject}}" placeholder="犬" required>
<label>関係</label><input name="relation" value="{{relation}}" placeholder="動作" required>
<label>内容</label><input name="content" value="{{content}}" placeholder="歩く" required>
<label>反復回数</label><input type="number" min="1" max="100" name="repeats" value="{{repeats}}">
<button>Coreへ経験を流す</button>
</form>
<p class="note">流れ：主体役＋犬 → 関係役＋動作 → 内容役＋歩く。各成分は別の文章でも同じ入口刺激を再利用します。</p>
{% if trained %}<p class="ok">保存しました：{{trained.input.label}}／活動ノード {{trained.all_nodes|length}}／通過Edge {{trained.all_edges|length}}</p>{% endif %}
</section>
<section class="card"><div class="eyebrow">Partial Cue Recall Probe</div><h2>主体だけから活動を観測</h2>
<form method="post">
<input type="hidden" name="action" value="probe">
<label>主体</label><input name="probe_subject" value="{{probe_subject}}" placeholder="犬" required>
<label>関係</label><input name="probe_relation" value="{{probe_relation}}" placeholder="動作" required>
<button>候補を注入せずCoreを動かす</button>
</form>
<p class="note">保存経路を候補としてCoreへ渡しません。Coreを動かした後、観測者が過去経験との重なりを比較します。</p>
{% if probe %}
<div><span class="pill">入口 {{probe.source_nodes}}</span><span class="pill">活動 {{probe.activated_nodes|length}} nodes</span><span class="pill">通過 {{probe.traversed_edges|length}} edges</span></div>
{% if probe.matches %}<table><tr><th>過去の内容</th><th>経路重なり</th><th>経験数</th></tr>{% for item in probe.matches %}<tr><td>{{item.content}}</td><td><span class="score">{{'%.1f'|format(item.score*100)}}%</span></td><td>{{item.experiences}}</td></tr>{% endfor %}</table>{% else %}<p>この主体・関係に対応する過去経験はまだありません。</p>{% endif %}
{% endif %}
</section></div>
<section class="card"><div class="eyebrow">Design Boundary</div><h2>このv2で与えるもの／Coreに任せるもの</h2>
<p><b>Encoderが与える：</b> 同じ「犬」、同じ「歩く」、同じ「動作関係」を別の経験でも再利用できる構造。</p>
<p><b>Encoderが与えない：</b> 犬は動物、歩くは移動、犬と猫は似ている、といった答え。</p>
<p><b>Coreに任せる：</b> 経験の重なりによる経路形成、共有、分岐、収束、部分刺激からの再活動。</p>
<form method="post" onsubmit="return confirm('Semantic v2専用のCoreとDBを初期化します。旧brain.jsonとmemory.dbは変更しません。よろしいですか？')"><input type="hidden" name="action" value="reset"><button class="danger">Semantic v2実験だけ初期化</button></form>
</section>
</main></body></html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    subject = "犬"
    relation = "動作"
    content = "歩く"
    repeats = 1
    probe_subject = "犬"
    probe_relation = "動作"
    trained = None
    probe = None
    error = ""

    if request.method == "POST":
        action = request.form.get("action", "")
        try:
            if action == "train":
                subject = request.form.get("subject", "")
                relation = request.form.get("relation", "")
                content = request.form.get("content", "")
                repeats = max(1, min(100, int(request.form.get("repeats", "1"))))
                trained = semantic.train(subject, relation, content, repeats)
            elif action == "probe":
                probe_subject = request.form.get("probe_subject", "")
                probe_relation = request.form.get("probe_relation", "")
                probe = semantic.recall_probe(probe_subject, probe_relation)
            elif action == "reset":
                semantic.reset_experiment()
        except Exception as exc:
            error = str(exc)

    return render_template_string(
        PAGE,
        stats=semantic.stats(), subject=subject, relation=relation, content=content,
        repeats=repeats, trained=trained, probe=probe,
        probe_subject=probe_subject, probe_relation=probe_relation, error=error,
    )


if __name__ == "__main__":
    semantic.initialize_db()
    url = "http://127.0.0.1:5092"
    webbrowser.open(url)
    serve(app, host="127.0.0.1", port=5092, threads=6)
