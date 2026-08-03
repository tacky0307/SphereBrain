from __future__ import annotations

from flask import Flask, redirect, render_template_string, request, url_for

from sphere_world import SphereWorld
from sphere_world_brain import SphereWorldBrain


app = Flask(__name__)
world = SphereWorld(player_position=0, enemy_position=2)
controller: SphereWorldBrain | None = None
last_decision: dict | None = None
last_event: dict | None = None


TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereWorld 0.1</title>
<style>
:root{--bg:#07111f;--panel:#11243b;--line:#31577c;--text:#eff6ff;--muted:#9eb3ca;--cyan:#6edcff;--orange:#ef9256;--green:#6be49b;--yellow:#ffd56a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1250px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;margin-top:18px}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.cell{min-height:150px;border:2px solid var(--line);border-radius:18px;background:#081725;display:flex;align-items:center;justify-content:center;font-size:44px;font-weight:900;position:relative}.cell small{position:absolute;left:10px;top:8px;font-size:13px;color:var(--muted)}.player{color:var(--cyan)}.enemy{color:var(--orange)}.touch{box-shadow:0 0 0 3px var(--yellow) inset}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.three{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}button,select{padding:11px 15px;border-radius:9px;border:1px solid var(--line);background:#0a1a2a;color:var(--text)}button{background:var(--orange);border:0;font-weight:800;cursor:pointer;margin:4px}.primary{background:var(--green);color:#052012}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;margin:3px;color:var(--cyan)}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left}.score{font-size:22px;font-weight:900}.raw{font-family:Consolas,monospace;background:#071522;border:1px solid var(--line);padding:12px;border-radius:10px;white-space:pre-wrap}p{color:var(--muted)}@media(max-width:800px){.grid,.three{grid-template-columns:1fr}.cell{min-height:100px}}
</style></head><body>
<header><div class="wrap"><h1>SphereWorld 0.1</h1><p>世界状態 → Encoder → SphereBrain v2.1 → Raw Output → 行動候補 → 世界更新</p></div></header>
<main class="wrap">
<section class="card">
<h2>3マスの世界</h2>
<div class="world">
{% for cell in snapshot['cells'] %}<div class="cell {{'touch' if snapshot['touching'] and cell}}"><small>{{['左','中央','右'][loop.index0]}}</small>{% if 'P' in cell %}<span class="player">P</span>{% endif %}{% if 'E' in cell %}<span class="enemy">E</span>{% endif %}</div>{% endfor %}
</div>
<p><span class="pill">turn {{snapshot['turn']}}</span><span class="pill">Player {{snapshot['player']['position']}}</span><span class="pill">Enemy {{snapshot['enemy']['position']}}</span><span class="pill">接触 {{'あり' if snapshot['touching'] else 'なし'}}</span></p>
</section>
<section class="card"><h2>操作</h2>
<form method="post" action="{{url_for('reset')}}" style="display:inline"><select name="player"><option value="0">Player 左</option><option value="1">Player 中央</option><option value="2">Player 右</option></select><select name="enemy"><option value="2">Enemy 右</option><option value="1">Enemy 中央</option><option value="0">Enemy 左</option></select><button>世界をリセット</button></form>
<form method="post" action="{{url_for('think')}}" style="display:inline"><button class="primary">SphereBrainに考えさせる</button></form>
{% if decision %}<form method="post" action="{{url_for('act')}}" style="display:inline"><input type="hidden" name="action" value="{{decision['selected_action']}}"><button>選択行動を実行：{{decision['selected_action']}}</button></form>{% endif %}
<p>初回の「考えさせる」は専用コピーCoreの準備に少し時間がかかります。保存済みCoreとDBは変更しません。</p></section>
{% if decision %}
<section class="card"><h2>SphereBrainの判断</h2><div class="grid"><div><h3>選択行動</h3><div class="score">{{decision['selected_action']}}</div><p>内容刺激を与えず、世界状態＋「次行動」から自由伝播した結果を、行動経験の経路と比較しています。</p></div><div><h3>Raw Output</h3><span class="pill">subject nodes {{decision['subject_nodes']}}</span><span class="pill">relation nodes {{decision['relation_nodes']}}</span><span class="pill">raw nodes {{decision['raw_nodes']}}</span><span class="pill">raw edges {{decision['raw_edges']}}</span></div></div>
<table><tr><th>候補</th><th>総合</th><th>Node</th><th>Edge</th><th>共通Node</th><th>共通Edge</th></tr>{% for row in decision['candidates'] %}<tr><td>{{row['action']}}</td><td class="score">{{'%.1f'|format(row['score']*100)}}%</td><td>{{'%.1f'|format(row['node_score']*100)}}%</td><td>{{'%.1f'|format(row['edge_score']*100)}}%</td><td>{{row['common_nodes']}}</td><td>{{row['common_edges']}}</td></tr>{% endfor %}</table>
<details><summary>Raw Top Nodes</summary><div class="raw">{% for node,value in decision['raw_top_nodes'] %}{{node}} : {{'%.6f'|format(value)}}{% if not loop.last %}\n{% endif %}{% endfor %}</div></details>
</section>{% endif %}
{% if event %}<section class="card"><h2>世界の変化</h2><p>turn {{event['turn']}}：{{event['action']}}</p><div class="raw">before: {{event['before']['state_key']}}\nafter : {{event['after']['state_key']}}</div></section>{% endif %}
<section class="card"><h2>この画面で見ていること</h2><div class="three"><div><h3>World</h3><p>PlayerとEnemyの位置を保持し、選ばれた行動を実行する。</p></div><div><h3>SphereBrain</h3><p>現在の世界状態から、経験した行動経路へRaw Outputがどれだけ近いかを出す。</p></div><div><h3>Decoder</h3><p>最も近い経路を「左へ移動・右へ移動・停止」へ翻訳する。</p></div></div></section>
</main></body></html>'''


@app.get('/')
def index():
    return render_template_string(TEMPLATE, snapshot=world.snapshot(), decision=last_decision, event=last_event)


@app.post('/reset')
def reset():
    global last_decision, last_event
    world.reset(int(request.form.get('player', 0)), int(request.form.get('enemy', 2)))
    last_decision = None
    last_event = None
    return redirect(url_for('index'))


@app.post('/think')
def think():
    global controller, last_decision, last_event
    if controller is None:
        controller = SphereWorldBrain(repeats=12)
    last_decision = controller.decide(world)
    last_event = None
    return redirect(url_for('index'))


@app.post('/act')
def act():
    global last_event, last_decision
    action = request.form.get('action', '停止')
    last_event = world.apply_action(action)
    last_decision = None
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5025, debug=False)
