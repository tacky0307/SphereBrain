from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from sphere_world_showcase import DemoState, SphereWorldShowcase

app = Flask(__name__)
showcase = SphereWorldShowcase(repeats=12)

TEMPLATE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain × SphereWorld</title>
<style>
:root{--bg:#06111f;--panel:#0e2038;--panel2:#132a48;--line:#294c73;--text:#eef6ff;--muted:#9db4ce;--cyan:#61d7ff;--orange:#ff9c5a;--green:#70e6a2;--red:#ff7d8d;--yellow:#ffe07a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 50% 0,#102a48 0,#06111f 42%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}button,select{font:inherit}.wrap{max-width:1180px;margin:auto;padding:26px}.hero{padding:48px 0 24px;text-align:center}.eyebrow{color:var(--cyan);letter-spacing:.16em;font-weight:800;font-size:13px}.hero h1{font-size:clamp(34px,6vw,72px);line-height:1.05;margin:10px 0 16px}.hero p{max-width:780px;margin:auto;color:var(--muted);font-size:18px}.hero strong{color:var(--text)}.card{background:rgba(14,32,56,.94);border:1px solid var(--line);border-radius:22px;padding:22px;margin-top:20px}.demo{border-color:#3c6e9d}.section-title{display:flex;align-items:flex-end;justify-content:space-between;gap:16px}.section-title h2{margin:0;font-size:25px}.section-title p{margin:0;color:var(--muted)}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.cell{height:190px;border:1px solid #426a94;border-radius:20px;background:linear-gradient(180deg,#173656,#0b1c31);position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}.cell-label{position:absolute;left:12px;top:10px;color:var(--muted);font-size:13px}.actor{width:86px;height:86px;border-radius:24px;display:flex;align-items:center;justify-content:center;font-size:38px;font-weight:900;border:2px solid currentColor;transition:all .35s ease}.player{color:var(--cyan);background:rgba(97,215,255,.12)}.enemy{color:var(--orange);background:rgba(255,156,90,.12)}.both{display:flex;gap:10px}.controls{display:grid;grid-template-columns:1fr 1fr auto auto;gap:12px;align-items:end}.controls label{display:block;color:var(--muted);font-size:13px}.controls select{width:100%;margin-top:6px;background:#071522;color:var(--text);border:1px solid #3c648b;border-radius:11px;padding:11px}.btn{border:0;border-radius:12px;padding:12px 17px;font-weight:800;cursor:pointer}.primary{background:var(--orange);color:white}.secondary{background:#173957;color:var(--text);border:1px solid #3e688e}.btn:disabled{opacity:.45;cursor:not-allowed}.status{display:grid;grid-template-columns:1.25fr .75fr;gap:16px;margin-top:18px}.decision{background:#081726;border:1px solid var(--line);border-radius:18px;padding:19px}.decision .answer{font-size:34px;font-weight:900;color:var(--green);margin:5px 0}.badge{display:inline-block;padding:5px 10px;border-radius:999px;border:1px solid var(--line);color:var(--cyan);font-size:13px}.facts{display:grid;gap:7px}.fact{background:#081726;border:1px solid var(--line);border-radius:10px;padding:9px 11px;font-family:Consolas,monospace;font-size:14px}.candidate-list{display:grid;gap:10px;margin-top:16px}.candidate{display:grid;grid-template-columns:120px 1fr 72px;gap:12px;align-items:center}.track{height:12px;border-radius:99px;background:#071522;overflow:hidden;border:1px solid #24466a}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));transition:width .4s}.candidate small{color:var(--muted)}.flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin-top:20px}.node{border:1px solid var(--line);border-radius:16px;padding:16px;background:#081726;text-align:center;min-height:112px}.node b{display:block;color:var(--cyan);margin-bottom:8px}.arrow{font-size:28px;color:var(--orange)}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}.training{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.training-item{background:#081726;border:1px solid var(--line);border-radius:15px;padding:15px}.training-item .action{color:var(--green);font-size:20px;font-weight:800}.compare{width:100%;border-collapse:collapse;margin-top:12px}.compare th,.compare td{border-bottom:1px solid var(--line);padding:12px;text-align:left;vertical-align:top}.compare th{color:var(--cyan)}.compare td:first-child{font-weight:800}.callout{border-left:4px solid var(--orange);background:#081726;padding:17px;border-radius:12px;color:var(--muted)}.callout strong{color:var(--text)}.footer{color:var(--muted);text-align:center;padding:32px 0}.loading{color:var(--yellow)}@media(max-width:850px){.controls,.status,.grid2{grid-template-columns:1fr}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg);text-align:center}.training{grid-template-columns:1fr}.world{gap:8px}.cell{height:145px}.actor{width:68px;height:68px;font-size:29px}.candidate{grid-template-columns:105px 1fr 58px}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero">
<div class="eyebrow">EXPERIENCE SHAPES INTELLIGENCE</div>
<h1>世界を見て、<br>経験から動く脳。</h1>
<p><strong>SphereBrain</strong>は、答えをルールで直接書くのではなく、経験によって形成された経路から次の行動を選びます。</p>
</section>

<section class="card demo">
<div class="section-title"><div><h2>動かしてみる</h2><p>P＝Player、E＝Enemy。位置を自由に変えてください。</p></div><span class="badge">SphereWorld 0.2</span></div>
<div class="world" id="world"></div>
<div class="controls">
<label>Playerの位置<select id="player"><option value="0">左</option><option value="1">中央</option><option value="2">右</option></select></label>
<label>Enemyの位置<select id="enemy"><option value="0">左</option><option value="1">中央</option><option value="2" selected>右</option></select></label>
<button class="btn primary" id="think">SphereBrainに考えさせる</button>
<button class="btn secondary" id="execute" disabled>選択行動を実行</button>
</div>
<div class="status">
<div class="decision">
<div class="badge" id="modeBadge">未判定</div>
<div class="answer" id="answer">—</div>
<div id="message" class="loading">最初の判断時だけ、専用コピーCoreを準備します。</div>
<div class="candidate-list" id="candidates"></div>
</div>
<div><h3>World Encoderが入力した情報</h3><div class="facts" id="facts"><div class="fact">判断するとここに表示されます</div></div></div>
</div>
</section>

<section class="card">
<div class="section-title"><div><h2>中で起きていること</h2><p>ゲーム側が正解を直接返しているのではありません。</p></div></div>
<div class="flow">
<div class="node"><b>WORLD</b>PとEの現在位置</div><div class="arrow">→</div>
<div class="node"><b>ENCODER</b>位置・相対位置・接触へ分解</div><div class="arrow">→</div>
<div class="node"><b>CORE</b>経験経路と共鳴・伝播</div><div class="arrow">→</div>
<div class="node"><b>DECODER</b>最も近い行動経験を読む</div>
</div>
<div class="callout" style="margin-top:18px"><strong>画面の％は確率ではありません。</strong> 現在の世界から生じたNode・Edgeの経路が、各行動を経験したときの経路とどれだけ重なったかを示す「類似度」です。</div>
</section>

<section class="card">
<div class="section-title"><div><h2>教えたのは、たった3つの経験</h2><p>全9配置を暗記させていません。</p></div></div>
<div class="training">
{% for item in training %}<div class="training-item"><div>P：{{item.player}} ／ E：{{item.enemy}}</div><div class="action">→ {{item.action}}</div><details><summary>Encoder入力</summary>{% for fact in item.facts %}<div class="fact">{{fact}}</div>{% endfor %}</details></div>{% endfor %}
</div>
<p style="color:var(--muted);margin-bottom:0">残り6配置は未経験です。それでも「Enemyより左・右・同じ」という再利用可能な世界構造を使い、9配置すべてで行動を一般化しました。</p>
</section>

<section class="card">
<div class="section-title"><div><h2>LLMとの違い</h2><p>優劣ではなく、作ろうとしている知性の形が違います。</p></div></div>
<table class="compare"><tr><th></th><th>大規模言語モデル</th><th>SphereBrain</th></tr>
<tr><td>主な経験</td><td>大量の言語データ</td><td>対象世界で起きた構造化経験</td></tr>
<tr><td>形成されるもの</td><td>言語・知識パターンを持つ巨大な重み</td><td>経験で強化・再利用される活動経路</td></tr>
<tr><td>得意な出力</td><td>文章、説明、幅広い知識応答</td><td>限定世界での短い判断・制御</td></tr>
<tr><td>Encoder / Decoder</td><td>主に言語トークンを入出力</td><td>専門世界に合わせて感覚器と行動器を設計</td></tr>
<tr><td>今回の例</td><td>状況を文章で説明できる</td><td>世界状態を受け、経験経路から実際にPを動かす</td></tr></table>
<div class="callout" style="margin-top:18px"><strong>SphereBrainの狙い：</strong> 世界中の知識を持つ巨大な万能脳ではなく、必要な世界を経験し、その世界に合った神経経路を育てて動く、小さな専門脳。</div>
</section>

<section class="card grid2">
<div><h2>今回使った入力情報</h2><div class="facts"><div class="fact">Player｜位置｜左 / 中央 / 右</div><div class="fact">Enemy｜位置｜左 / 中央 / 右</div><div class="fact">Player｜相対位置｜Enemyより左 / 右 / 同じ</div><div class="fact">Player｜接触｜している / していない</div><div class="fact">要求｜次行動</div></div></div>
<div><h2>出力候補</h2><div class="facts"><div class="fact">左へ移動</div><div class="fact">右へ移動</div><div class="fact">停止</div></div><p style="color:var(--muted)">Node類似度35%＋Edge類似度65%の総合値が最も高い行動を選びます。</p></div>
</section>
<div class="footer">SphereBrain / SphereWorld Presentation Demo — 保存済みCore・DBは変更しない専用コピーCoreで実行</div>
</div>
<script>
const labels=['左','中央','右'];
const player=document.getElementById('player');const enemy=document.getElementById('enemy');
const world=document.getElementById('world');const think=document.getElementById('think');const execute=document.getElementById('execute');
const answer=document.getElementById('answer');const message=document.getElementById('message');const facts=document.getElementById('facts');const candidates=document.getElementById('candidates');const modeBadge=document.getElementById('modeBadge');
let selectedAction=null;let turn=0;
function draw(){const p=Number(player.value),e=Number(enemy.value);world.innerHTML=labels.map((label,i)=>{let actor='';if(i===p&&i===e)actor='<div class="both"><div class="actor player">P</div><div class="actor enemy">E</div></div>';else if(i===p)actor='<div class="actor player">P</div>';else if(i===e)actor='<div class="actor enemy">E</div>';return `<div class="cell"><span class="cell-label">${label}</span>${actor}</div>`}).join('')}
[player,enemy].forEach(el=>el.addEventListener('change',()=>{draw();selectedAction=null;execute.disabled=true;answer.textContent='—';message.textContent='位置を変更しました。SphereBrainに考えさせてください。';candidates.innerHTML='';facts.innerHTML='<div class="fact">判断するとここに表示されます</div>';modeBadge.textContent='未判定'}));
think.addEventListener('click',async()=>{think.disabled=true;execute.disabled=true;answer.textContent='思考中…';message.textContent='現在の世界をEncoderで構造化し、Coreへ流しています。';try{const res=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player_position:Number(player.value),enemy_position:Number(enemy.value)})});const data=await res.json();if(!res.ok)throw new Error(data.error||'判定に失敗しました');selectedAction=data.selected_action;answer.textContent=selectedAction;modeBadge.textContent=data.trained_state?'学習済み配置':'未経験配置';message.textContent=`判断差 ${(data.margin*100).toFixed(1)}pt ／ Raw Node ${data.raw_nodes} ／ Raw Edge ${data.raw_edges}`;facts.innerHTML=data.facts.map(v=>`<div class="fact">${v}</div>`).join('');candidates.innerHTML=data.candidates.map((c,i)=>`<div class="candidate"><div><b>${c.action}</b><br><small>Node ${(c.node_score*100).toFixed(1)} / Edge ${(c.edge_score*100).toFixed(1)}</small></div><div class="track"><div class="fill" style="width:${(c.score*100).toFixed(1)}%"></div></div><div>${(c.score*100).toFixed(1)}%</div></div>`).join('');execute.disabled=false}catch(err){answer.textContent='エラー';message.textContent=err.message}finally{think.disabled=false}});
execute.addEventListener('click',()=>{if(!selectedAction)return;let p=Number(player.value);if(selectedAction==='左へ移動')p=Math.max(0,p-1);if(selectedAction==='右へ移動')p=Math.min(2,p+1);player.value=String(p);turn+=1;draw();message.textContent=`Turn ${turn}: ${selectedAction} を実行しました。変化した世界をもう一度見せられます。`;execute.disabled=true;selectedAction=null});
draw();
</script>
</body>
</html>'''


@app.get('/')
def index():
    return render_template_string(TEMPLATE, training=showcase.training_experiences())


@app.post('/api/decide')
def decide():
    try:
        payload = request.get_json(silent=True) or {}
        result = showcase.decide(
            int(payload.get('player_position', 0)),
            int(payload.get('enemy_position', 2)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': f'{type(exc).__name__}: {exc}'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5028, debug=False)
