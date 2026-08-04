from __future__ import annotations

from threading import Lock

from flask import Flask, jsonify, render_template_string, request

from sphere_world_puzzle import PUZZLES, PuzzleSphereBrain, PuzzleWorld

app = Flask(__name__)
brain_lock = Lock()
brain = PuzzleSphereBrain(repeats=5)
world = PuzzleWorld("straight")

PAGE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereWorld Puzzle</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5;--red:#ff7d8d;--yellow:#ffe27a}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:24px}.hero{text-align:center;padding:30px 0 18px}.hero h1{font-size:clamp(34px,6vw,66px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.eyebrow{color:var(--cyan);letter-spacing:.15em;font-size:12px;font-weight:800}.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:20px;padding:20px}.maze{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:18px 0}.cell{aspect-ratio:1;border:1px solid #42688f;border-radius:16px;background:linear-gradient(180deg,#173858,#0b1a2d);display:flex;align-items:center;justify-content:center;position:relative}.wall{background:repeating-linear-gradient(45deg,#17202d,#17202d 9px,#222f40 9px,#222f40 18px)}.actor{width:70%;height:70%;display:flex;align-items:center;justify-content:center;border-radius:22px;font-size:38px;font-weight:900}.player{color:var(--cyan);border:2px solid var(--cyan);background:rgba(104,216,255,.13)}.goal{color:var(--green);border:2px dashed var(--green);background:rgba(116,229,165,.12)}.both{box-shadow:0 0 0 5px rgba(255,226,122,.28)}.controls{display:grid;grid-template-columns:1fr auto auto auto;gap:10px}.controls select,.btn{border-radius:11px;padding:11px 14px;font:inherit}.controls select{background:#081522;color:var(--text);border:1px solid #365b80}.btn{border:0;font-weight:800;cursor:pointer}.primary{background:var(--orange);color:white}.secondary{background:#183957;color:var(--text);border:1px solid #42688f}.btn:disabled{opacity:.45;cursor:not-allowed}.speech{border-left:4px solid var(--orange);background:#081522;border-radius:12px;padding:16px;margin-bottom:15px}.speech .line{font-size:29px;font-weight:900;color:var(--green)}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--cyan);font-size:13px}.candidate{display:grid;grid-template-columns:110px 1fr 60px;gap:10px;align-items:center;margin:10px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts{display:grid;gap:7px;max-height:310px;overflow:auto}.fact{padding:9px 11px;background:#081522;border:1px solid var(--line);border-radius:9px;font-family:Consolas,monospace;font-size:13px}.log{display:grid;gap:8px;max-height:270px;overflow:auto}.log-item{background:#081522;border:1px solid var(--line);border-radius:10px;padding:10px}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:18px}.flow div{background:#081522;border:1px solid var(--line);padding:12px;border-radius:11px;text-align:center}.flow b{color:var(--cyan);display:block;margin-bottom:5px}.note{color:var(--muted);font-size:14px}.voice{display:flex;align-items:center;gap:8px;margin-top:12px}@media(max-width:900px){.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr}.maze{max-width:520px;margin:18px auto}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero"><div class="eyebrow">EXPERIENCE → JUDGMENT → ACTION → SPEECH</div><h1>SphereWorld Puzzle</h1><p>SphereBrainが迷路を見て、判断し、動き、その判断を言葉にします。</p></section>
<div class="grid">
<section class="card">
<h2>パズル世界</h2>
<div class="maze" id="maze"></div>
<div class="controls">
<select id="puzzle">{% for key,item in puzzles.items() %}<option value="{{key}}">{{item.name}}</option>{% endfor %}</select>
<button class="btn primary" id="step">一手考える</button>
<button class="btn secondary" id="auto">自動で進む</button>
<button class="btn secondary" id="reset">リセット</button>
</div>
<label class="voice"><input type="checkbox" id="voice" checked> SphereBrainの判断を音声で話す</label>
<div class="flow"><div><b>WORLD</b>迷路と現在位置</div><div><b>ENCODER</b>Goal方向と障害物</div><div><b>CORE</b>経験経路から判断</div><div><b>DECODER</b>Pを動かす</div><div><b>LANGUAGE</b>判断を発話する</div></div>
<p class="note">ゲーム側は移動可能かどうかだけを管理します。次にどちらへ動くかは、SphereBrainの候補経路比較から選びます。</p>
</section>
<section class="card">
<div class="speech"><div class="eyebrow">SPHEREBRAIN SAYS</div><div class="line" id="speech">準備ができました。</div></div>
<div class="badges"><span class="badge" id="turn">0手</span><span class="badge" id="raw">Raw Node — / Edge —</span><span class="badge" id="state">待機中</span></div>
<h3>行動候補</h3><div id="candidates"></div>
<h3>Encoderが見た世界</h3><div class="facts" id="facts"></div>
<h3>発話履歴</h3><div class="log" id="log"><div class="log-item">まだ行動していません。</div></div>
</section>
</div>
</div>
<script>
const maze=document.getElementById('maze'),puzzle=document.getElementById('puzzle'),step=document.getElementById('step'),auto=document.getElementById('auto'),reset=document.getElementById('reset');
const speech=document.getElementById('speech'),facts=document.getElementById('facts'),candidates=document.getElementById('candidates'),turn=document.getElementById('turn'),raw=document.getElementById('raw'),state=document.getElementById('state'),log=document.getElementById('log'),voice=document.getElementById('voice');
let world=null, timer=null;
function say(text){speech.textContent=text;if(voice.checked&&'speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='ja-JP';u.rate=.92;speechSynthesis.speak(u)}}
function draw(){if(!world)return;const walls=new Set(world.walls.map(x=>x.join(',')));maze.innerHTML='';for(let r=0;r<world.rows;r++){for(let c=0;c<world.cols;c++){const cell=document.createElement('div');cell.className='cell'+(walls.has(`${r},${c}`)?' wall':'');const isP=world.player[0]===r&&world.player[1]===c,isG=world.goal[0]===r&&world.goal[1]===c;if(!walls.has(`${r},${c}`)){if(isP){const a=document.createElement('div');a.className='actor player'+(isG?' both':'');a.textContent='P';cell.appendChild(a)}else if(isG){const a=document.createElement('div');a.className='actor goal';a.textContent='G';cell.appendChild(a)}}maze.appendChild(cell)}}turn.textContent=`${world.turn}手`;state.textContent=world.solved?'GOAL':'進行中'}
function renderDecision(d){raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('');candidates.innerHTML=d.candidates.map((x,i)=>`<div class="candidate"><b>${x.action}</b><div class="track"><div class="fill" style="width:${(x.score*100).toFixed(1)}%"></div></div><span>${(x.score*100).toFixed(1)}%</span></div>`).join('');log.innerHTML=d.history.map((x,i)=>`<div class="log-item">${i+1}. ${x}</div>`).join('')||'<div class="log-item">まだ行動していません。</div>';say(d.speech)}
async function call(path,body={}){const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();if(!res.ok)throw new Error(data.error||'処理に失敗しました');world=data.world;draw();if(data.decision)renderDecision(data.decision);return data}
async function load(){const res=await fetch('/api/state');const data=await res.json();world=data.world;draw();facts.innerHTML='<div class="fact">一手考えると表示されます。</div>'}
puzzle.addEventListener('change',async()=>{stopAuto();await call('/api/reset',{puzzle:puzzle.value});say('パズルを変更しました。')});reset.addEventListener('click',async()=>{stopAuto();await call('/api/reset',{puzzle:puzzle.value});say('最初の位置に戻りました。')});
step.addEventListener('click',async()=>{step.disabled=true;try{const d=await call('/api/step');if(d.world.solved)stopAuto()}catch(e){say(e.message)}finally{step.disabled=false}});
function stopAuto(){if(timer){clearInterval(timer);timer=null;auto.textContent='自動で進む'}}
auto.addEventListener('click',()=>{if(timer){stopAuto();return}auto.textContent='自動停止';timer=setInterval(async()=>{try{const d=await call('/api/step');if(d.world.solved)stopAuto()}catch(e){stopAuto();say(e.message)}},1500)});
load();
</script>
</body></html>'''

history: list[str] = []


@app.get("/")
def index():
    return render_template_string(PAGE, puzzles=PUZZLES)


@app.get("/api/state")
def state_api():
    return jsonify({"world": world.to_dict(), "history": history})


@app.post("/api/reset")
def reset_api():
    global world, history
    data = request.get_json(silent=True) or {}
    world = PuzzleWorld(str(data.get("puzzle", "straight")))
    history = []
    return jsonify({"world": world.to_dict(), "history": history})


@app.post("/api/step")
def step_api():
    global history
    with brain_lock:
        decision = brain.decide(world)
    action = decision["selected_action"]
    moved = world.move(action)
    if world.solved:
        spoken = "ゴールに到着しました。"
    elif moved:
        spoken = decision["speech"]
    else:
        spoken = f"{action}と判断しましたが、進めませんでした。"
    history.append(spoken)
    decision["speech"] = spoken
    decision["history"] = list(history)
    return jsonify({"world": world.to_dict(), "decision": decision})


if __name__ == "__main__":
    from waitress import serve
    import webbrowser
    webbrowser.open("http://127.0.0.1:5031")
    print("SphereWorld Puzzle: http://127.0.0.1:5031")
    serve(app, host="127.0.0.1", port=5031, threads=6)
