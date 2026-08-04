from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from sphere_ball_center import (
    GRID_SIZE,
    POSITIONS,
    TARGET_POSITION,
    TRAINING,
    BallCenterBrain,
)

app = Flask(__name__)
brain = BallCenterBrain(repeats=8)

PAGE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Ball Center 4x4</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5;--yellow:#ffe27a;--red:#ff7d8d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1260px;margin:auto;padding:28px}.hero{text-align:center;padding:26px 0}.hero h1{font-size:clamp(38px,6vw,68px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:22px;padding:24px}.world{display:grid;grid-template-columns:repeat({{grid_size}},1fr);gap:10px;margin:18px 0}.cell{aspect-ratio:1;border:1px solid #365b80;border-radius:15px;background:#081522;position:relative;display:flex;align-items:center;justify-content:center}.cell small{position:absolute;top:7px;left:9px;color:var(--muted);font-size:12px}.ball{width:54%;height:54%;border-radius:50%;background:var(--cyan);box-shadow:0 0 22px rgba(104,216,255,.35)}.target{border:2px solid var(--green);box-shadow:inset 0 0 0 2px rgba(116,229,165,.12)}.target:after{content:'TARGET';position:absolute;right:7px;bottom:6px;color:var(--green);font-size:10px;font-weight:800}.controls{display:grid;grid-template-columns:1fr auto auto;gap:10px;align-items:end}select,.btn{padding:12px 14px;border-radius:12px;font:inherit}select{width:100%;background:#081522;color:var(--text);border:1px solid #365b80}.btn{border:0;color:white;font-weight:800;cursor:pointer}.primary{background:var(--orange)}.secondary{background:#173957;border:1px solid #3e688e}.speech{border-left:4px solid var(--orange);background:#081522;border-radius:12px;padding:18px;margin-bottom:16px}.speech strong{font-size:28px;color:var(--green)}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--cyan);font-size:13px;margin:3px}.candidate{display:grid;grid-template-columns:110px 1fr 65px;gap:10px;align-items:center;margin:12px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts,.training{display:grid;gap:8px}.fact,.train{background:#081522;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-family:Consolas,monospace}.note{color:var(--muted)}.ok{color:var(--green)}.ng{color:var(--red)}details{margin-top:14px}summary{cursor:pointer;color:var(--cyan)}@media(max-width:900px){.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr}.world{max-width:620px;margin:18px auto}}
</style></head><body><div class="wrap">
<section class="hero"><div style="color:var(--cyan);letter-spacing:.14em;font-weight:800">POSITION → ROUTE → ONE-STEP ACTION</div><h1>Ball Target 4×4</h1><p>ボールの位置を見て、経験経路から目標マスへ1マスずつ寄せます。</p></section>
<div class="grid"><section class="card"><h2>4×4のボール世界</h2><div class="world" id="world"></div><div class="controls"><select id="position">{% for item in positions %}<option value="{{item}}">{{item}}</option>{% endfor %}</select><button class="btn primary" id="think">一手考える</button><button class="btn secondary" id="auto">自動で進む</button></div><p class="note">目標は {{target}}。16位置を各8回経験し、現在位置に最も近い経験経路から上下左右・停止を選びます。</p><details><summary>与えた16経験を見る</summary><div class="training">{% for p,a in training.items() %}<div class="train">ボール｜位置｜{{p}} → {{a}}</div>{% endfor %}</div></details></section>
<section class="card"><div class="speech"><div style="color:var(--cyan);font-weight:800;letter-spacing:.12em">SPHEREBRAIN SAYS</div><strong id="speech">位置を選んでください。</strong></div><div><span class="badge" id="judge">未判定</span><span class="badge" id="raw">Raw Node — / Edge —</span><span class="badge" id="route">一致経路 —</span></div><h3>行動候補</h3><div id="candidates"></div><h3>Encoder入力</h3><div class="facts" id="facts"></div></section></div></div>
<script>
const size={{grid_size}},target='{{target}}';const position=document.getElementById('position'),world=document.getElementById('world'),think=document.getElementById('think'),auto=document.getElementById('auto'),speech=document.getElementById('speech'),judge=document.getElementById('judge'),raw=document.getElementById('raw'),route=document.getElementById('route'),candidates=document.getElementById('candidates'),facts=document.getElementById('facts');let timer=null;
function labels(){const out=[];for(let r=1;r<=size;r++)for(let c=1;c<=size;c++)out.push(`${r},${c}`);return out}
function draw(p){world.innerHTML=labels().map(x=>`<div class="cell ${x===target?'target':''}"><small>${x}</small>${x===p?'<div class="ball"></div>':''}</div>`).join('')}
position.addEventListener('change',()=>{draw(position.value);stopAuto()});
async function step(){think.disabled=true;speech.textContent='考えています…';try{const res=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({position:position.value})});const d=await res.json();if(!res.ok)throw new Error(d.error||'失敗しました');speech.textContent=d.speech;judge.textContent=d.correct?'期待行動と一致':'期待行動と不一致';judge.className='badge '+(d.correct?'ok':'ng');raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;route.textContent=`一致経路 ${d.selected_route_position} (${(d.selected_route_score*100).toFixed(1)}%)`;candidates.innerHTML=d.candidates.map(x=>`<div class="candidate"><b>${x.action}</b><div class="track"><div class="fill" style="width:${(x.normalized*100).toFixed(1)}%"></div></div><span>${(x.normalized*100).toFixed(1)}%</span></div>`).join('');facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('');position.value=d.next_position;draw(d.next_position);if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(d.speech);u.lang='ja-JP';speechSynthesis.speak(u)}if(d.next_position===target)stopAuto();return d}catch(e){speech.textContent=e.message;stopAuto();return null}finally{think.disabled=false}}
think.addEventListener('click',step);
function stopAuto(){if(timer){clearInterval(timer);timer=null;auto.textContent='自動で進む'}}
auto.addEventListener('click',()=>{if(timer){stopAuto();return}auto.textContent='自動停止';timer=setInterval(step,1400)});draw(position.value);
</script></body></html>'''

@app.get('/')
def index():
    return render_template_string(
        PAGE,
        positions=POSITIONS,
        training=TRAINING,
        target=TARGET_POSITION,
        grid_size=GRID_SIZE,
    )

@app.post('/api/decide')
def decide():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(brain.decide(str(data.get('position', '1,1'))))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

if __name__ == '__main__':
    from waitress import serve
    import webbrowser
    webbrowser.open('http://127.0.0.1:5038')
    print('Ball Target 4x4: http://127.0.0.1:5038')
    serve(app, host='127.0.0.1', port=5038, threads=6)
