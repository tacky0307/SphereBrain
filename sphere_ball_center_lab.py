from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from sphere_ball_center import POSITIONS, TRAINING, BallCenterBrain

app = Flask(__name__)
brain = BallCenterBrain(repeats=8)

PAGE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Ball Center</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:28px}.hero{text-align:center;padding:26px 0}.hero h1{font-size:clamp(38px,6vw,68px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:22px;padding:24px}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.cell{height:170px;border:1px solid #365b80;border-radius:18px;background:#081522;position:relative;display:flex;align-items:center;justify-content:center}.cell small{position:absolute;top:10px;left:12px;color:var(--muted)}.ball{width:76px;height:76px;border-radius:50%;background:var(--cyan);box-shadow:0 0 22px rgba(104,216,255,.35)}.center{border-color:var(--green)}select,.btn{padding:12px 14px;border-radius:12px;font:inherit}select{width:100%;background:#081522;color:var(--text);border:1px solid #365b80}.btn{border:0;background:var(--orange);color:white;font-weight:800;cursor:pointer;margin-top:12px}.speech{border-left:4px solid var(--orange);background:#081522;border-radius:12px;padding:18px;margin-bottom:16px}.speech strong{font-size:28px;color:var(--green)}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--cyan);font-size:13px;margin-right:6px}.candidate{display:grid;grid-template-columns:110px 1fr 65px;gap:10px;align-items:center;margin:12px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts,.training{display:grid;gap:8px}.fact,.train{background:#081522;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-family:Consolas,monospace}.note{color:var(--muted)}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<section class="hero"><div style="color:var(--cyan);letter-spacing:.14em;font-weight:800">POSITION → ROUTE → ACTION</div><h1>Ball Center Experiment</h1><p>ボールの位置を見て、経験経路から中央へ寄せる行動を選びます。</p></section>
<div class="grid"><section class="card"><h2>ボールの位置</h2><div class="world" id="world"></div><select id="position">{% for item in positions %}<option>{{item}}</option>{% endfor %}</select><button class="btn" id="think">SphereBrainに考えさせる</button><p class="note">教えた経験は3つだけです。左→右へ移動、中央→停止、右→左へ移動。</p><h3>与えた経験</h3><div class="training">{% for p,a in training.items() %}<div class="train">ボール｜位置｜{{p}} → {{a}}</div>{% endfor %}</div></section>
<section class="card"><div class="speech"><div style="color:var(--cyan);font-weight:800;letter-spacing:.12em">SPHEREBRAIN SAYS</div><strong id="speech">位置を選んでください。</strong></div><div><span class="badge" id="judge">未判定</span><span class="badge" id="raw">Raw Node — / Edge —</span></div><h3>行動候補</h3><div id="candidates"></div><h3>Encoder入力</h3><div class="facts" id="facts"></div></section></div></div>
<script>
const labels=['左','中央','右'];const position=document.getElementById('position'),world=document.getElementById('world'),think=document.getElementById('think'),speech=document.getElementById('speech'),judge=document.getElementById('judge'),raw=document.getElementById('raw'),candidates=document.getElementById('candidates'),facts=document.getElementById('facts');
function draw(p){world.innerHTML=labels.map(x=>`<div class="cell ${x==='中央'?'center':''}"><small>${x}</small>${x===p?'<div class="ball"></div>':''}</div>`).join('')}
position.addEventListener('change',()=>draw(position.value));
think.addEventListener('click',async()=>{think.disabled=true;speech.textContent='考えています…';try{const res=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({position:position.value})});const d=await res.json();if(!res.ok)throw new Error(d.error||'失敗しました');speech.textContent=d.speech;judge.textContent=d.correct?'経験経路と一致':'期待行動と不一致';raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;candidates.innerHTML=d.candidates.map(x=>`<div class="candidate"><b>${x.action}</b><div class="track"><div class="fill" style="width:${(x.normalized*100).toFixed(1)}%"></div></div><span>${(x.normalized*100).toFixed(1)}%</span></div>`).join('');facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('');draw(d.next_position);if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(d.speech);u.lang='ja-JP';speechSynthesis.speak(u)}}catch(e){speech.textContent=e.message}finally{think.disabled=false}});draw(position.value);
</script></body></html>'''

@app.get('/')
def index():
    return render_template_string(PAGE, positions=POSITIONS, training=TRAINING)

@app.post('/api/decide')
def decide():
    data=request.get_json(silent=True) or {}
    try:
        return jsonify(brain.decide(str(data.get('position','左'))))
    except Exception as exc:
        return jsonify({'error':str(exc)}),400

if __name__=='__main__':
    from waitress import serve
    import webbrowser
    webbrowser.open('http://127.0.0.1:5038')
    print('Ball Center Experiment: http://127.0.0.1:5038')
    serve(app,host='127.0.0.1',port=5038,threads=6)
