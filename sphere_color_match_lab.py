from __future__ import annotations

import random
import webbrowser
from threading import Lock, Timer

from flask import Flask, jsonify, render_template_string, request
from waitress import serve

from sphere_color_match import COLORS, SphereColorMatchBrain

app = Flask(__name__)
brain_lock = Lock()
brain = SphereColorMatchBrain(repeats=8)

COLOR_HEX = {"赤": "#ef5350", "青": "#42a5f5", "緑": "#66bb6a"}

PAGE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Color Match</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5;--red:#ff7d8d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1120px;margin:auto;padding:26px}.hero{text-align:center;padding:30px 0 20px}.eyebrow{color:var(--cyan);letter-spacing:.15em;font-size:12px;font-weight:800}.hero h1{font-size:clamp(36px,6vw,66px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.grid{display:grid;grid-template-columns:.9fr 1.1fr;gap:18px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:21px;padding:22px}.target-wrap{text-align:center}.target{width:220px;height:220px;border-radius:34px;margin:18px auto;border:7px solid rgba(255,255,255,.2);box-shadow:0 20px 55px rgba(0,0,0,.35)}.controls{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}.btn{border:0;border-radius:11px;padding:12px 17px;font:inherit;font-weight:800;cursor:pointer}.primary{background:var(--orange);color:white}.secondary{background:#173957;color:var(--text);border:1px solid #3e688e}.speech{border-left:4px solid var(--orange);background:#081522;border-radius:12px;padding:17px;margin-bottom:15px}.speech strong{display:block;font-size:30px;color:var(--green);margin-top:5px}.choices{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}.choice{border:3px solid transparent;border-radius:17px;padding:14px;background:#081522;text-align:center}.choice.selected{border-color:white;box-shadow:0 0 0 3px rgba(255,255,255,.13)}.swatch{height:90px;border-radius:13px;margin-bottom:9px}.candidate{display:grid;grid-template-columns:70px 1fr 64px;gap:10px;align-items:center;margin:11px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts,.training{display:grid;gap:7px}.fact,.train{padding:9px 11px;background:#081522;border:1px solid var(--line);border-radius:9px;font-family:Consolas,monospace;font-size:13px}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--cyan);font-size:13px}.note{color:var(--muted);font-size:14px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:18px}.flow div{background:#081522;border:1px solid var(--line);padding:12px;border-radius:11px;text-align:center}.flow b{display:block;color:var(--cyan);margin-bottom:5px}@media(max-width:850px){.grid{grid-template-columns:1fr}.target{width:180px;height:180px}.flow{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero"><div class="eyebrow">THREE EXPERIENCES → ROUTE MATCH</div><h1>SphereColor Match</h1><p>見本の色をCoreへ流し、赤・青・緑を経験した経路のうち、最も近いものを選びます。</p></section>
<div class="grid">
<section class="card target-wrap">
<h2>見本の色</h2>
<div class="target" id="target"></div>
<div class="controls"><button class="btn secondary" id="random">色を変える</button><button class="btn primary" id="think">SphereBrainに選ばせる</button></div>
<div class="flow"><div><b>INPUT</b>見本色</div><div><b>ENCODER</b>見本｜色｜○色</div><div><b>CORE</b>経験経路を伝播</div><div><b>DECODER</b>最も近い色</div></div>
<p class="note">保存済みCoreのコピーを使います。赤・青・緑を各8回だけ追加学習し、元のCoreとDBは変更しません。</p>
<h3>与えた3経験</h3><div class="training">{% for color in colors %}<div class="train">見本｜色｜{{color}} × 8回</div>{% endfor %}</div>
</section>
<section class="card">
<div class="speech"><div class="eyebrow">SPHEREBRAIN SAYS</div><strong id="speech">色を選んでください。</strong></div>
<div class="badges"><span class="badge" id="judge">未判定</span><span class="badge" id="raw">Raw Node — / Edge —</span></div>
<div class="choices" id="choices">{% for color in colors %}<div class="choice" data-color="{{color}}"><div class="swatch" style="background:{{color_hex[color]}}"></div><b>{{color}}</b></div>{% endfor %}</div>
<h3>経路の近さ</h3><div id="candidates"></div>
<h3>Encoder入力</h3><div class="facts" id="facts"><div class="fact">判定すると表示されます。</div></div>
</section>
</div>
</div>
<script>
const colors={{colors|tojson}},hex={{color_hex|tojson}};
const target=document.getElementById('target'),speech=document.getElementById('speech'),judge=document.getElementById('judge'),raw=document.getElementById('raw'),candidates=document.getElementById('candidates'),facts=document.getElementById('facts');
let current=colors[0];
function setColor(color){current=color;target.style.background=hex[color];speech.textContent='まだ選んでいません。';judge.textContent='未判定';raw.textContent='Raw Node — / Edge —';candidates.innerHTML='';facts.innerHTML='<div class="fact">判定すると表示されます。</div>';document.querySelectorAll('.choice').forEach(x=>x.classList.remove('selected'))}
function randomColor(){let next=current;while(next===current&&colors.length>1)next=colors[Math.floor(Math.random()*colors.length)];setColor(next)}
document.getElementById('random').addEventListener('click',randomColor);
document.getElementById('think').addEventListener('click',async()=>{const btn=document.getElementById('think');btn.disabled=true;speech.textContent='考えています…';try{const res=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({color:current})});const d=await res.json();if(!res.ok)throw new Error(d.error||'判定に失敗しました');speech.textContent=d.speech;judge.textContent=d.correct?'一致':'不一致';raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;document.querySelectorAll('.choice').forEach(x=>x.classList.toggle('selected',x.dataset.color===d.selected_color));candidates.innerHTML=d.candidates.map(x=>`<div class="candidate"><b>${x.color}</b><div class="track"><div class="fill" style="width:${(x.normalized*100).toFixed(1)}%"></div></div><span>${(x.normalized*100).toFixed(1)}%</span></div>`).join('');facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('');if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(d.speech);u.lang='ja-JP';speechSynthesis.speak(u)}}catch(e){speech.textContent=e.message}finally{btn.disabled=false}});
setColor(current);
</script>
</body>
</html>'''


@app.get("/")
def index():
    return render_template_string(PAGE, colors=COLORS, color_hex=COLOR_HEX)


@app.post("/api/decide")
def decide_api():
    payload = request.get_json(silent=True) or {}
    try:
        with brain_lock:
            result = brain.decide(str(payload.get("color", "赤")))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


def open_browser() -> None:
    webbrowser.open("http://127.0.0.1:5037")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    print("SphereColor Match: http://127.0.0.1:5037")
    serve(app, host="127.0.0.1", port=5037, threads=4)
