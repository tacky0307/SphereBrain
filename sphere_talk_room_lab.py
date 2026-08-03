from __future__ import annotations

from threading import Lock

from flask import Flask, jsonify, render_template_string, request

from sphere_talk_room import SCENES, SphereTalkBrain

app = Flask(__name__)
brain_lock = Lock()
brain = SphereTalkBrain(repeats=6)

PAGE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereTalk Room — Experimental</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5;--red:#ff7d8d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:24px}.hero{text-align:center;padding:30px 0 18px}.hero h1{font-size:clamp(34px,6vw,64px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.eyebrow{color:var(--cyan);letter-spacing:.15em;font-size:12px;font-weight:800}.grid{display:grid;grid-template-columns:.9fr 1.1fr;gap:18px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:20px;padding:20px}.scene-list{display:grid;gap:10px}.scene-btn{width:100%;text-align:left;background:#081522;color:var(--text);border:1px solid var(--line);border-radius:12px;padding:14px;cursor:pointer;font:inherit}.scene-btn.active{border-color:var(--orange);box-shadow:0 0 0 2px rgba(255,157,82,.16)}.scene-btn small{display:block;color:var(--muted);margin-top:5px}.utterance{background:#081522;border-left:4px solid var(--orange);border-radius:12px;padding:18px;margin:18px 0}.speaker{color:var(--cyan);font-size:13px;font-weight:800}.quote{font-size:24px;font-weight:800;margin-top:7px}.btn{border:0;border-radius:11px;padding:12px 17px;background:var(--orange);color:white;font-weight:800;cursor:pointer;font:inherit}.btn:disabled{opacity:.45}.speech{background:#081522;border-left:4px solid var(--green);border-radius:12px;padding:18px;margin-bottom:15px}.speech .line{font-size:30px;font-weight:900;color:var(--green)}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--cyan);font-size:13px}.candidate{display:grid;grid-template-columns:80px 1fr 64px;gap:10px;align-items:center;margin:10px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts{display:grid;gap:7px;max-height:300px;overflow:auto}.fact{padding:9px 11px;background:#081522;border:1px solid var(--line);border-radius:9px;font-family:Consolas,monospace;font-size:13px}.flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:18px}.flow div{background:#081522;border:1px solid var(--line);padding:12px;border-radius:11px;text-align:center}.flow b{color:var(--cyan);display:block;margin-bottom:5px}.note{color:var(--muted);font-size:14px}.correct{color:var(--green)}.wrong{color:var(--red)}@media(max-width:900px){.grid{grid-template-columns:1fr}.flow{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero"><div class="eyebrow">PRIVATE RESEARCH PROTOTYPE</div><h1>SphereTalk Room</h1><p>会話を世界情報として受け取り、Coreの判断PORTを人間の言葉へ変換します。</p></section>
<div class="grid">
<section class="card">
<h2>会話場面を選ぶ</h2>
<div class="scene-list" id="sceneList">
{% for key,item in scenes.items() %}<button class="scene-btn{% if loop.first %} active{% endif %}" data-key="{{key}}"><b>{{item.name}}</b><small>{{item.speaker}}：「{{item.utterance}}」</small></button>{% endfor %}
</div>
<div class="utterance"><div class="speaker" id="speaker">—</div><div class="quote" id="utterance">場面を選んでください。</div></div>
<button class="btn" id="think">SphereBrainに返事を考えさせる</button>
<div class="flow"><div><b>CONVERSATION</b>相手の発言</div><div><b>ENCODER</b>主張・証拠・危険</div><div><b>CORE</b>経験経路で判断</div><div><b>TALK PORT</b>同意・否定など</div><div><b>LANGUAGE</b>言葉へ変換</div></div>
<p class="note">文章を自由生成しているのではありません。Coreが選んだ会話PORTを、Language Decoderが現在の場面に合う日本語へ翻訳します。</p>
</section>
<section class="card">
<div class="speech"><div class="eyebrow">SPHEREBRAIN SAYS</div><div class="line" id="speech">まだ判断していません。</div></div>
<div class="badges"><span class="badge" id="port">PORT —</span><span class="badge" id="raw">Raw Node — / Edge —</span><span class="badge" id="judge">未判定</span></div>
<h3>会話PORT</h3><div id="candidates"></div>
<h3>Encoderが受け取った情報</h3><div class="facts" id="facts"><div class="fact">判断すると表示されます。</div></div>
</section>
</div>
</div>
<script>
const buttons=[...document.querySelectorAll('.scene-btn')],think=document.getElementById('think');
const speaker=document.getElementById('speaker'),utterance=document.getElementById('utterance'),speech=document.getElementById('speech'),port=document.getElementById('port'),raw=document.getElementById('raw'),judge=document.getElementById('judge'),candidates=document.getElementById('candidates'),facts=document.getElementById('facts');
let current=buttons[0]?.dataset.key||'agree';
function choose(btn){buttons.forEach(x=>x.classList.remove('active'));btn.classList.add('active');current=btn.dataset.key;const small=btn.querySelector('small').textContent;const parts=small.split('：「');speaker.textContent=parts[0];utterance.textContent=(parts[1]||'').replace(/」$/,'');speech.textContent='まだ判断していません。';port.textContent='PORT —';judge.textContent='未判定';candidates.innerHTML='';facts.innerHTML='<div class="fact">判断すると表示されます。</div>'}
buttons.forEach(btn=>btn.addEventListener('click',()=>choose(btn)));if(buttons[0])choose(buttons[0]);
think.addEventListener('click',async()=>{think.disabled=true;speech.textContent='考えています…';try{const res=await fetch('/api/respond',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scene:current})});const d=await res.json();if(!res.ok)throw new Error(d.error||'判断に失敗しました');speaker.textContent=d.speaker;utterance.textContent=d.utterance;speech.textContent=d.speech;port.textContent=`PORT ${d.selected_port}`;raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;judge.textContent=d.correct?'期待判断と一致':'期待判断と不一致';judge.className='badge '+(d.correct?'correct':'wrong');candidates.innerHTML=d.candidates.map(x=>`<div class="candidate"><b>${x.port}</b><div class="track"><div class="fill" style="width:${(x.score*100).toFixed(1)}%"></div></div><span>${(x.score*100).toFixed(1)}%</span></div>`).join('');facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('');if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(d.speech);u.lang='ja-JP';u.rate=.92;speechSynthesis.speak(u)}}catch(e){speech.textContent=e.message}finally{think.disabled=false}});
</script>
</body></html>'''


@app.get("/")
def index():
    return render_template_string(PAGE, scenes=SCENES)


@app.post("/api/respond")
def respond_api():
    data = request.get_json(silent=True) or {}
    scene = str(data.get("scene", "agree"))
    try:
        with brain_lock:
            result = brain.respond(scene)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    from waitress import serve
    import webbrowser
    webbrowser.open("http://127.0.0.1:5034")
    print("SphereTalk Room experimental: http://127.0.0.1:5034")
    serve(app, host="127.0.0.1", port=5034, threads=6)
