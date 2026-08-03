from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from sphere_talk_strength import CHARACTERS, TRAINING_COMPARISONS, SphereTalkStrengthBrain

app = Flask(__name__)
brain = SphereTalkStrengthBrain(repeats=8)

PAGE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereTalk Strength Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5;--red:#ff7d8d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1150px;margin:auto;padding:24px}.hero{text-align:center;padding:32px 0 18px}.hero h1{font-size:clamp(34px,6vw,64px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.eyebrow{color:var(--cyan);letter-spacing:.15em;font-size:12px;font-weight:800}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:20px;padding:20px}.question{font-size:28px;font-weight:800;line-height:1.5;background:#081522;border:1px solid var(--line);border-radius:16px;padding:18px;margin:18px 0}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:10px}.controls select,.btn{border-radius:11px;padding:11px 14px;font:inherit}.controls select{background:#081522;color:var(--text);border:1px solid #365b80}.btn{border:0;font-weight:800;cursor:pointer;background:var(--orange);color:white}.speech{border-left:4px solid var(--orange);background:#081522;border-radius:12px;padding:16px;margin-bottom:15px}.speech .line{font-size:28px;font-weight:900;color:var(--green)}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--cyan);font-size:13px}.candidate{display:grid;grid-template-columns:80px 1fr 60px;gap:10px;align-items:center;margin:10px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts,.training{display:grid;gap:7px}.fact,.training-item{padding:9px 11px;background:#081522;border:1px solid var(--line);border-radius:9px;font-family:Consolas,monospace;font-size:13px}.note{color:var(--muted);font-size:14px}.training-wrap{margin-top:18px}.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr}.question{font-size:22px}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero"><div class="eyebrow">QUESTION → CORE → ANSWER PORT → LANGUAGE</div><h1>SphereTalk Strength Lab</h1><p>言葉を組み合わせて質問し、SphereBrainの回答PORTを日本語へ変換します。</p></section>
<div class="grid">
<section class="card">
<h2>質問を組み立てる</h2>
<div class="controls">
<select id="subject">{% for c in characters %}<option value="{{c}}">{{c}}</option>{% endfor %}</select>
<select id="target">{% for c in characters %}<option value="{{c}}">{{c}}</option>{% endfor %}</select>
<button class="btn" id="ask">質問する</button>
</div>
<div class="question" id="question">魔王は魔王より強いですか？</div>
<p class="note">文章の答えを直接選んでいるのではなく、肯定・否定・同等・不明の回答PORTをCoreが選びます。</p>
<div class="training-wrap"><h3>与えた比較経験</h3><div class="training">{% for a,b,answer in training %}<div class="training-item">{{a}} は {{b}} より強い？ → {{answer}}</div>{% endfor %}</div></div>
</section>
<section class="card">
<div class="speech"><div class="eyebrow">SPHEREBRAIN SAYS</div><div class="line" id="speech">質問してください。</div></div>
<div class="meta"><span class="badge" id="mode">未判定</span><span class="badge" id="raw">Raw Node — / Edge —</span></div>
<h3>回答PORT</h3><div id="candidates"></div>
<h3>Encoder入力</h3><div class="facts" id="facts"><div class="fact">質問すると表示されます。</div></div>
</section>
</div>
</div>
<script>
const subject=document.getElementById('subject'),target=document.getElementById('target'),ask=document.getElementById('ask');
const question=document.getElementById('question'),speech=document.getElementById('speech'),mode=document.getElementById('mode'),raw=document.getElementById('raw'),candidates=document.getElementById('candidates'),facts=document.getElementById('facts');
function refreshQuestion(){question.textContent=`${subject.value}は${target.value}より強いですか？`}
subject.addEventListener('change',refreshQuestion);target.addEventListener('change',refreshQuestion);
function say(text){speech.textContent=text;if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='ja-JP';u.rate=.92;speechSynthesis.speak(u)}}
ask.addEventListener('click',async()=>{ask.disabled=true;speech.textContent='考えています…';try{const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:subject.value,target:target.value})});const d=await res.json();if(!res.ok)throw new Error(d.error||'回答に失敗しました');question.textContent=d.question;say(d.speech);mode.textContent=d.trained_pair?'経験済みの組合せ':'未経験の組合せ';raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;candidates.innerHTML=d.candidates.map(x=>`<div class="candidate"><b>${x.answer}</b><div class="track"><div class="fill" style="width:${(x.score*100).toFixed(1)}%"></div></div><span>${(x.score*100).toFixed(1)}%</span></div>`).join('');facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('')}catch(e){say(e.message)}finally{ask.disabled=false}});
refreshQuestion();
</script>
</body></html>'''


@app.get("/")
def index():
    return render_template_string(PAGE, characters=CHARACTERS, training=TRAINING_COMPARISONS)


@app.post("/api/ask")
def ask_api():
    data = request.get_json(silent=True) or {}
    try:
        result = brain.answer(str(data.get("subject", "")), str(data.get("target", "")))
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    from waitress import serve
    import webbrowser
    webbrowser.open("http://127.0.0.1:5035")
    print("SphereTalk Strength Lab: http://127.0.0.1:5035")
    serve(app, host="127.0.0.1", port=5035, threads=6)
