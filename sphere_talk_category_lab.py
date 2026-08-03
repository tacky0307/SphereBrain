from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from sphere_talk_category import CATEGORIES, SUBJECTS, TRAINING_FACTS, SphereTalkCategoryBrain

app = Flask(__name__)
brain = SphereTalkCategoryBrain(repeats=8)

PAGE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereTalk Category Lab</title>
<style>
:root{--bg:#07111f;--panel:#10223b;--line:#2b4d70;--text:#eef6ff;--muted:#9db2c9;--cyan:#68d8ff;--orange:#ff9d52;--green:#74e5a5}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#173453,#07111f 44%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:28px}.hero{text-align:center;padding:26px 0}.hero h1{font-size:clamp(38px,6vw,68px);margin:8px 0}.hero p{color:var(--muted);font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{background:rgba(16,34,59,.96);border:1px solid var(--line);border-radius:22px;padding:24px}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:12px}.controls select,.btn{padding:12px 14px;border-radius:12px;font:inherit}.controls select{background:#081522;color:var(--text);border:1px solid #365b80}.btn{border:0;background:var(--orange);color:white;font-weight:800;cursor:pointer}.question{margin-top:20px;padding:22px;border-radius:16px;background:#081522;border:1px solid var(--line);font-size:28px;font-weight:900}.speech{border-left:4px solid var(--orange);background:#081522;border-radius:12px;padding:18px;margin-bottom:16px}.speech strong{font-size:28px;color:var(--green)}.badge{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:5px 10px;color:var(--cyan);font-size:13px;margin-right:6px}.candidate{display:grid;grid-template-columns:90px 1fr 65px;gap:10px;align-items:center;margin:12px 0}.track{height:11px;background:#071522;border:1px solid #294a6c;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.facts,.training{display:grid;gap:8px}.fact,.train{background:#081522;border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-family:Consolas,monospace}.note{color:var(--muted)}@media(max-width:900px){.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero"><div style="color:var(--cyan);letter-spacing:.14em;font-weight:800">RELATION EXPERIENCE → ANSWER</div><h1>SphereTalk Category Lab</h1><p>「犬は動物ですか？」のような所属関係を、経験経路から答える研究用実験です。</p></section>
<div class="grid">
<section class="card">
<h2>質問を組み立てる</h2>
<div class="controls">
<select id="subject">{% for item in subjects %}<option>{{item}}</option>{% endfor %}</select>
<select id="category">{% for item in categories %}<option>{{item}}</option>{% endfor %}</select>
<button class="btn" id="ask">質問する</button>
</div>
<div class="question" id="question">犬は動物ですか？</div>
<p class="note">一致・不一致・不明の文章を直接選ばず、現在経路と経験済みの分類経路を比較します。</p>
<h3>与えた分類経験</h3>
<div class="training">{% for s,c,a in training %}<div class="train">{{s}}｜種類｜{{c}} → {{a}}</div>{% endfor %}</div>
</section>
<section class="card">
<div class="speech"><div style="color:var(--cyan);font-weight:800;letter-spacing:.12em">SPHEREBRAIN SAYS</div><strong id="speech">質問してください。</strong></div>
<div><span class="badge" id="mode">未判定</span><span class="badge" id="raw">Raw Node — / Edge —</span></div>
<h3>回答経路</h3><div id="candidates"></div>
<h3>Encoder入力</h3><div class="facts" id="facts"></div>
</section>
</div>
</div>
<script>
const subject=document.getElementById('subject'),category=document.getElementById('category'),ask=document.getElementById('ask');
const question=document.getElementById('question'),speech=document.getElementById('speech'),mode=document.getElementById('mode'),raw=document.getElementById('raw'),candidates=document.getElementById('candidates'),facts=document.getElementById('facts');
function updateQuestion(){question.textContent=`${subject.value}は${category.value}ですか？`}
subject.addEventListener('change',updateQuestion);category.addEventListener('change',updateQuestion);
ask.addEventListener('click',async()=>{ask.disabled=true;speech.textContent='考えています…';try{const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({subject:subject.value,category:category.value})});const d=await res.json();if(!res.ok)throw new Error(d.error||'失敗しました');speech.textContent=d.speech;mode.textContent=d.trained_pair?'経験済みの組合せ':(d.trained_subject?'主体は経験済み・組合せ未経験':'未経験主体');raw.textContent=`Raw Node ${d.raw_nodes} / Edge ${d.raw_edges}`;candidates.innerHTML=d.candidates.map(x=>`<div class="candidate"><b>${x.answer}</b><div class="track"><div class="fill" style="width:${(x.score*100).toFixed(1)}%"></div></div><span>${(x.score*100).toFixed(1)}%</span></div>`).join('');facts.innerHTML=d.facts.map(x=>`<div class="fact">${x}</div>`).join('');if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(d.speech);u.lang='ja-JP';speechSynthesis.speak(u)}}catch(e){speech.textContent=e.message}finally{ask.disabled=false}});
updateQuestion();
</script>
</body></html>'''


@app.get("/")
def index():
    return render_template_string(PAGE, subjects=SUBJECTS, categories=CATEGORIES, training=TRAINING_FACTS)


@app.post("/api/ask")
def ask_api():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(brain.answer(str(data.get("subject", "犬")), str(data.get("category", "動物"))))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    from waitress import serve
    import webbrowser
    webbrowser.open("http://127.0.0.1:5036")
    print("SphereTalk Category Lab: http://127.0.0.1:5036")
    serve(app, host="127.0.0.1", port=5036, threads=6)
