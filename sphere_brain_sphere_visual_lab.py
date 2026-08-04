from __future__ import annotations

import json
import webbrowser
from threading import Timer

from flask import Flask, render_template_string
from waitress import serve

from sphere_brain_sphere_visual import build_sphere_visual_dataset


app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain — Experience Forms the Sphere</title>
<style>
:root{--bg:#040915;--panel:#0b1630;--line:#203c66;--text:#edf6ff;--muted:#91a8c5;--cyan:#65dcff;--orange:#ff9d59;--green:#71e6a3;--yellow:#ffe178;--purple:#b99aff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#10284a 0,#040915 48%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}.wrap{max-width:1420px;margin:auto;padding:24px}.hero{text-align:center;padding:34px 0 22px}.eyebrow{color:var(--cyan);letter-spacing:.16em;font-size:12px;font-weight:800}.hero h1{font-size:clamp(34px,5vw,70px);line-height:1.04;margin:9px 0 14px}.hero p{max-width:840px;margin:auto;color:var(--muted);font-size:17px}.layout{display:grid;grid-template-columns:370px minmax(0,1fr);gap:18px}.card{background:rgba(11,22,48,.94);border:1px solid var(--line);border-radius:20px;padding:18px}.card h2{margin:0 0 12px;font-size:20px}.experience-list{display:grid;gap:11px}.experience{width:100%;text-align:left;border:1px solid var(--line);border-radius:14px;padding:13px;background:#071226;color:var(--text);cursor:pointer}.experience.active{border-color:var(--yellow);box-shadow:0 0 0 2px rgba(255,225,120,.12)}.experience .num{color:var(--cyan);font-size:12px;font-weight:800}.experience .action{color:var(--green);font-size:18px;font-weight:850;margin-top:5px}.facts{display:grid;gap:6px;margin-top:9px}.fact{font-family:Consolas,monospace;font-size:12px;color:#bdd3ea;background:#050d1c;border:1px solid #182d4c;border-radius:8px;padding:7px}.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.btn{border:1px solid var(--line);background:#132c4b;color:var(--text);padding:9px 12px;border-radius:10px;cursor:pointer}.btn.primary{background:var(--orange);border-color:transparent;color:white;font-weight:800}.stage{position:relative;min-height:760px;overflow:hidden;background:radial-gradient(circle at 50% 47%,rgba(38,104,164,.23),rgba(4,9,21,.3) 52%,rgba(4,9,21,.85) 73%)}canvas{display:block;width:100%;height:680px}.overlay{position:absolute;left:18px;right:18px;top:16px;display:flex;justify-content:space-between;gap:12px;pointer-events:none}.badge{border:1px solid var(--line);background:rgba(5,13,28,.78);border-radius:999px;padding:7px 11px;color:var(--cyan);font-size:13px}.legend{position:absolute;left:20px;bottom:18px;background:rgba(5,13,28,.8);border:1px solid var(--line);border-radius:14px;padding:11px 13px;font-size:13px;color:var(--muted)}.legend div{margin:4px 0}.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px}.line{display:inline-block;width:28px;height:3px;margin-right:7px;vertical-align:middle}.summary{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:18px}.metric{background:#071226;border:1px solid var(--line);border-radius:14px;padding:13px}.metric b{display:block;font-size:24px;color:var(--cyan)}.metric span{color:var(--muted);font-size:12px}.callout{margin-top:18px;border-left:4px solid var(--orange);background:#071226;border-radius:12px;padding:16px;color:var(--muted)}.callout strong{color:var(--text)}@media(max-width:980px){.layout{grid-template-columns:1fr}.stage{min-height:630px}canvas{height:560px}.summary{grid-template-columns:repeat(3,1fr)}}@media(max-width:560px){.wrap{padding:13px}.summary{grid-template-columns:repeat(2,1fr)}canvas{height:470px}.stage{min-height:540px}.overlay{flex-direction:column}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero">
<div class="eyebrow">EXPERIENCE SHAPES INTELLIGENCE</div>
<h1>これだけの経験が、<br>球体の中に経路をつくった。</h1>
<p>左の3つの経験だけをSphereBrainへ与えました。右の球体は、その経験を観測したときに実際に活動したNodeと通過したEdgeを、Core内の実座標で描いたものです。</p>
</section>
<div class="layout">
<aside class="card">
<h2>経路形成に使った入力</h2>
<div class="experience-list" id="experienceList"></div>
<div class="controls"><button class="btn primary" id="showAll">3経験を重ねる</button><button class="btn" id="pause">回転を止める</button><button class="btn" id="reset">向きを戻す</button></div>
<div class="callout"><strong>見方：</strong> 経験を選ぶと、その経験で通った経路だけが強調されます。3経験を重ねると、複数の経験で再利用された共有経路が太く光ります。</div>
</aside>
<main>
<section class="card stage">
<canvas id="sphere"></canvas>
<div class="overlay"><div class="badge" id="mode">3経験の経路を重ねて表示</div><div class="badge" id="counts"></div></div>
<div class="legend"><div><span class="dot" style="background:var(--cyan)"></span>活動Node</div><div><span class="line" style="background:var(--cyan)"></span>1経験だけの経路</div><div><span class="line" style="background:var(--yellow);height:5px"></span>複数経験で共有された経路</div><div><span class="line" style="background:#3b506f"></span>非選択経路</div></div>
</section>
<section class="summary" id="summary"></section>
<div class="callout"><strong>重要：</strong> この図は球体らしく見せるために作った架空の配線ではありません。Coreが持つ3次元Node座標と、3経験の観測時に返された実際の <code>activated_nodes</code>・<code>traversed_edges</code> を描画しています。</div>
</main>
</div>
</div>
<script>
const DATA={{ data_json|safe }};
const canvas=document.getElementById('sphere');const ctx=canvas.getContext('2d');
const mode=document.getElementById('mode');const counts=document.getElementById('counts');
let selected='all',running=true,rx=-0.20,ry=0.35,drag=false,lastX=0,lastY=0;
const nodeMap=new Map(DATA.nodes.map(n=>[n.id,n]));
function resize(){const r=canvas.getBoundingClientRect();const d=Math.min(2,window.devicePixelRatio||1);canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0)}
function rotate(p){let x=p.x,y=p.y,z=p.z;let cy=Math.cos(ry),sy=Math.sin(ry);let x1=x*cy-z*sy,z1=x*sy+z*cy;let cx=Math.cos(rx),sx=Math.sin(rx);return{x:x1,y:y*cx-z1*sx,z:y*sx+z1*cx}}
function project(p,w,h){const q=rotate(p),scale=Math.min(w,h)*.39;const depth=1.85+q.z*.34;return{x:w/2+q.x*scale/depth,y:h/2+q.y*scale/depth,z:q.z}}
function isActive(tags){return selected==='all'||tags.includes(selected)}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);const cx=w/2,cy=h/2,r=Math.min(w,h)*.34;let g=ctx.createRadialGradient(cx-r*.25,cy-r*.3,r*.08,cx,cy,r);g.addColorStop(0,'rgba(85,180,255,.18)');g.addColorStop(.72,'rgba(18,56,98,.09)');g.addColorStop(1,'rgba(3,10,24,.03)');ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.fill();ctx.strokeStyle='rgba(101,220,255,.25)';ctx.lineWidth=1.2;ctx.stroke();
const pts=new Map(DATA.nodes.map(n=>[n.id,project(n,w,h)]));
const edges=[...DATA.edges].sort((a,b)=>((pts.get(a.a)?.z||0)+(pts.get(a.b)?.z||0))-((pts.get(b.a)?.z||0)+(pts.get(b.b)?.z||0)));
for(const e of edges){const a=pts.get(e.a),b=pts.get(e.b);if(!a||!b)continue;const active=isActive(e.experiences);const shared=e.shared>=2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);if(active){ctx.strokeStyle=shared?'rgba(255,225,120,.90)':'rgba(101,220,255,.52)';ctx.lineWidth=shared?Math.min(6,2.4+e.shared+e.usage*.05):Math.min(3.4,1+e.weight*2);}else{ctx.strokeStyle='rgba(56,76,105,.18)';ctx.lineWidth=.7}ctx.stroke()}
const nodes=[...DATA.nodes].sort((a,b)=>(pts.get(a.id)?.z||0)-(pts.get(b.id)?.z||0));for(const n of nodes){const p=pts.get(n.id),active=isActive(n.experiences);const rad=active?(n.shared>=2?4.4:3):1.6;ctx.beginPath();ctx.arc(p.x,p.y,rad,0,Math.PI*2);ctx.fillStyle=active?(n.shared>=2?'rgba(255,225,120,.98)':'rgba(101,220,255,.88)'):'rgba(65,84,112,.3)';ctx.fill();if(active&&n.shared>=2){ctx.strokeStyle='rgba(255,225,120,.3)';ctx.lineWidth=5;ctx.stroke()}}
if(running&&!drag)ry+=.0022;requestAnimationFrame(draw)}
function renderExperiences(){const el=document.getElementById('experienceList');el.innerHTML=DATA.experiences.map(e=>`<button class="experience ${selected===e.id?'active':''}" data-id="${e.id}"><div class="num">EXPERIENCE ${e.number}</div><div>P：${e.player} ／ E：${e.enemy}</div><div class="action">→ ${e.action}</div><div class="facts">${e.facts.map(f=>`<div class="fact">${f}</div>`).join('')}</div><div style="margin-top:8px;color:var(--muted);font-size:12px">活動 ${e.node_count} Node ／ ${e.edge_count} Edge</div></button>`).join('');el.querySelectorAll('.experience').forEach(b=>b.addEventListener('click',()=>{selected=b.dataset.id;const e=DATA.experiences.find(x=>x.id===selected);mode.textContent=`EXPERIENCE ${e.number}：P${e.player} / E${e.enemy} → ${e.action}`;renderExperiences();updateCounts()}))}
function updateCounts(){const ns=DATA.nodes.filter(n=>isActive(n.experiences)).length,es=DATA.edges.filter(e=>isActive(e.experiences)).length;counts.textContent=`表示 ${ns} Node / ${es} Edge`}
document.getElementById('showAll').addEventListener('click',()=>{selected='all';mode.textContent='3経験の経路を重ねて表示';renderExperiences();updateCounts()});document.getElementById('pause').addEventListener('click',e=>{running=!running;e.target.textContent=running?'回転を止める':'回転を再開'});document.getElementById('reset').addEventListener('click',()=>{rx=-.20;ry=.35});
canvas.addEventListener('pointerdown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;ry+=(e.clientX-lastX)*.008;rx+=(e.clientY-lastY)*.008;lastX=e.clientX;lastY=e.clientY});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);
const s=DATA.summary;document.getElementById('summary').innerHTML=[['入力経験',s.experience_count],['入力事実',s.input_fact_count],['Core全体Node',s.core_node_count],['表示Node',s.displayed_node_count],['表示Edge',s.displayed_edge_count],['共有Edge',s.shared_edge_count]].map(x=>`<div class="metric"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('');
window.addEventListener('resize',resize);resize();renderExperiences();updateCounts();draw();
</script>
</body>
</html>'''


@app.route("/")
def index():
    dataset = build_sphere_visual_dataset(repeats=12)
    return render_template_string(
        TEMPLATE,
        data_json=json.dumps(dataset.to_dict(), ensure_ascii=False),
    )


def open_browser() -> None:
    webbrowser.open("http://127.0.0.1:5030")


if __name__ == "__main__":
    Timer(1.0, open_browser).start()
    print("SphereBrain Sphere Visual: http://127.0.0.1:5030")
    serve(app, host="127.0.0.1", port=5030, threads=4)
