from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from sphere_world_route_match import SphereWorldRouteMatch

app = Flask(__name__)
viewer = SphereWorldRouteMatch(repeats=12)

TEMPLATE = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Route Match Viewer</title>
<style>
:root{--bg:#050912;--panel:#0b1423;--panel2:#101d30;--line:#243b59;--text:#edf6ff;--muted:#96abc2;--cyan:#5ee1ff;--orange:#ffa45f;--green:#69e89b;--red:#ff7088;--yellow:#ffe176}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -20%,#173c64 0,#050912 46%);color:var(--text);font-family:Inter,"Noto Sans JP",system-ui,sans-serif}button,select{font:inherit}.wrap{max-width:1240px;margin:auto;padding:24px}.hero{text-align:center;padding:34px 0 18px}.eyebrow{font-size:12px;letter-spacing:.18em;color:var(--cyan);font-weight:800}.hero h1{font-size:clamp(32px,5vw,62px);margin:8px 0 12px}.hero p{max-width:800px;margin:auto;color:var(--muted);font-size:17px}.card{background:rgba(11,20,35,.95);border:1px solid var(--line);border-radius:22px;padding:20px;margin-top:18px}.topgrid{display:grid;grid-template-columns:.8fr 1.2fr;gap:18px}.world{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:16px 0}.cell{height:150px;border:1px solid #365b82;border-radius:18px;background:linear-gradient(180deg,#132943,#091523);display:flex;align-items:center;justify-content:center;position:relative}.cell span{position:absolute;top:8px;left:10px;color:var(--muted);font-size:12px}.actor{width:70px;height:70px;border-radius:20px;border:2px solid currentColor;display:flex;align-items:center;justify-content:center;font-size:31px;font-weight:900}.player{color:var(--cyan);background:rgba(94,225,255,.11)}.enemy{color:var(--orange);background:rgba(255,164,95,.11)}.both{display:flex;gap:7px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:10px}.controls label{font-size:13px;color:var(--muted)}select{width:100%;margin-top:5px;background:#07111e;color:var(--text);border:1px solid #385b7f;border-radius:10px;padding:10px}.buttons{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px}.btn{border:0;border-radius:11px;padding:11px 14px;font-weight:800;cursor:pointer}.primary{background:var(--orange);color:white}.secondary{background:#16324d;color:var(--text);border:1px solid #365c81}.preset{background:#10253b;color:var(--cyan);border:1px solid #2d557c}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:#07111e;border:1px solid var(--line);border-radius:14px;padding:13px}.metric b{display:block;font-size:25px;margin-top:4px}.metric small{color:var(--muted)}.answer{font-size:38px;font-weight:900;color:var(--green);margin:10px 0}.facts{display:grid;gap:7px}.fact{padding:8px 10px;background:#07111e;border:1px solid var(--line);border-radius:9px;font-family:Consolas,monospace;font-size:13px}.lanes{display:grid;gap:18px}.lane{border:1px solid var(--line);border-radius:18px;padding:16px;background:#081321}.lane.winner{border-color:var(--green);box-shadow:inset 0 0 0 1px rgba(105,232,155,.18)}.lanehead{display:grid;grid-template-columns:150px 1fr auto;gap:14px;align-items:center}.action{font-size:22px;font-weight:900}.score{font-size:28px;font-weight:900;color:var(--cyan)}.bar{height:11px;border-radius:99px;background:#040a12;overflow:hidden;border:1px solid #213a56}.bar>div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green))}.stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:9px;color:var(--muted);font-size:12px}.chip{padding:4px 8px;border:1px solid var(--line);border-radius:999px}.routes{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.routebox{background:#050c16;border:1px solid #1c314a;border-radius:13px;padding:12px}.routebox h4{margin:0 0 10px;font-size:13px;color:var(--muted)}.route{display:flex;align-items:center;gap:2px;flex-wrap:wrap;min-height:52px}.edge{display:flex;align-items:center}.node{width:27px;height:27px;border-radius:50%;border:1px solid #41617f;color:#a9bfd4;display:flex;align-items:center;justify-content:center;font-size:9px;background:#0d1c2d}.link{width:18px;height:3px;background:#2a425c}.edge.shared .node{border-color:var(--yellow);color:var(--yellow);background:rgba(255,225,118,.08)}.edge.shared .link{height:6px;background:var(--yellow);box-shadow:0 0 9px rgba(255,225,118,.45)}.legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:13px}.swatch{display:inline-block;width:24px;height:4px;background:#2a425c;vertical-align:middle;margin-right:6px}.swatch.shared{height:7px;background:var(--yellow)}.explain{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.step{background:#07111e;border:1px solid var(--line);border-radius:14px;padding:14px}.step b{color:var(--cyan)}.callout{border-left:4px solid var(--orange);background:#07111e;padding:15px;border-radius:10px;color:var(--muted)}.callout strong{color:var(--text)}.loading{color:var(--yellow)}@media(max-width:900px){.topgrid,.routes,.explain{grid-template-columns:1fr}.summary{grid-template-columns:1fr 1fr}.lanehead{grid-template-columns:1fr}.cell{height:120px}.actor{width:58px;height:58px;font-size:25px}}
</style>
</head>
<body>
<div class="wrap">
<section class="hero"><div class="eyebrow">REAL CORE ROUTE OBSERVER</div><h1>答えではなく、<br>通った経路を見る。</h1><p>現在の世界から発生した活動経路を、右移動・左移動・停止を経験したときの実経路と比較します。</p></section>
<section class="topgrid">
<div class="card">
<h2 style="margin-top:0">世界を選ぶ</h2>
<div class="world" id="world"></div>
<div class="controls"><label>Player<select id="player"><option value="0">左</option><option value="1">中央</option><option value="2">右</option></select></label><label>Enemy<select id="enemy"><option value="0">左</option><option value="1">中央</option><option value="2" selected>右</option></select></label></div>
<div class="buttons"><button class="btn primary" id="inspect">経路を観察する</button><button class="btn preset" data-p="0" data-e="2">学習済み：左 / 右</button><button class="btn preset" data-p="1" data-e="2">未経験：中央 / 右</button><button class="btn preset" data-p="2" data-e="1">未経験：右 / 中央</button></div>
</div>
<div class="card">
<div id="badge" style="color:var(--cyan);font-weight:800">未観察</div>
<div class="answer" id="answer">—</div>
<div id="message" class="loading">経路を観察すると、実際のNodeとEdgeを表示します。</div>
<div class="summary" style="margin-top:15px"><div class="metric"><small>世界文脈Node</small><b id="contextNodes">—</b></div><div class="metric"><small>Raw Node</small><b id="rawNodes">—</b></div><div class="metric"><small>Raw Edge</small><b id="rawEdges">—</b></div><div class="metric"><small>正解判定</small><b id="correct">—</b></div></div>
<h3>Encoder入力</h3><div class="facts" id="facts"><div class="fact">まだ入力されていません</div></div>
</div>
</section>
<section class="card"><div style="display:flex;justify-content:space-between;gap:12px;align-items:end;flex-wrap:wrap"><div><h2 style="margin:0">Route Match</h2><p style="margin:5px 0 0;color:var(--muted)">現在経路と、3つの行動経験経路を比較</p></div><div class="legend"><span><i class="swatch shared"></i>両方で通った実Edge</span><span><i class="swatch"></i>片方だけのEdge</span></div></div><div class="lanes" id="lanes" style="margin-top:16px"><div class="callout">「経路を観察する」を押してください。</div></div></section>
<section class="card"><h2 style="margin-top:0">この画面が示すこと</h2><div class="explain"><div class="step"><b>1. 現在経路</b><p>世界事実をCoreへ流したとき、本当に通過したNodeとEdgeです。</p></div><div class="step"><b>2. 経験経路</b><p>右・左・停止を経験した状態を、同じRaw Output段階で再生した経路です。</p></div><div class="step"><b>3. 行動選択</b><p>共通Node 35%＋共通Edge 65%で、最も近い経験経路を選びます。</p></div></div><div class="callout" style="margin-top:14px"><strong>単純な条件分岐との違い：</strong>この表示は「Pが左なら右」と書いた説明図ではありません。Coreが実際に返した <code>activated_nodes</code> と <code>traversed_edges</code> をそのまま比較し、共通Edgeを強調しています。</div></section>
</div>
<script>
const labels=['左','中央','右'];const player=document.getElementById('player');const enemy=document.getElementById('enemy');const world=document.getElementById('world');
function drawWorld(){const p=Number(player.value),e=Number(enemy.value);world.innerHTML=labels.map((l,i)=>{let a='';if(i===p&&i===e)a='<div class="both"><div class="actor player">P</div><div class="actor enemy">E</div></div>';else if(i===p)a='<div class="actor player">P</div>';else if(i===e)a='<div class="actor enemy">E</div>';return `<div class="cell"><span>${l}</span>${a}</div>`}).join('')}
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function routeHtml(items){if(!items.length)return '<span style="color:var(--muted)">表示対象Edgeなし</span>';return items.map(e=>`<div class="edge ${e.shared?'shared':''}"><div class="node">${e.from}</div><div class="link"></div><div class="node">${e.to}</div></div>`).join('')}
function laneHtml(lane,index){return `<div class="lane ${index===0?'winner':''}"><div class="lanehead"><div class="action">${esc(lane.action)} ${index===0?'✓':''}</div><div><div class="bar"><div style="width:${(lane.score*100).toFixed(1)}%"></div></div><div class="stats"><span class="chip">共通Node ${lane.shared_nodes}</span><span class="chip">共通Edge ${lane.shared_edges}</span><span class="chip">Node類似 ${(lane.node_score*100).toFixed(1)}%</span><span class="chip">Edge類似 ${(lane.edge_score*100).toFixed(1)}%</span></div></div><div class="score">${(lane.score*100).toFixed(1)}%</div></div><div class="routes"><div class="routebox"><h4>現在の活動経路（抽出）</h4><div class="route">${routeHtml(lane.current_route)}</div></div><div class="routebox"><h4>${esc(lane.action)}を経験した経路（抽出）</h4><div class="route">${routeHtml(lane.prototype_route)}</div></div></div></div>`}
async function inspect(){const btn=document.getElementById('inspect');btn.disabled=true;document.getElementById('answer').textContent='観察中…';document.getElementById('message').textContent='Coreを伝播し、実経路を比較しています。';try{const res=await fetch('/api/inspect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({player_position:Number(player.value),enemy_position:Number(enemy.value)})});const d=await res.json();if(!res.ok)throw new Error(d.error||'観察に失敗しました');document.getElementById('badge').textContent=d.trained?'学習済み配置':'未経験配置';document.getElementById('answer').textContent=d.selected;document.getElementById('message').textContent=`期待行動 ${d.expected}。最も重なった経験経路は「${d.selected}」です。`;document.getElementById('contextNodes').textContent=d.world_context_nodes;document.getElementById('rawNodes').textContent=d.raw_nodes;document.getElementById('rawEdges').textContent=d.raw_edges;document.getElementById('correct').textContent=d.correct?'○':'×';document.getElementById('facts').innerHTML=d.facts.map(f=>`<div class="fact">${esc(f)}</div>`).join('');document.getElementById('lanes').innerHTML=d.lanes.map(laneHtml).join('')+`<p style="color:var(--muted);font-size:12px">${esc(d.note)}</p>`}catch(err){document.getElementById('answer').textContent='エラー';document.getElementById('message').textContent=err.message}finally{btn.disabled=false}}
[player,enemy].forEach(el=>el.addEventListener('change',drawWorld));document.getElementById('inspect').addEventListener('click',inspect);document.querySelectorAll('.preset').forEach(btn=>btn.addEventListener('click',()=>{player.value=btn.dataset.p;enemy.value=btn.dataset.e;drawWorld();inspect()}));drawWorld();
</script>
</body></html>'''


@app.get("/")
def index():
    return render_template_string(TEMPLATE)


@app.post("/api/inspect")
def inspect():
    payload = request.get_json(silent=True) or {}
    try:
        result = viewer.inspect(
            int(payload.get("player_position", 0)),
            int(payload.get("enemy_position", 2)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5029, debug=False)
