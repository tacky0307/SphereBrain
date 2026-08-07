from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import webbrowser

from .core_agent import CoreAgent
from .world import SphereWorld

HOST = "127.0.0.1"
PORT = 8764
BRAIN_PATH = Path("data/sphereworld_v01/brain.json")

_lock = threading.Lock()
_world: SphereWorld | None = None
_agent: CoreAgent | None = None
_seed = 7


def _new_session(seed: int = 7, reset_core: bool = False) -> None:
    global _world, _agent, _seed
    _seed = int(seed)
    _world = SphereWorld(size=7, seed=_seed)
    _agent = CoreAgent.load_or_create(BRAIN_PATH, reset=reset_core)
    _agent.save()


def _require_state() -> tuple[SphereWorld, CoreAgent]:
    if _world is None or _agent is None:
        _new_session()
    assert _world is not None and _agent is not None
    return _world, _agent


def _state_payload(last: dict | None = None) -> dict:
    world, agent = _require_state()
    grid = []
    for r in range(world.size):
        row = []
        for c in range(world.size):
            if (r, c) == world.agent:
                row.append("O")
            else:
                row.append(world.tile_at((r, c)).value)
        grid.append(row)
    return {
        "grid": grid,
        "turn": world.turn,
        "energy": world.energy,
        "max_energy": world.max_energy,
        "food": world.food_eaten,
        "alive": world.energy > 0,
        "sense": world.sense(),
        "core": agent.core_stats(),
        "seed": _seed,
        "last": last,
    }


def _move(action: str) -> dict:
    world, agent = _require_state()
    if world.energy <= 0:
        return _state_payload({"error": "Sphere is not alive. Start a new world."})
    senses = world.sense()
    result = world.step(action)
    agent.experience(senses, action, result.outcome)
    agent.save()
    return _state_payload({
        "mode": "teach",
        "action": action,
        "tile": result.tile.name.lower(),
        "outcome": result.outcome,
        "energy_delta": result.energy_delta,
    })


def _auto() -> dict:
    world, agent = _require_state()
    if world.energy <= 0:
        return _state_payload({"error": "Sphere is not alive. Start a new world."})
    senses = world.sense()
    decision = agent.choose_action(senses)
    result = world.step(decision.action)
    agent.experience(senses, decision.action, result.outcome)
    agent.save()
    ranked = sorted(decision.scores.items(), key=lambda item: item[1], reverse=True)
    return _state_payload({
        "mode": "auto",
        "action": decision.action,
        "tile": result.tile.name.lower(),
        "outcome": result.outcome,
        "energy_delta": result.energy_delta,
        "scores": [{"action": name, "score": score} for name, score in ranked],
    })


HTML = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereWorld v0.1</title>
<style>
:root{--bg:#0b1018;--card:#141c29;--line:#28364b;--text:#edf4ff;--muted:#91a2bb;--good:#5ee1ad;--bad:#ff718f;--food:#ffd66b;--blue:#79aaff}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 50% 0,#16243a 0,var(--bg) 52%);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh}
.wrap{width:min(1080px,94vw);margin:26px auto}.top{display:flex;justify-content:space-between;gap:16px;align-items:end;flex-wrap:wrap}h1{margin:0;font-size:34px}.sub{color:var(--muted);margin-top:6px}.layout{display:grid;grid-template-columns:minmax(420px,1fr) 360px;gap:18px;margin-top:18px}@media(max-width:850px){.layout{grid-template-columns:1fr}}
.card{background:rgba(20,28,41,.94);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 20px 60px rgba(0,0,0,.25)}
.hud{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}.stat{background:#0f1621;border:1px solid #253249;border-radius:12px;padding:10px}.stat b{font-size:22px;display:block}.stat span{font-size:12px;color:var(--muted)}
.board{display:grid;grid-template-columns:repeat(7,1fr);gap:7px;aspect-ratio:1/1}.cell{display:flex;align-items:center;justify-content:center;border-radius:11px;border:1px solid #2a3950;background:#101925;font-size:24px;font-weight:800}.agent{background:#173650;border-color:#4a83b4;color:#dff1ff;box-shadow:inset 0 0 20px rgba(87,175,255,.18)}.food{color:var(--food)}.danger{color:var(--bad);background:#291824}.wall{background:#202734;color:#74849a}
.controls{display:grid;grid-template-columns:repeat(3,70px);justify-content:center;gap:8px;margin:18px 0}.controls button,.wide button{border:0;border-radius:11px;padding:12px;font-weight:800;cursor:pointer;background:#e8f0ff;color:#0d1420}.controls .ghost,.wide .ghost{background:#25334a;color:var(--text)}.controls .auto{background:var(--blue);color:#08111d}.wide{display:flex;gap:8px;flex-wrap:wrap;justify-content:center}
.row{padding:9px 0;border-bottom:1px solid #223047}.label{color:var(--muted);font-size:12px}.value{font-weight:700;margin-top:3px}.log{background:#0c131d;border:1px solid #253248;border-radius:12px;padding:12px;min-height:100px;white-space:pre-wrap;font-family:ui-monospace,Consolas,monospace;font-size:13px}.dead{color:var(--bad);font-weight:900}.alive{color:var(--good);font-weight:900}
</style></head><body><div class="wrap">
<div class="top"><div><h1>SphereWorld v0.1</h1><div class="sub">World → numeric stimulus → real SphereBrain Core → action → experience</div></div><div id="life"></div></div>
<div class="layout"><main class="card"><div class="hud"><div class="stat"><b id="turn">0</b><span>TURN</span></div><div class="stat"><b id="energy">0</b><span>ENERGY</span></div><div class="stat"><b id="food">0</b><span>FOOD</span></div><div class="stat"><b id="seed">0</b><span>WORLD SEED</span></div></div><div id="board" class="board"></div>
<div class="controls"><span></span><button onclick="move('N')">▲</button><span></span><button onclick="move('W')">◀</button><button class="ghost" onclick="move('STAY')">●</button><button onclick="move('E')">▶</button><span></span><button onclick="move('S')">▼</button><span></span></div>
<div class="wide"><button class="auto" onclick="autoStep()">Core AUTO 1手</button><button class="ghost" onclick="newWorld(false)">新しい世界</button><button class="ghost" onclick="newWorld(true)">Coreも初期化</button></div></main>
<aside class="card"><h2 style="margin-top:0">Real Core</h2><div class="row"><div class="label">EXPERIENCED NODES</div><div class="value" id="nodes">-</div></div><div class="row"><div class="label">USED EDGES</div><div class="value" id="edges">-</div></div><div class="row"><div class="label">TOTAL EDGE USAGE</div><div class="value" id="usage">-</div></div><div class="row"><div class="label">MAX EDGE WEIGHT</div><div class="value" id="weight">-</div></div><h3>現在の刺激</h3><div id="sense" class="log"></div><h3>直前の結果</h3><div id="last" class="log">まだ行動していません。</div></aside></div></div>
<script>
const sym={'.':'','F':'F','!':'!','#':'#','O':'O'};
async function req(url,body){const r=await fetch(url,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});return await r.json()}
function render(s){document.getElementById('turn').textContent=s.turn;document.getElementById('energy').textContent=s.energy+'/'+s.max_energy;document.getElementById('food').textContent=s.food;document.getElementById('seed').textContent=s.seed;document.getElementById('life').innerHTML=s.alive?'<span class="alive">● ALIVE</span>':'<span class="dead">● DEAD</span>';
const b=document.getElementById('board');b.innerHTML='';for(const row of s.grid){for(const x of row){const d=document.createElement('div');d.className='cell '+(x==='O'?'agent':x==='F'?'food':x==='!'?'danger':x==='#'?'wall':'');d.textContent=sym[x];b.appendChild(d)}}
document.getElementById('nodes').textContent=s.core.experienced_nodes;document.getElementById('edges').textContent=s.core.used_edges;document.getElementById('usage').textContent=s.core.total_edge_usage;document.getElementById('weight').textContent=Number(s.core.max_edge_weight).toFixed(4);document.getElementById('sense').textContent=JSON.stringify(s.sense,null,2);document.getElementById('last').textContent=s.last?JSON.stringify(s.last,null,2):'まだ行動していません。'}
async function load(){render(await req('/api/state'))}async function move(a){render(await req('/api/move',{action:a}))}async function autoStep(){render(await req('/api/auto',{}))}async function newWorld(resetCore){const seed=Math.floor(Math.random()*100000);render(await req('/api/reset',{seed,reset_core:resetCore}))}load();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/":
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/api/state":
            with _lock:
                self._json(_state_payload())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid json"}, 400)
            return
        try:
            with _lock:
                if self.path == "/api/move":
                    self._json(_move(str(body.get("action", ""))))
                elif self.path == "/api/auto":
                    self._json(_auto())
                elif self.path == "/api/reset":
                    _new_session(int(body.get("seed", 7)), bool(body.get("reset_core", False)))
                    self._json(_state_payload({"message": "new world"}))
                else:
                    self.send_error(404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="SphereWorld v0.1 localhost web UI")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-core", action="store_true")
    args = parser.parse_args()
    _new_session(args.seed, args.reset_core)
    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"SphereWorld web: {url}")
    print(f"Core file: {BRAIN_PATH}")
    print("Ctrl+C to stop")
    if args.host in {"127.0.0.1", "localhost"}:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _, agent = _require_state()
        agent.save()


if __name__ == "__main__":
    main()
