from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import SphereWordEngine


HTML = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereWord v0.1</title>
<style>
:root{--bg:#0b1020;--panel:#121a2b;--line:#263450;--text:#eef4ff;--muted:#94a3bd;--accent:#7dd3fc;--good:#86efac;--hot:#fbbf24}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:radial-gradient(circle at 25% 0,#18284a 0,var(--bg) 46%);color:var(--text);min-height:100vh}
.wrap{width:min(980px,calc(100% - 28px));margin:28px auto}.hero{margin-bottom:18px}.hero h1{font-size:clamp(30px,6vw,58px);margin:0;letter-spacing:-.04em}.hero p{color:var(--muted);margin:7px 0 0}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:rgba(18,26,43,.92);border:1px solid var(--line);border-radius:22px;padding:20px;box-shadow:0 24px 70px rgba(0,0,0,.28)}
label{display:block;font-size:13px;color:var(--muted);margin:0 0 7px}.row{display:flex;gap:9px}.row input{flex:1}input{width:100%;background:#0c1424;border:1px solid #344563;border-radius:13px;padding:13px 14px;color:var(--text);font-size:16px;outline:none}input:focus{border-color:var(--accent)}button{border:0;border-radius:13px;padding:12px 15px;font-weight:800;cursor:pointer;background:#e8f1ff;color:#111827}button.secondary{background:#25324b;color:var(--text)}button:disabled{opacity:.45;cursor:not-allowed}.round{margin:18px 0;padding:18px;border-radius:18px;background:#0d1526;border:1px solid #2a3a59;text-align:center}.round .prompt{font-size:13px;color:var(--muted)}.round b{display:block;font-size:36px;margin:6px 0}.hint{color:var(--muted);font-size:13px}.guesslist{margin-top:16px;display:grid;gap:8px}.guess{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;padding:11px 13px;border-radius:13px;background:#0d1524;border:1px solid #263752}.bar{height:7px;background:#172237;border-radius:99px;overflow:hidden;margin-top:7px}.bar span{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--good));width:0}.result{font-weight:800}.correct{color:var(--good)}.stats{display:grid;grid-template-columns:1fr 1fr;gap:9px}.stat{padding:12px;border:1px solid #293a58;border-radius:14px;background:#0d1524}.stat b{display:block;font-size:23px}.tiny{font-size:12px;color:var(--muted);line-height:1.6}.status{min-height:22px;margin-top:10px;color:var(--hot)}.loading:after{content:' …';animation:dots 1s infinite}@keyframes dots{50%{opacity:.25}}
</style>
</head>
<body><div class="wrap">
<div class="hero"><h1>SphereWord</h1><p>LLMが候補をつくり、秘密の連想語はSphereBrain Coreが選ぶ。</p></div>
<div class="grid">
<section class="card">
<label>お題</label><div class="row"><input id="prompt" placeholder="例：海" maxlength="40"><button id="start">ゲーム開始</button></div>
<div class="round"><div class="prompt">現在のお題</div><b id="roundPrompt">まだ始まっていません</b><div class="hint">Coreが選んだ秘密の連想語を当ててください。</div></div>
<label>推測</label><div class="row"><input id="guess" placeholder="例：船" maxlength="40" disabled><button id="guessBtn" disabled>推測する</button></div>
<div id="status" class="status"></div><div id="guesses" class="guesslist"></div>
<div style="margin-top:16px;display:flex;gap:9px"><button id="reveal" class="secondary" disabled>答えを見る</button><button id="newRound" class="secondary">履歴を消す</button></div>
</section>
<aside class="card"><h2 style="margin-top:0">Core Monitor</h2><div class="stats"><div class="stat">使用Edge<b id="edges">0</b></div><div class="stat">Edge通過<b id="usage">0</b></div><div class="stat">経験Node<b id="nodes">0</b></div><div class="stat">最大Weight<b id="weight">0</b></div></div><p class="tiny" id="models"></p><hr style="border:0;border-top:1px solid #263450;margin:18px 0"><p class="tiny">秘密語はLLMが直接決めません。LLMは連想候補を作るだけで、候補ごとの活動を本物のSphereBrain Coreへ通し、現在形成されている経路との親和性から1語を選びます。推測の近さもCore活動のNode/Edge重なりで判定します。</p><p class="tiny">遊ぶたびに推測と秘密語の経験がCoreへ入り、同じお題でも将来の選択が変わる可能性があります。</p></aside>
</div></div>
<script>
let active=false;
const $=id=>document.getElementById(id);
async function api(path, body={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.error||'エラーが発生しました');return j}
function busy(text){$('status').textContent=text;$('status').classList.add('loading')}
function clearBusy(){ $('status').classList.remove('loading') }
function renderStats(s){$('edges').textContent=s.used_edges;$('usage').textContent=s.total_edge_usage;$('nodes').textContent=s.experienced_nodes;$('weight').textContent=s.max_edge_weight;$('models').textContent=`Embedding: ${s.embedding_model} / LLM: ${s.word_model}`}
async function refresh(){try{const r=await fetch('/api/status');renderStats(await r.json())}catch(e){}}
$('start').onclick=async()=>{const p=$('prompt').value.trim();if(!p)return;busy('LLMが候補を作り、Coreが秘密語を選んでいます');$('start').disabled=true;try{const j=await api('/api/start',{prompt:p});active=true;$('roundPrompt').textContent=j.prompt;$('guess').disabled=false;$('guessBtn').disabled=false;$('reveal').disabled=false;$('guesses').innerHTML='';$('status').textContent='準備完了。自由に連想して当ててみて。';renderStats(j.stats);$('guess').focus()}catch(e){$('status').textContent=e.message}finally{clearBusy();$('start').disabled=false}}
async function doGuess(){if(!active)return;const g=$('guess').value.trim();if(!g)return;busy('Core活動を比較しています');$('guessBtn').disabled=true;try{const j=await api('/api/guess',{guess:g});const d=document.createElement('div');d.className='guess';d.innerHTML=`<div><b>${j.guess}</b><div class="bar"><span style="width:${j.percent}%"></span></div></div><div class="result ${j.correct?'correct':''}">${j.label}<br><small>${j.percent}%</small></div>`;$('guesses').prepend(d);$('guess').value='';$('status').textContent=j.correct?`正解！ SphereBrainが選んだ言葉は「${j.secret}」でした。`:'もう一語どうぞ。';if(j.correct){$('guess').disabled=true;$('guessBtn').disabled=true;$('reveal').disabled=true;active=false}renderStats(j.stats)}catch(e){$('status').textContent=e.message}finally{clearBusy();if(active)$('guessBtn').disabled=false;$('guess').focus()}}
$('guessBtn').onclick=doGuess;$('guess').addEventListener('keydown',e=>{if(e.key==='Enter')doGuess()});$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter')$('start').click()});
$('reveal').onclick=async()=>{try{const j=await api('/api/reveal');$('status').textContent=`答えは「${j.secret}」。候補は ${j.candidates.join(' / ')}`;$('guess').disabled=true;$('guessBtn').disabled=true;$('reveal').disabled=true;active=false}catch(e){$('status').textContent=e.message}}
$('newRound').onclick=()=>{$('guesses').innerHTML='';$('roundPrompt').textContent='まだ始まっていません';$('status').textContent='';$('guess').disabled=true;$('guessBtn').disabled=true;$('reveal').disabled=true;active=false;$('prompt').focus()};refresh();setInterval(refresh,5000);
</script></body></html>'''


class GameState:
    def __init__(self, engine: SphereWordEngine) -> None:
        self.engine = engine
        self.lock = threading.RLock()
        self.prompt: str | None = None
        self.secret: str | None = None
        self.candidates: list[str] = []
        self.scores: dict[str, float] = {}

    def start(self, prompt: str) -> dict:
        with self.lock:
            choice = self.engine.new_round(prompt)
            self.prompt = choice.prompt
            self.secret = choice.secret
            self.candidates = choice.candidates
            self.scores = choice.scores
            return {"prompt": choice.prompt, "stats": self.engine.stats()}

    def guess(self, word: str) -> dict:
        with self.lock:
            if not self.prompt or not self.secret:
                raise ValueError("先にゲームを開始してください。")
            result = self.engine.guess(self.prompt, self.secret, word)
            result["stats"] = self.engine.stats()
            if result["correct"]:
                result["secret"] = self.secret
            return result

    def reveal(self) -> dict:
        with self.lock:
            if not self.secret:
                raise ValueError("ゲームが始まっていません。")
            return {"secret": self.secret, "candidates": self.candidates}


STATE: GameState | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "SphereWord/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print("[SphereWord] " + fmt % args)

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        assert STATE is not None
        if self.path == "/":
            raw = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == "/api/status":
            self._json(STATE.engine.stats())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        assert STATE is not None
        try:
            body = self._body()
            if self.path == "/api/start":
                self._json(STATE.start(str(body.get("prompt", ""))))
            elif self.path == "/api/guess":
                self._json(STATE.guess(str(body.get("guess", ""))))
            elif self.path == "/api/reveal":
                self._json(STATE.reveal())
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SphereWord local browser game")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reset-core", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    global STATE
    STATE = GameState(SphereWordEngine(reset_core=args.reset_core))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"SphereWord v0.1: {url}")
    print("終了: Ctrl+C")
    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSphereWordを終了します。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
