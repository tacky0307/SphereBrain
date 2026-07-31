from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json, math, random, sqlite3, webbrowser
import numpy as np
from flask import Flask, render_template_string, request
from waitress import serve
from brain import SphereBrain

BASE=Path(__file__).resolve().parent; DATA=BASE/'data'
BRAIN_FILE=DATA/'brain.json'; MEMORY_DB=DATA/'memory.db'; PATTERN_DB=DATA/'pattern_candidates.db'; FEEDBACK_DB=DATA/'route_choice_feedback.db'
app=Flask(__name__)

@dataclass
class Candidate:
    key:str; text:str; nodes:list[int]; edges:list[tuple[int,int]]; decoy:bool=False; score:float=0.; percent:float=0.; label:str=''

def norm(a,b): return (a,b) if a<=b else (b,a)
def jload(v,d):
    try:return json.loads(v) if v else d
    except:return d
def key_for(edges): return sha256(';'.join(f'{a}-{b}' for a,b in edges).encode()).hexdigest()[:20]

def add_route(out,seen,text,nodes):
    clean=[]
    for n in nodes:
        n=int(n)
        if not clean or clean[-1]!=n: clean.append(n)
    edges=[norm(a,b) for a,b in zip(clean,clean[1:])]
    if len(edges)<2:return
    k=key_for(edges)
    if k in seen:return
    seen.add(k); out.append(Candidate(k,text or '(名称なし)',clean,edges))

def load_routes(limit=600):
    out=[]; seen=set(); counts={'memory':0,'reflection':0}
    if MEMORY_DB.exists():
        with sqlite3.connect(f'file:{MEMORY_DB.as_posix()}?mode=ro',uri=True,timeout=30) as c:
            c.row_factory=sqlite3.Row
            rows=c.execute("SELECT input_text,activated_nodes,traversed_edges FROM memories WHERE kind='input' ORDER BY id DESC LIMIT ?",(limit,)).fetchall()
        before=len(out)
        for r in rows:
            raw=jload(r['traversed_edges'],[]); nodes=[]
            if len(raw)>=2:
                for a,b in raw:
                    if not nodes:nodes.extend([int(a),int(b)])
                    elif nodes[-1]==int(a):nodes.append(int(b))
                    else:nodes.extend([int(a),int(b)])
            else:nodes=[int(n) for n in jload(r['activated_nodes'],[])]
            add_route(out,seen,r['input_text'] or '',nodes)
        counts['memory']=len(out)-before
    if PATTERN_DB.exists():
        with sqlite3.connect(f'file:{PATTERN_DB.as_posix()}?mode=ro',uri=True,timeout=30) as c:
            c.row_factory=sqlite3.Row
            latest=c.execute('SELECT MAX(run_id) FROM reflection_runs').fetchone()[0]
            rows=[] if latest is None else c.execute("SELECT pattern_json,target_texts,classification FROM reflection_pattern_snapshots WHERE run_id=? ORDER BY pattern_id",(latest,)).fetchall()
        before=len(out)
        for r in rows:
            nodes=[int(n) for n in jload(r['pattern_json'],[])]
            texts=[str(x) for x in jload(r['target_texts'],[])] or ['Reflection経路']
            for t in texts:add_route(out,seen,f"{t} [{r['classification'] or 'pattern'}]",nodes)
        counts['reflection']=len(out)-before
    return out,counts

def init_feedback():
    with sqlite3.connect(FEEDBACK_DB) as c:c.execute('CREATE TABLE IF NOT EXISTS route_feedback(prefix TEXT NOT NULL,route_key TEXT NOT NULL,positive INTEGER NOT NULL DEFAULT 0,negative INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(prefix,route_key))')
def fbias(prefix,k):
    init_feedback()
    with sqlite3.connect(FEEDBACK_DB) as c:r=c.execute('SELECT positive,negative FROM route_feedback WHERE prefix=? AND route_key=?',(prefix,k)).fetchone()
    if not r:return 0.
    p,n=map(int,r); return (p-n)/(p+n+3.)
def decoys(real,count,seed):
    if len(real)<2:return []
    rng=random.Random(int.from_bytes(sha256(seed.encode()).digest()[:8],'big')); out=[]; used={x.key for x in real}
    for _ in range(count*30):
        if len(out)>=count:break
        a,b=rng.sample(real,2); nodes=a.nodes[:max(2,len(a.nodes)//2)]+b.nodes[-max(2,len(b.nodes)//2):]
        if len(nodes)<4:continue
        edges=[norm(x,y) for x,y in zip(nodes,nodes[1:])]; k=key_for(edges)
        if k in used:continue
        used.add(k); out.append(Candidate(k,'偽経路（実経験断片を組み替え）',nodes,edges,True))
    return out

def score(brain,sources,prefix,c):
    ws=[]; us=[]; valid={n for n in c.nodes if 0<=n<brain.node_count}
    for a,b in c.edges:
        if 0<=a<brain.node_count and 0<=b<brain.node_count and brain.adjacency[a,b]:ws.append(float(brain.weights[a,b])); us.append(int(brain.usage[a,b]))
    strength=float(np.mean(ws)) if ws else 0.; familiar=float(np.mean([u/(u+5.) for u in us])) if us else 0.; entry=0.
    if valid:
        arr=np.array(sorted(valid)); vals=[]
        for s in sources:
            d=np.linalg.norm(brain.positions[arr]-brain.positions[s],axis=1); vals.append(1/(1+float(np.min(d))))
        entry=float(np.mean(vals))
    return .46*strength+.29*familiar+.20*entry+.18*fbias(prefix,c.key)-(.04 if c.decoy else 0.)

def evaluate(text,count,decoy_count):
    if not BRAIN_FILE.exists():raise RuntimeError('data/brain.json がありません。')
    brain=SphereBrain.load(BRAIN_FILE); sources=[int(n) for n in brain.text_to_sources(text)]; prefix=sha256(','.join(map(str,sorted(sources))).encode()).hexdigest()[:20]
    routes,counts=load_routes()
    if not routes:raise RuntimeError('memory.dbとpattern_candidates.dbの両方を調べましたが候補経路がありません。Reflectionを一度実行してください。')
    for c in routes:c.score=score(brain,sources,prefix,c)
    routes.sort(key=lambda c:(-c.score,c.key)); cand=routes[:max(1,count-decoy_count)]+decoys(routes[:60],decoy_count,text)
    for c in cand:c.score=score(brain,sources,prefix,c)
    cand.sort(key=lambda c:(-c.score,c.key)); peak=max(c.score for c in cand); ex=[math.exp((c.score-peak)*7) for c in cand]; total=sum(ex) or 1
    for i,(c,v) in enumerate(zip(cand,ex)):c.label=chr(65+i); c.percent=100*v/total
    return {'prefix':prefix,'sources':sources,'candidates':cand,'source_counts':counts}

def teach(text,payload,correct_key):
    brain=SphereBrain.load(BRAIN_FILE); sources=[int(n) for n in brain.text_to_sources(text)]; prefix=sha256(','.join(map(str,sorted(sources))).encode()).hexdigest()[:20]; init_feedback()
    with sqlite3.connect(FEEDBACK_DB) as c:
        for item in payload:
            ok=str(item['key'])==correct_key
            c.execute('INSERT INTO route_feedback(prefix,route_key,positive,negative) VALUES(?,?,?,?) ON CONFLICT(prefix,route_key) DO UPDATE SET positive=positive+excluded.positive,negative=negative+excluded.negative',(prefix,str(item['key']),1 if ok else 0,0 if ok else 1))
    for item in payload:
        ok=str(item['key'])==correct_key
        for a,b in item.get('edges',[]):
            a=int(a); b=int(b)
            if not(0<=a<brain.node_count and 0<=b<brain.node_count and brain.adjacency[a,b]):continue
            if ok:
                w=float(brain.weights[a,b]); brain.weights[a,b]=brain.weights[b,a]=min(1,w+.04*(1-w)); brain.usage[a,b]+=1; brain.usage[b,a]+=1
            else:brain.weights[a,b]=brain.weights[b,a]=max(.05,float(brain.weights[a,b])*.99)
    brain.save(BRAIN_FILE); return '○の経路を強化し、ほかの候補を×として記録しました。'

PAGE='''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Route Choice Lab</title><style>:root{--bg:#07111f;--panel:#10223a;--line:#284a70;--text:#edf4ff;--muted:#9ab0ca;--cyan:#69dcff;--green:#8ce3a9;--red:#ff8585}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui}.wrap{max-width:1180px;margin:auto;padding:22px}.card{background:linear-gradient(180deg,#122744,#0d1d31);border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 180px 180px;gap:12px}input{width:100%;background:#071522;color:var(--text);border:1px solid #345c86;border-radius:10px;padding:12px;font-size:16px}button{background:#ed8447;border:0;color:white;border-radius:10px;padding:12px 18px;font-weight:800}.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.12em}.muted{color:var(--muted)}.candidate{display:grid;grid-template-columns:70px 120px 1fr 150px;gap:14px;align-items:center;padding:15px 0;border-bottom:1px solid var(--line)}.letter{font-size:32px;font-weight:900;color:var(--cyan)}.percent{font-size:28px;font-weight:900}.route{font-family:monospace;color:#bfeeff}.real{color:var(--green)}.decoy{color:var(--red)}@media(max-width:760px){.grid,.candidate{grid-template-columns:1fr}}</style></head><body><main class="wrap"><div class="card"><div class="eyebrow">ROUTE CHOICE LEARNING LAB v0.2</div><h1>経路を提示し、○と×を教える</h1><p class="muted">候補元は通常記憶とReflection安定経路の両方です。</p></div><div class="card"><form method="post"><input type="hidden" name="action" value="evaluate"><div class="grid"><div><div class="eyebrow">PARTIAL INPUT</div><h2>途中入力</h2><input name="text" value="{{text}}" required></div><div><div class="eyebrow">CANDIDATES</div><h2>候補数</h2><input name="count" type="number" min="3" max="8" value="{{count}}"></div><div><div class="eyebrow">DECOYS</div><h2>偽経路数</h2><input name="decoys" type="number" min="0" max="4" value="{{decoys}}"></div></div><p><button>経路候補を評価する</button></p></form></div>{% if message %}<div class="card real">{{message}}</div>{% endif %}{% if error %}<div class="card decoy">{{error}}</div>{% endif %}{% if result %}<div class="card"><p class="muted">候補元：通常記憶 {{result.source_counts.memory}}本 / Reflection {{result.source_counts.reflection}}本</p>{% for c in result.candidates %}<div class="candidate"><div class="letter">{{c.label}}</div><div class="percent">{{'%.1f'|format(c.percent)}}%</div><div><div class="route">{% for e in c.edges[:10] %}{{e[0]}}→{{e[1]}}{% if not loop.last %} / {% endif %}{% endfor %}</div><div class="{% if c.decoy %}decoy{% else %}real{% endif %}">{% if c.decoy %}偽経路{% else %}実経路{% endif %} — {{c.text}}</div></div><form method="post"><input type="hidden" name="action" value="teach"><input type="hidden" name="text" value="{{text}}"><input type="hidden" name="payload" value='{{payload}}'><input type="hidden" name="correct_key" value="{{c.key}}"><button>{{c.label}} が○</button></form></div>{% endfor %}</div>{% endif %}</main></body></html>'''

@app.route('/',methods=['GET','POST'])
def index():
    text='犬は'; count=5; decoy_count=2; result=None; error=''; message=''; payload=''
    if request.method=='POST':
        text=request.form.get('text','').strip(); action=request.form.get('action','evaluate')
        try:
            if action=='teach':message=teach(text,json.loads(request.form.get('payload','[]')),request.form.get('correct_key',''))
            else:
                count=max(3,min(8,int(request.form.get('count','5')))); decoy_count=max(0,min(4,int(request.form.get('decoys','2')))); decoy_count=min(decoy_count,count-1); result=evaluate(text,count,decoy_count); payload=json.dumps([{'key':c.key,'edges':c.edges} for c in result['candidates']],separators=(',',':'))
        except Exception as e:error=str(e)
    return render_template_string(PAGE,text=text,count=count,decoys=decoy_count,result=result,error=error,message=message,payload=payload)

def main():webbrowser.open('http://127.0.0.1:5077'); serve(app,host='127.0.0.1',port=5077,threads=4)
if __name__=='__main__':main()
