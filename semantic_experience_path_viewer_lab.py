from __future__ import annotations

from flask import Flask, render_template_string, request

import semantic_experience_path_viewer as viewer

app = Flask(__name__)

TEMPLATE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SphereBrain Experience Path Viewer</title>
<style>
:root{--bg:#07111f;--panel:#13233b;--line:#315478;--text:#eef5ff;--muted:#9db2ca;--a:#ff9c5a;--b:#71d7ff;--common:#7de3a3;--old:#ff8295;--new:#ffd76a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}.wrap{max-width:1500px;margin:auto;padding:24px}header{background:#0b192b;border-bottom:1px solid var(--line)}h1,h2,h3{margin-top:0}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:20px;margin-top:18px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.inputs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}label{display:block;color:var(--muted);margin:4px 0}input{width:100%;padding:11px;border:1px solid #41658b;border-radius:9px;background:#081522;color:var(--text)}button{margin-top:14px;padding:12px 20px;border:0;border-radius:10px;background:var(--a);color:white;font-weight:800;cursor:pointer}.hero{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}.metric{background:#0a192b;border:1px solid var(--line);border-radius:14px;padding:16px;text-align:center}.big{font-size:34px;font-weight:900}.old{color:var(--old)}.new{color:var(--new)}.good{color:var(--common)}p{color:var(--muted)}.legend{display:flex;gap:14px;flex-wrap:wrap}.dot{display:inline-block;width:13px;height:13px;border-radius:50%;margin-right:6px}.pipeline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.stage{background:#0a192b;border:1px solid var(--line);border-radius:15px;padding:16px}.tokens{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;margin-bottom:15px}.token{padding:9px;border-radius:10px;text-align:center;font-weight:800}.ta{background:rgba(255,156,90,.18);border:1px solid var(--a)}.tb{background:rgba(113,215,255,.16);border:1px solid var(--b)}.arrow{color:var(--muted)}.road{position:relative;height:112px;margin:12px 0}.line{position:absolute;left:8%;right:8%;height:12px;border-radius:999px}.common{top:50px;background:var(--common)}.aline{top:22px;background:var(--a)}.bline{top:78px;background:var(--b)}.join{position:absolute;left:46%;top:28px;width:8%;height:60px;border-left:5px solid var(--common);border-right:5px solid var(--common);opacity:.65}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stat{background:#102139;border-radius:10px;padding:9px;text-align:center}.read{margin-top:12px;font-weight:800;color:var(--common)}details{margin-top:10px}summary{cursor:pointer;color:#8edfff}.raw{font-family:Consolas,monospace;background:#071522;padding:10px;border-radius:9px;white-space:pre-wrap;word-break:break-word}.bad{color:var(--old)}@media(max-width:1000px){.grid,.hero,.pipeline,.inputs{grid-template-columns:1fr}}
</style></head><body>
<header><div class="wrap"><h1>Experience Path Viewer</h1><p>旧v2とv2.1で、2つの経験がどのように同じ道・別の道を通るかを、人の目で見える形にします。</p></div></header>
<main class="wrap">
<section class="card"><form method="post"><div class="grid"><div><h3>経験A</h3><div class="inputs"><div><label>主体</label><input name="ls" value="{{ls}}"></div><div><label>関係</label><input name="lr" value="{{lr}}"></div><div><label>内容</label><input name="lc" value="{{lc}}"></div></div></div><div><h3>経験B</h3><div class="inputs"><div><label>主体</label><input name="rs" value="{{rs}}"></div><div><label>関係</label><input name="rr" value="{{rr}}"></div><div><label>内容</label><input name="rc" value="{{rc}}"></div></div></div></div><button>経験の流れを見る</button></form><p>観測のみ。CoreとDBは変更しません。</p></section>
{% if error %}<section class="card"><p class="bad">{{error}}</p></section>{% endif %}
{% if result %}
<section class="card"><h2>何が一歩進んだのか</h2><div class="hero"><div class="metric"><div>旧v2 最終類似</div><div class="big old">{{'%.1f'|format(result['headline']['old_final']*100)}}%</div><p>{{result['headline']['old_message']}}</p></div><div class="metric"><div>v2.1 最終類似</div><div class="big new">{{'%.1f'|format(result['headline']['new_final']*100)}}%</div><p>{{result['headline']['new_message']}}</p></div><div class="metric"><div>結果まで残った経験差</div><div class="big good">{{'%.1f'|format(result['headline']['difference_retained']*100)}}pt</div><p>同じ場所へ潰れず、経験ごとの違いが残った量です。</p></div></div></section>
<div class="legend card"><span><i class="dot" style="background:var(--a)"></i>経験Aだけの道</span><span><i class="dot" style="background:var(--b)"></i>経験Bだけの道</span><span><i class="dot" style="background:var(--common)"></i>両方が共有する道</span></div>
{% for title,key in [('旧 Semantic Encoder v2','old'),('Semantic Encoder v2.1 Contextual','new')] %}
<section class="card"><h2>{{title}}</h2><div class="pipeline">{% for s in result[key] %}<article class="stage"><h3>{{s['name']}}</h3><div class="tokens"><div class="token ta">{{s['left_text']}}</div><div class="arrow">→ Core ←</div><div class="token tb">{{s['right_text']}}</div></div><div class="road"><div class="line aline" style="opacity:{{0.25 + (s['left_only_edges']/(s['left_only_edges']+s['common_edges']+1))*0.75}}"></div><div class="line common" style="opacity:{{0.25 + (s['common_edges']/(s['left_only_edges']+s['right_only_edges']+s['common_edges']+1))*0.75}}"></div><div class="line bline" style="opacity:{{0.25 + (s['right_only_edges']/(s['right_only_edges']+s['common_edges']+1))*0.75}}"></div><div class="join"></div></div><div class="stats"><div class="stat">最終類似<br><strong>{{'%.1f'|format(s['activation_similarity']*100)}}%</strong></div><div class="stat">共有Edge<br><strong>{{s['common_edges']}}</strong></div><div class="stat">固有Edge A/B<br><strong>{{s['left_only_edges']}} / {{s['right_only_edges']}}</strong></div></div><div class="read">{{s['reading']}}</div><details><summary>経路の一部を数値で見る</summary><div class="raw">共通: {{s['common_edge_sample']}}\nAのみ: {{s['left_edge_sample']}}\nBのみ: {{s['right_edge_sample']}}</div></details></article>{% endfor %}</div></section>
{% endfor %}
<section class="card"><h2>読み方</h2><p>緑の道は共通構造、橙と水色はそれぞれの経験だけが使った経路です。旧v2では橙・水色の道が途中にあっても、最終活性が100%へ揃いやすい。v2.1では前段階の経路を持ち越すため、関係・内容へ進んでも経験差が結果まで残ります。</p></section>
{% endif %}
</main></body></html>'''

@app.route('/', methods=['GET','POST'])
def index():
    ls=request.form.get('ls','犬'); lr=request.form.get('lr','種類'); lc=request.form.get('lc','動物')
    rs=request.form.get('rs','車'); rr=request.form.get('rr','種類'); rc=request.form.get('rc','人工物')
    result=None; error=''
    if request.method=='POST':
        try: result=viewer.build_view(ls,lr,lc,rs,rr,rc)
        except Exception as exc: error=f'{type(exc).__name__}: {exc}'
    return render_template_string(TEMPLATE,ls=ls,lr=lr,lc=lc,rs=rs,rr=rr,rc=rc,result=result,error=error)

if __name__=='__main__': app.run(host='127.0.0.1',port=5024,debug=False)
