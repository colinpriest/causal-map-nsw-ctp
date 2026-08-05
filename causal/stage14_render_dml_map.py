"""Stage 14 — render what the DoubleML estimates do and do not show about cause.

The LLM-elicited map (stage 11) is a graph: nodes, directed edges, and the evidence for
each. This one cannot be that, and the difference is the point of having both.

An earlier version of this page drew a star -- the target at the centre, an arrow from every
treatment -- and called it a causal map. It was not one. Every arrow's DIRECTION was
asserted by the analyst, not found by the method, and every magnitude was conditional on an
adjustment set that came from the DAG. Remove those two borrowed things and a bar chart of
adjusted associations is left, wearing arrowheads it had not earned.

So the star is gone. What replaces it is the one causal statement these estimates actually
support: SAME treatment, SAME outcome, SAME learner, SAME data, and the answer changes with
what you condition on -- with two treatments changing sign. That is a statement about
IDENTIFICATION, which is the causal part, and it is the reason the graph matters.

Each treatment is drawn twice, once per adjustment set:

    naive   adjust for every other column -- which conditions on mediators and on the
            collider at Lump Sum, so it answers a different question than it appears to
    dag     adjust for the treatment's parents in the assembled graph, the backdoor set

Where those two disagree, the disagreement is attributable to identification rather than
to estimation, since the learner and the data are held fixed.

Run:  python causal/stage14_render_dml_map.py
Out:  causal/ctp_identification_contrast.html
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "provenance" / "tabpfn_dml.json"
GRAPH = HERE / "provenance" / "banded_graph.json"
OUT = HERE / "ctp_identification_contrast.html"


def main() -> int:
    doc = json.loads(SRC.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH.read_text(encoding="utf-8")) if GRAPH.exists() else {}
    elicited = {(e["source"], e["target"]) for e in graph.get("edges", [])}

    rows = []
    for r in doc["results"]:
        t = r["treatment"]
        entry = dict(treatment=t, dag_parents=r.get("dag_parents", []),
                     in_elicited_graph=any(s == t for s, _ in elicited),
                     has_edge_to_target=(t, doc["target"]) in elicited,
                     estimates={})
        for kind in ("naive", "dag"):
            block = r.get(kind) or {}
            if "skipped" in block:
                entry["estimates"][kind] = dict(skipped=block["skipped"])
                continue
            per = {}
            for learner, v in block.items():
                if "error" in v or "ate" not in v or not v["ate"]:
                    per[learner] = dict(error=v.get("error", "no estimate"))
                    continue
                i = len(v["ate"]) - 1                      # top level vs reference
                lo, hi = v["ate_ci"][i]
                per[learner] = dict(
                    ate=round(v["ate"][i], 4), ci=[round(lo, 4), round(hi, 4)],
                    se=round(v["ate_se"][i], 4), seconds=v.get("seconds"),
                    significant=bool(lo > 0 or hi < 0),
                    levels=v["levels"], apo=[round(x, 3) for x in v["apo"]],
                    ate_all=[round(x, 4) for x in v["ate"]])
            entry["estimates"][kind] = per
        rows.append(entry)

    payload = dict(
        target=doc["target"], generated=doc["generated_at"],
        tabpfn_version=doc.get("tabpfn_version"), learners=doc["learners"],
        n_rows=doc["n_rows"], adjustment_sets=doc["adjustment_sets"],
        caveat=doc["caveat"], imputation=doc.get("imputation", ""),
        method=doc.get("method", ""), rows=rows,
        rendered=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    html = TEMPLATE.replace("__DATA__",
                            json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"treatments {len(rows)}   learners {doc['learners']}   "
          f"tabpfn {doc.get('tabpfn_version')}")
    for r in rows:
        n = r["estimates"].get("naive", {})
        d = r["estimates"].get("dag", {})
        tn = (n.get("TabPFN") or {}).get("ate")
        td = (d.get("TabPFN") or {}).get("ate")
        print(f"  {r['treatment']:32} naive={tn if tn is not None else '  --':>8} "
              f"dag={td if td is not None else '  --':>8}")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSW CTP — what conditioning changes</title>
<style>
:root{color-scheme:light dark;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--page:#f9f9f7;
--surface:#fff;--line:rgba(11,11,11,.12);--pos:#0b7d3e;--neg:#d03b3b;--band:rgba(11,11,11,.03)}
@media(prefers-color-scheme:dark){:root{--ink:#fff;--ink2:#c3c2b7;--page:#0d0d0d;
--surface:#1a1a19;--line:rgba(255,255,255,.14);--pos:#3fbf74;--neg:#e66767;--band:rgba(255,255,255,.04)}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px}
.wrap{max-width:1400px;margin:0 auto}h1{font-size:23px;margin:0 0 6px;letter-spacing:-.01em}
p.sub{margin:0;color:var(--ink2);max-width:92ch}
.warn{border:1px solid #fab219;border-left-width:4px;border-radius:6px;padding:10px 14px;
margin:14px 0;background:color-mix(in srgb,#fab219 11%,var(--surface));font-size:13.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.tile .v{font-size:22px;letter-spacing:-.02em}.tile .k{font-size:11.5px;color:var(--muted);
text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:22px}
.panel h2{font-size:15px;margin:0;padding:14px 18px 0}
.panel p.note{margin:4px 18px 10px;color:var(--ink2);font-size:13px}
.scroll{overflow-x:auto}
svg{display:block;width:100%;height:auto}
text{font:12px system-ui;fill:var(--ink)}text.small{font-size:10.5px;fill:var(--muted)}
rect.node{fill:var(--surface);stroke:var(--line);stroke-width:1.3}
g.node{cursor:pointer}g.node:hover rect{stroke:#2a78d6;stroke-width:2}
path.edge{fill:none;opacity:.75}path.edge:hover{opacity:1}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.sig{font-weight:600}.ns{color:var(--muted)}
code{background:var(--band);padding:1px 5px;border-radius:4px;font-size:.9em}
#detail{padding:14px 18px;font-size:13.5px;color:var(--ink2);min-height:70px;border-top:1px solid var(--line)}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);padding:0 18px 12px}
</style></head><body><div class="wrap">
<h1>NSW CTP — what conditioning changes</h1>
<p class="sub">Double machine learning estimates the effect of a treatment on
<code>log(Lump Sum)</code> — but only once someone has said which variable is the treatment and
what to adjust for. This page shows what changes when that second choice changes. An ATE of
0.10 is roughly a 10% change in the award.</p>
<div class="warn" id="warn"></div>
<div class="tiles" id="tiles"></div>

<div class="panel">
  <h2>The estimate moves when the adjustment set moves</h2>
  <p class="note">Same treatment, same outcome, same learner, same 540 rows. The only thing that
  changes between the two columns is <strong>what is conditioned on</strong> — every other
  variable on the left, the treatment&rsquo;s parents in the DAG on the right. Lines crossing
  zero changed sign. This is the causal content of the analysis: identification, not estimation.</p>
  <div class="legend" id="legend"></div>
  <div class="scroll"><svg id="map" viewBox="0 0 1420 640" preserveAspectRatio="xMidYMid meet"></svg></div>
  <div id="detail">Select a treatment.</div>
</div>

<div class="panel">
  <h2>The same thing as numbers</h2>
  <div class="scroll"><table id="cmp"></table></div>
</div>

<div class="panel">
  <h2>Learner comparison</h2>
  <p class="note">There is no ground truth here, unlike the DoubleML documentation example
  which uses synthetic data with oracle effects. A narrower interval is not necessarily a
  better estimate — on a misspecified adjustment set it is a confident wrong answer.</p>
  <div class="scroll"><table id="learn"></table></div>
</div>

<script id="d" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const SVG='http://www.w3.org/2000/svg',el=(t,a={})=>{const e=document.createElementNS(SVG,t);
for(const k in a)e.setAttribute(k,a[k]);return e};
const fmt=v=>v===undefined||v===null?'—':(v>=0?'+':'')+v.toFixed(3);

document.getElementById('warn').innerHTML=
 `<strong>These are not discovered causal relationships.</strong> Every direction here —
  treatment to outcome — was chosen by the analyst, and every magnitude depends on an
  adjustment set supplied by the DAG. ${D.caveat} What the page shows is the consequence of
  that choice, which is the part the method does speak to.`;

const rows=D.rows, est=(r,k,l)=>((r.estimates[k]||{})[l]||{});
const nSig=rows.filter(r=>est(r,'naive','TabPFN').significant).length;
const nDag=rows.filter(r=>est(r,'dag','TabPFN').significant).length;
[[rows.length,'treatments'],[D.learners.length,'learners'],[D.tabpfn_version||'—','tabpfn'],
 [nSig,'significant (naive)'],[nDag,'significant (dag)'],[D.n_rows,'rows']]
 .forEach(([v,k])=>{const t=document.createElement('div');t.className='tile';
  t.innerHTML=`<div class="v">${v}</div><div class="k">${k}</div>`;
  document.getElementById('tiles').appendChild(t)});

document.getElementById('legend').innerHTML=
 '<span><b style="color:var(--neg)">red</b> = the estimate changes sign</span>'+
 '<span>filled dot = 95% interval excludes zero</span>'+
 '<span>hollow dot = it does not</span>'+
 '<span>estimates shown for TabPFN-3</span>';

// Layout must fit the viewBox: nodes are 200 wide, so the furthest node centre can sit
// at most (1420/2 - 100 - margin) from the centre. An earlier version placed them at
// 1.55*R from a centre of 690 inside a 900px-min-width svg with no viewBox, which put the
// right-hand column past the edge where it was simply clipped.
const svg=document.getElementById('map');
// A slope chart, not a star. The earlier star drew an arrow from every treatment to the
// target and called itself a causal map -- but those directions were asserted by the
// analyst, so the picture claimed what the method had not established. What the estimates
// genuinely show is that the answer depends on the adjustment set, so that is what is drawn.
const L=430, R=1010, TOP=60, BOT=590;
const pairs=rows.map(r=>({name:r.treatment,
  a:est(r,'naive','TabPFN'), b:est(r,'dag','TabPFN'), parents:r.dag_parents}))
  .filter(p=>p.a.ate!==undefined||p.b.ate!==undefined);
const vals=pairs.flatMap(p=>[p.a.ate,p.b.ate]).filter(v=>v!==undefined);
const lo=Math.min(0,...vals), hi=Math.max(0,...vals), pad=(hi-lo)*0.08||0.1;
const Y=v=>BOT-((v-(lo-pad))/((hi+pad)-(lo-pad)))*(BOT-TOP);

[[L,'adjust for everything else','(no graph)'],[R,'adjust for the DAG parents','(backdoor set)']]
 .forEach(([x,t1,t2])=>{
  svg.appendChild(el('line',{x1:x,y1:TOP-16,x2:x,y2:BOT+10,stroke:'var(--line)','stroke-width':1.5}));
  const a=el('text',{x:x,y:TOP-34,'text-anchor':'middle'});a.textContent=t1;svg.appendChild(a);
  const b=el('text',{x:x,y:TOP-20,'text-anchor':'middle',class:'small'});b.textContent=t2;svg.appendChild(b)});

// zero line: the only value where a sign change is visible
svg.appendChild(el('line',{x1:L-60,y1:Y(0),x2:R+60,y2:Y(0),stroke:'var(--muted)',
  'stroke-dasharray':'5 4','stroke-width':1}));
const z=el('text',{x:L-72,y:Y(0)+4,'text-anchor':'end',class:'small'});
z.textContent='no effect';svg.appendChild(z);

const detail=document.getElementById('detail');
pairs.forEach(p=>{
 const flip=(p.a.ate!==undefined&&p.b.ate!==undefined&&Math.sign(p.a.ate)!==Math.sign(p.b.ate));
 const col=flip?'var(--neg)':'var(--ink2)';
 const g=el('g',{class:'node'});
 if(p.a.ate!==undefined&&p.b.ate!==undefined){
  g.appendChild(el('line',{x1:L,y1:Y(p.a.ate),x2:R,y2:Y(p.b.ate),stroke:col,
   'stroke-width':flip?3:1.6,opacity:flip?1:.6}))}
 [[L,p.a,'end',-10],[R,p.b,'start',10]].forEach(([x,v,anch,dx])=>{
  if(v.ate===undefined)return;
  g.appendChild(el('circle',{cx:x,cy:Y(v.ate),r:v.significant?6:4.5,
   fill:v.significant?col:'var(--surface)',stroke:col,'stroke-width':1.8}));
  const lab=el('text',{x:x+dx,y:Y(v.ate)+4,'text-anchor':anch,class:flip?'':'small'});
  lab.textContent=(anch==='end'?p.name+'  ':'')+fmt(v.ate);
  lab.setAttribute('fill',flip?'var(--neg)':'var(--ink2)');
  g.appendChild(lab)});
 g.addEventListener('click',()=>{
  detail.innerHTML=`<strong>${p.name}</strong>`+
   `<div style="margin-top:6px">adjusting for every other column: ATE ${fmt(p.a.ate)} `+
   (p.a.ci?`[${fmt(p.a.ci[0])}, ${fmt(p.a.ci[1])}]`:'')+
   (p.a.significant?' <span class="sig">significant</span>':' <span class="ns">not significant</span>')+
   ' — conditions on mediators and on the collider at the target</div>'+
   `<div>adjusting for ${p.parents.length?'the DAG parents ('+p.parents.join(', ')+')':'nothing — the graph makes this exogenous'}: `+
   (p.b.ate!==undefined?`ATE ${fmt(p.b.ate)} [${fmt(p.b.ci[0])}, ${fmt(p.b.ci[1])}]`:'not estimated')+'</div>'+
   (flip?'<div style="margin-top:6px;color:var(--neg)"><strong>Sign change.</strong> Nothing about the data or the learner differs — only what was conditioned on.</div>':'')});
 svg.appendChild(g)});

const cmp=document.getElementById('cmp');
cmp.innerHTML='<thead><tr><th>Treatment</th><th>DAG parents</th><th>Naive ATE</th>'+
 '<th>DAG ATE</th><th>Difference</th><th>Sign flip</th></tr></thead><tbody>'+
 rows.map(r=>{const nv=est(r,'naive','TabPFN'),dg=est(r,'dag','TabPFN');
  const diff=(nv.ate!==undefined&&dg.ate!==undefined)?dg.ate-nv.ate:undefined;
  const flip=(nv.ate!==undefined&&dg.ate!==undefined&&Math.sign(nv.ate)!==Math.sign(dg.ate));
  return `<tr><td>${r.treatment}</td><td class="small">${r.dag_parents.join(', ')||'—'}</td>
   <td class="num ${nv.significant?'sig':'ns'}">${fmt(nv.ate)}</td>
   <td class="num ${dg.significant?'sig':'ns'}">${fmt(dg.ate)}</td>
   <td class="num">${diff===undefined?'—':fmt(diff)}</td>
   <td>${flip?'<span style="color:var(--neg)">yes</span>':''}</td></tr>`}).join('')+'</tbody>';

const lt=document.getElementById('learn');
lt.innerHTML='<thead><tr><th>Treatment</th>'+D.learners.map(l=>`<th>${l} ATE</th><th>± CI width</th>`).join('')+
 '</tr></thead><tbody>'+rows.map(r=>`<tr><td>${r.treatment}</td>`+
 D.learners.map(l=>{const v=est(r,'naive',l);
  const w=v.ci?(v.ci[1]-v.ci[0]):undefined;
  return `<td class="num ${v.significant?'sig':'ns'}">${fmt(v.ate)}</td>
          <td class="num small">${w===undefined?'—':w.toFixed(3)}</td>`}).join('')+
 '</tr>').join('')+'</tbody>';
</script></div></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
