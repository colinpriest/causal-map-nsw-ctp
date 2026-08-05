"""Stage 14 — render the DoubleML causal map.

The LLM-elicited map (stage 11) is a graph: nodes, directed edges, and the evidence for
each. This one cannot be that, and the difference is the point of having both.

DoubleML estimates the effect of ONE treatment on ONE outcome, given a direction and an
adjustment set that it does not supply. Run it over every discrete column and you get a
star: the target at the centre, each treatment attached by an edge whose WIDTH and COLOUR
carry an estimated effect size and its uncertainty. There are no edges between treatments,
because nothing in the method produces one.

Each treatment is drawn twice, once per adjustment set:

    naive   adjust for every other column -- which conditions on mediators and on the
            collider at Lump Sum, so it answers a different question than it appears to
    dag     adjust for the treatment's parents in the assembled graph, the backdoor set

Where those two disagree, the disagreement is attributable to identification rather than
to estimation, since the learner and the data are held fixed.

Run:  python causal/stage14_render_dml_map.py
Out:  causal/ctp_tabpfn_dml_map.html
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "provenance" / "tabpfn_dml.json"
GRAPH = HERE / "provenance" / "banded_graph.json"
OUT = HERE / "ctp_tabpfn_dml_map.html"


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
<title>NSW CTP — DoubleML + TabPFN causal map</title>
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
<h1>NSW CTP — DoubleML + TabPFN causal map</h1>
<p class="sub">Average Treatment Effect of each discrete column on <code>log(Lump Sum)</code>,
estimated by double machine learning. An ATE of 0.10 is roughly a 10% change in the award.</p>
<div class="warn" id="warn"></div>
<div class="tiles" id="tiles"></div>

<div class="panel">
  <h2>The map</h2>
  <p class="note">A star, not a graph. Edge width is |ATE|, colour is direction, solid means the
  95% interval excludes zero. There are no edges between treatments because DoubleML does not
  produce any — it estimates one treatment against one outcome at a time.</p>
  <div class="legend" id="legend"></div>
  <div class="scroll"><svg id="map" viewBox="0 0 1420 760" preserveAspectRatio="xMidYMid meet"></svg></div>
  <div id="detail">Select a treatment.</div>
</div>

<div class="panel">
  <h2>Naive vs DAG adjustment</h2>
  <p class="note">Same learner, same data, same treatment — only the adjustment set differs.
  The gap is attributable to identification, not estimation.</p>
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
 `<strong>This is not a causal graph.</strong> DoubleML estimates an effect size given a
  direction and an adjustment set that it cannot supply. ${D.caveat}`;

const rows=D.rows, est=(r,k,l)=>((r.estimates[k]||{})[l]||{});
const nSig=rows.filter(r=>est(r,'naive','TabPFN').significant).length;
const nDag=rows.filter(r=>est(r,'dag','TabPFN').significant).length;
[[rows.length,'treatments'],[D.learners.length,'learners'],[D.tabpfn_version||'—','tabpfn'],
 [nSig,'significant (naive)'],[nDag,'significant (dag)'],[D.n_rows,'rows']]
 .forEach(([v,k])=>{const t=document.createElement('div');t.className='tile';
  t.innerHTML=`<div class="v">${v}</div><div class="k">${k}</div>`;
  document.getElementById('tiles').appendChild(t)});

document.getElementById('legend').innerHTML=
 '<span><b style="color:var(--pos)">green</b> = raises the award</span>'+
 '<span><b style="color:var(--neg)">red</b> = lowers it</span>'+
 '<span>solid = 95% CI excludes zero · dashed = does not</span>'+
 '<span>width ∝ |ATE|</span>';

// Layout must fit the viewBox: nodes are 200 wide, so the furthest node centre can sit
// at most (1420/2 - 100 - margin) from the centre. An earlier version placed them at
// 1.55*R from a centre of 690 inside a 900px-min-width svg with no viewBox, which put the
// right-hand column past the edge where it was simply clipped.
const svg=document.getElementById('map'),CX=710,CY=380;
const n=rows.length, RX=580, RY=300;
svg.appendChild(el('rect',{x:CX-95,y:CY-24,width:190,height:48,rx:9,
  fill:'var(--band)',stroke:'var(--line)','stroke-width':2}));
const ct=el('text',{x:CX,y:CY+5,'text-anchor':'middle'});ct.textContent=D.target;
svg.appendChild(ct);
const detail=document.getElementById('detail');
const maxAte=Math.max(...rows.map(r=>Math.abs(est(r,'naive','TabPFN').ate||0)),0.05);

rows.forEach((r,i)=>{
 const ang=(i/n)*2*Math.PI-Math.PI/2;
 const x=CX+RX*Math.cos(ang), y=CY+RY*Math.sin(ang);
 const e=est(r,'naive','TabPFN'), a=e.ate;
 if(a!==undefined){
  const w=1+5*Math.abs(a)/maxAte;
  const p=el('path',{d:`M${x},${y} Q${(x+CX)/2},${(y+CY)/2} ${CX+(x>CX?95:-95)},${CY}`,
   class:'edge',stroke:a>=0?'var(--pos)':'var(--neg)','stroke-width':w});
  if(!e.significant)p.setAttribute('stroke-dasharray','6 4');
  svg.appendChild(p)}
 const g=el('g',{class:'node'});
 g.appendChild(el('rect',{x:x-100,y:y-19,width:200,height:38,rx:7,class:'node'}));
 const t=el('text',{x:x,y:y-2,'text-anchor':'middle'});
 t.textContent=r.treatment.length>27?r.treatment.slice(0,26)+'…':r.treatment;g.appendChild(t);
 const s=el('text',{x:x,y:y+12,'text-anchor':'middle',class:'small'});
 s.textContent=a===undefined?'not estimable':`ATE ${fmt(a)}`;g.appendChild(s);
 g.addEventListener('click',()=>{
  const nv=est(r,'naive','TabPFN'), dg=est(r,'dag','TabPFN');
  detail.innerHTML=`<strong>${r.treatment}</strong>`+
   `<div style="margin-top:6px">naive adjustment (all other columns): ATE ${fmt(nv.ate)} `+
   (nv.ci?`[${fmt(nv.ci[0])}, ${fmt(nv.ci[1])}]`:'')+
   (nv.significant?' <span class="sig">significant</span>':' <span class="ns">not significant</span>')+'</div>'+
   `<div>DAG adjustment (${r.dag_parents.length?r.dag_parents.join(', '):'no parents in the graph'}): `+
   (dg.ate!==undefined?`ATE ${fmt(dg.ate)} [${fmt(dg.ci[0])}, ${fmt(dg.ci[1])}]`:'not estimated')+'</div>'+
   (nv.apo?`<div class="small" style="margin-top:6px">APO by level: ${nv.apo.join(' · ')}</div>`:'')});
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
