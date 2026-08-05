"""Stage 11 — render the assembled graph as a single self-contained page for review.

This exists to be argued with. Everything on it is traceable: click an edge and you get
the provision quoted, or the mechanism reasoned and the prediction it survived. Nothing is
drawn that has no evidence behind it.

Edges carry an effect estimate where the graph identifies one (stage 15, `--edges`):
colour for sign, dashing for whether the 95% interval excludes zero, and the number itself
on the edge. An edge with no estimate is drawn grey and unlabelled rather than being given a
default, because "not estimated" and "estimated at zero" are different things and a reader
must be able to tell them apart.

The magnitudes are NOT comparable across edges. A discrete treatment reports a top-vs-base
level contrast; a continuous one reports a per-unit coefficient. Sign and significance mean
the same thing throughout; the numbers do not, and each is labelled with its kind.

Two more things the page deliberately shows that a causal diagram usually hides:

  WHAT IS MISSING. Band order permits 130 ordered pairs; a handful have evidence. Drawing
  only the survivors makes a sparse graph look like a complete one, so the coverage of
  each band transition is stated outright.

  HOW GOOD EACH EDGE IS. Statute, a tested reasoned prior, and a measurement assumption
  are not the same kind of claim and are not drawn the same way.

Run:  python causal/stage11_render_dag.py
Out:  causal/ctp_reviewed_dag.html
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "provenance"
OUT = HERE / "ctp_reviewed_dag.html"

CLASS_STYLE = {
    "statute": ("#0b7d3e", "solid", "Statute — provision quoted verbatim"),
    "reasoned_prior_tested": ("#2a78d6", "solid", "Reasoned prior that passed its own "
                                                  "pre-registered prediction"),
    "measurement": ("#898781", "dashed", "Measurement — an indicator and the latent "
                                         "quantity it traces"),
}
BAND_X = {1: 90, 2: 310, 3: 530, 4: 750, 5: 970, 6: 1180}


def main() -> int:
    g = json.loads((P / "banded_graph.json").read_text(encoding="utf-8"))

    # effect per edge, where the graph identifies one
    eff_path = P / "dag_edge_effects.json"
    effects = {}
    if eff_path.exists():
        ed = json.loads(eff_path.read_text(encoding="utf-8"))
        for e in ed["effects"]:
            tot = e.get("total", {})
            if "ate" in tot:
                ci = tot.get("ci") or [float("nan")] * 2
                effects[(e["treatment"], e["outcome"])] = dict(
                    ate=round(tot["ate"], 3),
                    ci=[None if c != c else round(c, 3) for c in ci],
                    significant=bool(tot.get("significant")),
                    estimand=tot.get("estimand", ""),
                    n=tot.get("n"), backdoor=e.get("backdoor_set", []))
    for e in g["edges"]:
        e["effect"] = effects.get((e["source"], e["target"]))
    latent = set(g.get("latent", []))
    roles = g["roles"]

    # vertical placement within each band
    by_band = defaultdict(list)
    for col, r in roles.items():
        by_band[r["band"]].append(col)
    pos = {}
    for band, cols in by_band.items():
        cols.sort()
        step = 620 / (len(cols) + 1)
        for i, c in enumerate(cols):
            pos[c] = (BAND_X[band], 70 + step * (i + 1))

    # coverage per band transition
    cover = defaultdict(lambda: [0, 0])
    names = list(roles)
    for a in names:
        for b in names:
            if a == b or roles[a]["band"] > roles[b]["band"]:
                continue
            if roles[a]["band"] == roles[b]["band"] and a >= b:
                continue
            cover[(roles[a]["band"], roles[b]["band"])][1] += 1
    for e in g["edges"]:
        cover[(roles[e["source"]]["band"], roles[e["target"]]["band"])][0] += 1

    payload = dict(
        bands=g["bands"], roles=roles, edges=g["edges"], latent=sorted(latent),
        violations=g.get("violations", []), pos={k: list(v) for k, v in pos.items()},
        coverage=[dict(frm=k[0], to=k[1], have=v[0], permitted=v[1])
                  for k, v in sorted(cover.items()) if v[1]],
        n_permitted=g["n_permitted"], n_edges=g["n_edges"],
        n_with_effect=sum(1 for e in g["edges"] if e.get("effect")),
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        classes={k: dict(colour=v[0], dash=v[1], label=v[2]) for k, v in CLASS_STYLE.items()},
    )

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))
                            .replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"nodes {len(roles)} ({len(latent)} latent)   edges {g['n_edges']}   "
          f"permitted pairs {g['n_permitted']}")
    print("\ncoverage by band transition:")
    for c in payload["coverage"]:
        print(f"  band {c['frm']} -> {c['to']}   {c['have']:2d} of {c['permitted']:3d} "
              f"permitted pairs have evidence")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSW CTP — assembled causal graph for review</title>
<style>
:root{color-scheme:light dark;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--page:#f9f9f7;--surface:#fff;--line:rgba(11,11,11,.12);--band:rgba(11,11,11,.03)}
@media(prefers-color-scheme:dark){:root{--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--page:#0d0d0d;--surface:#1a1a19;--line:rgba(255,255,255,.14);--band:rgba(255,255,255,.04)}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px}
.wrap{max-width:1420px;margin:0 auto}h1{font-size:23px;margin:0 0 6px;letter-spacing:-.01em}
p.sub{margin:0 0 4px;color:var(--ink2);max-width:90ch}
.warn{border:1px solid #fab219;border-left-width:4px;border-radius:6px;padding:10px 14px;
margin:14px 0;background:color-mix(in srgb,#fab219 11%,var(--surface));font-size:13.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:11px 13px}
.tile .v{font-size:24px;letter-spacing:-.02em}.tile .k{font-size:11.5px;color:var(--muted);
text-transform:uppercase;letter-spacing:.05em;margin-top:2px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink2);margin:10px 0}
.legend i{display:inline-block;width:26px;height:0;border-top-width:2.5px;vertical-align:middle;margin-right:6px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.scroll{overflow-x:auto}svg{display:block;min-width:1290px}
text{font:12px system-ui;fill:var(--ink)}
text.small{font-size:10.5px;fill:var(--muted)}
text.bandlab{font-size:11px;font-weight:600;fill:var(--muted);text-transform:uppercase;letter-spacing:.06em}
rect.node{fill:var(--surface);stroke:var(--line);stroke-width:1.4}
rect.node.latent{stroke-dasharray:5 3;stroke-width:1.8;fill:none}
g.node{cursor:pointer}g.node:hover rect{stroke:#2a78d6;stroke-width:2.2}
path.edge{fill:none;opacity:.62}path.edge:hover{opacity:1;stroke-width:4}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
code{background:var(--band);padding:1px 5px;border-radius:4px;font-size:.9em}
#detail{padding:14px 18px;font-size:13.5px;color:var(--ink2);min-height:78px;border-top:1px solid var(--line)}
h2{font-size:15px;margin:26px 0 8px}
blockquote{margin:6px 0;padding:6px 12px;border-left:3px solid var(--line);color:var(--ink2);font-size:13px}
</style></head><body><div class="wrap">
<h1>NSW CTP — assembled causal graph</h1>
<p class="sub">Every edge is backed by evidence and traceable. Click any edge or node.
Chronology runs left to right and no edge may run backwards.</p>
<div class="warn" id="sparse"></div>
<div class="tiles" id="tiles"></div>
<div class="legend" id="legend"></div>
<div class="panel"><div class="scroll"><svg id="map" height="760"></svg></div>
<div id="detail">Select an edge or a node.</div></div>
<h2>Coverage — how much of the permitted space has evidence</h2>
<div class="panel"><table id="cov"></table></div>
<h2>Directions corrected by chronology</h2>
<div class="panel"><table id="viol"></table></div>
<script id="d" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('d').textContent);
const SVG='http://www.w3.org/2000/svg', el=(t,a={})=>{const e=document.createElementNS(SVG,t);
for(const k in a)e.setAttribute(k,a[k]);return e};
document.getElementById('sparse').innerHTML=
 `<strong>This graph is sparse by construction.</strong> Chronology permits ${D.n_permitted}
  ordered pairs; ${D.n_edges} have evidence. Absence of an edge here means <em>nothing was
  found</em>, not that no relationship exists — most pairs have simply not been examined yet.`;
[[Object.keys(D.roles).length,'nodes'],[D.latent.length,'latent'],[D.n_edges,'edges with evidence'],
 [D.n_with_effect,'edges with an effect estimate'],
 [D.n_permitted,'pairs permitted'],[D.violations.length,'directions corrected']]
 .forEach(([v,k])=>{const t=document.createElement('div');t.className='tile';
  t.innerHTML=`<div class="v">${v}</div><div class="k">${k}</div>`;
  document.getElementById('tiles').appendChild(t)});
const lg=document.getElementById('legend');
for(const [k,c] of Object.entries(D.classes)){const s=document.createElement('span');
 s.innerHTML=`<i style="border-top-style:${c.dash};border-color:${c.colour}"></i>${c.label}`;
 lg.appendChild(s)}
[['<i style="border-top-style:solid;border-color:#0b7d3e"></i>raises the outcome'],
 ['<i style="border-top-style:solid;border-color:#d03b3b"></i>lowers it'],
 ['<i style="border-top-style:solid;border-color:#a8a69e"></i>no estimate available'],
 ['solid = 95% CI excludes zero · dashed = includes zero, or not estimated'],
 ['number on the edge = the estimate; magnitudes are <b>not</b> comparable across edges'],
 ['<i style="border-top-style:dashed;border-color:var(--muted)"></i>dashed box = unobserved (no column)']]
 .forEach(([h])=>{const s=document.createElement('span');s.innerHTML=h;lg.appendChild(s)});

const svg=document.getElementById('map'), P=D.pos;
const W=176,H=38;
D.bands.forEach(b=>{const xs=b.columns.map(c=>P[c]&&P[c][0]).filter(Boolean);
 if(!xs.length)return;const x=xs[0];
 svg.appendChild(el('rect',{x:x-W/2-14,y:44,width:W+28,height:700,fill:'var(--band)',rx:8}));
 const t=el('text',{x:x,y:32,class:'bandlab','text-anchor':'middle'});
 t.textContent=`${b.n}. ${b.label}`;svg.appendChild(t)});

const defs=el('defs');svg.appendChild(defs);
[['pos','#0b7d3e'],['neg','#d03b3b'],['none','#a8a69e']].forEach(([k,c])=>{
 const m=el('marker',{id:'a-'+k,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,
  markerHeight:7,orient:'auto-start-reverse'});
 m.appendChild(el('path',{d:'M0,0 L10,5 L0,10 z',fill:c}));defs.appendChild(m)});

const detail=document.getElementById('detail');
const POS='#0b7d3e', NEG='#d03b3b', NONE='#a8a69e';
D.edges.forEach(e=>{const [x1,y1]=P[e.source],[x2,y2]=P[e.target];
 const cls=e.evidence.some(v=>v.evidence_class==='statute')?'statute':
   (e.evidence[0].evidence_class);
 const st=D.classes[cls]||D.classes.reasoned_prior_tested;
 const sx=x1+W/2, ex=x2-W/2, mx=(sx+ex)/2;
 const same=Math.abs(x1-x2)<2;
 const d=same?`M${x1+W/2},${y1} C${x1+W/2+70},${y1} ${x2+W/2+70},${y2} ${x2+W/2},${y2}`
             :`M${sx},${y1} C${mx},${y1} ${mx},${y2} ${ex},${y2}`;
 // colour = sign of the estimated effect, dash = 95% interval includes zero.
 // No estimate means grey and unlabelled: "not estimated" is not "zero".
 const ef=e.effect;
 const colour=ef?(ef.ate>=0?POS:NEG):NONE;
 const p=el('path',{d:d,class:'edge',stroke:colour,'stroke-width':
   ef?(ef.significant?3.2:2):1.6,'marker-end':'url(#a-'+(ef?(ef.ate>=0?'pos':'neg'):'none')+')'});
 if(!ef||!ef.significant)p.setAttribute('stroke-dasharray','6 4');
 p.addEventListener('click',()=>{
  const efd=e.effect;
  detail.innerHTML=`<strong>${e.source} → ${e.target}</strong>`+
   (efd?`<div style="margin-top:6px">effect <strong style="color:${efd.ate>=0?POS:NEG}">`+
     `${efd.ate>=0?'+':''}${efd.ate}</strong>`+
     (efd.ci[0]!==null?` [${efd.ci[0]}, ${efd.ci[1]}]`:'')+
     (efd.significant?' <span class="sig">95% CI excludes zero</span>':
      ' <span class="ns">95% CI includes zero</span>')+
     `<div class="small">${efd.estimand} · n=${efd.n} · adjusted for `+
     `${efd.backdoor.length?efd.backdoor.join(', '):'nothing (exogenous)'}</div></div>`
    :'<div class="small" style="margin-top:6px">no effect estimate: the graph does not identify one, or the source is latent</div>')+
   e.evidence.map(v=>{
   let s=`<div style="margin-top:8px"><code>${v.evidence_class}</code>`;
   if(v.direction_corrected)s+=` <span style="color:#d03b3b">direction corrected — originally claimed ${v.originally_claimed}</span>`;
   if(v.provision)s+=` · s ${v.provision} · <em>${v.relation}</em>`;
   if(v.concept)s+=`<blockquote>concept: ${v.concept}<br>“${v.quote}”</blockquote>`;
   if(v.mechanism)s+=`<blockquote>${v.mechanism}</blockquote>`;
   if(v.confidence)s+=`<div class="small">confidence ${v.confidence}/5 · magnitude ${v.magnitude}/5 · ${Math.round(v.agreement*100)}% sample agreement · test: ${v.test}</div>`;
   if(v.note)s+=`<blockquote>${v.note}</blockquote>`;
   return s+'</div>'}).join('')});
 svg.appendChild(p);
 if(ef){                                    // the number itself, on the edge
  const lx=same?(x1+W/2+58):((sx+ex)/2), ly=same?((y1+y2)/2):((y1+y2)/2-7);
  const bg=el('rect',{x:lx-25,y:ly-9,width:50,height:15,rx:4,
   fill:'var(--surface)',stroke:colour,'stroke-width':1,opacity:.95});
  const tx=el('text',{x:lx,y:ly+3,'text-anchor':'middle',class:'small'});
  tx.textContent=(ef.ate>=0?'+':'')+ef.ate.toFixed(2);
  tx.setAttribute('fill',colour);
  tx.setAttribute('font-weight',ef.significant?'600':'400');
  svg.appendChild(bg);svg.appendChild(tx)}
 });

Object.entries(D.roles).forEach(([c,r])=>{const [x,y]=P[c];
 const g=el('g',{class:'node'});
 const isLat=D.latent.includes(c);
 g.appendChild(el('rect',{x:x-W/2,y:y-H/2,width:W,height:H,rx:7,
  class:'node'+(isLat?' latent':'')}));
 const t=el('text',{x:x,y:y-1,'text-anchor':'middle'});
 t.textContent=c.length>26?c.slice(0,25)+'…':c;g.appendChild(t);
 const s=el('text',{x:x,y:y+12,'text-anchor':'middle',class:'small'});
 s.textContent=isLat?'unobserved':r.role;g.appendChild(s);
 g.addEventListener('click',()=>{
  const ins=D.edges.filter(e=>e.target===c).map(e=>e.source);
  const outs=D.edges.filter(e=>e.source===c).map(e=>e.target);
  detail.innerHTML=`<strong>${c}</strong> · band ${r.band} ${r.band_label} · role <code>${r.role}</code>`+
   (isLat?'<div class="small">Unobserved: carried in the graph, no column, no values.</div>':
    `<div class="small">marginal with award ${r.marginal_with_target ?? '—'} · unique contribution ${r.unique_contribution_to_target ?? '—'}</div>`)+
   `<div style="margin-top:8px">causes: ${outs.join(', ')||'<em>nothing with evidence</em>'}</div>`+
   `<div>caused by: ${ins.join(', ')||'<em>nothing with evidence</em>'}</div>`});
 svg.appendChild(g)});

const cov=document.getElementById('cov');
cov.innerHTML='<thead><tr><th>Transition</th><th>Edges found</th><th>Pairs permitted</th><th>Coverage</th></tr></thead><tbody>'+
 D.coverage.map(c=>`<tr><td>band ${c.frm} → ${c.to}</td><td>${c.have}</td><td>${c.permitted}</td>
  <td>${(100*c.have/c.permitted).toFixed(0)}%</td></tr>`).join('')+'</tbody>';
const vt=document.getElementById('viol');
vt.innerHTML='<thead><tr><th>Claimed</th><th>Bands</th><th>Resolution</th></tr></thead><tbody>'+
 (D.violations.length?D.violations.map(v=>`<tr><td>${v.source} → ${v.target}</td>
  <td>b${v.band_source} → b${v.band_target}</td><td>${v.resolution}</td></tr>`).join('')
  :'<tr><td colspan="3"><em>none</em></td></tr>')+'</tbody>';
</script></div></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
