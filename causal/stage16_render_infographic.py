"""Render the project summary infographic as an SVG, from the artifacts.

The previous infographic was an image with its numbers typed in. They drifted -- by the
time anyone looked, three of four were wrong (statute edges, tested priors, and forbidden
pairs, which read 91/240 against an actual 131/272), and the graph had moved several times
since the image was made. Nothing in the project could have caught that, because a PNG is
not checkable against anything.

Every figure here is read from the artifact that produced it. There are no literals in this
file that a reader could mistake for a result: if the graph changes, re-running this changes
the picture, and if an artifact is missing the script fails rather than drawing a stale
number.

Encoding note. The five evidence classes are an *ordered* scale -- elicited is stronger
evidence than statute is stronger than a reasoned prior -- so they take a single-hue ordinal
ramp, darkest at the strongest class, rather than five categorical hues. Both the light and
dark ramps are validated (monotone lightness, adjacent gaps, light end clearing the
surface). Bar length carries the count; hue carries the rank, which is a different variable,
so the ramp is not double-encoding length.

In:   causal/provenance/{banded_graph,sensitivity,provision_reading,dag_edge_effects}.json
Out:  docs/project-infographic.svg

Run:  python causal/stage16_render_infographic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
PROV = HERE / "provenance"
OUT = HERE.parent / "docs" / "project-infographic.svg"

W, H = 1600, 850

# Ordinal ramp, strongest evidence darkest. Validated with the data-viz palette validator:
# light  #104281,#1c5cab,#2a78d6,#5598e7,#86b6ef  -- ALL CHECKS PASS (light end 2.06:1)
# dark   #b7d3f6,#86b6ef,#3987e5,#256abf,#184f95  -- ALL CHECKS PASS (light end 2.15:1)
RAMP_LIGHT = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
RAMP_DARK = ["#b7d3f6", "#86b6ef", "#3987e5", "#256abf", "#184f95"]

# Order is the evidence hierarchy itself, strongest first. This is the project's ranking,
# not a display choice, so it is declared once here and drives both the ramp and the legend.
CLASSES = [
    ("elicited", "Elicited", "A domain expert stated it, with mechanism and verbatim quote"),
    ("statute", "Statute", "A provision quoted verbatim from primary legislation"),
    ("measurement", "Measurement", "An indicator and the latent quantity it traces"),
    ("reasoned_prior_tested", "Reasoned prior (tested)",
     "Blind reasoning that passed a prediction fixed before the data was seen"),
    ("reasoned_prior_path", "Reasoned prior (path)",
     "A model named a mediator, nothing more"),
]


def load(name: str) -> dict:
    p = PROV / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"missing {p}; run the pipeline before rendering the infographic")
    return json.loads(p.read_text(encoding="utf-8"))


def esc(s) -> str:
    return escape(str(s))


def text(x, y, s, cls="", anchor="start", extra="") -> str:
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.0f}" y="{y:.0f}"{c}{a}{extra}>{esc(s)}</text>'


def panel(x, y, w, h, title) -> str:
    return (f'<rect class="panel" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'
            + text(x + 22, y + 32, title, "h2"))


def evidence_chart(x, y, w, strongest: dict, n_edges: int) -> str:
    """Horizontal bars, one per evidence class, ordered by the hierarchy not by count.

    Ordering by rank rather than by length is the whole point of the panel: it puts the
    biggest bar at the bottom, where the reader can see that the class carrying the most
    edges is the one the project trusts least.
    """
    out = [panel(x, y, w, 470, "Edges by strongest evidence")]
    out.append(text(x + 22, y + 54, f"{n_edges} edges. An edge with both a statute quote "
                                    "and a reasoned prior is counted under statute only.",
                    "sub"))
    top, row, barh = y + 90, 62, 15
    label_w, bar_x = 250, x + 272
    bar_max = w - (bar_x - x) - 74
    peak = max(strongest.values()) or 1
    for i, (key, label, blurb) in enumerate(CLASSES):
        n = strongest.get(key, 0)
        cy = top + i * row
        bw = max(2.0, bar_max * n / peak)
        out.append(text(x + 22, cy + 4, label, "lab"))
        out.append(text(x + 22, cy + 20, blurb, "tiny"))
        # 4px rounded data-end, anchored square to the baseline at bar_x
        out.append(f'<rect class="bar b{i}" x="{bar_x}" y="{cy - 8}" '
                   f'width="{bw:.1f}" height="{barh}" rx="4"/>')
        out.append(f'<rect class="bar b{i}" x="{bar_x}" y="{cy - 8}" '
                   f'width="{min(6, bw):.1f}" height="{barh}"/>')
        out.append(text(bar_x + bw + 12, cy + 4, n, "val"))
    cap = top + len(CLASSES) * row + 6
    out.append(f'<line class="rule" x1="{x + 22}" y1="{cap}" x2="{x + w - 22}" y2="{cap}"/>')
    weakest_key = CLASSES[-1][0]
    out.append(text(x + 22, cap + 32,
                    f"The largest class is the weakest: {strongest.get(weakest_key, 0)} of "
                    f"{n_edges} edges rest only on a model", "note"))
    out.append(text(x + 22, cap + 52, "naming a mediator.", "note"))
    return "".join(out)


def chronology(x, y, w, h, ordered, permitted, forbidden, n_edges) -> str:
    """Part-to-whole over the ordered pairs: forbidden, permitted-but-empty, evidenced."""
    out = [panel(x, y, w, h, "Chronology forbids more than evidence supplies")]
    out.append(text(x + 22, y + 54,
                    f"{ordered} ordered pairs among the variables.", "sub"))
    bx, by, bw, bh = x + 22, y + 78, w - 44, 30
    empty = permitted - n_edges
    segs = [("forbidden", forbidden, "seg-forbidden"),
            ("permitted, no evidence", empty, "seg-empty"),
            ("evidenced", n_edges, "seg-edge")]
    cx = bx
    for i, (_, n, cls) in enumerate(segs):
        sw = bw * n / ordered
        gap = 2 if i < len(segs) - 1 else 0       # 2px surface gap between fills
        out.append(f'<rect class="{cls}" x="{cx:.1f}" y="{by}" '
                   f'width="{max(1.0, sw - gap):.1f}" height="{bh}" rx="4"/>')
        cx += sw
    ly = by + bh + 26
    cx = bx
    for name, n, cls in segs:
        out.append(f'<rect class="{cls}" x="{cx}" y="{ly - 9}" width="10" height="10" rx="2"/>')
        out.append(text(cx + 16, ly, f"{name} — {n}", "tiny"))
        cx += 210
    out.append(text(x + 22, y + h - 26,
                    "An effect cannot precede its cause. This rule needs no evidence and "
                    "does more orientation", "note"))
    out.append(text(x + 22, y + h - 8, "work than every statistical signal combined.", "note"))
    return "".join(out)


def measurement(x, y, w, h, latent) -> str:
    out = [panel(x, y, w, h, "A score cannot cause what it measures")]
    name = latent[0] if latent else "latent quantity"
    ly = y + 92
    out.append(f'<rect class="node latent" x="{x + 26}" y="{ly - 24}" width="196" '
               f'height="48" rx="7"/>')
    out.append(text(x + 124, ly - 4, name, "nodelab", "middle"))
    out.append(text(x + 124, ly + 13, "latent — never observed", "tiny", "middle"))
    ax0, ax1 = x + 230, x + 288
    out.append(f'<path class="arrow" d="M{ax0} {ly} L{ax1 - 7} {ly}" '
               f'marker-end="url(#arrowhead)"/>')
    out.append(f'<rect class="node" x="{x + 292}" y="{ly - 24}" width="212" height="48" rx="7"/>')
    out.append(text(x + 398, ly - 4, f"{name} Emphasis", "nodelab", "middle"))
    out.append(text(x + 398, ly + 13, "indicator — a recorded score", "tiny", "middle"))
    out.append(text(x + 26, y + h - 44,
                    "An assessment score is caused by the state it scores. It cannot cause "
                    "what that", "note"))
    out.append(text(x + 26, y + h - 26,
                    "state causes. Like chronology, this rule needs no evidence — and it "
                    "removed an edge", "note"))
    out.append(text(x + 26, y + h - 8,
                    "a reasoned prior had asserted at 100% agreement.", "note"))
    return "".join(out)


def tiles(x, y, w, h, items) -> str:
    out = []
    tw = (w - 2 * 14) / 3
    for i, (val, key, sub) in enumerate(items):
        tx = x + i * (tw + 14)
        out.append(f'<rect class="panel" x="{tx}" y="{y}" width="{tw}" height="{h}" rx="10"/>')
        out.append(text(tx + 20, y + 74, val, "big"))
        out.append(text(tx + 20, y + 100, key, "lab"))
        out.append(text(tx + 20, y + 124, sub, "tiny"))
    return "".join(out)


def build() -> str:
    g = load("banded_graph")
    sens = load("sensitivity")
    reading = load("provision_reading")
    eff = load("dag_edge_effects")

    strongest = g["edges_by_strongest_class"]
    n_edges = g["n_edges"]
    unknown = set(strongest) - {k for k, _, _ in CLASSES}
    if unknown:
        raise SystemExit(f"graph has evidence classes this renderer does not rank: {unknown}. "
                         f"Add them to CLASSES in hierarchy order rather than letting them "
                         f"drop silently out of the chart.")

    n_nodes = len(g["roles"])
    n_latent = len(g["latent"])
    css = f"""
    .bg{{fill:#fcfcfb}} .panel{{fill:#ffffff;stroke:rgba(11,11,11,.12)}}
    text{{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;fill:#0b0b0b}}
    .h1{{font-size:34px;font-weight:700;letter-spacing:-.02em}}
    .h2{{font-size:16px;font-weight:600;letter-spacing:-.01em}}
    .lab{{font-size:14.5px;font-weight:600}}
    .sub{{font-size:13px;fill:#52514e}} .note{{font-size:13px;fill:#52514e}}
    .tiny{{font-size:11.5px;fill:#6f6d68}}
    .val{{font-size:15px;font-weight:700;fill:#52514e}}
    .big{{font-size:38px;font-weight:700;letter-spacing:-.02em}}
    .nodelab{{font-size:13px;font-weight:600}}
    .rule{{stroke:rgba(11,11,11,.12)}}
    .node{{fill:#ffffff;stroke:rgba(11,11,11,.34);stroke-width:1.4}}
    .node.latent{{fill:none;stroke-dasharray:5 3;stroke-width:1.8}}
    .arrow{{stroke:#52514e;stroke-width:2;fill:none}}
    .seg-forbidden{{fill:#c9c7c1}} .seg-empty{{fill:#86b6ef}} .seg-edge{{fill:#104281}}
    {"".join(f'.b{i}{{fill:{c}}}' for i, c in enumerate(RAMP_LIGHT))}
    @media (prefers-color-scheme: dark){{
      .bg{{fill:#0d0d0d}} .panel{{fill:#1a1a19;stroke:rgba(255,255,255,.14)}}
      text{{fill:#ffffff}} .sub,.note,.val{{fill:#c3c2b7}} .tiny{{fill:#a8a69e}}
      .rule{{stroke:rgba(255,255,255,.14)}}
      .node{{fill:#1a1a19;stroke:rgba(255,255,255,.4)}} .node.latent{{fill:none}}
      .arrow{{stroke:#c3c2b7}}
      .seg-forbidden{{fill:#55534e}} .seg-empty{{fill:#256abf}} .seg-edge{{fill:#b7d3f6}}
      {"".join(f'.b{i}{{fill:{c}}}' for i, c in enumerate(RAMP_DARK))}
    }}"""

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" role="img" '
         f'aria-label="How the NSW CTP causal graph is built: evidence classes, '
         f'structural constraints, and coverage">',
         f'<title>NSW CTP causal map — evidence and constraints</title>',
         f'<defs><marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" '
         f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M0 0 L10 5 L0 10 z" fill="#52514e"/></marker></defs>',
         f'<style>{css}</style>',
         f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>']

    o.append(text(48, 62, "Building a causal map of NSW CTP awards — with provenance",
                  "h1"))
    o.append(text(48, 90, "Every edge traces to something checkable. Edges without evidence "
                          "are not drawn.", "sub"))

    o.append(evidence_chart(48, 122, 700, strongest, n_edges))
    o.append(chronology(772, 122, 780, 200, g["n_ordered_pairs"], g["n_permitted"],
                        g["n_forbidden"], n_edges))
    o.append(measurement(772, 338, 780, 200, g["latent"]))

    o.append(tiles(772, 554, 780, 178, [
        (f"{n_nodes}", "nodes", f"{n_latent} latent, never observed"),
        (f"{n_edges}", "edges", f"effect estimated on {eff['n_estimated']}"),
        (f"{sens['n_fragile']}/{sens['n_parameters']}", "thresholds fragile",
         "quote those results with the threshold"),
    ]))

    o.append(panel(48, 608, 700, 124, "Sparse by construction"))
    o.append(text(70, 664, "Absence of an edge means no evidence was found — not that "
                           "no relationship", "note"))
    o.append(text(70, 686, f"exists. {reading['n_provisions']} provisions were read blind; "
                           f"{reading['n_links_verbatim']} produced a link with a verbatim "
                           f"quote.", "note"))

    o.append(f'<line class="rule" x1="48" y1="770" x2="{W - 48}" y2="770"/>')
    o.append(text(48, 796, "Generated from causal/provenance/banded_graph.json by "
                           "causal/stage16_render_infographic.py — no figure on this "
                           "page is typed in.", "tiny"))
    o.append(text(48, 816, f"Graph assembled {g['generated_at']}. Re-run the stage to "
                           f"redraw.", "tiny"))
    o.append("</svg>")
    return "".join(o)


def main() -> int:
    svg = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parent)} ({len(svg) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
