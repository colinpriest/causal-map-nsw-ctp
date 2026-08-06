"""Export the reviewed DAG as a standalone SVG for embedding in Markdown.

``stage11_render_dag.py`` writes an interactive HTML page whose ``<svg>`` is empty on disk --
the nodes and edges are built by JavaScript at load time. GitHub will not run that, so a
README cannot embed the HTML. This loads the page in a headless browser, waits for the
script to build the graph, and writes the resulting SVG out on its own.

Two things have to be repaired on the way out. The page's SVG carries no ``viewBox`` (it is
sized by CSS in the browser), so it must be given one from the rendered bounding box or it
renders clipped. And its colours come from page-level CSS variables, which do not reach an
SVG loaded as an ``<img>`` -- so the handful of rules the graph depends on are inlined with
fixed values, plus a background rect so the dark strokes stay legible on GitHub's dark theme.

In:   causal/ctp_reviewed_dag.html
Out:  docs/ctp_reviewed_dag.svg

Run:  python causal/stage11b_export_dag_svg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SRC = HERE / "ctp_reviewed_dag.html"
OUT = HERE.parent / "docs" / "ctp_reviewed_dag.svg"

# The page's light-mode palette, redeclared on the SVG root. The graph's elements carry
# literal `fill="var(--band)"` style attributes written by the renderer, and a custom
# property that resolves to nothing falls back to black -- which is how the band strips came
# out solid black the first time this ran. Declaring them here is what keeps that from
# happening; the rules below cover the elements styled by page-level selectors instead.
PALETTE = {
    "--ink": "#0b0b0b", "--ink2": "#52514e", "--muted": "#7d7b76",
    "--page": "#f9f9f7", "--surface": "#ffffff",
    "--line": "rgba(11,11,11,.22)", "--band": "rgba(11,11,11,.03)",
}

INLINE_CSS = (
    "svg{" + "".join(f"{k}:{v};" for k, v in PALETTE.items()) + "}"
    "text{font:12px system-ui,-apple-system,'Segoe UI',sans-serif;fill:#0b0b0b}"
    "text.small{font-size:10.5px;fill:#7d7b76}"
    "text.bandlab{font-size:11px;font-weight:600;fill:#7d7b76;letter-spacing:.06em}"
    "rect.node{fill:#ffffff;stroke:rgba(11,11,11,.32);stroke-width:1.4}"
    "rect.node.latent{stroke-dasharray:5 3;stroke-width:1.8;fill:none}"
    "path.edge{fill:none;opacity:.72}"
)

EXTRACT = """(css) => {
  const src = document.getElementById('map');
  if (!src || src.childElementCount === 0) return null;
  const bb = src.getBBox(), pad = 16;
  const x = Math.floor(bb.x - pad), y = Math.floor(bb.y - pad);
  const w = Math.ceil(bb.width + pad * 2), h = Math.ceil(bb.height + pad * 2);
  const n = src.cloneNode(true);
  n.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  n.setAttribute('viewBox', [x, y, w, h].join(' '));
  n.setAttribute('width', w);
  n.setAttribute('height', h);
  n.removeAttribute('id');
  const bg = `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="#f9f9f7"/>`;
  n.innerHTML = `<style>${css}</style>` + bg + n.innerHTML;
  return {svg: n.outerHTML, w, h, nodes: src.childElementCount};
}"""


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run stage11_render_dag.py first")
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(SRC.as_uri())
        # The build is synchronous on load, but wait on the result rather than a timeout so
        # a script error surfaces as a failure here instead of a silently empty SVG.
        page.wait_for_function("document.getElementById('map').childElementCount > 0",
                               timeout=15000)
        result = page.evaluate(EXTRACT, INLINE_CSS)
        browser.close()

    if not result:
        print("the page rendered no graph")
        return 1

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(result["svg"], encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parent)} "
          f"({result['w']}x{result['h']}, {result['nodes']} elements, "
          f"{len(result['svg']) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
