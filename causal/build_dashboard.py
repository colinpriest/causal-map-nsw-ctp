"""
SUPERSEDED. This built the original hand-authored 49-edge DAG. That graph has been
replaced by the evidence-based pipeline in stages 0-15, whose output is
causal/provenance/banded_graph.json. Kept as a record of what the project started from --
a graph whose edges rested on nothing checkable -- because the contrast with the current
one is the point. Nothing downstream reads anything this produces.
"""

"""Render causal/ctp_causal_dashboard.html from the DAG JSON.

The template holds all markup/CSS/JS; the graph is injected as a single JSON
payload so the dashboard and the generator read the same source of truth.

Run:  python causal/build_ctp_causal_dag.py && python causal/build_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "dashboard_template.html"
DATA = HERE / "ctp_causal_dag.json"
OUT = HERE / "ctp_causal_dashboard.html"


def main():
    dag = json.loads(DATA.read_text(encoding="utf-8"))
    payload = json.dumps(dag, separators=(",", ":"))
    # the payload lives inside <script type="application/json">; only </script>
    # can terminate it early
    payload = payload.replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DAG_JSON__", payload)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
