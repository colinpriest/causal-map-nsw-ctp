"""Generate ctp/dictionary.md from ctp/columns.yaml, and check it against the data.

The definitions live in one machine-readable file so that every stage describing a column
to a model reads the same words. This script renders the human-readable dictionary and
verifies the file against the actual table: names must match the CSV exactly, declared
ordinal ranges must contain the observed values, and missingness is filled in from the data
rather than transcribed by hand.

Run:  python causal/build_dictionary.py
Out:  ctp/dictionary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "ctp" / "columns.yaml"
CSV = ROOT / "ctp" / "ctp.csv"
OUT = ROOT / "ctp" / "dictionary.md"


def load() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


def main() -> int:
    spec = load()
    df = pd.read_csv(CSV)
    names = [c["name"] for c in spec["columns"]]

    problems = []
    if names != list(df.columns):
        missing = [n for n in names if n not in df.columns]
        extra = [c for c in df.columns if c not in names]
        if missing or extra:
            problems.append(f"name mismatch: not in CSV {missing}, undocumented {extra}")

    for c in spec["columns"]:
        s = df.get(c["name"])
        if s is None:
            continue
        c["missing_rate"] = round(float(s.isna().mean()), 4)
        if "range" in c:
            lo, hi = c["range"]
            obs = s.dropna()
            if len(obs) and (obs.min() < lo or obs.max() > hi):
                problems.append(f"{c['name']}: declared range {c['range']} but observed "
                                f"[{obs.min()}, {obs.max()}]")

    for l in spec.get("latent", []):
        if l["measured_by"] not in names:
            problems.append(f"latent {l['name']}: indicator {l['measured_by']!r} "
                            "is not a column")

    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1

    lines = [
        f"# Data Dictionary — `{spec['dataset']}`",
        "",
        "<!-- GENERATED FILE. Edit ctp/columns.yaml, then run "
        "`python causal/build_dictionary.py`. -->",
        "",
        f"**Target:** `{spec['target']}` · **Rows:** {spec['n_rows']} · "
        f"**Columns:** {len(names)}",
        "",
        "**Source.** NSW CTP impairment lump-sum awards from Personal Injury Commission "
        "decisions, scraped from [AustLII](https://www.austlii.edu.au/) and "
        "LLM-structured.",
        "",
        "Each definition states **what the column measures** and nothing else. None says "
        "which phase of a claim a column belongs to, and none says what drives its level — "
        "those are conclusions the analysis derives, not inputs it is given. Where a "
        "definition implies an ordering it is because the term carries it: a *pre-existing* "
        "condition is by definition one the claimant had beforehand.",
        "",
        "`Recorded` is how the value was produced. It is separate from what the column is "
        "about: the eight ordinals were assigned by a language model reading each decision, "
        "which is a fact about their derivation and does not make them measurements *of* "
        "the decision.",
        "",
        "## Columns",
        "",
        "| Column | Definition | Type | Missing | Recorded |",
        "|---|---|---|---|---|",
    ]
    for c in spec["columns"]:
        rng = f" ({c['range'][0]}–{c['range'][1]})" if "range" in c else ""
        defn = " ".join(c["definition"].split())
        if c.get("indicator_of"):
            defn += f" **Indicator of the latent `{c['indicator_of']}`.**"
        lines.append(f"| `{c['name']}` | {defn} | {c['dtype']}{rng} | "
                     f"{c.get('missing_rate', 0):.1%} | {c['recorded']} |")

    if spec.get("latent"):
        lines += [
            "",
            "## Unobserved variables",
            "",
            "A latent node is carried in the causal graph but has no column. It exists "
            "because a measured column is an *indicator* of something rather than the "
            "thing itself, and conflating the two forces a false choice about when the "
            "variable arises. Nothing downstream may treat a latent node as data — it has "
            "no values and constrains structure only.",
            "",
        ]
        for l in spec["latent"]:
            lines += [
                f"### `{l['name']}` — unobserved",
                "",
                f"{' '.join(l['definition'].split())}",
                "",
                f"- **Measured by:** `{l['measured_by']}`",
                f"- **Why latent:** {' '.join(l['why_latent'].split())}",
                "",
            ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"columns documented: {len(names)}   latent: {len(spec.get('latent', []))}")
    print("checks passed: names match CSV, ordinal ranges contain observed values, "
          "every indicator resolves to a column")
    return 0


if __name__ == "__main__":
    sys.exit(main())
