"""Stage 3a — which variables do cases turn on when a given provision is cited?

The raw workbook records, per case, the provisions the tribunal actually relied on
('Regulatory Sections' -- 341 distinct citations, 99.4% filled). That is a direct handle
on the scheme's own rulebook, from the decisions themselves rather than from a search
engine's summary of the legislation.

This stage does the DERIVATION half of the statutory link, with no API and no assertions:
for every provision cited often enough, compare the cases that cite it against the cases
that do not, on every modelling variable. Two comparisons per (provision, variable):

  value      rank-biserial correlation from Mann-Whitney U -- do citing cases sit higher
             or lower on this variable?
  recorded   difference in blank rate -- does citing the provision change whether the
             variable is recorded at all? This is how a statutory GATE shows up: a gate
             does not shift a value, it decides whether the head is reached.

Benjamini-Hochberg controls the false discovery rate across the whole grid.

What this stage CANNOT do is say what a provision means. A link here says "cases citing
s 4.11 differ on WPI %", not "s 4.11 makes WPI % an input". Stage 3b fetches the provision
text so the second claim can be made and checked against the first. A provision->variable
edge needs BOTH: the text must name the input, and the cases must actually differ on it.

Run:  python causal/stage3a_provision_links.py
Out:  causal/provenance/provision_links.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "ctp" / "raw" / "ctp_impairment_lump_sum.xlsx"
CSV = ROOT / "ctp" / "ctp.csv"
OUT = Path(__file__).resolve().parent / "provenance" / "provision_links.json"

MIN_CITES = 20        # provisions cited in fewer cases are not tested
MIN_GROUP = 10        # smallest comparison group
FDR = 0.05
EFFECT_MIN = 0.15     # |rank-biserial| worth reporting as a value link
GATE_MIN = 0.10       # blank-rate difference worth reporting as a gate link
DEFINITIONAL = 0.90   # |effect| at or above this means the provision and the variable
                      # are the same fact recorded twice, not a mechanism

# "s 6.23(2)(b) Motor Accident Injuries Act 2017" -> ("6.23", "Motor Accident Injuries Act 2017")
# Subsections are folded into the parent section: they are variants of one rule, and
# splitting them fragments the counts below anything testable.
CITE = re.compile(r"^(?:s|ss|cl|r|reg|sch|pt)\s*\.?\s*([0-9]+[A-Za-z]*(?:\.[0-9]+)?)"
                  r"(?:\([^)]*\))*\s+(.*?)\s*$", re.IGNORECASE)


def normalise(raw_cite: str) -> tuple[str, str] | None:
    m = CITE.match(raw_cite.strip())
    if not m:
        return None
    section, act = m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()
    if not act or len(act) < 4:
        return None
    return section, act


def rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Effect size and p from Mann-Whitney U. +1 means x stochastically dominates y."""
    if len(x) < MIN_GROUP or len(y) < MIN_GROUP:
        return float("nan"), float("nan")
    if len(np.unique(np.concatenate([x, y]))) < 2:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 2.0 * u.statistic / (len(x) * len(y)) - 1.0, float(u.pvalue)


def bh(pvals: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg. NaN p-values never pass."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    keep = [False] * len(pvals)
    m = len(order)
    cut = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= alpha * rank / m:
            cut = rank
    for rank, i in enumerate(order, start=1):
        if rank <= cut:
            keep[i] = True
    return keep


def main() -> None:
    raw = pd.read_excel(XLSX)
    csv = pd.read_csv(CSV)
    assert len(raw) == len(csv) == 540

    num = csv.copy()
    num["Claimant Gender"] = (num["Claimant Gender"] == "Male").astype(float)
    num["Nature"] = (num["Nature"] == "Damages").astype(float)
    variables = list(csv.columns)

    # ---- parse citations --------------------------------------------------
    cited_by: dict[tuple[str, str], set[int]] = defaultdict(set)
    unparsed = Counter()
    for i, cell in enumerate(raw["Regulatory Sections"].fillna("")):
        for part in re.split(r"[;|,]\s*", str(cell)):
            part = part.strip()
            if not part:
                continue
            key = normalise(part)
            if key is None:
                unparsed[part] += 1
            else:
                cited_by[key].add(i)

    tested = {k: v for k, v in cited_by.items() if len(v) >= MIN_CITES}
    total_cites = sum(len(v) for v in cited_by.values()) + sum(unparsed.values())
    print(f"citations: {len(cited_by)} distinct provisions, "
          f"{len(tested)} cited in >= {MIN_CITES} cases")
    print(f"unparsed: {sum(unparsed.values())} of {total_cites} citation strings "
          f"({sum(unparsed.values()) / total_cites:.1%}) - top 10:")
    for t, n in unparsed.most_common(10):
        print(f"    {n:4d}  {t[:78]}")

    # ---- test every (provision, variable) --------------------------------
    records, pv_value, pv_gate = [], [], []
    for (section, act), rows in sorted(tested.items(), key=lambda kv: -len(kv[1])):
        mask = np.zeros(len(csv), dtype=bool)
        mask[sorted(rows)] = True
        for var in variables:
            col = num[var]
            a = col[mask].dropna().to_numpy(float)
            b = col[~mask].dropna().to_numpy(float)
            eff, p = rank_biserial(a, b)

            blank_cite = float(col[mask].isna().mean())
            blank_other = float(col[~mask].isna().mean())
            gate = blank_cite - blank_other
            n1, n2 = int(mask.sum()), int((~mask).sum())
            if col.isna().any() and min(n1, n2) >= MIN_GROUP:
                tab = [[int(col[mask].isna().sum()), int(col[mask].notna().sum())],
                       [int(col[~mask].isna().sum()), int(col[~mask].notna().sum())]]
                gp = float(stats.fisher_exact(tab).pvalue) if min(map(min, tab)) >= 0 else float("nan")
            else:
                gp = float("nan")

            records.append(dict(
                section=section, act=act, provision=f"{section} {act}",
                n_citing=n1, variable=var,
                effect=None if np.isnan(eff) else round(float(eff), 3),
                p_value=None if np.isnan(p) else float(f"{p:.3g}"),
                n_value=[len(a), len(b)],
                blank_rate_citing=round(blank_cite, 3),
                blank_rate_other=round(blank_other, 3),
                gate_effect=round(gate, 3),
                gate_p=None if np.isnan(gp) else float(f"{gp:.3g}"),
            ))
            pv_value.append(p)
            pv_gate.append(gp)

    keep_v = bh(pv_value, FDR)
    keep_g = bh(pv_gate, FDR)
    for rec, kv, kg in zip(records, keep_v, keep_g):
        e, g = rec["effect"], rec["gate_effect"]
        # A provision whose citation is all but perfectly predicted by a variable's value
        # is not evidence that the provision drives the variable -- the two are the same
        # fact recorded twice. s 6.23 MAIA governs settlement approval, so citing it and
        # being a Settlement Approval coincide by definition. Flag, do not use.
        rec["definitional"] = bool(e is not None and abs(e) >= DEFINITIONAL)
        rec["value_link"] = bool(kv and e is not None and abs(e) >= EFFECT_MIN
                                 and not rec["definitional"])
        rec["gate_link"] = bool(kg and abs(g) >= GATE_MIN)

    value_links = [r for r in records if r["value_link"]]
    gate_links = [r for r in records if r["gate_link"]]
    definitional = [r for r in records if r["definitional"]]

    doc = dict(
        thresholds=dict(min_cites=MIN_CITES, min_group=MIN_GROUP, fdr=FDR,
                        effect_min=EFFECT_MIN, gate_min=GATE_MIN,
                        definitional=DEFINITIONAL),
        n_provisions_distinct=len(cited_by), n_provisions_tested=len(tested),
        provisions_tested=[dict(provision=f"{s} {a}", n_citing=len(r))
                           for (s, a), r in sorted(tested.items(), key=lambda kv: -len(kv[1]))],
        # Every cited provision, at any count. MIN_CITES gates the STATISTICAL test, which
        # needs group sizes -- it must not gate which provisions get READ. Citation
        # frequency tracks what is disputed, not what is fundamental: s 4.6 MAIA sets the
        # cap on loss-of-earnings damages and is cited 5 times, because nobody argues
        # about it. Thresholding the reading step discards exactly the load-bearing rules.
        all_provisions=[dict(provision=f"{s} {a}", n_citing=len(r))
                        for (s, a), r in sorted(cited_by.items(), key=lambda kv: -len(kv[1]))],
        unparsed_citations=[dict(text=t, n=n) for t, n in unparsed.most_common(20)],
        n_tests=len(records), n_value_links=len(value_links), n_gate_links=len(gate_links),
        n_definitional=len(definitional),
        definitional_pairs=[dict(provision=r["provision"], variable=r["variable"],
                                 effect=r["effect"]) for r in definitional],
        records=records,
        caveat=("A link is an association between citing a provision and a variable. It is "
                "NOT evidence that the provision makes the variable an input -- cases citing "
                "a provision differ in many ways at once. Stage 3b must confirm the "
                "provision text actually names the input before an edge is drawn."),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"tests={len(records)}  value links={len(value_links)}  gate links={len(gate_links)}\n")
    print("STRONGEST VALUE LINKS (citing cases sit higher/lower on the variable)")
    print(f"{'provision':44}{'variable':32}{'eff':>7}{'n':>6}")
    for r in sorted(value_links, key=lambda r: -abs(r["effect"]))[:14]:
        print(f"{r['provision'][:43]:44}{r['variable']:32}{r['effect']:7.3f}{r['n_citing']:6d}")
    print("\nSTRONGEST GATE LINKS (citing changes whether the variable is recorded)")
    print(f"{'provision':44}{'variable':32}{'d_blank':>8}{'n':>6}")
    for r in sorted(gate_links, key=lambda r: -abs(r["gate_effect"]))[:14]:
        print(f"{r['provision'][:43]:44}{r['variable']:32}{r['gate_effect']:8.3f}{r['n_citing']:6d}")


if __name__ == "__main__":
    main()
