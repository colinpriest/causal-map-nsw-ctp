"""Stage 1 — measure every pairwise association in ctp.csv, and rank by materiality.

All 120 unordered pairs are computed. Nothing is dropped: screening candidate edges on
marginal association would discard exactly the relationships confounding hides, and admit
plenty that confounding invents. Association is an annotation and a budget-allocation
signal here, not an admission test.

Four numbers per pair:

  rho          pairwise-complete Spearman, log1p on the dollar columns
  rho_partial  the same, residualised on every *other* column both members are observed
               with -- a cheap conditional-independence read
  tail         SIGNED upper-tail dependence. Rank correlation is a whole-of-distribution
               measure and understates variables that only bite in the extreme -- which on
               a target spanning $1k to $3.5M is where the money is. Two coefficients are
               computed at the q-quantile of the ranks:
                   lambda_UU = P(b high | a high)   co-movement in the upper tails
                   lambda_UL = P(b low  | a high)   high a with low b
               Each is compared against the independence baseline (1 - q); the larger
               excess is reported, positive for UU and negative for UL. Columns whose
               distribution is too lumpy to form a tail at all -- Liability Clarity sits
               91.9% on one level -- return null with a reason rather than a number.
  materiality  |rho| x min(|rho(a, target)|, |rho(b, target)|)

Materiality drives what the later, expensive stages look at. A pair that barely moves the
award does not earn an LLM research budget, and saying so with a number is more honest
than quietly skipping it.

Pairs where both members are LLM-coded are flagged: see stage 0. Their association is
partly the coder's, and no downstream stage should treat it as scheme mechanism.

Run:  python causal/stage1_associations.py
Out:  causal/provenance/associations.json
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "ctp" / "ctp.csv"
LEDGER = Path(__file__).resolve().parent / "provenance" / "measurement_ledger.json"
OUT = Path(__file__).resolve().parent / "provenance" / "associations.json"

TARGET = "Lump Sum"
HEAVY = {"Lump Sum", "Non-Economic Loss", "Future Economic Loss", "Claimant Weekly Income"}
# The statutory-quantum core: these set the award, so they are researched regardless of
# where the arithmetic ranks them.
QUANTUM = {"Lump Sum", "Non-Economic Loss", "Future Economic Loss", "WPI %"}
MIN_N = 30


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Claimant Gender"] = (d["Claimant Gender"] == "Male").astype(float)
    d["Nature"] = (d["Nature"] == "Damages").astype(float)
    for c in HEAVY:
        d[c] = np.log1p(d[c])
    return d


def spearman(d: pd.DataFrame, a: str, b: str):
    s = d[[a, b]].dropna()
    if len(s) < MIN_N or s[a].nunique() < 2 or s[b].nunique() < 2:
        return None, len(s)
    return float(stats.spearmanr(s[a], s[b]).statistic), len(s)


def tail_dependence(d: pd.DataFrame, a: str, b: str, q: float = 0.8):
    """Signed upper-tail dependence of b on a, as excess over independence.

    Returns (signal, detail). `signal` is positive when the upper tails co-move and
    negative when a's upper tail pairs with b's lower tail; None when either column
    cannot form a tail (too few distinct values, or one level dominating the ranks).
    """
    s = d[[a, b]].dropna()
    base = 1.0 - q
    if len(s) < 60:
        return None, dict(reason="fewer than 60 complete pairs", n=len(s))
    ua = s[a].rank(pct=True)
    ub = s[b].rank(pct=True)
    hi_a = ua > q
    n_tail = int(hi_a.sum())
    # A column with a dominant level puts most mass on one average rank, so no tail forms.
    if n_tail < 15:
        return None, dict(reason="no upper tail (ties dominate the ranks)",
                          n=len(s), n_tail=n_tail)
    if (ub > q).sum() < 15 or (ub < base).sum() < 15:
        return None, dict(reason="target column has no usable tail", n=len(s), n_tail=n_tail)
    lam_uu = float((ub[hi_a] > q).mean())
    lam_ul = float((ub[hi_a] < base).mean())
    exc_uu, exc_ul = lam_uu - base, lam_ul - base
    signal = exc_uu if abs(exc_uu) >= abs(exc_ul) else -exc_ul
    return round(signal, 3), dict(lambda_uu=round(lam_uu, 3), lambda_ul=round(lam_ul, 3),
                                  baseline=round(base, 3), n=len(s), n_tail=n_tail, q=q)


def partial(d: pd.DataFrame, a: str, b: str, others: list[str]):
    """Spearman between a and b, residualised on the other columns, complete cases only."""
    cols = [a, b] + others
    s = d[cols].dropna()
    if len(s) < MIN_N or s[a].nunique() < 2 or s[b].nunique() < 2:
        return None, len(s)
    z = s[others].to_numpy(float)
    keep = [i for i in range(z.shape[1]) if np.ptp(z[:, i]) > 0]
    if not keep:
        return spearman(s, a, b)
    X = np.column_stack([np.ones(len(s)), z[:, keep]])

    def resid(v):
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

    ra, rb = resid(s[a].to_numpy(float)), resid(s[b].to_numpy(float))
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None, len(s)
    return float(stats.spearmanr(ra, rb).statistic), len(s)


def main() -> None:
    raw = pd.read_csv(CSV)
    d = prepare(raw)
    cols = list(raw.columns)
    kinds = {k: v["kind"] for k, v in json.loads(
        LEDGER.read_text(encoding="utf-8"))["columns"].items()}

    to_target = {}
    for c in cols:
        if c == TARGET:
            continue
        r, _ = spearman(d, c, TARGET)
        to_target[c] = abs(r) if r is not None else 0.0

    rows = []
    for a, b in itertools.combinations(cols, 2):
        r, n = spearman(d, a, b)
        others = [c for c in cols if c not in (a, b)]
        rp, npart = partial(d, a, b, others)
        tail, tail_detail = tail_dependence(d, a, b)
        mat = 0.0 if r is None else abs(r) * min(to_target.get(a, 1.0), to_target.get(b, 1.0))
        both_coded = kinds.get(a) == "coded" and kinds.get(b) == "coded"

        if a in QUANTUM and b in QUANTUM:
            tier = "A"
        elif mat >= 0.10:
            tier = "B"
        elif mat >= 0.04:
            tier = "C"
        else:
            tier = "D"
        # Rank correlation is a body measure. A pair that is unremarkable overall but
        # concentrated in the extreme still moves the award, so it is promoted.
        tail_promoted = tier in ("C", "D") and tail is not None and abs(tail) >= 0.08
        if tail_promoted:
            tier = "B"

        rows.append(dict(
            a=a, b=b, tier=tier,
            rho=None if r is None else round(r, 3), n=n,
            rho_partial=None if rp is None else round(rp, 3), n_partial=npart,
            tail=tail, tail_detail=tail_detail, tail_promoted=tail_promoted,
            materiality=round(mat, 4),
            both_llm_coded=both_coded,
            # a marginal that vanishes once everything else is held fixed is a confounding
            # signature, and the pair should not be researched as a direct mechanism
            marginal_only=(r is not None and rp is not None
                           and abs(r) > 0.15 and abs(rp) < 0.05),
        ))

    rows.sort(key=lambda x: -x["materiality"])
    counts = {t: sum(r["tier"] == t for r in rows) for t in "ABCD"}
    research = [r for r in rows if r["tier"] in ("A", "B")]

    doc = dict(
        n_pairs=len(rows), target=TARGET, tier_counts=counts,
        n_to_research=len(research),
        tier_definitions={
            "A": "both members are quantum/gate columns - researched regardless of magnitude",
            "B": "materiality >= 0.10 - researched",
            "C": "materiality 0.04-0.10 - recorded, not researched",
            "D": "materiality < 0.04 - immaterial to the award at n=540, not researched",
        },
        association_to_target={k: round(v, 3) for k, v in
                               sorted(to_target.items(), key=lambda x: -x[1])},
        pairs=rows,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"pairs={len(rows)}  tiers={counts}  to research={len(research)}")
    flagged = [r for r in research if r["both_llm_coded"]]
    collapsed = [r for r in rows if r["marginal_only"]]
    promoted = [r for r in rows if r["tail_promoted"]]
    no_tail = [r for r in rows if r["tail"] is None]
    print(f"of those, both-LLM-coded (coder-confounded): {len(flagged)}")
    print(f"marginal collapses under adjustment (confounding signature): {len(collapsed)}")
    print(f"promoted on tail dependence alone: {len(promoted)}")
    print(f"no measurable tail (lumpy distribution): {len(no_tail)}")

    def show(rs, title):
        print(f"\n{title}")
        print(f"{'':2}{'pair':58}{'rho':>7}{'partial':>8}{'tail':>7}{'mat':>7}")
        for r in rs:
            flag = "!" if r["both_llm_coded"] else " "
            rp = "     -" if r["rho_partial"] is None else f"{r['rho_partial']:8.3f}"
            tl = "      -" if r["tail"] is None else f"{r['tail']:7.3f}"
            ro = "      -" if r["rho"] is None else f"{r['rho']:7.3f}"
            print(f"{r['tier']}{flag}{r['a'] + ' ~ ' + r['b']:58}{ro}{rp}{tl}{r['materiality']:7.3f}")

    show(rows[:12], "TOP 12 BY MATERIALITY")
    if promoted:
        show(sorted(promoted, key=lambda r: -abs(r["tail"])),
             "PROMOTED BY TAIL DEPENDENCE (missed by rank correlation)")
    print("\n! = both columns LLM-coded; association is partly the coder's")


if __name__ == "__main__":
    main()
