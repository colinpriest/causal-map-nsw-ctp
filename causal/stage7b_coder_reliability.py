"""Stage 7b — is an association between two coded columns real, or one reader's halo?

Two questions, and the second is the one that matters.

RELIABILITY (per column)
    Do the two readers agree on the score? Reported as quadratic-weighted Cohen's kappa,
    Spearman, and exact agreement. A column both readers score the same way is measuring
    something in the text; a column they disagree on is measuring the reader.

CROSS-CODER ASSOCIATION (per pair) -- THE DECISIVE TEST
    Take a pair like Injury Burden Intensity ~ Psychological Injury Emphasis.

        within-coder    rho(A.injury, A.psych)   and   rho(B.injury, B.psych)
        cross-coder     rho(A.injury, B.psych)   and   rho(B.injury, A.psych)

    A relationship that exists in the world survives crossing the readers: coder A's injury
    score still predicts coder B's psychological score, because both are tracking the same
    underlying case. A halo -- one reader forming an impression of severity and marking
    several scales up together -- lives inside a single coder's columns and collapses when
    the readers are crossed, because A's impression cannot influence B's scoring.

    So: cross-coder rho near the within-coder rho means the association is real.
    Cross-coder rho near zero while within-coder rho is substantial means it is an artefact
    of the coding process. The ratio is reported as `survival`.

    This settles by measurement what stage 5 could not settle by asking. It needs no
    threshold of mine to interpret: the two numbers are compared to each other.

Attenuation caveat: cross-coder correlations are attenuated by measurement error in BOTH
columns, so some drop is expected even for a wholly real relationship. A survival ratio is
therefore a conservative floor, and the per-column reliabilities say how much attenuation
to expect.

Run:  python causal/stage7b_coder_reliability.py
Out:  causal/provenance/coder_reliability.json
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "provenance" / "recoded_sample.json"
ASSOC = HERE / "provenance" / "associations.json"
OUT = HERE / "provenance" / "coder_reliability.json"

MIN_N = 30
REAL = 0.60         # survival at or above this: the association crosses readers
ARTEFACT = 0.25     # survival at or below this: it does not


def weighted_kappa(x, y, lo, hi):
    """Cohen's kappa with quadratic weights, for ordinal scales."""
    levels = list(range(lo, hi + 1))
    k = len(levels)
    o = np.zeros((k, k))
    for a, b in zip(x, y):
        o[int(a) - lo, int(b) - lo] += 1
    n = o.sum()
    if n == 0:
        return None
    w = np.array([[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    den = (w * e).sum()
    return None if den == 0 else float(1 - (w * o).sum() / den)


def rho(x, y):
    s = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(s) < MIN_N or s["x"].nunique() < 2 or s["y"].nunique() < 2:
        return None
    return float(stats.spearmanr(s["x"], s["y"]).statistic)


def main() -> int:
    doc = json.loads(SAMPLE.read_text(encoding="utf-8"))
    codings = doc["codings"]
    scales = doc["scales"]
    cols = list(scales)

    A = pd.DataFrame([c["original"] for c in codings])      # original coder
    B = pd.DataFrame([c["recoded"] for c in codings])       # independent re-coder
    print(f"cases compared: {len(A)}\n")

    # ---- per-column reliability ------------------------------------------
    reliability = {}
    print(f"{'column':32}{'kappa_w':>9}{'rho':>8}{'exact':>8}{'n':>5}")
    for c in cols:
        s = pd.DataFrame({"a": A[c], "b": B[c]}).dropna()
        if len(s) < MIN_N:
            reliability[c] = dict(n=len(s), reason="too few complete pairs")
            continue
        lo, hi = scales[c]["min"], scales[c]["max"]
        kw = weighted_kappa(s["a"], s["b"], lo, hi)
        r = rho(s["a"], s["b"])
        exact = float((s["a"] == s["b"]).mean())
        reliability[c] = dict(n=int(len(s)), kappa_weighted=None if kw is None else round(kw, 3),
                              spearman=None if r is None else round(r, 3),
                              exact_agreement=round(exact, 3),
                              mean_original=round(float(s["a"].mean()), 2),
                              mean_recoded=round(float(s["b"].mean()), 2))
        print(f"{c:32}{kw if kw is None else round(kw, 3):>9}"
              f"{r if r is None else round(r, 3):>8}{exact:>8.2f}{len(s):>5}")

    # ---- cross-coder association: the decisive test ----------------------
    pairs = []
    for x, y in itertools.combinations(cols, 2):
        within_a, within_b = rho(A[x], A[y]), rho(B[x], B[y])
        cross_1, cross_2 = rho(A[x], B[y]), rho(B[x], A[y])
        if None in (within_a, within_b, cross_1, cross_2):
            continue
        within = (abs(within_a) + abs(within_b)) / 2
        cross = (abs(cross_1) + abs(cross_2)) / 2
        if within < 0.10:
            verdict, survival = "no_association", None
        else:
            survival = cross / within
            verdict = ("survives_coder_change" if survival >= REAL
                       else "coder_artefact" if survival <= ARTEFACT
                       else "attenuated_inconclusive")
        pairs.append(dict(
            a=x, b=y, within_original=round(within_a, 3), within_recoded=round(within_b, 3),
            cross_a_b=round(cross_1, 3), cross_b_a=round(cross_2, 3),
            within_mean=round(within, 3), cross_mean=round(cross, 3),
            survival=None if survival is None else round(survival, 3), verdict=verdict))

    pairs.sort(key=lambda p: -(p["survival"] or -1))
    counts = pd.Series([p["verdict"] for p in pairs]).value_counts().to_dict()

    out = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_cases=len(A), recoder_model=doc["model"], seed=doc["seed"],
        thresholds=dict(real=REAL, artefact=ARTEFACT, min_n=MIN_N),
        rubric_is_reconstructed=doc.get("rubric_is_reconstructed", True),
        limitation=doc.get("limitation"),
        reliability=reliability, verdict_counts=counts, pairs=pairs,
        note=("survival = cross-coder |rho| / within-coder |rho|. Cross-coder correlations "
              "are attenuated by measurement error in both columns, so survival is a "
              "conservative floor; read it alongside the per-column reliabilities."),
    )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT}")
    print(f"\ncross-coder verdicts: {counts}\n")
    print(f"{'pair':62}{'within':>8}{'cross':>8}{'surv':>7}")
    for p in pairs:
        if p["survival"] is None:
            continue
        print(f"{p['a'] + ' ~ ' + p['b']:62}{p['within_mean']:8.3f}"
              f"{p['cross_mean']:8.3f}{p['survival']:7.2f}  {p['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
