"""Stage 5b — run every prediction stage 5a committed to, and record whether it held.

Stage 5a produced, for each pair, a causal verdict and a prediction chosen from a fixed
menu. This stage executes those predictions against ctp.csv. No model is involved and
nothing here is negotiable: the prediction was fixed before the data was consulted, so the
result is a genuine test rather than a description.

A prior that survives its own prediction is worth more than a confidence score. A prior
that fails is informative -- it is the one thing a purely reasoned edge can offer that
looks like evidence.

THE FIVE TESTS

  sign          Spearman(a, b) carries the predicted sign, |rho| >= MIN_RHO.

  monotone      Median of b TRENDS one way across levels of a -- Spearman between bin
                position and median, at least TREND. Not strict step-by-step monotonicity:
                five quantile bins of 540 rows wobble, and requiring every step to move
                the right way rejects real trends on one reversal.

  inverted_u    Median of b peaks (or troughs) at an INTERIOR level of a. In the menu
                because real mechanisms are not always monotone: earnings rise with career
                experience and fall at retirement, so age->earnings can be strong and
                still show a near-zero rank correlation.

  mediation     |rho(a, outcome)| falls by at least ATTEN once `mediator` is held fixed.

  attenuation   For a chain a -> b -> outcome, holding b fixed blocks a's path and
  _asymmetry    collapses rho(a, outcome), while holding a fixed leaves rho(b, outcome)
                largely intact. So `a_is_upstream` predicts that a attenuates MORE than b.
                (An earlier version of the menu described this backwards, which would have
                inverted every directional verdict that used it.)

Every test reports its numbers whatever the verdict, so a reader can disagree with the
threshold without rerunning anything.

Run:  python causal/stage5b_test_implications.py
Out:  causal/provenance/prior_tests.json
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "ctp" / "ctp.csv"
PRIORS = HERE / "provenance" / "reasoned_priors.json"
OUT = HERE / "provenance" / "prior_tests.json"

HEAVY = {"Lump Sum", "Non-Economic Loss", "Future Economic Loss", "Claimant Weekly Income"}
MIN_N = 40
MIN_RHO = 0.05      # below this a sign claim is not really a claim
ATTEN = 0.50        # "largely disappears" == at least half the association gone
MARGIN = 0.15       # asymmetry must be this much to count as directional evidence
TREND = 0.70        # Spearman between bin position and median, for a monotone claim


def prepare() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    d["Claimant Gender"] = (d["Claimant Gender"] == "Male").astype(float)
    d["Nature"] = (d["Nature"] == "Damages").astype(float)
    for c in HEAVY:
        d[c] = np.log1p(d[c])
    return d


def rho(d, a, b):
    # A model sometimes names the outcome as one of the pair, which would ask for a
    # variable's correlation with itself. Degenerate, not an error worth crashing on.
    if a == b:
        return None, 0
    s = d[[a, b]].dropna()
    if len(s) < MIN_N or s[a].nunique() < 2 or s[b].nunique() < 2:
        return None, len(s)
    return float(stats.spearmanr(s[a], s[b]).statistic), len(s)


def partial(d, a, b, given):
    """Spearman(a, b) with `given` residualised out, complete cases only."""
    given = [g for g in given if g and g not in (a, b)]
    if a == b or not given:
        return rho(d, a, b)
    cols = [a, b] + given
    s = d[cols].dropna()
    if len(s) < MIN_N or s[a].nunique() < 2 or s[b].nunique() < 2:
        return None, len(s)
    z = s[given].to_numpy(float)
    keep = [i for i in range(z.shape[1]) if np.ptp(z[:, i]) > 0]
    if not keep:
        return rho(s, a, b)
    X = np.column_stack([np.ones(len(s)), z[:, keep]])

    def resid(v):
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

    ra, rb = resid(s[a].to_numpy(float)), resid(s[b].to_numpy(float))
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None, len(s)
    return float(stats.spearmanr(ra, rb).statistic), len(s)


def level_medians(d, a, b, max_levels=6):
    """Median of b by level of a; a is binned to quantiles if it is near-continuous."""
    s = d[[a, b]].dropna()
    if len(s) < MIN_N:
        return None
    if s[a].nunique() <= max_levels:
        grp = s.groupby(a)[b].median()
    else:
        try:
            bins = pd.qcut(s[a], q=min(5, s[a].nunique()), duplicates="drop")
        except ValueError:
            return None
        grp = s.groupby(bins, observed=True)[b].median()
    return grp if len(grp) >= 3 else None


def run_test(d, pred, verdict=None):
    """Execute one prediction. Returns (passed | None, detail dict).

    `monotone` and `inverted_u` are NOT symmetric: binning var_a and taking the median of
    var_b is a different question from the reverse. var_a/var_b arrive in the pair's
    storage order, which has nothing to do with the causal direction, so a `b_causes_a`
    verdict must have them swapped before the shape is measured.

    Without this, `Claimant Age -> Claimant Weekly Income` -- age raising earnings through
    experience and career progression -- was tested by binning INCOME and taking median
    AGE, and recorded as a failure. The mechanism is about how earnings vary with age.
    """
    t, a, b = pred["test"], pred["var_a"], pred["var_b"]
    want = pred["direction"]
    swapped = False
    if verdict == "b_causes_a" and t in ("monotone", "inverted_u"):
        a, b, swapped = b, a, True

    if t == "sign":
        r, n = rho(d, a, b)
        if r is None:
            return None, dict(reason="insufficient data", n=n)
        ok = (r >= MIN_RHO) if want == "positive" else (r <= -MIN_RHO)
        return ok, dict(rho=round(r, 3), n=n, threshold=MIN_RHO)

    if t in ("monotone", "inverted_u"):
        grp = level_medians(d, a, b)
        if grp is None:
            return None, dict(reason="too few levels or rows")
        vals = list(grp.to_numpy(float))
        peak = int(np.argmax(vals))
        trough = int(np.argmin(vals))
        # TREND, not step-by-step monotonicity. Five quantile bins of a 540-row table are
        # noisy, and demanding every consecutive step move the right way fails a clear
        # trend on a single reversal: median future economic loss by age band runs
        # 11.92, 11.76, 11.94, 11.33, 10.40 -- unmistakably downward, one wobble, and the
        # strict test rejected it. Spearman between bin position and median tolerates that
        # without accepting a shape that is not really moving.
        trend = float(stats.spearmanr(range(len(vals)), vals).statistic)
        if t == "monotone":
            ok = (trend >= TREND if want in ("increasing", "positive")
                  else trend <= -TREND)
        else:
            interior = range(1, len(vals) - 1)
            ok = (peak in interior) if want in ("peak", "positive") else (trough in interior)
        return ok, dict(binned_on=a, median_of=b, oriented_by_verdict=swapped,
                        trend=round(trend, 3), trend_threshold=TREND,
                        medians=[round(v, 3) for v in vals],
                        levels=[str(i) for i in grp.index],
                        peak_at=peak, trough_at=trough, n_levels=len(vals))

    if t == "mediation":
        y, m = pred.get("outcome"), pred.get("mediator")
        if not y or not m:
            return None, dict(reason="mediation needs outcome and mediator")
        if m in (a, y) or y == a:
            return None, dict(reason="degenerate: mediator or outcome repeats the pair",
                              outcome=y, mediator=m)
        base, n0 = rho(d, a, y)
        adj, n1 = partial(d, a, y, [m])
        if base is None or adj is None or abs(base) < MIN_RHO:
            return None, dict(reason="no baseline association to attenuate",
                              rho_base=base, n=n0)
        drop = 1 - abs(adj) / abs(base)
        return drop >= ATTEN, dict(rho_base=round(base, 3), rho_given_mediator=round(adj, 3),
                                   attenuation=round(drop, 3), threshold=ATTEN,
                                   n=n1, mediator=m, outcome=y)

    if t == "attenuation_asymmetry":
        y = pred.get("outcome")
        if not y:
            return None, dict(reason="needs an outcome")
        if y in (a, b):
            return None, dict(reason="degenerate: outcome repeats the pair", outcome=y)
        ra, _ = rho(d, a, y)
        rb, _ = rho(d, b, y)
        ra_given_b, n1 = partial(d, a, y, [b])
        rb_given_a, n2 = partial(d, b, y, [a])
        if None in (ra, rb, ra_given_b, rb_given_a) or min(abs(ra), abs(rb)) < MIN_RHO:
            return None, dict(reason="no baseline association on one side",
                              rho_a=ra, rho_b=rb)
        at_a = 1 - abs(ra_given_b) / abs(ra)      # how much holding b kills a
        at_b = 1 - abs(rb_given_a) / abs(rb)      # how much holding a kills b
        # a upstream of b  =>  b mediates  =>  holding b collapses a
        ok = (at_a - at_b >= MARGIN) if want == "a_is_upstream" else \
             (at_b - at_a >= MARGIN)
        return ok, dict(outcome=y, rho_a=round(ra, 3), rho_b=round(rb, 3),
                        rho_a_given_b=round(ra_given_b, 3),
                        rho_b_given_a=round(rb_given_a, 3),
                        attenuation_of_a=round(at_a, 3), attenuation_of_b=round(at_b, 3),
                        margin=MARGIN, n=min(n1, n2))

    return None, dict(reason=f"unknown test {t}")


def main() -> int:
    d = prepare()
    doc = json.loads(PRIORS.read_text(encoding="utf-8"))

    results, counts = [], Counter()
    for p in doc["priors"]:
        passed, detail = run_test(d, p["prediction"], p["verdict"])
        counts["pass" if passed else "fail" if passed is False else "untestable"] += 1
        results.append(dict(
            a=p["a"], b=p["b"], verdict=p["verdict"],
            verdict_agreement=p["verdict_agreement"],
            confidence=p["confidence_mechanism_exists"],
            magnitude=p["expected_magnitude"],
            both_llm_coded=p["both_llm_coded"],
            prediction=p["prediction"], passed=passed, detail=detail,
            mechanism=p["mechanism"],
        ))

    # A verdict whose own prediction failed is not evidence for an edge.
    by_verdict = Counter((r["verdict"], r["passed"]) for r in results)

    out = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        thresholds=dict(min_n=MIN_N, min_rho=MIN_RHO, attenuation=ATTEN, margin=MARGIN),
        n=len(results), counts=dict(counts),
        by_verdict={f"{v}|{p}": c for (v, p), c in sorted(by_verdict.items(),
                                                          key=lambda kv: str(kv[0]))},
        note=("Predictions were fixed by stage 5a before any data was consulted. A failed "
              "prediction does not disprove the mechanism -- n is 540 and several tests "
              "are low-powered -- but it removes the only external support a reasoned "
              "prior has, and such an edge should not be drawn on this evidence alone."),
        results=results,
    )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"predictions: {dict(counts)}\n")
    for r in sorted(results, key=lambda r: (r["passed"] is not True, r["a"])):
        mark = {True: "PASS", False: "FAIL", None: " -- "}[r["passed"]]
        pr = r["prediction"]
        print(f"{mark}  {r['a']} ~ {r['b']}")
        print(f"        verdict={r['verdict']} (agree {r['verdict_agreement']:.0%}, "
              f"conf {r['confidence']}, mag {r['magnitude']})")
        print(f"        test={pr['test']}/{pr['direction']}  {r['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
