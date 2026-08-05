"""Plot Claimant Age against Claimant Weekly Income, on observed values only.

Before plotting anything, this verifies that neither column has been imputed. Two checks:

  1. The modelling table's blanks must match the raw workbook's. `Claimant Weekly Income
     Basis == "Not stated"` is the raw field recording that earnings were never put in
     evidence; if ctp.csv has FEWER blanks than that, a value was filled in from somewhere.
  2. Ages must match the raw workbook row for row where both are present.

Anything imputed and then plotted would manufacture a relationship out of whatever rule did
the filling, so the checks run first and the script refuses to plot if they fail.

Zero incomes are reported separately rather than plotted as earnings. A recorded $0 means
the claimant was not earning, which is a different fact from a low wage, and dropping it
into a scatter of the age-earnings profile would drag the curve down at both ends.

Run:  python causal/plot_age_income.py
Out:  causal/provenance/age_income.png
      causal/provenance/age_income_check.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from scipy import stats                   # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "ctp" / "ctp.csv"
XLSX = ROOT / "ctp" / "raw" / "ctp_impairment_lump_sum.xlsx"
PNG = HERE / "provenance" / "age_income.png"
OUT = HERE / "provenance" / "age_income_check.json"

AGE, INC = "Claimant Age", "Claimant Weekly Income"


def main() -> int:
    d = pd.read_csv(CSV)
    raw = pd.read_excel(XLSX)
    assert len(d) == len(raw) == 540

    # ---- imputation checks -------------------------------------------------
    basis_not_stated = int((raw["Claimant Weekly Income Basis"] == "Not stated").sum())
    csv_inc_blank = int(d[INC].isna().sum())
    raw_inc_blank = int(raw[INC].isna().sum())
    csv_age_blank = int(d[AGE].isna().sum())
    raw_age_blank = int(raw[AGE].isna().sum())

    both = d[[AGE, INC]].dropna()
    zero_inc = int((both[INC] == 0).sum())
    earners = both[both[INC] > 0]

    # Ages must agree row for row where both tables have one. The modelling column is
    # Int64, so a raw half-year (one row records 86.5) rounds -- that is a type cast, not
    # a filled-in value, and is counted separately from a real mismatch.
    m = pd.DataFrame({"csv": d[AGE], "raw": pd.to_numeric(raw[AGE], errors="coerce")}).dropna()
    diff = (m["csv"] - m["raw"]).abs()
    age_rounded = int(((diff > 0) & (diff <= 0.5)).sum())
    age_mismatch = int((diff > 0.5).sum())

    checks = dict(
        n_rows=len(d),
        income_blank_in_csv=csv_inc_blank, income_blank_in_raw=raw_inc_blank,
        income_basis_not_stated=basis_not_stated,
        income_blanks_match_raw=csv_inc_blank == raw_inc_blank,
        income_blanks_match_basis=csv_inc_blank == basis_not_stated,
        age_blank_in_csv=csv_age_blank, age_blank_in_raw=raw_age_blank,
        age_blanks_match_raw=csv_age_blank == raw_age_blank,
        age_values_rounded_to_int=age_rounded,
        age_value_mismatches=age_mismatch,
        complete_pairs=int(len(both)), zero_income_rows=zero_inc,
        plotted_rows=int(len(earners)),
    )
    ok = (checks["income_blanks_match_raw"] and checks["age_blanks_match_raw"]
          and age_mismatch == 0)
    checks["no_imputation_detected"] = bool(ok)

    print("IMPUTATION CHECKS")
    for k, v in checks.items():
        print(f"  {k:28} {v}")
    if not ok:
        print("\nREFUSING TO PLOT: the modelling table does not match the raw workbook.")
        OUT.write_text(json.dumps(checks, indent=2), encoding="utf-8")
        return 1

    # ---- statistics on observed earners ------------------------------------
    x = earners[AGE].to_numpy(float)
    y = earners[INC].to_numpy(float)
    rho = float(stats.spearmanr(x, y).statistic)
    pear = float(stats.pearsonr(np.log1p(x), np.log1p(y)).statistic)

    bins = pd.qcut(earners[AGE], q=6, duplicates="drop")
    grp = earners.groupby(bins, observed=True)[INC].agg(["median", "count"])
    centres = [iv.mid for iv in grp.index]

    checks.update(spearman=round(rho, 3), pearson_loglog=round(pear, 3),
                  band_medians=[dict(age_mid=round(float(c), 1),
                                     median_income=round(float(mv), 1), n=int(n))
                                for c, (mv, n) in zip(centres, grp.to_numpy())])
    OUT.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.scatter(x, y, s=26, alpha=0.45, edgecolor="none", color="#2a78d6",
               label=f"observed earners (n={len(earners)})")
    ax.plot(centres, grp["median"], color="#e34948", lw=2.2, marker="o", ms=6,
            label="median by age sextile")

    lo = stats.linregress(x, np.log1p(y))
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, np.expm1(lo.intercept + lo.slope * xs), color="#898781", ls="--", lw=1.4,
            label=f"log-linear fit (slope {lo.slope:+.4f}/yr)")

    ax.set_xlabel("Claimant Age at injury (years)")
    ax.set_ylabel("Claimant Weekly Income ($)")
    ax.set_title("NSW CTP: age vs pre-accident weekly income\n"
                 f"observed values only, no imputation  |  Spearman {rho:+.3f}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01,
             f"{csv_inc_blank} of 540 rows have no income recorded ({csv_inc_blank/540:.1%}); "
             f"{csv_age_blank} have no age. {zero_inc} rows record $0 income and are excluded.",
             fontsize=7.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(PNG, dpi=150)

    print(f"\nSpearman {rho:+.3f}   log-log Pearson {pear:+.3f}   n={len(earners)}")
    print("\nmedian income by age sextile")
    for c, (mv, n) in zip(centres, grp.to_numpy()):
        print(f"  age ~{float(c):5.1f}   ${mv:9,.0f}   n={int(n)}")
    print(f"\nwrote {PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
