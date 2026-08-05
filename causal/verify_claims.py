"""Check every descriptive statistic asserted in the DAG's prose against ctp/ctp.csv.

The node ``note`` and edge ``mechanism`` strings in ``build_ctp_causal_dag.py`` quote
numbers ("47.7% missing for female claimants vs 37.2% for male"). Those numbers are
typed into prose and are never recomputed by the build, so nothing stops them drifting
away from the data — or from having been wrong to begin with.

This script recomputes each one and fails loudly on a mismatch.

**Scope.** This verifies the *descriptive* claims only. It says nothing about whether
any edge points the right way. The causal assertions were LLM-generated and are not
verifiable from 540 observational rows — see the provenance block in
``build_ctp_causal_dag.py``.

Run:  python causal/verify_claims.py
Exit: 0 if every claim reproduces, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "ctp" / "ctp.csv"


def tolerance(asserted: float) -> float:
    """Half a unit in the last place the claim is quoted to.

    "91% at the top level" is satisfied by anything rounding to 91; "42.0% missing"
    is not. Reading the precision off the asserted value keeps the check as strict as
    the prose actually is, and no stricter.
    """
    text = f"{asserted:g}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** -decimals


def main() -> int:
    d = pd.read_csv(CSV)
    lo = d[d["WPI %"] <= 10]
    hi = d[d["WPI %"] > 10]
    inc_missing = d["Claimant Weekly Income"].isna()
    both = d.dropna(subset=["Non-Economic Loss", "Future Economic Loss"])
    nel_by_nature = d.groupby("Nature")["WPI %"].apply(lambda s: s.isna().mean())
    inc_by_gender = d.groupby("Claimant Gender")["Claimant Weekly Income"].apply(
        lambda s: s.isna().mean())
    med_inc = d.groupby("Claimant Gender")["Claimant Weekly Income"].median()
    med_award = d.groupby("Nature")["Lump Sum"].median()
    heads = both["Non-Economic Loss"] + both["Future Economic Loss"]

    # (where the claim appears, the claim, asserted value, computed value)
    checks = [
        ("Nature -> R_WPI", "WPI blank rate, Damages", 42.0,
         100 * nel_by_nature["Damages"]),
        ("Nature -> R_WPI", "WPI blank rate, Settlement Approval", 49.4,
         100 * nel_by_nature["Settlement Approval"]),
        ("WPI % -> R_NEL", "NEL blank rate at WPI<=10", 18.4,
         100 * lo["Non-Economic Loss"].isna().mean()),
        ("WPI % -> R_NEL", "NEL blank rate at WPI>10", 3.0,
         100 * hi["Non-Economic Loss"].isna().mean()),
        ("Claimant Gender -> R_Income", "income blank rate, female", 47.7,
         100 * inc_by_gender["Female"]),
        ("Claimant Gender -> R_Income", "income blank rate, male", 37.2,
         100 * inc_by_gender["Male"]),
        ("R_Income -> R_FEL", "FEL blank | income blank", 50.4,
         100 * d.loc[inc_missing, "Future Economic Loss"].isna().mean()),
        ("R_Income -> R_FEL", "FEL blank | income present", 4.1,
         100 * d.loc[~inc_missing, "Future Economic Loss"].isna().mean()),
        ("Claimant Gender -> Claimant Weekly Income", "male median income premium %", 29.0,
         100 * (med_inc["Male"] / med_inc["Female"] - 1)),
        ("note: Non-Economic Loss", "share $0 at WPI<=10 (blanks in denominator)", 77.0,
         100 * (lo["Non-Economic Loss"] == 0).mean()),
        ("note: Liability Clarity", "share at top level", 91.9,
         100 * (d["Liability Clarity"] == 2).mean()),
        ("note: Nature", "Damages median award multiple", 2.6,
         med_award["Damages"] / med_award["Settlement Approval"]),
        ("spec 5.2", "share of recorded NEL exactly $0", 44.2,
         100 * (d["Non-Economic Loss"].dropna() == 0).mean()),
        ("spec 5.2", "share of recorded FEL exactly $0", 9.7,
         100 * (d["Future Economic Loss"].dropna() == 0).mean()),
        ("spec 5.3", "NEL+FEL <= Lump Sum, %", 95.4,
         100 * (heads <= both["Lump Sum"]).mean()),
        ("spec 5.3", "median (NEL+FEL)/Lump Sum", 0.79,
         (heads / both["Lump Sum"]).median()),
    ]

    failures = []
    print(f"{'':2} {'claim':44} {'asserted':>9} {'actual':>9} {'tol':>6}")
    print("-" * 75)
    for where, what, asserted, actual in checks:
        actual = float(actual)
        tol = tolerance(asserted)
        ok = abs(actual - asserted) <= tol
        print(f"{'ok' if ok else 'XX':2} {what:44} {asserted:9.2f} {actual:9.2f} {tol:6.3f}")
        if not ok:
            failures.append((where, what, asserted, actual, tol))

    print("-" * 75)
    if failures:
        print(f"FAIL: {len(failures)}/{len(checks)} asserted statistics do not reproduce.")
        for where, what, asserted, actual, tol in failures:
            print(f"  {where}: {what} - asserted {asserted}, computed {actual:.3f} "
                  f"(tolerance {tol})")
        return 1

    print(f"OK: all {len(checks)} asserted statistics reproduce from {CSV.name}.")
    print("NOTE: this verifies descriptive claims only. No edge direction is verified here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
