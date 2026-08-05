"""Stage 0 — how was each ctp.csv column actually measured?

Before any association can be read as evidence about the world, we need to know which
columns are facts the tribunal recorded and which are codings a language model assigned
by reading the decision. Two LLM-coded columns can correlate because the coder used one
cue for both; that is an artefact of the coding pass, not structure in the scheme.

Evidence comes from the raw workbook, which carries corroborating fields the modelling
table drops:

  Non-Economic Loss      <- 'Non-Economic Loss Status'   (Awarded / Nil / Not addressed)
  Future Economic Loss   <- 'Future Economic Loss Status'
  Claimant Weekly Income <- 'Claimant Weekly Income Basis' ("Not stated" == blank)
  Nature                 <- 'Result'                     (procedural outcome)

A column with a corroborating field is *extracted* — the value was read off the decision
and the workbook records how. A column with none of that, on an invented ordinal scale,
is *coded*.

Run:  python causal/stage0_measurement_ledger.py
Out:  causal/provenance/measurement_ledger.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "ctp" / "raw" / "ctp_impairment_lump_sum.xlsx"
CSV = ROOT / "ctp" / "ctp.csv"
OUT = Path(__file__).resolve().parent / "provenance" / "measurement_ledger.json"

# kind: "extracted" = read off the decision;  "coded" = LLM-assigned ordinal scale
# corroborator: raw-workbook column that independently evidences the value (or None)
LEDGER = {
    "Lump Sum": dict(kind="extracted", corroborator="Result",
                     note="Dollar total stated in the decision."),
    "WPI %": dict(kind="extracted", corroborator=None,
                  note="Whole Person Impairment from the medical assessment as recorded."),
    "Non-Economic Loss": dict(kind="extracted", corroborator="Non-Economic Loss Status",
                              note="Status separates a statutory Nil from a head never addressed."),
    "Future Economic Loss": dict(kind="extracted", corroborator="Future Economic Loss Status",
                                 note="Status separates Nil from not addressed."),
    "Claimant Weekly Income": dict(kind="extracted", corroborator="Claimant Weekly Income Basis",
                                   note="Basis records the earnings measure used, or 'Not stated'."),
    "Claimant Age": dict(kind="extracted", corroborator="Claimant Age At Decision",
                         note="Age at injury; a second age field exists at decision date."),
    "Claimant Gender": dict(kind="extracted", corroborator=None, note="Stated in the decision."),
    "Nature": dict(kind="extracted", corroborator="Result",
                   note="Procedural route: settlement approval vs damages assessment."),
    "Injury Burden Intensity": dict(kind="coded", corroborator=None, note="LLM ordinal 0-4."),
    "Treatment Burden": dict(kind="coded", corroborator=None, note="LLM ordinal 0-3."),
    "Work Impact Severity": dict(kind="coded", corroborator=None, note="LLM ordinal 0-3."),
    "Legal Procedural Complexity": dict(kind="coded", corroborator=None, note="LLM ordinal 0-3."),
    "Psychological Injury Emphasis": dict(kind="coded", corroborator=None, note="LLM ordinal 0-2."),
    "Liability Clarity": dict(kind="coded", corroborator=None, note="LLM ordinal 0-2."),
    "Causation Complexity": dict(kind="coded", corroborator=None, note="LLM ordinal 0-2."),
    "Pre-existing Condition Salience": dict(kind="coded", corroborator=None, note="LLM ordinal 0-2."),
}


def main() -> None:
    csv = pd.read_csv(CSV)
    raw = pd.read_excel(XLSX)
    assert len(csv) == len(raw) == 540, "row counts diverged; ledger assumes aligned tables"

    out = {}
    for col, spec in LEDGER.items():
        rec = dict(spec)
        rec["missing_rate"] = round(float(csv[col].isna().mean()), 4)
        corr = spec["corroborator"]
        if corr and corr in raw.columns:
            vc = raw[corr].astype(str).value_counts()
            rec["corroborator_levels"] = {k: int(v) for k, v in vc.head(6).items()}
            rec["corroborator_fill"] = round(float(raw[corr].notna().mean()), 4)
        out[col] = rec

    # Missingness in the modelling table is a recorded status in the raw workbook, not an
    # unexplained blank. Check that the two line up before anyone models it as latent.
    recovered = {
        "Non-Economic Loss": dict(
            csv_blank=int(csv["Non-Economic Loss"].isna().sum()),
            raw_not_addressed=int((raw["Non-Economic Loss Status"] == "Not addressed").sum()),
            raw_nil=int((raw["Non-Economic Loss Status"] == "Nil").sum())),
        "Future Economic Loss": dict(
            csv_blank=int(csv["Future Economic Loss"].isna().sum()),
            raw_not_addressed=int((raw["Future Economic Loss Status"] == "Not addressed").sum()),
            raw_nil=int((raw["Future Economic Loss Status"] == "Nil").sum())),
        "Claimant Weekly Income": dict(
            csv_blank=int(csv["Claimant Weekly Income"].isna().sum()),
            raw_not_stated=int((raw["Claimant Weekly Income Basis"] == "Not stated").sum())),
    }

    n_coded = sum(v["kind"] == "coded" for v in out.values())
    doc = dict(
        n_columns=len(out), n_extracted=len(out) - n_coded, n_coded=n_coded,
        columns=out,
        missingness_is_recorded=recovered,
        caveat=(
            f"{n_coded} of {len(out)} columns are LLM-assigned ordinals derived from the same "
            "decision text by the same pass. Associations between two coded columns are "
            "confounded by the coder and must not be read as scheme mechanism without "
            "independent evidence."),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"extracted={doc['n_extracted']}  coded={doc['n_coded']}")
    print("\ncoded (coder-confounded with each other):")
    for c, v in out.items():
        if v["kind"] == "coded":
            print(f"  {c}")
    print("\nmissingness recovered from the raw workbook:")
    for c, v in recovered.items():
        print(f"  {c:24} csv_blank={v['csv_blank']:4d}  raw={ {k: x for k, x in v.items() if k != 'csv_blank'} }")


if __name__ == "__main__":
    main()
