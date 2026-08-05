"""Check the statutory reader against provisions whose correct reading is known.

Tuning a prompt until it produces a known answer is calibration when the answer is a fact
about the text, and fitting when the answer is the causal graph you wanted. Every case here
is the first kind: a sentence from the Act, and what a correct reader should extract from
it. Anyone can verify a case by reading the quote.

Positive cases must be found. NEGATIVE cases must NOT be -- they are provisions where a
correct reader reports nothing, and they exist so that widening the reader's vocabulary
cannot be validated only against cases where an edge is wanted.

Run this after ANY change to the reader or challenger prompts in stage 3c.

Run:  python causal/check_calibration.py
Exit: 0 if every case passes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CASES = HERE / "calibration_provisions.yaml"
READING = HERE / "provenance" / "provision_reading.json"


def main() -> int:
    spec = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    doc = json.loads(READING.read_text(encoding="utf-8"))
    by_prov = {r["provision"]: r for r in doc["readings"]}

    rows, failed = [], 0
    for c in spec["cases"]:
        r = by_prov.get(c["provision"])
        negative = c["expect_input"] is None
        kept = [lk for lk in (r["links"] if r else []) if lk["quote_verbatim"]]

        if r is None:
            ok, got = None, "provision not read"
        elif negative:
            ok = not kept
            got = "no link" if ok else ", ".join(
                f"{lk['input']}->{lk['determines']} ({lk['relation']})" for lk in kept)
        else:
            match = [lk for lk in kept
                     if lk["input"] == c["expect_input"]
                     and lk["determines"] == c["expect_determines"]]
            ok = bool(match)
            if match:
                got = f"{match[0]['relation']}"
                if match[0]["relation"] != c["expect_relation"]:
                    got += f" (expected {c['expect_relation']})"
            else:
                rejected = [lk for lk in (r["links"] or [])
                            if not lk["quote_verbatim"]]
                got = ("nothing" if not r["links"] else
                       "; ".join(f"{lk['input']}->{lk['determines']} "
                                 f"[{lk.get('rejected') or 'kept'}]" for lk in r["links"]))
        failed += (ok is not True)
        rows.append((c["provision"], "NEGATIVE" if negative else
                     f"{c['expect_input']} -> {c['expect_determines']}", ok, got))

    print(f"{'provision':44}{'expected':52}{'result'}")
    for prov, exp, ok, got in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"{prov[:43]:44}{exp[:51]:52}{mark}")
        if not ok:
            print(f"{'':96}got: {got[:110]}")

    print(f"\n{len(rows) - failed}/{len(rows)} calibration cases pass")
    if failed:
        print("The reader or challenger is miscalibrated against provisions whose correct "
              "reading is checkable. Fix before trusting any statutory output.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
