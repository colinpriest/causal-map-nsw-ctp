"""Stage 7a — have a second, independent reader re-code the eight constructed columns.

Eight of the sixteen columns are ordinal scores a language model assigned by reading each
decision. Every association among them is therefore open to the same objection: a reader
who sees a passage describing a catastrophic case may mark several scales up at once, and
the columns would move together with nothing causal between them.

Stage 5 tried to settle this by asking a model, and the ablation showed the answer was
whatever the prompt suggested -- 28/28 `measurement_artifact` with the warning, 0/28
without. The question cannot be answered by asking. It needs a second reader.

WHAT THIS DOES
A seeded random sample of decisions is re-coded from the VERBATIM decision text in the raw
workbook (`Key Paragraphs`, plus `Catchwords`), by a model that has not seen the original
codes. Stage 7b then compares the two codings.

The decisive comparison is not per-column agreement, it is the CROSS-CODER association.
If `Injury Burden Intensity` and `Psychological Injury Emphasis` are related in the world,
then coder A's injury score should predict coder B's psychological score. If the
relationship is a halo inside one reader's head, it exists within each coder's columns and
vanishes when the coders are crossed. Stage 7b computes both.

A LIMITATION TO STATE PLAINLY
The original coding rubric is not recorded anywhere in this repository -- only the observed
ranges. The scale descriptions below are RECONSTRUCTED from the column names and ranges, so
disagreement here mixes genuine unreliability with rubric drift. That makes low agreement
ambiguous. It does not weaken the cross-coder test, which asks whether a relationship
survives a change of reader at all, and it makes high agreement strong evidence.

The second reader is also a language model, so this is independence of context and prompt,
not independence of kind. Two models may share a bias that no amount of resampling reveals.

Run:  python causal/stage7a_recode_sample.py [--n 80] [--model gpt-4o]
Out:  causal/provenance/llm_cache/recode-*.json
      causal/provenance/recoded_sample.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XLSX = ROOT / "ctp" / "raw" / "ctp_impairment_lump_sum.xlsx"
CACHE = HERE / "provenance" / "llm_cache"
OUT = HERE / "provenance" / "recoded_sample.json"

DEFAULT_N = 80
DEFAULT_MODEL = "gpt-4o"
SEED = 2026
MAX_CHARS = 9000

# Reconstructed from column names and observed ranges. NOT the original rubric -- see the
# limitation in the module docstring.
SCALES = {
    "Injury Burden Intensity": (0, 4, "overall physical burden of the injuries: 0 minimal, "
                                      "4 catastrophic/multiple major injuries"),
    "Treatment Burden": (0, 3, "extent of treatment, surgery and ongoing care: 0 none or "
                               "minimal, 3 extensive surgery and long-term care"),
    "Work Impact Severity": (0, 3, "loss of working capacity: 0 none, 3 total incapacity"),
    "Legal Procedural Complexity": (0, 3, "procedural complexity of the proceeding: "
                                          "0 straightforward, 3 heavily contested"),
    "Psychological Injury Emphasis": (0, 2, "weight the decision gives to psychological "
                                            "injury: 0 none, 2 central"),
    "Liability Clarity": (0, 2, "how clear fault is: 0 seriously disputed, 2 clear or "
                                "admitted"),
    "Causation Complexity": (0, 2, "difficulty attributing the condition to the accident: "
                                   "0 straightforward, 2 heavily contested"),
    "Pre-existing Condition Salience": (0, 2, "how prominent a prior condition is in the "
                                              "decision: 0 absent, 2 central"),
}

SYSTEM = (
    "You are coding NSW motor accident decisions for a research dataset. Read the decision "
    "text supplied and assign each ordinal score from the text alone.\n\n"
    "Score each dimension INDEPENDENTLY. A decision describing a severe case will score "
    "high on some dimensions and not others: severe physical injury does not by itself "
    "imply a contested proceeding, a disputed liability, or a prominent psychological "
    "component. Judge each scale against its own definition and resist letting an overall "
    "impression of severity carry across the scales.\n\n"
    "If the text does not support a judgement on a dimension, give the lowest score rather "
    "than guessing upward, and say so in `uncertain`."
)

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": list(SCALES) + ["uncertain"],
    "properties": {
        **{k: {"type": "integer", "minimum": lo, "maximum": hi, "description": desc}
           for k, (lo, hi, desc) in SCALES.items()},
        "uncertain": {"type": "array", "items": {"type": "string", "enum": list(SCALES)},
                      "description": "Dimensions the text did not really support."},
    },
}


def source_text(row) -> str:
    parts = []
    for col in ("Catchwords", "Key Paragraphs"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            parts.append(f"--- {col} ---\n{v.strip()}")
    return "\n\n".join(parts)[:MAX_CHARS]


def recode(client, model, case_id, text, refresh):
    scales = "\n".join(f"  {k} ({lo}-{hi}): {desc}" for k, (lo, hi, desc) in SCALES.items())
    user = (f"DECISION TEXT:\n{text}\n\n"
            f"Assign each score:\n{scales}")
    key = hashlib.sha256(json.dumps([model, SYSTEM, user, SCHEMA],
                                    sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"recode-{key}.json"
    if cached.exists() and not refresh:
        rec = json.loads(cached.read_text(encoding="utf-8"))
        return rec["response"], dict(rec["meta"], cache="hit")
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "recode", "strict": True, "schema": SCHEMA}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    meta = dict(model=resp.model, prompt_sha256=key, cache="miss", case=case_id,
                requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(dict(meta=meta, system=SYSTEM, user=user,
                                      response=parsed), indent=2), encoding="utf-8")
    return parsed, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found")
        return 1
    client = OpenAI()

    raw = pd.read_excel(XLSX)
    sample = raw.sample(n=min(args.n, len(raw)), random_state=SEED).sort_index()
    print(f"re-coding {len(sample)} of {len(raw)} decisions "
          f"(seed {SEED})  model={args.model}\n")

    rows, skipped = [], 0
    for idx, row in sample.iterrows():
        text = source_text(row)
        if len(text) < 400:
            skipped += 1
            continue
        codes, meta = recode(client, args.model, str(row.get("Case Name"))[:80],
                             text, args.refresh)
        rows.append(dict(
            row_index=int(idx), case=str(row.get("Case Name"))[:120],
            url=row.get("URL"), source_chars=len(text),
            original={k: (None if pd.isna(row[k]) else int(row[k])) for k in SCALES},
            recoded={k: int(codes[k]) for k in SCALES},
            recoder_uncertain=codes.get("uncertain", []),
            prompt_sha256=meta["prompt_sha256"],
        ))
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} coded")

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=args.model, seed=SEED, n_requested=args.n, n_coded=len(rows),
        n_skipped_short_text=skipped,
        source_fields=["Catchwords", "Key Paragraphs"], max_chars=MAX_CHARS,
        scales={k: dict(min=lo, max=hi, description=d) for k, (lo, hi, d) in SCALES.items()},
        rubric_is_reconstructed=True,
        limitation=("The original coding rubric is not recorded in this repository. These "
                    "scale descriptions were reconstructed from column names and observed "
                    "ranges, so disagreement mixes unreliability with rubric drift. The "
                    "second reader is also a language model: independent in context and "
                    "prompt, not in kind."),
        system_prompt=SYSTEM, codings=rows,
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(rows)} coded, {skipped} skipped for short text)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
