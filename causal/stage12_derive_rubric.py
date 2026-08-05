"""Stage 12 — recover the coding rubric from the codings themselves.

The eight ordinal columns were assigned by a language model reading each decision, and the
rubric it used is not recorded anywhere in this project. Stage 7 therefore had to invent
scale descriptions in order to re-code a sample, which means its disagreement figures mix
two different things: genuine unreliability, and my reconstruction differing from whatever
the original coder actually did. A kappa of 0.23 on `Treatment Burden` could be either.

The rubric cannot be retrieved, but it can be RECOVERED. Every level of every scale has
decisions sitting at it, and the decision text is in the raw workbook. So for each level a
model is shown the text of decisions the original coder placed there -- and none of the
text from any other level -- and asked to state what distinguishes them. It never sees the
level's number or name, only the documents, so it is describing the coder's behaviour
rather than rationalising a label.

The result is an empirical rubric: what the original coder was in fact responding to at
each level. Feeding that back into stage 7 makes the re-code a test of reliability rather
than a test of whether two readers guessed the same scale.

WHAT THIS CANNOT FIX
A recovered rubric describes what the coder DID, not what it was TOLD. If the original
instructions were poor, this reproduces the poor behaviour faithfully. It removes rubric
drift as a confound; it does not validate the scale.

Run:  python causal/stage12_derive_rubric.py [--per-level 6]
Out:  causal/provenance/derived_rubric.json
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
import yaml
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XLSX = ROOT / "ctp" / "raw" / "ctp_impairment_lump_sum.xlsx"
SPEC = ROOT / "ctp" / "columns.yaml"
CACHE = HERE / "provenance" / "llm_cache"
OUT = HERE / "provenance" / "derived_rubric.json"

DEFAULT_MODEL = "gpt-4o"
PER_LEVEL = 6
CHARS = 2200
SEED = 2026

SYSTEM = (
    "You are reverse-engineering how a set of documents was sorted into groups.\n\n"
    "You will see several groups of extracts from decided motor accident claims. Someone "
    "sorted them, and you are not told what the sorting criterion was, what the groups are "
    "called, or which end of any scale they represent. The groups are presented in the "
    "order the sorter used.\n\n"
    "For each group, state what the documents in it have in common that the documents in "
    "the OTHER groups do not. Describe what is actually present in the text -- the "
    "injuries, the treatment, the argument -- rather than guessing at an abstract label. "
    "If two adjacent groups are not distinguishable from their contents, say so plainly: "
    "that is a finding about the sorting, not a failure on your part."
)

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["criterion", "groups", "indistinguishable_pairs", "notes"],
    "properties": {
        "criterion": {"type": "string",
                      "description": "In one sentence, what the sorter appears to have "
                                     "been responding to."},
        "groups": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["group", "description", "distinguishing_features"],
                "properties": {
                    "group": {"type": "integer"},
                    "description": {"type": "string",
                                    "description": "What documents in this group look "
                                                   "like, in concrete terms."},
                    "distinguishing_features": {
                        "type": "array", "items": {"type": "string"},
                        "description": "What separates this group from its neighbours."},
                },
            },
        },
        "indistinguishable_pairs": {
            "type": "array", "items": {"type": "string"},
            "description": "Adjacent groups whose contents do not differ, e.g. '1 and 2'.",
        },
        "notes": {"type": "string"},
    },
}


def extract(row) -> str:
    parts = []
    for col in ("Catchwords", "Key Paragraphs"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            parts.append(" ".join(v.split()))
    return " ".join(parts)[:CHARS]


def derive(client, model, column, groups, refresh):
    blocks = []
    for lvl, docs in groups:
        blocks.append(f"=== GROUP {lvl + 1} ({len(docs)} documents) ===\n" +
                      "\n---\n".join(docs))
    user = ("Groups of extracts, in the sorter's own order:\n\n" + "\n\n".join(blocks) +
            "\n\nWhat distinguishes each group from the others?")
    key = hashlib.sha256(json.dumps([model, SYSTEM, user, SCHEMA],
                                    sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"rubric-{key}.json"
    if cached.exists() and not refresh:
        rec = json.loads(cached.read_text(encoding="utf-8"))
        return rec["response"], dict(rec["meta"], cache="hit")
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "rubric", "strict": True, "schema": SCHEMA}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    meta = dict(model=resp.model, prompt_sha256=key, cache="miss", column=column,
                requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(dict(meta=meta, system=SYSTEM, user=user,
                                      response=parsed), indent=2), encoding="utf-8")
    return parsed, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=PER_LEVEL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found")
        return 1
    client = OpenAI()

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    ordinals = [c for c in spec["columns"] if c.get("dtype") == "ordinal"]
    raw = pd.read_excel(XLSX)
    raw["_text"] = raw.apply(extract, axis=1)

    out = {}
    for c in ordinals:
        col = c["name"]
        lo, hi = c["range"]
        groups, counts = [], {}
        for lvl in range(lo, hi + 1):
            sub = raw[(raw[col] == lvl) & (raw["_text"].str.len() > 500)]
            counts[lvl] = int(len(sub))
            if len(sub) < 2:
                continue
            docs = sub.sample(n=min(args.per_level, len(sub)),
                              random_state=SEED)["_text"].tolist()
            groups.append((lvl - lo, docs))
        if len(groups) < 2:
            out[col] = dict(skipped="fewer than two usable levels", level_counts=counts)
            print(f"  {col:32} skipped (levels too sparse)")
            continue

        r, meta = derive(client, args.model, col, groups, args.refresh)
        levels = {}
        for g in r["groups"]:
            lvl = lo + g["group"] - 1
            levels[lvl] = dict(description=g["description"],
                               distinguishing=g["distinguishing_features"])
        out[col] = dict(criterion=r["criterion"], levels=levels,
                        indistinguishable=r.get("indistinguishable_pairs", []),
                        notes=r.get("notes", ""), level_counts=counts,
                        n_sampled_per_level=args.per_level, llm=meta)
        flag = ("  INDISTINGUISHABLE: " + ", ".join(r["indistinguishable_pairs"])
                if r.get("indistinguishable_pairs") else "")
        print(f"  {col:32} {r['criterion'][:60]}{flag}")

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=args.model, seed=SEED, chars_per_document=CHARS,
        method=("For each ordinal, decisions the original coder placed at each level were "
                "sampled and shown to a model as unlabelled groups, in order. The model "
                "was told neither the column name nor what the levels mean, so it "
                "describes what the coder responded to rather than rationalising a label."),
        limitation=("This recovers what the original coder DID, not what it was told. A "
                    "poor original rubric is reproduced faithfully. It removes rubric "
                    "drift as a confound in stage 7; it does not validate any scale."),
        system_prompt=SYSTEM, columns=out,
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    flagged = [c for c, v in out.items() if v.get("indistinguishable")]
    if flagged:
        print(f"columns with levels the text cannot separate: {', '.join(flagged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
