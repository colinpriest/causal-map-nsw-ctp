"""Stage 3c — read the statute and report what each provision makes an input to what.

Stage 3a found which variables cases differ on when a provision is cited: association.
Stage 3b retrieved the provision text: primary source. This stage is the only step in the
pipeline that asks a language model anything, and it is deliberately kept blind.

WHAT THE MODEL SEES
    * the verbatim text of ONE provision, from causal/provenance/provision_text.json
    * the list of 16 modelling variables, with neutral descriptions

WHAT THE MODEL DOES NOT SEE
    * any correlation, effect size or p-value from stage 3a
    * any case data, any other provision, any part of the existing DAG

That blindness is the point. If the model were shown that cases citing s 4.7 differ on
Claimant Weekly Income, it would find a reason for it. Reading the provision cold means
its answer is independent evidence, and stage 3d can require the two legs to agree.

ANTI-FABRICATION
Every claimed link must carry a `quote` that appears VERBATIM in the provision text. The
quote is checked programmatically against the source after the response returns; a link
whose quote cannot be found is discarded and recorded as `quote_not_found`. A model
cannot assert a statutory input without pointing at the words.

REPRODUCIBILITY
Requests are cached under causal/provenance/llm_cache/ keyed by a sha256 of
(model, prompt, schema). A re-run with an unchanged prompt costs nothing and returns the
identical answer. The cache is committed, so the LLM step is auditable rather than merely
repeatable-in-principle. temperature=0.

Run:  python causal/stage3c_read_provisions.py [--model gpt-4o] [--refresh]
Out:  causal/provenance/llm_cache/*.json   (verbatim request + response)
      causal/provenance/provision_reading.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEXT = HERE / "provenance" / "provision_text.json"
LEDGER = HERE / "provenance" / "measurement_ledger.json"
CACHE = HERE / "provenance" / "llm_cache"
OUT = HERE / "provenance" / "provision_reading.json"

DEFAULT_MODEL = "gpt-4o"

# Neutral descriptions. Deliberately free of any claim about what causes what.
# Definitions come from the data dictionary, not from a copy kept here. Stage 3c held its
# own wording for months and it drifted from ctp/columns.yaml -- which matters for the
# statutory leg specifically, because the reader has to recognise that the Act's
# "impairment of earning capacity" is the concept `Work Impact Severity` measures.
SPEC = ROOT / "ctp" / "columns.yaml"
_spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
VARIABLES = {c["name"]: " ".join(c["definition"].split()) for c in _spec["columns"]}


SYSTEM = (
    "You are reading a single provision of New South Wales motor accident legislation. "
    "Report ONLY what this provision's text states. Do not use background knowledge of "
    "the scheme, other provisions, case law, or what you expect the answer to be.\n\n"
    "The dataset variables are PROXIES FOR CONCEPTS, and the legislation will not use "
    "their names. Match on meaning, not wording. If a provision speaks of 'future earning "
    "capacity', that is the concept the variable 'Claimant Weekly Income' measures, and "
    "you should report the link. If it speaks of 'degree of permanent impairment', that is "
    "'WPI %'. Do not decline a link merely because the provision uses different words.\n\n"
    "Separate two things in your answer:\n"
    "  1. WHAT THE TEXT SAYS. Every link must carry a quote copied from the text you were "
    "given, containing the words that express the concept. You may elide with '...'. If "
    "you cannot copy such a quote, do not report the link.\n"
    "  2. HOW YOU MAPPED IT. State, separately, why the quoted concept corresponds to that "
    "dataset variable. This is your interpretation and will be recorded as such.\n\n"
    "A provision that says what a head of damage is FOR is stating an input to that head. "
    "An Act writes this as a list of what may be awarded -- 'the only damages that may be "
    "awarded for economic loss are ... damages for the deprivation or impairment of "
    "earning capacity' -- and that sentence makes impairment of earning capacity an input, "
    "with relation `defines_scope`. Do not pass over these because they read as "
    "definitions rather than rules; defining what a payment compensates IS specifying what "
    "determines it.\n\n"
    "A quoted phrase often names SEVERAL distinct quantities joined by 'or' or 'and'. "
    "Report one link per quantity, not one for the phrase. 'economic loss due to loss of "
    "earnings OR the deprivation or impairment of earning capacity' names two different "
    "things -- what the claimant was earning, and how much earning capacity was lost -- "
    "and different dataset variables measure each. Collapsing them into a single link "
    "silently drops one of the statutory inputs.\n\n"
    "A concept the provision names that no dataset variable measures should be listed in "
    "`unmapped_concepts`, not forced onto a variable that does not fit.\n\n"
    "TWO TRAPS TO AVOID.\n"
    "  (a) If the THING THE PROVISION GOVERNS is not one of the dataset variables, report "
    "NO links at all, whatever else the provision mentions. A provision about an insurer's "
    "duty to PAY FOR treatment does not govern how much treatment a claimant received: "
    "those are different quantities. Do not map preconditions onto variables just because "
    "the provision has preconditions.\n"
    "  (b) Do not import a variable's name into your reading. A provision that scopes "
    "payments to injury 'caused by the fault of the driver' states a condition; it says "
    "nothing about how DIFFICULT causation is to establish, so it is not evidence about a "
    "variable measuring difficulty. Ask whether the provision discusses the quantity the "
    "variable measures, not whether it uses a similar word."
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["provision_subject", "determines", "inputs", "links", "notes"],
    "properties": {
        "provision_subject": {
            "type": "string",
            "description": "One sentence: what this provision is about, in its own terms.",
        },
        "determines": {
            "type": "array", "items": {"type": "string", "enum": list(VARIABLES)},
            "description": "Variables whose value or availability this provision governs. "
                           "Empty if none.",
        },
        "inputs": {
            "type": "array", "items": {"type": "string", "enum": list(VARIABLES)},
            "description": "Variables this provision names as bearing on that "
                           "determination. Empty if none.",
        },
        "links": {
            "type": "array",
            "description": "One entry per input->determined pair the text supports, "
                           "matching on MEANING rather than wording.",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["concept_in_text", "quote", "input", "mapping_rationale",
                             "determines", "relation", "confidence"],
                "properties": {
                    "concept_in_text": {
                        "type": "string",
                        "description": "The concept as the provision expresses it, e.g. "
                                       "'future earning capacity'.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Copied from the provision text and containing the "
                                       "words of concept_in_text. Elide with '...' if long.",
                    },
                    "input": {"type": "string", "enum": list(VARIABLES)},
                    "mapping_rationale": {
                        "type": "string",
                        "description": "Why that concept corresponds to that dataset "
                                       "variable. Your interpretation, recorded as such.",
                    },
                    "determines": {"type": "string", "enum": list(VARIABLES)},
                    "relation": {
                        "type": "string",
                        "enum": ["defines_scope", "gate", "formula_input", "limit",
                                 "adjustment", "other"],
                        "description": "defines_scope: the provision states that this head "
                                       "of damage IS FOR the input -- it compensates that "
                                       "thing. This is the commonest statutory shape and "
                                       "the easiest to miss, because an Act states it as "
                                       "an enumeration of what may be awarded rather than "
                                       "as a rule about what determines what. "
                                       "'The only damages that may be awarded for economic "
                                       "loss are ... damages for impairment of earning "
                                       "capacity' MAKES impairment of earning capacity an "
                                       "input to that head. "
                                       "gate: decides whether the head is available at all. "
                                       "formula_input: enters a calculation of the amount. "
                                       "limit: caps or floors it. adjustment: discounts or "
                                       "increases it.",
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
        "unmapped_concepts": {
            "type": "array", "items": {"type": "string"},
            "description": "Concepts the provision names that no dataset variable measures.",
        },
        "notes": {"type": "string", "description": "Anything ambiguous. May be empty."},
    },
}
SCHEMA["required"].append("unmapped_concepts")


def normalise(s: str) -> str:
    """Whitespace- and punctuation-insensitive form, for verbatim quote checking."""
    s = s.replace("—", "-").replace("–", "-")
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


MIN_FRAGMENT = 12   # characters; shorter fragments match by accident


def verify_quote(quote: str, source: str) -> tuple[bool, str, list[str]]:
    """Is `quote` genuinely lifted from `source`?

    Models quote statute the way lawyers do, eliding the middle with an ellipsis:
    "No damages for non-economic loss may be awarded...unless the degree of permanent
    impairment...is greater than 10%." Every word of that IS in the provision, so a plain
    substring test rejects a perfectly honest quote.

    Each fragment between ellipses must therefore appear verbatim AND in order. That still
    makes a fabricated quote essentially impossible -- the model cannot invent words -- while
    accepting the way the text is actually cited.
    """
    hay = normalise(source)
    parts = [normalise(p) for p in re.split(r"\s*(?:\.{3}|…|\.\s\.\s\.)\s*", quote)]
    parts = [p for p in parts if len(p) >= MIN_FRAGMENT]
    if not parts:
        return False, "no fragment long enough to verify", []
    pos = 0
    for frag in parts:
        found = hay.find(frag, pos)
        if found < 0:
            return False, f"fragment not in source: {frag[:60]!r}", parts
        pos = found + len(frag)
    return True, "elided" if len(parts) > 1 else "exact", parts


def ask(client: OpenAI, model: str, provision: str, text: str,
        refresh: bool) -> tuple[dict, dict]:
    variables = "\n".join(f"  - {k}: {v}" for k, v in VARIABLES.items())
    user = (
        f"PROVISION: {provision}\n\n"
        f"--- BEGIN PROVISION TEXT ---\n{text}\n--- END PROVISION TEXT ---\n\n"
        f"The dataset records these variables about decided claims:\n{variables}\n\n"
        "Using ONLY the provision text above, report which of these variables the "
        "provision governs, and which concepts it names as bearing on that determination. "
        "Match on MEANING: the provision will not use these variable names. Quote the "
        "words that carry the concept, then say separately how you mapped it."
    )
    key = hashlib.sha256(
        json.dumps([model, SYSTEM, user, SCHEMA], sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"{key}.json"
    if cached.exists() and not refresh:
        rec = json.loads(cached.read_text(encoding="utf-8"))
        return rec["response"], dict(rec["meta"], cache="hit")

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "provision_reading", "strict": True, "schema": SCHEMA}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    meta = dict(model=resp.model, prompt_sha256=key,
                requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens, cache="miss")
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(
        dict(meta=meta, system=SYSTEM, user=user, response=parsed), indent=2),
        encoding="utf-8")
    return parsed, meta


CHALLENGE_SYSTEM = (
    "You are checking a claim about a statutory provision. Refute it if it can be refuted, "
    "but ONLY on the two grounds set out below. Do not invent stricter grounds.\n\n"
    "WHAT THE CLAIM MEANS. It asserts the provision makes one quantity bear on another in "
    "one of these specific ways. Judge it against the relation actually claimed:\n"
    "  defines_scope - the provision states that the second IS COMPENSATION FOR the "
    "first. An Act specifies this by enumerating what a head of damage may be awarded "
    "for, and that enumeration IS a statement of what the head responds to. Do NOT refuse "
    "such a claim on the ground that the provision 'only says damages may be awarded for "
    "such loss' -- that is precisely what defines_scope asserts, and refusing it there "
    "rejects the commonest statutory shape there is.\n"
    "  gate          - makes the second available or unavailable by reference to the "
    "first, e.g. a threshold.\n"
    "  formula_input - the first enters a calculation of the second.\n"
    "  limit         - the first caps or floors the second. A provision that disregards "
    "earnings above a maximum DOES limit an earnings-based award: that is a sound limit "
    "claim, not a failed determination claim.\n"
    "  adjustment    - the first discounts or increases the second. A contributory "
    "negligence provision DOES adjust damages by reference to fault: that is a sound "
    "adjustment claim.\n\n"
    "TWO THINGS THAT ARE NOT GROUNDS FOR REFUSAL.\n"
    "  1. The provision does not use the variable's name. It never will -- legislation "
    "does not use dataset column names, and requiring it would refute every true claim. "
    "'Net weekly earnings' IS pre-accident weekly income. 'Contributory negligence' IS "
    "about clarity of fault. 'Degree of permanent impairment' IS whole person impairment. "
    "Conceptual correspondence is required of you, not optional.\n"
    "  2. The relationship is indirect, partial, conditional, or shared with other "
    "factors. Almost every statutory rule is. Only the claimed relation has to hold.\n\n"
    "THE TWO REAL GROUNDS FOR REFUSAL.\n"
    "  (a) SUBJECT MISMATCH. The provision governs a different quantity from the one the "
    "claim says it bears on. A provision about an insurer's duty to PAY FOR treatment does "
    "not govern HOW MUCH TREATMENT the claimant underwent. A provision conferring "
    "jurisdiction to assess a claim does not make one assessed quantity determine another.\n"
    "  (b) WORD MATCHING. The claim followed a word rather than a meaning -- reading "
    "'injury caused by the fault of the driver' (a fault condition) as being about how "
    "DIFFICULT causation is to establish.\n\n"
    "Refuse on (a) or (b). Otherwise the claim stands."
)

CHALLENGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["objection", "failure_mode", "sound"],
    "properties": {
        "objection": {"type": "string",
                      "description": "The strongest case against the claim, in one or two "
                                     "sentences. Required even if you end up accepting it."},
        "failure_mode": {"type": "string",
                         "enum": ["subject_mismatch", "word_matching", "precondition_only",
                                  "none"]},
        "sound": {"type": "boolean"},
    },
}


def challenge(client: OpenAI, model: str, provision: str, text: str,
              link: dict, refresh: bool) -> tuple[dict, dict]:
    user = (
        f"PROVISION: {provision}\n\n"
        f"--- BEGIN PROVISION TEXT ---\n{text}\n--- END PROVISION TEXT ---\n\n"
        f"THE CLAIM: this provision makes '{link['input']}' "
        f"({VARIABLES[link['input']]}) bear on '{link['determines']}' "
        f"({VARIABLES[link['determines']]}), as a {link['relation']}.\n"
        f"Quoted in support: \"{link['quote']}\"\n"
        f"Mapping offered: {link['mapping_rationale']}\n\n"
        "Refute this claim if it can be refuted."
    )
    key = hashlib.sha256(json.dumps(
        [model, CHALLENGE_SYSTEM, user, CHALLENGE_SCHEMA],
        sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"challenge-{key}.json"
    if cached.exists() and not refresh:
        rec = json.loads(cached.read_text(encoding="utf-8"))
        return rec["response"], dict(rec["meta"], cache="hit")

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": CHALLENGE_SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "challenge", "strict": True, "schema": CHALLENGE_SCHEMA}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    meta = dict(model=resp.model, prompt_sha256=key, cache="miss",
                requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(
        dict(meta=meta, system=CHALLENGE_SYSTEM, user=user, response=parsed), indent=2),
        encoding="utf-8")
    return parsed, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment or .env")
        return 1
    client = OpenAI()

    doc = json.loads(TEXT.read_text(encoding="utf-8"))
    available = [p for p in doc["provisions"] if p["extracted"]]
    print(f"provisions with text: {len(available)}  model: {args.model}\n")

    readings, kept, dropped = [], 0, 0
    for p in available:
        reading, meta = ask(client, args.model, p["provision"], p["text"], args.refresh)

        checked, seen_pairs = [], set()
        for link in reading.get("links", []):
            ok, how, frags = verify_quote(link["quote"], p["text"])
            # A link from a variable to itself is not a mechanism; the model occasionally
            # emits one when a provision's subject and object are the same concept.
            if ok and link["input"] == link["determines"]:
                ok, how = False, "self-loop"
            # The same pair emitted twice under different relations is one claim, not two.
            pair = (link["input"], link["determines"])
            if ok and pair in seen_pairs:
                ok, how = False, "duplicate_pair"
            if ok:
                seen_pairs.add(pair)

            # Adversarial pass: a fresh call that must try to refute the link. Prompting
            # the reader not to word-match did not stop it word-matching, so the check is
            # made structural instead of another instruction.
            verdict = None
            if ok:
                verdict, vmeta = challenge(client, args.model, p["provision"],
                                           p["text"], link, args.refresh)
                if not verdict["sound"]:
                    ok, how = False, f"refuted:{verdict['failure_mode']}"

            checked.append(dict(link, quote_verbatim=ok, quote_match=how,
                                quote_fragments=frags, challenge=verdict,
                                rejected=None if ok else how))
            kept += ok
            dropped += not ok

        readings.append(dict(
            provision=p["provision"], section=p["section"], act=p["act"],
            source_file=p.get("source_file"), source_sha256=p.get("source_sha256"),
            subject=reading.get("provision_subject", ""),
            determines=reading.get("determines", []),
            inputs=reading.get("inputs", []),
            links=checked, unmapped_concepts=reading.get("unmapped_concepts", []),
            notes=reading.get("notes", ""),
            llm=meta,
        ))
        good = [c for c in checked if c["quote_verbatim"]]
        bad = len(checked) - len(good)
        print(f"  {p['provision']:44} links={len(good):2d}"
              f"{f'  ({bad} rejected)' if bad else '':16} [{meta['cache']}]")
        print(f"      {reading.get('provision_subject', '')[:96]}")
        for c in good:
            print(f"      {c['input']} -> {c['determines']}  "
                  f"({c['relation']}, {c['confidence']})")
            print(f"        concept: {c['concept_in_text'][:82]}")

    out = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=args.model, n_provisions=len(readings),
        n_links_verbatim=kept, n_links_rejected=dropped,
        blindness=("The model saw only one provision's verbatim text plus neutral variable "
                   "descriptions. It saw no correlations, no case data, no other provision "
                   "and no part of the existing DAG."),
        variables=VARIABLES, system_prompt=SYSTEM, schema=SCHEMA,
        readings=readings,
    )
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"links with verbatim support: {kept}   rejected as unquoted: {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
