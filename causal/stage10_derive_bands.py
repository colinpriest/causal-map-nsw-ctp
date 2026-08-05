"""Stage 10 — derive the claim chronology instead of asserting it.

Stage 9 enforces a six-phase temporal order over the columns, and that order does more
orientation work than every statistical signal in the pipeline. It was also hand-written,
which makes it the largest remaining unsourced assertion in the project -- and the one
error found in it (band 1 treated as simultaneous, deleting age -> earnings) shows the
cost of getting it wrong.

So the phases are derived here, blind, and the hand-written version becomes something to
compare against rather than something to reproduce.

WHAT THE MODEL SEES
    Column names and neutral descriptions. Nothing else.

WHAT IT DOES NOT SEE
    The existing six bands, their names, their count, any correlation, any edge, or any
    part of the DAG. It is not asked to validate a proposed ordering -- it proposes its
    own, including how many phases there are.

WHY BLIND MATTERS HERE
    Shown the six bands and asked "is this right?", a model will agree. The only way the
    exercise means anything is if it has to construct the chronology unaided, and the
    comparison is then made in code.

VALIDATION, NONE OF IT MINE
    n independent samples          per-column agreement. A column several samples place in
                                   different phases is genuinely ambiguous, and saying so
                                   is more useful than a confident single answer.
    every column assigned once     structural, checked in code
    phases totally ordered         structural
    event-log consistency          where stage 2 found an evidenced event proxy for a
                                   column, the derived order must not contradict the
                                   observed dates. Thin -- only two columns have one --
                                   but it is the single external check available.
    comparison to the hand bands   agreement corroborates the assertion; disagreement is
                                   a question, not a verdict either way

Run:  python causal/stage10_derive_bands.py [--samples 5] [--model gpt-4o]
Out:  causal/provenance/llm_cache/bands-*.json
      causal/provenance/derived_bands.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "provenance" / "llm_cache"
TEMPORAL = HERE / "provenance" / "temporal_order.json"
LEDGER = HERE / "provenance" / "measurement_ledger.json"
OUT = HERE / "provenance" / "derived_bands.json"

DEFAULT_SAMPLES = 5
DEFAULT_MODEL = "gpt-4o"
MAX_PHASES = 8
# Definitions are NOT kept here. They live in ctp/columns.yaml, the project's data
# dictionary, and every stage that describes a column to a model reads that one file.
# Keeping a private copy per prompt is how wording drifts: an earlier version of this
# script told the model the ordinals measure "how prominently this features in the
# reasons", which contradicted the dictionary and dragged the injury columns downstream.
#
# Latent variables are loaded too. A latent node has no column and no values, but it still
# has to be PLACED. `Psychological Injury Emphasis` measures how much the psychological
# damage was discussed in the proceeding, while the injury it traces arose at the accident;
# placing the observed column at either point misstates the other. Asking the model to
# place both, with the indicator never earlier than the thing it indicates, resolves the
# split rather than forcing one answer onto one column.
SPEC = ROOT / "ctp" / "columns.yaml"


def load_spec():
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    variables = {c["name"]: " ".join(c["definition"].split()) for c in spec["columns"]}
    recorded = {c["name"]: c["recorded"] for c in spec["columns"]}
    latent = {l["name"]: " ".join(l["definition"].split()) for l in spec.get("latent", [])}
    indicates = {l["measured_by"]: l["name"] for l in spec.get("latent", [])}
    return variables, recorded, latent, indicates


VARIABLES, RECORDED, LATENT, INDICATES = load_spec()
PLACEABLE = {**VARIABLES, **LATENT}


# The hand-written version. Loaded only for COMPARISON, after the model has answered.
ASSERTED = {
    "Claimant Age": 1, "Claimant Gender": 1, "Claimant Weekly Income": 1,
    "Pre-existing Condition Salience": 1,
    "Injury Burden Intensity": 2, "Psychological Injury Emphasis": 2, "Liability Clarity": 2,
    "WPI %": 3, "Treatment Burden": 3, "Work Impact Severity": 3,
    "Causation Complexity": 4, "Nature": 4, "Legal Procedural Complexity": 4,
    "Non-Economic Loss": 5, "Future Economic Loss": 5,
    "Lump Sum": 6,
}

SYSTEM = (
    "You are working out the chronology of a compulsory third party motor accident claim "
    "in New South Wales, from a list of things recorded about decided claims.\n\n"
    "Divide the life of a claim into ordered PHASES, and place every recorded item in the "
    "phase at which the thing it measures COMES INTO EXISTENCE -- not when it is written "
    "down, and not when it is used.\n\n"
    "Work out from each definition what the measured thing IS, and place it when that "
    "thing comes into being. Some definitions carry their own timing: a condition "
    "described as pre-existing is one the claimant had beforehand. Others do not, and "
    "for those you must reason about the claim.\n\n"
    "Some entries are NOT OBSERVED. They have no column and no values, but they are real "
    "quantities the data traces only indirectly, and they must be placed too. Where an "
    "unobserved quantity has an observed indicator, place the unobserved thing when IT "
    "arose and the indicator when the OBSERVATION was made: an injury and the later "
    "discussion of that injury are different events. An indicator can never be earlier "
    "than the thing it indicates.\n\n"
    "That distinction decides most of the hard cases:\n"
    "  * A medical assessment performed years later measures a physical state that arose "
    "at the time of the injury. Ask when the QUANTITY was fixed, not when it was measured.\n"
    "  * Something recorded in the final decision may describe a fact established long "
    "before it.\n"
    "  * A procedural fact about how the claim was run belongs to the phase in which the "
    "claim was run, not to the injury.\n\n"
    "Decide how many phases there are. Use as few as genuinely capture the order and no "
    "more; a phase containing one item is fine if that item really stands alone in time. "
    "Items that arise at the same stage belong in the same phase, and a phase may contain "
    "items that cause one another -- being contemporaneous is not being simultaneous.\n\n"
    "For every item, say in one sentence WHEN the thing it measures came into existence. "
    "That sentence is the justification for its placement and will be recorded."
)


def schema(n_max: int) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["phases", "assignments"],
        "properties": {
            "phases": {
                "type": "array", "minItems": 2, "maxItems": n_max,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["order", "name", "description"],
                    "properties": {
                        "order": {"type": "integer", "minimum": 1, "maximum": n_max,
                                  "description": "1 is earliest. Consecutive, no gaps."},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "assignments": {
                "type": "array", "minItems": len(PLACEABLE), "maxItems": len(PLACEABLE),
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["column", "phase_order", "when_it_arises"],
                    "properties": {
                        "column": {"type": "string", "enum": list(PLACEABLE)},
                        "phase_order": {"type": "integer", "minimum": 1, "maximum": n_max},
                        "when_it_arises": {"type": "string"},
                    },
                },
            },
        },
    }


def elicit(client, model, sample, refresh):
    def describe(k, v):
        s = f"  - {k}: {v}"
        if k in RECORDED:
            s += f"\n      [how recorded: {RECORDED[k]}]"
        if k in LATENT:
            trace = next(i for i, l in INDICATES.items() if l == k)
            s += ("\n      [NOT OBSERVED: no column, no values. Its only measured trace "
                  f"is {trace}]")
        if k in INDICATES:
            s += f"\n      [this is the observed indicator of {INDICATES[k]}]"
        return s

    items = "\n".join(describe(k, v) for k, v in PLACEABLE.items())
    user = (f"Recorded about each decided claim:\n{items}\n\n"
            "Define the ordered phases of a claim, then place every item in one.")
    sch = schema(MAX_PHASES)
    key = hashlib.sha256(json.dumps([model, SYSTEM, user, sch, sample],
                                    sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"bands-{key}.json"
    if cached.exists() and not refresh:
        rec = json.loads(cached.read_text(encoding="utf-8"))
        return rec["response"], dict(rec["meta"], cache="hit")
    resp = client.chat.completions.create(
        model=model, temperature=1.0 if sample else 0, seed=sample,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "claim_phases", "strict": True, "schema": sch}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    meta = dict(model=resp.model, prompt_sha256=key, sample=sample, cache="miss",
                requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(dict(meta=meta, system=SYSTEM, user=user,
                                      response=parsed), indent=2), encoding="utf-8")
    return parsed, meta


def normalise(sample_doc):
    """Map each column to a rank in 1..n, and check the sample is structurally sound."""
    phases = sorted(sample_doc["phases"], key=lambda p: p["order"])
    orders = [p["order"] for p in phases]
    if orders != list(range(1, len(orders) + 1)):
        return None, "phase orders are not consecutive from 1"
    assigned = {a["column"]: a["phase_order"] for a in sample_doc["assignments"]}
    if set(assigned) != set(PLACEABLE):
        return None, "not every column assigned exactly once"
    if any(v not in orders for v in assigned.values()):
        return None, "a column was placed in a phase that was not defined"
    return dict(n_phases=len(phases), phases=phases, assigned=assigned,
                reasons={a["column"]: a["when_it_arises"]
                         for a in sample_doc["assignments"]}), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found")
        return 1
    client = OpenAI()

    samples, rejected = [], []
    for s in range(args.samples):
        doc, meta = elicit(client, args.model, s, args.refresh)
        norm, err = normalise(doc)
        if err:
            rejected.append(dict(sample=s, reason=err))
            continue
        norm["llm"] = meta
        samples.append(norm)
    print(f"samples accepted: {len(samples)}  rejected: {len(rejected)}")
    if not samples:
        print("no structurally valid sample")
        return 1

    n_phases = Counter(s["n_phases"] for s in samples)
    print(f"phase counts proposed: {dict(n_phases)}\n")

    # Rank each column by its MEAN normalised position, so samples that chose a different
    # number of phases can still be pooled.
    positions = defaultdict(list)
    for s in samples:
        for c, p in s["assigned"].items():
            positions[c].append((p - 1) / max(s["n_phases"] - 1, 1))
    mean_pos = {c: sum(v) / len(v) for c, v in positions.items()}
    order = sorted(PLACEABLE, key=lambda c: mean_pos[c])

    # agreement: share of samples placing the column in its modal RELATIVE position
    agreement = {}
    for c in PLACEABLE:
        rounded = [round(p, 1) for p in positions[c]]
        top = Counter(rounded).most_common(1)[0][1]
        agreement[c] = round(top / len(rounded), 2)

    # comparison with the hand-written bands, by rank correlation of the orderings
    # Compared on OBSERVED columns only -- the asserted band table has no latent nodes.
    observed = [c for c in order if c in VARIABLES]
    derived_rank = {c: i for i, c in enumerate(order)}
    from scipy import stats as st
    rho = float(st.spearmanr([ASSERTED[c] for c in observed],
                             [derived_rank[c] for c in observed]).statistic)

    # external check: event-log orderings from stage 2, where a column has an evidenced proxy
    consistency = []
    if TEMPORAL.exists():
        tdoc = json.loads(TEMPORAL.read_text(encoding="utf-8"))
        for vp in tdoc.get("variable_pairs", []):
            e, l = vp["earlier"], vp["later"]
            if e in mean_pos and l in mean_pos:
                consistency.append(dict(
                    observed=f"{e} before {l}", share=vp["share"], n=vp["n_cases"],
                    derived_agrees=mean_pos[e] <= mean_pos[l]))

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=args.model, samples_requested=args.samples,
        samples_accepted=len(samples), samples_rejected=rejected,
        blindness=("The model saw column names, definitions, and how each was "
                   "recorded. It was not "
                   "shown the hand-written bands, their count, or any statistic."),
        latent=LATENT, indicator_of=INDICATES,
        phase_counts_proposed=dict(n_phases),
        derived_order=[dict(rank=i + 1, column=c, mean_position=round(mean_pos[c], 3),
                            agreement=agreement[c], observed=c in VARIABLES,
                            asserted_band=ASSERTED.get(c))
                       for i, c in enumerate(order)],
        spearman_vs_asserted=round(rho, 3),
        event_log_consistency=consistency,
        per_sample=[dict(n_phases=s["n_phases"],
                         phases=[dict(order=p["order"], name=p["name"]) for p in s["phases"]],
                         assigned=s["assigned"], reasons=s["reasons"],
                         prompt_sha256=s["llm"]["prompt_sha256"]) for s in samples],
        note=("mean_position is the column's average place in the ordering, rescaled to "
              "0-1 so samples proposing different phase counts can be pooled. Agreement "
              "is the share of samples placing it at its modal position."),
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"{'rank':>4}  {'column':32}{'pos':>6}{'agree':>7}{'asserted':>9}")
    for i, c in enumerate(order):
        band = f"{ASSERTED[c]:9d}" if c in ASSERTED else "        -"
        tag = "" if c in VARIABLES else "   <-- LATENT"
        print(f"{i + 1:>4}  {c:32}{mean_pos[c]:6.2f}{agreement[c]:7.0%}{band}{tag}")
    print(f"\nSpearman(derived order, asserted bands) = {rho:+.3f}")
    if consistency:
        agree = sum(c["derived_agrees"] for c in consistency)
        print(f"event-log orderings consistent with derived: {agree}/{len(consistency)}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
