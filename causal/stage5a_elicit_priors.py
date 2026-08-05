"""Stage 5a — elicit a blind causal prior AND a prediction that can be run against the data.

Eight of the sixteen columns are constructs built to be predictive of outcomes, not
concepts the scheme defines. Statute has no opinion about `Psychological Injury Emphasis`,
so stage 3 can never reach it, and a correlation cannot orient it. This stage supplies the
third evidence class -- reasoned, and deliberately the weakest of the three, so it is
labelled `reasoned_prior` and never merged with provision-backed edges.

WHAT MAKES THIS MORE THAN AN OPINION
A model asked "which way does this go?" will produce a fluent case for whichever direction
it picks. That is precisely how the graph this project is replacing came to exist. Three
things are done about it:

  1. BOTH DIRECTIONS ARGUED FIRST. The model must write the best case for A->B and the
     best case for B->A before it is allowed to choose. It cannot commit and then
     rationalise.
  2. THE ANSWER SPACE IS WIDER THAN A->B / B->A. For two columns scored by one model
     reading one decision, the likeliest explanation is often neither: a common cause, or
     a measurement artefact where the coder read one passage and marked several scales up
     together. Both are explicit verdicts. A common cause must be NAMED.
  3. A TESTABLE IMPLICATION IS REQUIRED, chosen from a fixed menu the next stage can
     actually execute. The prior is thereby a falsifiable prediction rather than a score.

The menu deliberately includes `inverted_u`. Age and earnings rise together through a
career and fall at retirement, so a mechanism that is real can still show a near-zero rank
correlation. A menu of monotone tests alone would refute true mechanisms.

BLINDNESS
The model sees variable names, neutral descriptions, and -- where relevant -- the fact
that both columns came from one reader and one document. It sees no correlation, no effect
size, no tail statistic, and no part of any DAG. Prompts are hashed and cached so the
blindness is auditable rather than asserted.

THE CODER WARNING DETERMINES THE ANSWER -- DO NOT READ THESE VERDICTS AS FINDINGS
For pairs where both columns are LLM-coded, the prompt states that fact and says to weigh
`measurement_artifact` seriously. Run with that warning, all 28 coded pairs return
`measurement_artifact`. Run `--no-coder-warning` over the same 28 pairs and NONE do: they
come back 15 common_cause, 11 indirect, 2 a_causes_b.

A complete flip. The verdict on coded pairs is set by the instruction, not by reasoning
about the variables, so neither run is evidence about the world. Both are kept --
reasoned_priors.json and reasoned_priors_ablation.json -- and any consumer must treat a
coded-pair verdict as unresolved. The question "is this association real or an artefact of
one model scoring several scales off one passage?" cannot be settled by asking a model; it
needs re-coding by an independent reader and an inter-coder agreement statistic.

The ablation is why `--no-coder-warning` exists. Keep running it whenever this prompt
changes: a prompt that decides its own answer is worth knowing about before the output is
believed.

Run:  python causal/stage5a_elicit_priors.py [--model gpt-4o] [--samples 3]
Out:  causal/provenance/llm_cache/prior-*.json
      causal/provenance/reasoned_priors.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSOC = HERE / "provenance" / "associations.json"
LEDGER = HERE / "provenance" / "measurement_ledger.json"
CACHE = HERE / "provenance" / "llm_cache"
OUT = HERE / "provenance" / "reasoned_priors.json"

DEFAULT_MODEL = "gpt-4o"
DEFAULT_SAMPLES = 3

VARIABLES = {
    "Lump Sum": "the total dollar amount awarded to the claimant",
    "WPI %": "whole person impairment, as a percentage, from a medical assessment",
    "Non-Economic Loss": "dollars awarded for non-economic loss (pain and suffering)",
    "Future Economic Loss": "dollars awarded for future loss of earning capacity",
    "Claimant Weekly Income": "the claimant's pre-accident weekly earnings, in dollars",
    "Claimant Age": "the claimant's age at the date of injury, in years",
    "Claimant Gender": "the claimant's recorded gender",
    "Nature": "the procedural route: settlement approval, or damages assessment",
    "Injury Burden Intensity": "ordinal 0-4 for the overall physical burden of injuries",
    "Treatment Burden": "ordinal 0-3 for the extent of treatment, surgery and care",
    "Work Impact Severity": "ordinal 0-3 for loss of working capacity",
    "Legal Procedural Complexity": "ordinal 0-3 for procedural complexity of the proceeding",
    "Psychological Injury Emphasis": "ordinal 0-2 for weight given to psychological injury",
    "Liability Clarity": "ordinal 0-2 for how clear fault is",
    "Causation Complexity": "ordinal 0-2 for difficulty attributing the condition to the accident",
    "Pre-existing Condition Salience": "ordinal 0-2 for how prominent a prior condition is",
}

SYSTEM = (
    "You are reasoning about how compulsory third party motor accident claims work, to "
    "decide how two recorded quantities are related. You have NOT been shown any data, and "
    "you must not guess at what the data shows. Reason from mechanism.\n\n"
    "PROCEDURE, in this order:\n"
    "  1. Write the strongest case that A causes B.\n"
    "  2. Write the strongest case that B causes A.\n"
    "  3. Only then choose a verdict. Arguing both sides first is mandatory: a case "
    "written after choosing is a rationalisation.\n\n"
    "THE VERDICT IS NOT LIMITED TO A->B OR B->A:\n"
    "  a_causes_b / b_causes_a  - a direct causal relationship.\n"
    "  indirect                 - A affects B only THROUGH a named third variable. Use "
    "this when the honest answer is a chain: age does not directly set future economic "
    "loss, it sets how many working years remain, which sets the loss.\n"
    "  common_cause             - both are driven by something else, which you must NAME.\n"
    "  measurement_artifact     - the two only move together because of how they were "
    "recorded, not because of anything in the world.\n"
    "  none                     - no relationship worth drawing.\n\n"
    "TWO SCORES, kept separate because they are different questions:\n"
    "  confidence_mechanism_exists - 1 to 5, how sure you are the mechanism is real.\n"
    "  expected_magnitude          - 1 to 5, how large the effect would be if real.\n"
    "A statutory arithmetic identity is 5/5. A plausible behavioural story that would "
    "move an award slightly is 4/1.\n\n"
    "FINALLY, commit to a prediction the data can refute. Choose the test from the menu "
    "that would most sharply distinguish your verdict from its alternatives. Do not choose "
    "a test your verdict would pass trivially. Note that a real mechanism can be "
    "NON-MONOTONE -- earnings rise with career experience and fall at retirement -- so if "
    "you expect a peak in the middle, say `inverted_u`, not `monotone`."
)

TESTS = {
    "sign": "Spearman correlation between var_a and var_b has the predicted sign.",
    "monotone": "The median of var_b moves consistently in one direction across levels "
                "of var_a.",
    "inverted_u": "The median of var_b peaks at intermediate levels of var_a and is lower "
                  "at both extremes (or the inverse, a U).",
    "mediation": "The association between var_a and `outcome` largely disappears once "
                 "`mediator` is held fixed.",
    "attenuation_asymmetry": "If var_a acts on `outcome` THROUGH var_b, then holding var_b "
                             "fixed blocks var_a's path and collapses its association with "
                             "`outcome`, while holding var_a fixed leaves var_b's "
                             "association largely intact. Predict `a_is_upstream` when you "
                             "expect var_a's association to collapse the more of the two.",
}

VERDICTS = ["a_causes_b", "b_causes_a", "indirect", "common_cause",
            "measurement_artifact", "none"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["case_for_a_causes_b", "case_for_b_causes_a", "verdict", "mechanism",
                 "named_common_cause", "mediator", "confidence_mechanism_exists",
                 "expected_magnitude", "test", "test_outcome", "test_mediator",
                 "predicted_direction", "test_rationale"],
    "properties": {
        "case_for_a_causes_b": {"type": "string"},
        "case_for_b_causes_a": {"type": "string"},
        "verdict": {"type": "string", "enum": VERDICTS},
        "mechanism": {"type": "string",
                      "description": "One or two sentences naming the pathway."},
        "named_common_cause": {
            "type": ["string", "null"],
            "description": "Required when verdict is common_cause. May be a variable name "
                           "or an unobserved factor.",
        },
        "mediator": {
            "type": ["string", "null"], "enum": list(VARIABLES) + [None],
            "description": "Required when verdict is indirect.",
        },
        "confidence_mechanism_exists": {"type": "integer", "minimum": 1, "maximum": 5},
        "expected_magnitude": {"type": "integer", "minimum": 1, "maximum": 5},
        "test": {"type": "string", "enum": list(TESTS)},
        "test_outcome": {
            "type": ["string", "null"], "enum": list(VARIABLES) + [None],
            "description": "Required for mediation and attenuation_asymmetry.",
        },
        "test_mediator": {
            "type": ["string", "null"], "enum": list(VARIABLES) + [None],
            "description": "Required for mediation.",
        },
        "predicted_direction": {
            "type": "string",
            "enum": ["positive", "negative", "increasing", "decreasing", "peak", "trough",
                     "attenuates", "a_is_upstream", "b_is_upstream"],
        },
        "test_rationale": {"type": "string",
                           "description": "Why this test discriminates your verdict from "
                                          "the alternatives."},
    },
}


def elicit(client, model, a, b, both_coded, sample, refresh, warn_coder=True):
    coded_note = ""
    if both_coded and warn_coder:
        coded_note = (
            "\n\nIMPORTANT: both of these columns were assigned by ONE model reading ONE "
            "decision document. They were not measured independently. A reader who sees a "
            "passage describing a severe case may mark several scales up at once, which "
            "would make the columns move together with no causal relationship between "
            "them. Weigh `measurement_artifact` seriously here."
        )
    menu = "\n".join(f"  {k}: {v}" for k, v in TESTS.items())
    user = (
        f"A = {a}: {VARIABLES[a]}\n"
        f"B = {b}: {VARIABLES[b]}\n"
        f"{coded_note}\n\n"
        f"Other recorded variables you may refer to as mediators or common causes:\n"
        + "\n".join(f"  - {k}: {v}" for k, v in VARIABLES.items() if k not in (a, b))
        + f"\n\nTEST MENU:\n{menu}\n\n"
        "Argue both directions, choose a verdict, score it, and commit to a test."
    )
    key = hashlib.sha256(json.dumps(
        [model, SYSTEM, user, SCHEMA, sample], sort_keys=True).encode()).hexdigest()[:24]
    cached = CACHE / f"prior-{key}.json"
    if cached.exists() and not refresh:
        rec = json.loads(cached.read_text(encoding="utf-8"))
        return rec["response"], dict(rec["meta"], cache="hit")

    resp = client.chat.completions.create(
        model=model, temperature=1.0 if sample else 0,   # vary only across samples
        seed=sample,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "causal_prior", "strict": True, "schema": SCHEMA}},
    )
    parsed = json.loads(resp.choices[0].message.content)
    meta = dict(model=resp.model, prompt_sha256=key, sample=sample, cache="miss",
                requested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(dict(meta=meta, system=SYSTEM, user=user,
                                      response=parsed), indent=2), encoding="utf-8")
    return parsed, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    # Ablation. The coder warning tells the model both columns came from one reader and to
    # weigh `measurement_artifact` seriously -- and it then returned that verdict for every
    # coded pair. Whether that is a finding or an artefact of the instruction can only be
    # settled by asking again without the warning.
    ap.add_argument("--no-coder-warning", action="store_true")
    ap.add_argument("--coded-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found")
        return 1
    client = OpenAI()

    assoc = json.loads(ASSOC.read_text(encoding="utf-8"))
    kinds = {k: v["kind"] for k, v in
             json.loads(LEDGER.read_text(encoding="utf-8"))["columns"].items()}

    # Material pairs, plus every coder-derived pair regardless of materiality -- those are
    # the ones statute cannot reach and where measurement confounding is worst.
    pairs = [p for p in assoc["pairs"]
             if p["tier"] in ("A", "B") or p["both_llm_coded"]]
    if args.coded_only:
        pairs = [p for p in pairs if p["both_llm_coded"]]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"pairs: {len(pairs)}  samples each: {args.samples}  model: {args.model}\n")

    results = []
    for p in pairs:
        a, b = p["a"], p["b"]
        samples = []
        for s in range(args.samples):
            r, meta = elicit(client, args.model, a, b, p["both_llm_coded"], s,
                             args.refresh, warn_coder=not args.no_coder_warning)
            samples.append(dict(r, llm=meta))
        verdicts = Counter(s["verdict"] for s in samples)
        top, n_top = verdicts.most_common(1)[0]
        agreement = n_top / len(samples)
        primary = next(s for s in samples if s["verdict"] == top)

        results.append(dict(
            a=a, b=b, tier=p["tier"], both_llm_coded=p["both_llm_coded"],
            verdict=top, verdict_agreement=round(agreement, 2),
            verdict_spread={k: v for k, v in verdicts.items()},
            mechanism=primary["mechanism"],
            named_common_cause=primary.get("named_common_cause"),
            mediator=primary.get("mediator"),
            confidence_mechanism_exists=primary["confidence_mechanism_exists"],
            expected_magnitude=primary["expected_magnitude"],
            case_for_a_causes_b=primary["case_for_a_causes_b"],
            case_for_b_causes_a=primary["case_for_b_causes_a"],
            prediction=dict(test=primary["test"], var_a=a, var_b=b,
                            outcome=primary.get("test_outcome"),
                            mediator=primary.get("test_mediator"),
                            direction=primary["predicted_direction"],
                            rationale=primary["test_rationale"]),
            samples=[dict(verdict=s["verdict"], test=s["test"],
                          direction=s["predicted_direction"],
                          confidence=s["confidence_mechanism_exists"],
                          magnitude=s["expected_magnitude"],
                          prompt_sha256=s["llm"]["prompt_sha256"]) for s in samples],
        ))
        flag = "!" if p["both_llm_coded"] else " "
        print(f"{flag}{a} ~ {b}")
        print(f"    {top:22} agree={agreement:.0%}  conf={primary['confidence_mechanism_exists']}"
              f" mag={primary['expected_magnitude']}  test={primary['test']}"
              f"/{primary['predicted_direction']}")

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=args.model, samples_per_pair=args.samples,
        evidence_class="reasoned_prior",
        weaker_than=("statute and case text: there is no external referent and no quote. "
                     "Never merge with provision-backed edges."),
        blindness=("The model saw variable names and neutral descriptions only. No "
                   "correlation, effect size, tail statistic or DAG was shown."),
        tests=TESTS, verdicts=VERDICTS, system_prompt=SYSTEM,
        n_pairs=len(results),
        verdict_counts=dict(Counter(r["verdict"] for r in results)),
        priors=results,
    )
    doc["coder_warning_shown"] = not args.no_coder_warning
    out_path = Path(args.out) if args.out else OUT
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("verdicts:", doc["verdict_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
