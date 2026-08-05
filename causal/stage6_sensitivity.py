"""Stage 6 — sweep every threshold in the pipeline and report which ones decide the answer.

Four times in building this pipeline a criterion that looked principled turned out to be
excluding the most fundamental relationships:

  MIN_CITES = 20            dropped s 4.6 MAIA, which caps loss-of-earnings damages
  "name the variable"       dropped Claimant Weekly Income -> Future Economic Loss
  challenger directness     dropped s 4.6 again, and s 4.17 contributory negligence
  strict monotonicity       dropped both of the worked examples it was asked about

and once in the other direction the coder warning produced 28 identical verdicts out of
28, which vanished entirely when the warning was removed.

Every one of those was invisible until something outside the pipeline forced a check. The
lesson is not that those particular numbers were wrong; it is that a threshold nobody
varies is indistinguishable from a threshold that is deciding the outcome. So this stage
re-applies every threshold across a range and reports what moves.

A parameter is FRAGILE when a step within its swept range changes a headline count by more
than FRAGILE_FRAC. Fragile does not mean wrong -- it means the conclusion should never be
quoted without the threshold quoted beside it.

Nothing here calls an API: every statistic needed was recorded by the stage that produced
it, so thresholds are re-applied to stored numbers rather than recomputed.

Run:  python causal/stage6_sensitivity.py
Out:  causal/provenance/sensitivity.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "provenance"
OUT = P / "sensitivity.json"

FRAGILE_FRAC = 0.25     # a step that moves a headline count by >25% is fragile


def load(name):
    return json.loads((P / name).read_text(encoding="utf-8"))


def sweep(name, current, values, fn, unit):
    """Apply fn at each value; flag if any adjacent step moves the count a lot."""
    rows = [dict(value=v, count=fn(v)) for v in values]
    counts = [r["count"] for r in rows]
    fragile = False
    for i in range(1, len(counts)):
        base = max(counts[i - 1], 1)
        if abs(counts[i] - counts[i - 1]) / base > FRAGILE_FRAC:
            fragile = True
    at_current = fn(current)
    return dict(parameter=name, current=current, unit=unit, count_at_current=at_current,
                sweep=rows, fragile=fragile,
                span=[min(counts), max(counts)])


def main() -> int:
    assoc = load("associations.json")
    temporal = load("temporal_order.json")
    links = load("provision_links.json")
    tests = load("prior_tests.json")
    results = []

    # ---- stage 1: which pairs get researched ------------------------------
    pairs = assoc["pairs"]

    def n_research(cut):
        return sum(1 for p in pairs
                   if p["tier"] == "A" or p["materiality"] >= cut
                   or (p["tail"] is not None and abs(p["tail"]) >= 0.08))
    results.append(sweep("stage1.materiality_cut", 0.10,
                         [0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30],
                         n_research, "pairs researched"))

    def n_tail_promoted(cut):
        return sum(1 for p in pairs
                   if p["tier"] not in ("A",) and p["tail"] is not None
                   and abs(p["tail"]) >= cut and p["materiality"] < 0.10)
    results.append(sweep("stage1.tail_promotion", 0.08,
                         [0.04, 0.06, 0.08, 0.12, 0.16, 0.20],
                         n_tail_promoted, "pairs promoted on tail alone"))

    # ---- stage 2: how much of the temporal evidence survives --------------
    ev_links = temporal.get("event_variable_links", {})
    rejected = temporal.get("event_variable_rejected", [])
    allrho = ([abs(v["rho"]) for v in ev_links.values()]
              + [abs(r["rho"]) for r in rejected])

    def n_event_links(cut):
        return sum(1 for r in allrho if r >= cut)
    results.append(sweep("stage2.link_min", 0.35,
                         [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
                         n_event_links, "event->variable links"))

    # ---- stage 3a: statutory association tests ---------------------------
    recs = links["records"]

    def n_value(cut):
        return sum(1 for r in recs
                   if r["effect"] is not None and abs(r["effect"]) >= cut
                   and abs(r["effect"]) < 0.90 and r.get("p_value", 1) is not None
                   and (r["p_value"] or 1) < 0.05)
    results.append(sweep("stage3a.effect_min", 0.15,
                         [0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
                         n_value, "value links"))

    def n_gate(cut):
        return sum(1 for r in recs if abs(r["gate_effect"]) >= cut
                   and (r.get("gate_p") or 1) < 0.05)
    results.append(sweep("stage3a.gate_min", 0.10,
                         [0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
                         n_gate, "gate links"))

    def n_definitional(cut):
        return sum(1 for r in recs
                   if r["effect"] is not None and abs(r["effect"]) >= cut)
    results.append(sweep("stage3a.definitional", 0.90,
                         [0.80, 0.85, 0.90, 0.95, 0.99],
                         n_definitional, "pairs excluded as definitional"))

    # MIN_CITES gates which provisions are TESTED. It cannot be re-applied from stored
    # records -- untested provisions have none -- so it is reported from the citation
    # counts instead. This is the parameter that dropped s 4.6, and it is here so that
    # never happens silently again.
    allprov = links.get("all_provisions", [])
    results.append(sweep("stage3a.min_cites", 20,
                         [1, 2, 5, 10, 20, 30, 50],
                         lambda c: sum(1 for p in allprov if p["n_citing"] >= c),
                         "provisions statistically testable"))

    # ---- stage 5b: which reasoned priors survive their predictions --------
    tr = tests["results"]

    def n_pass(thresholds):
        mr, at, mg, td = thresholds
        n = 0
        for r in tr:
            d, t = r["detail"], r["prediction"]["test"]
            if r["passed"] is None:
                continue
            if t == "sign" and "rho" in d:
                want = r["prediction"]["direction"]
                n += (d["rho"] >= mr) if want == "positive" else (d["rho"] <= -mr)
            elif t in ("monotone",) and "trend" in d:
                want = r["prediction"]["direction"]
                n += (d["trend"] >= td) if want in ("increasing", "positive") \
                    else (d["trend"] <= -td)
            elif t == "inverted_u" and "peak_at" in d:
                n += r["passed"] is True
            elif t == "mediation" and "attenuation" in d:
                n += d["attenuation"] >= at
            elif t == "attenuation_asymmetry" and "attenuation_of_a" in d:
                want = r["prediction"]["direction"]
                diff = d["attenuation_of_a"] - d["attenuation_of_b"]
                n += (diff >= mg) if want == "a_is_upstream" else (-diff >= mg)
        return n

    results.append(sweep("stage5b.trend", 0.70, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                         lambda v: n_pass((0.05, 0.50, 0.15, v)), "priors passing"))
    results.append(sweep("stage5b.min_rho", 0.05, [0.02, 0.05, 0.10, 0.15, 0.20],
                         lambda v: n_pass((v, 0.50, 0.15, 0.70)), "priors passing"))
    results.append(sweep("stage5b.attenuation", 0.50, [0.30, 0.40, 0.50, 0.60, 0.70],
                         lambda v: n_pass((0.05, v, 0.15, 0.70)), "priors passing"))
    results.append(sweep("stage5b.margin", 0.15, [0.05, 0.10, 0.15, 0.20, 0.30],
                         lambda v: n_pass((0.05, 0.50, v, 0.70)), "priors passing"))

    # ---- prompt instructions, not numeric: report the known ablation ------
    ablation = None
    abl_path = P / "reasoned_priors_ablation.json"
    if abl_path.exists():
        main_priors = {(p["a"], p["b"]): p["verdict"] for p in
                       load("reasoned_priors.json")["priors"] if p["both_llm_coded"]}
        abl = {(p["a"], p["b"]): p["verdict"]
               for p in json.loads(abl_path.read_text(encoding="utf-8"))["priors"]}
        agree = sum(1 for k in abl if main_priors.get(k) == abl[k])
        ablation = dict(
            instruction="stage5a coder warning",
            pairs=len(abl), verdicts_unchanged=agree,
            fragile=agree / max(len(abl), 1) < 0.75,
            note=("The warning tells the model both columns came from one reader and to "
                  "weigh measurement_artifact seriously. Removing it changes this many "
                  "verdicts, which bounds how much of the verdict is the instruction."),
        )

    fragile = [r["parameter"] for r in results if r["fragile"]]
    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        fragile_fraction=FRAGILE_FRAC,
        n_parameters=len(results), n_fragile=len(fragile), fragile_parameters=fragile,
        prompt_ablation=ablation,
        parameters=results,
        note=("Fragile means a conclusion moves materially within the swept range, not "
              "that it is wrong. A fragile parameter's result must be quoted with its "
              "threshold attached."),
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}\n")
    print(f"{'parameter':30}{'current':>9}{'now':>6}{'range':>12}  fragile")
    for r in results:
        rng = f"{r['span'][0]}-{r['span'][1]}"
        print(f"{r['parameter']:30}{r['current']:>9}{r['count_at_current']:>6}{rng:>12}"
              f"  {'FRAGILE' if r['fragile'] else ''}")
    if ablation:
        print(f"\nprompt ablation - {ablation['instruction']}: "
              f"{ablation['verdicts_unchanged']}/{ablation['pairs']} verdicts unchanged"
              f"  {'FRAGILE' if ablation['fragile'] else ''}")
    print(f"\n{len(fragile)} of {len(results)} numeric parameters are fragile:")
    for f in fragile:
        r = next(x for x in results if x["parameter"] == f)
        print(f"  {f}: {r['unit']} runs {r['span'][0]}-{r['span'][1]} across the sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
