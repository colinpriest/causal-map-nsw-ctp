"""Stage 9 — enforce the claim chronology, and assemble the evidence that respects it.

Six bands, in the order a claim actually happens:

    1 CLAIMANT, PRE-ACCIDENT      fixed before the crash
    2 ACCIDENT & INJURY           the crash and the harm it caused
    3 CLINICAL COURSE & CAPACITY  how the injury played out
    4 DISPUTE & FORUM             what got contested, and where
    5 HEADS OF DAMAGE             the quantum components recorded
    6 AWARD                       the total

EVERY EDGE RUNS FORWARD. An edge from a later band to an earlier one is not a weak edge or
an uncertain one; it is impossible, because the effect would precede its cause. This is the
one constraint in the whole pipeline that needs no evidence and admits no exceptions, and
it does more orientation work than every statistical signal combined.

WHY THIS IS ENFORCED RATHER THAN DERIVED
Stage 8 tried to establish position from the data and could not: conditioning on a mediator
and conditioning on a confounder are statistically identical, and the immutable-root anchor
turned out too weakly associated with the award to resolve much. Chronology settles by
construction what correlation cannot settle at all. The bands are a claim about the world's
order of events, not about the data, so no amount of data can overturn them -- but data CAN
reveal that a column was put in the wrong band, which is what the violation report is for.

WHAT THIS STAGE PRODUCES
  * a permitted-edge mask over all 240 ordered pairs
  * every piece of evidence gathered so far, filtered to band-permitted direction
  * VIOLATIONS: evidence pointing backwards through the bands. Each one is either a
    misplaced column or a bad inference, and is reported rather than silently dropped
  * roles that follow from band position, no longer guessed from correlation structure

Run:  python causal/stage9_enforce_bands.py
Out:  causal/provenance/banded_graph.json
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict

import yaml
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "provenance"
OUT = P / "banded_graph.json"
ELICITED = HERE / "elicited_edges.yaml"

BANDS = [
    (1, "claimant", "Claimant, pre-accident",
     ["Claimant Age", "Claimant Gender", "Claimant Weekly Income",
      "Pre-existing Condition Salience"]),
    # `Psychological Injury` is LATENT -- no column, no values. It sits here because the
    # injury arose at the accident. Its observed indicator, `Psychological Injury
    # Emphasis`, measures how much the damage was DISCUSSED, so it moves to the dispute
    # band. Splitting them removes a false choice: one column could not sit in both places
    # and the old table had it at band 2, describing the discussion as though it were the
    # injury.
    (2, "injury", "Accident & injury",
     ["Injury Burden Intensity", "Psychological Injury", "Liability Clarity"]),
    (3, "clinical", "Clinical course & capacity",
     ["WPI %", "Treatment Burden", "Work Impact Severity"]),
    (4, "dispute", "Dispute & forum",
     ["Causation Complexity", "Nature", "Legal Procedural Complexity",
      "Psychological Injury Emphasis"]),
    (5, "heads", "Heads of damage",
     ["Non-Economic Loss", "Future Economic Loss"]),
    (6, "award", "Award", ["Lump Sum"]),
]

BAND_OF = {col: n for n, _, _, cols in BANDS for col in cols}
# Unobserved: carried in the graph, never treated as data.
LATENT = {"Psychological Injury"}
# for reporting each edge under its STRONGEST evidence, not every class it has
RANK_REPORT = {"elicited": 5, "statute": 4, "measurement": 4,
               "reasoned_prior_tested": 2, "reasoned_prior_path": 1,
               "reasoned_prior_common_cause": 1}
# An indicator is caused by the thing it indicates. This is a measurement edge, not a
# claim about the scheme, and it is asserted by the data dictionary.
MEASURES = [("Psychological Injury", "Psychological Injury Emphasis")]
LABEL = {n: label for n, _, label, _ in BANDS}


# Fixed at birth: causally prior to everything, including the other pre-accident columns.
# Band 1 is NOT a set of simultaneous facts -- age precedes and drives both accumulated
# earnings (career progression) and accumulated degenerative change. An earlier version of
# this file forbade all within-band-1 edges, which silently deleted exactly those two
# mechanisms. Roots are the columns nothing can cause, not the columns in band 1.
IMMUTABLE = ["Claimant Age", "Claimant Gender"]


def permitted(src: str, dst: str) -> bool:
    """Forward in time. Same-band edges are allowed throughout -- two facts can sit in the
    same phase of a claim and still stand in a causal relation -- except that nothing may
    point INTO an immutable column."""
    a, b = BAND_OF[src], BAND_OF[dst]
    if dst in IMMUTABLE:
        return False
    return a <= b and src != dst


def _mediator(rec) -> str | None:
    for k in ("mediator", "test_mediator"):
        v = rec.get(k) or rec.get("prediction", {}).get(k)
        if v:
            return v
    return None


def _resolve(name: str, cols: list[str]) -> str | None:
    """Match a free-text common-cause name to a column. The model writes 'Pre-existing
    Condition' for the column 'Pre-existing Condition Salience'."""
    if not name:
        return None
    n = name.strip().lower()
    for c in cols:
        if c.lower() == n:
            return c
    for c in cols:
        if n in c.lower() or c.lower() in n:
            return c
    return None


def load(name):
    p = P / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    cols = list(BAND_OF)
    assert len(cols) == 17, f"expected 16 columns + 1 latent, banded {len(cols)}"

    # ---- the mask ---------------------------------------------------------
    ordered = [(a, b) for a, b in itertools.permutations(cols, 2)]
    allowed = [(a, b) for a, b in ordered if permitted(a, b)]

    # ---- collect evidence, oriented ---------------------------------------
    edges, violations, undirected = {}, [], []

    def add(src, dst, cls, detail):
        if permitted(src, dst):
            e = edges.setdefault((src, dst), dict(source=src, target=dst, evidence=[]))
            # An edge appearing as a leg in five different paths is ONE piece of support
            # mentioned five times, not five reasons to believe it. Collapse them into a
            # single entry that lists the paths, so the evidence count means what a reader
            # takes it to mean.
            if cls.endswith("_path"):
                existing = next((v for v in e["evidence"]
                                 if v["evidence_class"] == cls), None)
                if existing:
                    # an entry may predate this collapsing (e.g. added by another branch)
                    existing.setdefault("paths", [existing.pop("path", None)])
                    if detail.get("path") not in existing["paths"]:
                        existing["paths"].append(detail.get("path"))
                    existing["n_paths"] = len(existing["paths"])
                    return
                detail = dict(detail)
                detail["paths"] = [detail.pop("path", None)]
                detail["n_paths"] = 1
            e["evidence"].append(dict(evidence_class=cls, **detail))
            return
        # Evidence pointing backwards still establishes that the PAIR is related; only the
        # direction is refuted, and chronology refutes it decisively. So reverse it rather
        # than discard it. WPI % -> Injury Burden Intensity came back at 100% agreement
        # from the reasoned priors, and it is backwards: injury burden is a fact of the
        # crash, WPI is a later assessment OF that burden. Reversed, it is correct.
        # A backwards PATH leg refutes the path; it is never reversed. This check must
        # come BEFORE the repeat-violation branch below, which also reverses -- leaving it
        # after meant a path leg seen twice got reversed by that branch instead, which is
        # how `Injury Burden Intensity -> Treatment Burden` ended up citing
        # "Treatment Burden -> Injury Burden Intensity -> Lump Sum" as its support.
        if cls.endswith("_path"):
            for existing in violations:
                if (existing.get("source"), existing.get("target"),
                        existing.get("evidence_class")) == (src, dst, cls):
                    existing["times_seen"] = existing.get("times_seen", 1) + 1
                    return
            violations.append(dict(
                source=src, target=dst, evidence_class=cls, times_seen=1,
                band_source=BAND_OF[src], band_target=BAND_OF[dst], detail=detail,
                resolution="dropped: a leg of this path runs backwards, so the path claim "
                           "is refuted rather than reversed"))
            return

        # Each `indirect` path re-adds the same backwards edge, so the same violation was
        # logged six times. One row per (pair, evidence class), with a count.
        key = (src, dst, cls)
        for existing in violations:
            if (existing.get("source"), existing.get("target"),
                    existing.get("evidence_class")) == key:
                existing["times_seen"] = existing.get("times_seen", 1) + 1
                if permitted(dst, src):
                    e = edges.setdefault((dst, src),
                                         dict(source=dst, target=src, evidence=[]))
                    if not any(ev.get("originally_claimed") == f"{src} -> {dst}"
                               for ev in e["evidence"]):
                        e["evidence"].append(dict(evidence_class=cls,
                                                  direction_corrected=True,
                                                  originally_claimed=f"{src} -> {dst}",
                                                  **detail))
                return
        rec = dict(source=src, target=dst, evidence_class=cls, times_seen=1,
                   band_source=BAND_OF[src], band_target=BAND_OF[dst], detail=detail)
        if permitted(dst, src):
            rec["resolution"] = "reversed to run forward in time"
            e = edges.setdefault((dst, src), dict(source=dst, target=src, evidence=[]))
            e["evidence"].append(dict(evidence_class=cls, direction_corrected=True,
                                      originally_claimed=f"{src} -> {dst}", **detail))
        else:
            rec["resolution"] = "dropped: neither direction is permitted"
        violations.append(rec)

    # Human domain claims first: they outrank everything else, and where a machine-derived
    # edge contradicts one it is the machine-derived edge that goes.
    elicited_pairs = set()
    if ELICITED.exists():
        el = yaml.safe_load(ELICITED.read_text(encoding="utf-8"))
        for e in el.get("edges", []):
            add(e["source"], e["target"], "elicited",
                dict(who=el.get("elicited_by"), date=str(e.get("date")),
                     mechanism=e["mechanism"], verbatim=e.get("verbatim", "")))
            elicited_pairs.add((e["source"], e["target"]))

    for src, dst in MEASURES:
        add(src, dst, "measurement",
            dict(note="the indicator is caused by the latent quantity it measures; "
                      "declared in ctp/columns.yaml"))

    stat = load("statutory_edges.json")
    if stat:
        for e in stat.get("edges", []):
            add(e["source"], e["target"], "statute",
                dict(provision=e["provision"], relation=e["relation"],
                     grade=e.get("grade"), concept=e["statute"]["concept"],
                     quote=e["statute"]["quote"]))

    # Whether an association between two reader-scored columns is real is settled by
    # MEASUREMENT, not by asking a model. Stage 7 re-coded 100 decisions with an
    # independent reader and crossed the coders: an association that survives the crossing
    # is in the world, one that collapses is in the reader's head. Nothing survives here on
    # a model's opinion about measurement artefacts -- an earlier run let exactly that
    # opinion suppress every ordinal-to-ordinal edge, including Pre-existing Condition
    # Salience, which then had no children at all.
    survival = {}
    rel = load("coder_reliability.json")
    if rel:
        for pr in rel["pairs"]:
            survival[frozenset((pr["a"], pr["b"]))] = pr

    tested = load("prior_tests.json")
    if tested:
        for r in tested["results"]:
            if r["passed"] is False:
                continue
            v, a, b = r["verdict"], r["a"], r["b"]
            cross = survival.get(frozenset((a, b)))
            if cross and cross["verdict"] == "coder_artefact":
                violations.append(dict(source=a, target=b,
                                       evidence_class="reasoned_prior_tested",
                                       band_source=BAND_OF.get(a), band_target=BAND_OF.get(b),
                                       resolution="dropped: does not survive a change of "
                                                  "coder (stage 7)"))
                continue

            base = dict(confidence=r["confidence"], magnitude=r["magnitude"],
                        agreement=r["verdict_agreement"], test=r["prediction"]["test"],
                        mechanism=r["mechanism"][:200], verdict=v)
            if cross:
                base["coder_survival"] = cross.get("survival")
                base["coder_verdict"] = cross["verdict"]

            def earlier(x, y):
                """Which of the pair is upstream, from band order alone.

                Returns None when both sit in the SAME band. Band order says nothing about
                direction within a band, and an earlier version silently fell back to the
                pair's storage order -- which produced `Legal Procedural Complexity ->
                Causation Complexity` from nothing but alphabetical position, and with it a
                cycle through Nature. A verdict that carries no direction of its own cannot
                be given one by a tie-break.
                """
                ba, bb = BAND_OF.get(x, 99), BAND_OF.get(y, 99)
                if ba == bb:
                    return None
                return (x, y) if ba < bb else (y, x)

            if v in ("a_causes_b", "b_causes_a") and r["passed"] is True:
                src, dst = (a, b) if v == "a_causes_b" else (b, a)
                add(src, dst, "reasoned_prior_tested", base)

            # An `indirect` verdict is not "no relationship" -- it asserts a PATH,
            # A -> M -> B, with M named. That is two edges. Discarding these threw away 42
            # of 77 verdicts and left Pre-existing Condition Salience with no children at
            # all, which is plainly wrong: a prior condition is the single biggest source
            # of causation argument in the scheme. Direction is taken from band order,
            # since the verdict itself does not carry one.
            elif v == "indirect" and r.get("prediction", {}).get("mediator") or                     (v == "indirect" and _mediator(r)):
                m = _mediator(r)
                pair = earlier(a, b)
                if m and m not in (a, b) and pair:
                    src, dst = pair
                    add(src, m, "reasoned_prior_path", dict(base, path=f"{src} -> {m} -> {dst}"))
                    add(m, dst, "reasoned_prior_path", dict(base, path=f"{src} -> {m} -> {dst}"))
                elif m and not pair:
                    undirected.append(dict(a=a, b=b, via=m, verdict=v,
                                           reason="same band: no direction available"))

            elif v == "common_cause":
                c = (r.get("named_common_cause") or "").strip()
                match = _resolve(c, list(BAND_OF))
                if match in (a, b):
                    # The model named one of the pair as the cause of both -- that is a
                    # direct edge, mislabelled. `Pre-existing Condition Salience ~
                    # Causation Complexity` came back this way, naming "Pre-existing
                    # Condition" as the common cause of itself and the other.
                    src = match
                    dst = b if match == a else a
                    add(src, dst, "reasoned_prior_tested",
                        dict(base, note="verdict was common_cause naming one of the pair; "
                                        "read as a direct edge"))
                elif match:
                    add(match, a, "reasoned_prior_common_cause", dict(base, common_cause=match))
                    add(match, b, "reasoned_prior_common_cause", dict(base, common_cause=match))

    # ---- acyclicity: a DAG that is not acyclic is not a DAG ----------------
    # The band constraint guarantees no cycle ACROSS bands but says nothing about within a
    # band. Nothing checked it until a cycle appeared:
    #   Causation Complexity -> Nature -> Legal Procedural Complexity -> Causation Complexity
    # Same-band edges are the only ones that can do this, so when a cycle is found the
    # same-band edge resting on the weakest evidence is removed and recorded.
    RANK = {"elicited": 5, "statute": 3, "measurement": 3, "reasoned_prior_tested": 2,
            "reasoned_prior_path": 1, "reasoned_prior_common_cause": 1}

    # A machine-derived edge pointing the opposite way to a human claim is simply wrong.
    # `Legal Procedural Complexity -> Causation Complexity` arrived this way, from a model
    # naming Legal Procedural Complexity as a mediator, and it survived a cycle break that
    # removed a better edge instead.
    contradicted = []
    for src, dst in list(edges):
        if (dst, src) in elicited_pairs and (src, dst) not in elicited_pairs:
            contradicted.append(dict(
                source=src, target=dst,
                classes=sorted({ev["evidence_class"] for ev in edges[(src, dst)]["evidence"]}),
                reason=f"contradicts the elicited edge {dst} -> {src}"))
            del edges[(src, dst)]

    def find_cycle(edge_keys):
        parents = defaultdict(set)
        nodes = {n for e in edge_keys for n in e}
        for s, d in edge_keys:
            parents[d].add(s)
        rem = {n: set(parents[n]) for n in nodes}
        while rem:
            ready = [k for k, v in rem.items() if not v]
            if not ready:
                return set(rem)
            for k in ready:
                del rem[k]
            for v in rem.values():
                v.difference_update(ready)
        return set()

    broken = []
    while True:
        stuck = find_cycle(list(edges))
        if not stuck:
            break
        candidates = [k for k in edges
                      if k[0] in stuck and k[1] in stuck
                      and BAND_OF[k[0]] == BAND_OF[k[1]]]
        if not candidates:
            candidates = [k for k in edges if k[0] in stuck and k[1] in stuck]
        weakest = min(candidates, key=lambda k: max(
            RANK.get(ev["evidence_class"], 0) for ev in edges[k]["evidence"]))
        broken.append(dict(source=weakest[0], target=weakest[1],
                           classes=sorted({ev["evidence_class"]
                                           for ev in edges[weakest]["evidence"]}),
                           reason="removed to break a cycle within a band"))
        del edges[weakest]

    # ---- roles follow from band position ---------------------------------
    roles_src = load("variable_roles.json") or {"roles": {}}
    roles = {}
    for c in cols:
        band = BAND_OF[c]
        stats_ = roles_src["roles"].get(c, {})   # latent nodes have no statistics
        uniq = stats_.get("unique_contribution_to_target")
        marg = stats_.get("marginal_with_target")
        # A root is a column nothing may point into, not a column in band 1. Claimant
        # Weekly Income sits in band 1 and has age and gender as causes.
        if c in LATENT:
            role = "latent"
        elif c in IMMUTABLE:
            role = "root"
        elif band == 6:
            role = "target"
        elif uniq is not None and marg is not None and abs(marg) >= 0.20 and uniq < 0.10:
            # strong marginal, nothing of its own: the award's association with it is
            # carried entirely by other columns
            role = "pass_through" if band <= 3 else "recorder"
        else:
            role = "intermediate"
        roles[c] = dict(band=band, band_label=LABEL[band], role=role,
                        marginal_with_target=marg, unique_contribution_to_target=uniq,
                        n_parents_permitted=sum(1 for s in cols if permitted(s, c)),
                        n_children_permitted=sum(1 for t in cols if permitted(c, t)))

    by_role = {}
    for c, r in roles.items():
        by_role.setdefault(r["role"], []).append(c)

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        bands=[dict(n=n, id=i, label=l, columns=c) for n, i, l, c in BANDS],
        latent=sorted(LATENT),
        rule=("Every edge runs from a band to the same band or a later one. Band 1 is "
              "internally acyclic by fiat: pre-accident facts are not treated as causing "
              "one another."),
        n_ordered_pairs=len(ordered), n_permitted=len(allowed),
        n_forbidden=len(ordered) - len(allowed),
        n_edges=len(edges), n_violations=len(violations),
        n_evidence_items=sum(len(e["evidence"]) for e in edges.values()),
        edges_by_strongest_class=dict(sorted(Counter(
            max((ev["evidence_class"] for ev in e["evidence"]),
                key=lambda c: RANK_REPORT.get(c, 0)) for e in edges.values()).items())),
        roles=roles, roles_by_type=by_role,
        edges=sorted(edges.values(), key=lambda e: (BAND_OF[e["source"]],
                                                    BAND_OF[e["target"]])),
        acyclic=True, broken_to_break_cycles=broken,
        contradicted_by_elicited=contradicted,
        undirected_same_band=undirected,
        violations=violations,
        immutable=IMMUTABLE,
        violation_note=("Evidence pointing backwards through the bands. Where the reverse "
                        "direction is permitted the edge is REVERSED and kept, because the "
                        "pair is still related and only the direction was refuted. Every "
                        "correction is recorded on the edge as direction_corrected."),
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}\n")
    print(f"ordered pairs {len(ordered)}   permitted {len(allowed)}   "
          f"forbidden {len(ordered) - len(allowed)}")
    print(f"edges with evidence: {len(edges)}   violations: {len(violations)}\n")

    for n, _, label, band_cols in BANDS:
        print(f"  {n}  {label}")
        for c in band_cols:
            r = roles[c]
            uq = r["unique_contribution_to_target"]
            print(f"       {c:32}{r['role']:14}"
                  f"{'' if uq is None else f'unique={uq:.2f}'}")

    if edges:
        print("\nEDGES SUPPORTED BY EVIDENCE, ALL FORWARD:")
        for e in doc["edges"]:
            classes = sorted({ev["evidence_class"] for ev in e["evidence"]})
            print(f"  b{BAND_OF[e['source']]}->b{BAND_OF[e['target']]}  "
                  f"{e['source']} -> {e['target']:32} {','.join(classes)}")
    if violations:
        print("\nVIOLATIONS (evidence pointing backwards):")
        for v in violations:
            print(f"  b{v['band_source']}->b{v['band_target']}  "
                  f"{v['source']} -> {v['target']}  [{v['evidence_class']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
