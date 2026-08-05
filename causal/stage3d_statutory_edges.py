"""Stage 3d — keep only the edges both legs support.

Two independent lines of evidence have been gathered about the same provisions:

  LEG 1 (stage 3c)  The provision text NAMES a concept as bearing on a determination.
                    Evidence: a verbatim quote from primary source, plus a recorded
                    semantic mapping from that concept to a dataset variable.

  LEG 2 (stage 3a)  Cases that CITE that provision actually differ on that variable --
                    either in its value, or in whether it is recorded at all.

Neither is sufficient alone. Leg 1 without leg 2 is a rule nobody applied in this corpus.
Leg 2 without leg 1 is a correlation among cases that happen to cite the same section --
they differ in many ways at once, and citing a provision is not a random assignment.

Stage 3c was run blind: it never saw a single number from stage 3a. So agreement between
the legs is genuine corroboration rather than one leg being fitted to the other.

Edges are emitted at three grades:

  confirmed     both legs. Statute names it and the corpus shows it.
  statute_only  the provision names it; the corpus does not corroborate. STILL AN EDGE.
  data_only     cases differ but no provision supports it. NOT an edge -- a candidate for
                the case-text stage, which can read the reasoning directly.

WHY `statute_only` IS AN EDGE, NOT A REJECT
An earlier version required both legs. That was wrong, and wrong in a way that removed
precisely the rules the scheme runs on. Leg 2 can only speak about provisions the tribunal
CITES often enough to compare -- and citation frequency tracks what is DISPUTED, not what
governs. s 4.6 MAIA caps damages for loss of earnings and is cited in 5 of 540 decisions,
because nobody argues about it; s 4.7, an evidentiary standard for the assumptions behind
a future-loss award, is cited 111 times because that is what gets fought over. Requiring
corpus corroboration therefore selects for contested provisions and discards the
load-bearing ones.

So absence of leg 2 is recorded as "not corroborated here", never as refutation. A
provision cited too rarely to test is marked `untestable` rather than `uncorroborated`,
because the two mean different things.

Run:  python causal/stage3d_statutory_edges.py
Out:  causal/provenance/statutory_edges.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
READING = HERE / "provenance" / "provision_reading.json"
LINKS = HERE / "provenance" / "provision_links.json"
OUT = HERE / "provenance" / "statutory_edges.json"


def main() -> int:
    reading = json.loads(READING.read_text(encoding="utf-8"))
    links = json.loads(LINKS.read_text(encoding="utf-8"))

    # LEG 2, indexed by (provision, variable)
    empirical: dict[tuple[str, str], dict] = {}
    for r in links["records"]:
        empirical[(r["provision"], r["variable"])] = r

    confirmed, text_only = [], []
    for rd in reading["readings"]:
        prov = rd["provision"]
        for lk in rd["links"]:
            if not lk["quote_verbatim"]:
                continue
            ev_in = empirical.get((prov, lk["input"]))
            ev_out = empirical.get((prov, lk["determines"]))

            def support(ev):
                if not ev:
                    return None
                if ev.get("value_link"):
                    return dict(kind="value", effect=ev["effect"], p=ev["p_value"],
                                n_citing=ev["n_citing"])
                if ev.get("gate_link"):
                    return dict(kind="recorded", effect=ev["gate_effect"], p=ev["gate_p"],
                                n_citing=ev["n_citing"])
                return None

            s_in, s_out = support(ev_in), support(ev_out)
            # Distinguish "the corpus was able to test this and found nothing" from "the
            # corpus could not test it at all". Only the first is evidence of absence.
            testable = ev_in is not None or ev_out is not None
            corroboration = ("corroborated" if (s_in or s_out)
                             else "uncorroborated" if testable else "untestable")
            rec = dict(
                source=lk["input"], target=lk["determines"], relation=lk["relation"],
                provision=prov, act=rd["act"], section=rd["section"],
                grade="confirmed" if (s_in or s_out) else "statute_only",
                corroboration=corroboration,
                n_citing=(ev_in or ev_out or {}).get("n_citing", 0),
                statute=dict(concept=lk["concept_in_text"], quote=lk["quote"],
                             quote_match=lk["quote_match"],
                             mapping_rationale=lk["mapping_rationale"],
                             confidence=lk["confidence"],
                             source_file=rd.get("source_file"),
                             source_sha256=rd.get("source_sha256")),
                data=dict(on_source=s_in, on_target=s_out),
                llm=dict(model=rd["llm"]["model"], prompt_sha256=rd["llm"]["prompt_sha256"]),
            )
            (confirmed if (s_in or s_out) else text_only).append(rec)

    # LEG 2 with no leg 1 anywhere: candidates for the case-text stage, not edges.
    supported_pairs = {(c["provision"], c["source"]) for c in confirmed} | \
                      {(c["provision"], c["target"]) for c in confirmed}
    data_only = defaultdict(list)
    for (prov, var), r in empirical.items():
        if (prov, var) in supported_pairs:
            continue
        if r.get("value_link") or r.get("gate_link"):
            data_only[var].append(dict(
                provision=prov, n_citing=r["n_citing"],
                kind="value" if r.get("value_link") else "recorded",
                effect=r["effect"] if r.get("value_link") else r["gate_effect"]))

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        method=("Leg 1: provision text names the concept (verbatim quote + recorded "
                "semantic mapping). Leg 2: cases citing that provision differ on the "
                "variable. Stage 3c was blind to stage 3a."),
        n_confirmed=len(confirmed), n_statute_only=len(text_only),
        n_edges=len(confirmed) + len(text_only),
        n_data_only_variables=len(data_only),
        confirmed=confirmed, statute_only=text_only,
        edges=confirmed + text_only,
        data_only={k: sorted(v, key=lambda d: -abs(d["effect"]))[:5]
                   for k, v in sorted(data_only.items())},
        caveat=("A confirmed edge is grounded, not verified. The semantic mapping from "
                "statutory concept to dataset column is a model's interpretation, recorded "
                "in `statute.mapping_rationale` so it can be disputed. No practitioner has "
                "reviewed any of it."),
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}\n")
    print(f"CONFIRMED (statute names it AND cases differ): {len(confirmed)}")
    for c in confirmed:
        d = c["data"]["on_source"] or c["data"]["on_target"]
        print(f"  {c['source']} -> {c['target']}   [{c['relation']}]")
        print(f"      s {c['provision']}")
        print(f"      concept : {c['statute']['concept'][:76]}")
        print(f"      data    : {d['kind']} effect={d['effect']:+.3f} n={d['n_citing']}")
    print(f"\nTEXT ONLY (provision says so, corpus shows no difference): {len(text_only)}")
    for c in text_only:
        print(f"  {c['source']} -> {c['target']}   s {c['provision']}")
    print(f"\nDATA ONLY (no statutory text; candidates for the case-text stage): "
          f"{len(data_only)} variables")
    for var, rows in list(doc["data_only"].items())[:8]:
        top = rows[0]
        print(f"  {var:32} best: s {top['provision'][:34]:36} {top['effect']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
