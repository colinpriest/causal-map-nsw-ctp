"""Stage 2 — measure the claim chronology from the dated event logs.

The raw workbook carries a per-case event log ('Event History': ``date | actor | event``).
That gives a real, checkable chronology instead of an asserted one.

Three steps, in increasing order of how much can go wrong:

  2a  EVENT ordering        pure data. For each ordered pair of event types, the share of
                            cases where the first occurrence of one precedes the other.
                            Nothing is asserted; nothing can be got wrong except arithmetic.

  2b  EVENT -> VARIABLE     derived, not declared. An event type is linked to a modelling
                            variable only if the per-case *count* of that event associates
                            with the variable's value at |rho| >= LINK_MIN. The rho is kept
                            as the link's provenance.

  2c  VARIABLE ordering     lifted from 2a through 2b, and ONLY for variables 2b actually
                            evidences. Most variables have no well-evidenced event proxy,
                            so most get no temporal claim. That is the correct outcome, not
                            a gap to paper over.

An earlier version of this script hard-coded the event->variable mapping by hand. It was
wrong -- it filed IME (an independent medical examination, the most common event at 732
occurrences) under Treatment Burden, when the data links it to Legal Procedural Complexity
-- and the variable orderings it produced were largely an artefact of that guess. The
mapping is now derived and its evidence recorded.

Caveat that applies to every number here: an event log records when something was DONE,
not when the underlying state arose. A WPI assessment dated after surgery does not make
surgery its cause. Chronology bounds orientation; it does not supply it.

Run:  python causal/stage2_temporal.py
Out:  causal/provenance/temporal_order.json
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "ctp" / "raw" / "ctp_impairment_lump_sum.xlsx"
CSV = ROOT / "ctp" / "ctp.csv"
OUT = Path(__file__).resolve().parent / "provenance" / "temporal_order.json"

MIN_EVENT_N = 80      # event types rarer than this are not analysed
MIN_CASES = 40        # pairs co-occurring in fewer cases than this are not reported
DECISIVE = 0.90       # share above which an ordering is treated as established
LINK_MIN = 0.35       # |rho| an event-count must reach to proxy a variable

# Variables an event could plausibly evidence. Candidate space only -- which links
# survive is decided by the data in 2b.
CANDIDATES = [
    "Injury Burden Intensity", "Treatment Burden", "Work Impact Severity",
    "Legal Procedural Complexity", "Psychological Injury Emphasis", "Liability Clarity",
    "Causation Complexity", "Pre-existing Condition Salience", "WPI %",
]


def parse_first(cell) -> dict[str, pd.Timestamp]:
    first: dict[str, pd.Timestamp] = {}
    for line in str(cell).splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        when = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(when):
            continue
        ev = parts[2]
        if ev not in first or when < first[ev]:
            first[ev] = when
    return first


def parse_counts(cell) -> Counter:
    c = Counter()
    for line in str(cell).splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            c[parts[2]] += 1
    return c


def orderings(per_case: list[dict], min_cases: int) -> list[dict]:
    votes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for first in per_case:
        for a, b in itertools.combinations(sorted(first), 2):
            if first[a] == first[b]:
                continue
            votes[(a, b)].append(1 if first[a] < first[b] else 0)
    out = []
    for (a, b), obs in votes.items():
        if len(obs) < min_cases:
            continue
        share = sum(obs) / len(obs)
        earlier, later = (a, b) if share >= 0.5 else (b, a)
        s = share if share >= 0.5 else 1 - share
        out.append(dict(earlier=earlier, later=later, share=round(s, 3),
                        n_cases=len(obs), decisive=s >= DECISIVE))
    return sorted(out, key=lambda r: -r["share"])


def main() -> None:
    raw = pd.read_excel(XLSX)
    csv = pd.read_csv(CSV)
    logs = raw["Event History"].fillna("")
    firsts = [parse_first(v) for v in logs]
    counts = [parse_counts(v) for v in logs]

    freq = Counter()
    for c in counts:
        freq.update(c)
    analysed = sorted(t for t, n in freq.items() if n >= MIN_EVENT_N)

    # ---- 2a: event-level ordering (pure data) ----------------------------
    event_pairs = orderings(firsts, MIN_CASES)

    # ---- 2b: derive event -> variable links ------------------------------
    mat = pd.DataFrame([{t: c.get(t, 0) for t in analysed} for c in counts])
    links, rejected = {}, []
    for t in analysed:
        scored = []
        for v in CANDIDATES:
            s = pd.concat([mat[t], csv[v]], axis=1).dropna()
            if len(s) < 60 or s.iloc[:, 0].nunique() < 2 or s.iloc[:, 1].nunique() < 2:
                continue
            r = float(stats.spearmanr(s.iloc[:, 0], s.iloc[:, 1]).statistic)
            scored.append((v, round(r, 3), len(s)))
        if not scored:
            continue
        scored.sort(key=lambda x: -abs(x[1]))
        var, r, n = scored[0]
        if abs(r) >= LINK_MIN:
            links[t] = dict(variable=var, rho=r, n=n,
                            runners_up=[dict(variable=v, rho=rr) for v, rr, _ in scored[1:3]])
        else:
            rejected.append(dict(event=t, best_variable=var, rho=r,
                                 reason=f"|rho| < {LINK_MIN}"))

    # ---- 2c: variable ordering, only through evidenced links -------------
    var_firsts = []
    for first in firsts:
        seen: dict[str, pd.Timestamp] = {}
        for ev, when in first.items():
            link = links.get(ev)
            if not link:
                continue
            v = link["variable"]
            if v not in seen or when < seen[v]:
                seen[v] = when
        var_firsts.append(seen)
    variable_pairs = orderings(var_firsts, MIN_CASES)

    covered = sorted({l["variable"] for l in links.values()})
    doc = dict(
        n_cases=len(firsts),
        thresholds=dict(min_event_n=MIN_EVENT_N, min_cases=MIN_CASES,
                        decisive=DECISIVE, link_min=LINK_MIN),
        event_pairs=event_pairs,
        event_variable_links=links,
        event_variable_rejected=rejected,
        variables_covered=covered,
        variables_uncovered=[v for v in CANDIDATES if v not in covered],
        variable_pairs=variable_pairs,
        caveat=("Event logs record when something was DONE, not when the underlying state "
                "arose. Chronology bounds orientation; it does not supply it. Variable-level "
                "orderings exist only for variables with an event proxy at "
                f"|rho| >= {LINK_MIN}; the rest carry no temporal claim."),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"2a  event orderings:   {len(event_pairs)} pairs, "
          f"{sum(r['decisive'] for r in event_pairs)} decisive")
    print(f"2b  evidenced links:   {len(links)} of {len(analysed)} event types "
          f"(|rho| >= {LINK_MIN})")
    for t, l in sorted(links.items(), key=lambda kv: -abs(kv[1]["rho"])):
        print(f"      {t:24} -> {l['variable']:30} rho={l['rho']:6.3f}")
    print(f"    variables covered:   {', '.join(covered) or 'none'}")
    print(f"    variables UNcovered: {', '.join(doc['variables_uncovered'])}")
    print(f"2c  variable orderings: {len(variable_pairs)} pairs, "
          f"{sum(r['decisive'] for r in variable_pairs)} decisive")
    for r in variable_pairs:
        mark = "*" if r["decisive"] else " "
        print(f"    {mark}{r['earlier']:30}-> {r['later']:30}{r['share']:6.3f}  n={r['n_cases']}")


if __name__ == "__main__":
    main()
