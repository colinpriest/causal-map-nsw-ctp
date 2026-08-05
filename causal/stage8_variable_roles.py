"""Stage 8 — classify each column as root, mediator, collider or recorder, from the data.

"Which columns are intermediate?" has a statistical answer, but only once one thing is
supplied from outside: conditioning on a MEDIATOR and conditioning on a CONFOUNDER look
identical. Both sit on the path between two variables, and conditioning on either destroys
the dependence between them. Nothing in a correlation matrix tells them apart.

What separates them is position, and position needs an anchor. The anchor used here is
immutability: `Claimant Age` and `Claimant Gender` are fixed before the accident, so
nothing in the claim can cause them. They are roots by the nature of the measurement, not
by assertion. `Lump Sum` is the target. Everything else is then classified by behaviour.

FOUR SIGNATURES

  screening      Conditioning on V destroys dependence between other pairs. V is on paths
                 between them -- mediator or confounder, not yet distinguished.

  collider       Conditioning on V CREATES dependence between pairs that were independent,
                 or flips the sign. This is the signature of a common effect, and it is
                 the opposite of screening. `Lump Sum` should light up here: its two
                 recorded heads correlate +0.51 and go to -0.46 once it is held fixed.

  mediates       Conditioning on V collapses the association between a ROOT and the
  _root_target   TARGET. Since a root cannot be caused by anything, V cannot be a
                 confounder of that pair -- it must lie between them. This is what makes
                 the classification directional rather than merely topological.

  unique         |rho(V, target)| once every other column is held fixed. A variable with a
  _contribution  strong marginal association and near-zero unique contribution is
                 downstream of the real drivers -- it RECORDS the process rather than
                 driving it. `Legal Procedural Complexity` is the candidate.

The output is a role per column with the numbers behind it, not a verdict. Where the
signatures disagree the column is marked `ambiguous` rather than forced.

Run:  python causal/stage8_variable_roles.py
Out:  causal/provenance/variable_roles.json
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
CSV = ROOT_DIR / "ctp" / "ctp.csv"
OUT = HERE / "provenance" / "variable_roles.json"

TARGET = "Lump Sum"
# Fixed before the accident: nothing recorded in the claim can cause them. This is the one
# externally supplied fact the classification needs, and it is a property of what the
# columns measure rather than a causal assertion about the scheme.
ROOTS = ["Claimant Age", "Claimant Gender"]

HEAVY = {"Lump Sum", "Non-Economic Loss", "Future Economic Loss", "Claimant Weekly Income"}
MIN_N = 50
MIN_BASE = 0.10     # a pair must be associated this much before screening means anything
SCREEN = 0.35       # mean attenuation to count as a screener
UNIQUE = 0.10       # unique contribution to the target above which a column is a driver
# A collider does not merely amplify on average -- conditioning on it CREATES dependence
# between variables that were marginally independent. Averaging over all pairs washes that
# out and flagged 7 of 16 columns. Count the induced pairs instead.
INDEP = 0.10        # |rho| below this counts as marginally independent
INDUCED = 0.20      # |rho| above this, once conditioned, counts as induced dependence
N_INDUCED = 3       # induced pairs needed to call a collider
# Roots barely move the target directly (Claimant Age ~ Lump Sum is -0.094), so anchoring
# mediation on root->target finds nothing. Anchor on root->ANY variable the root is
# materially associated with; a root cannot be an effect, so anything that screens such a
# pair lies downstream of the root.
ROOT_LINK = 0.15


def prepare():
    d = pd.read_csv(CSV)
    d["Claimant Gender"] = (d["Claimant Gender"] == "Male").astype(float)
    d["Nature"] = (d["Nature"] == "Damages").astype(float)
    for c in HEAVY:
        d[c] = np.log1p(d[c])
    return d


def rho(d, a, b, given=(), rows=None):
    """Spearman(a, b), optionally residualised on `given`.

    `rows` forces the calculation onto a specific row set. This matters more than it
    looks: conditioning on a column that is 42% missing drops 42% of the sample, so
    comparing a pairwise-complete marginal against a listwise-complete partial measures
    conditioning AND selection together. On a table that is 27.6% complete that difference
    swamps the effect being tested -- it made Claimant Weekly Income look like a stronger
    collider than the target. Base and adjusted must be computed on the same rows.
    """
    given = [g for g in given if g not in (a, b)]
    s = d[[a, b] + given]
    if rows is not None:
        s = s.loc[rows]
    s = s.dropna()
    if len(s) < MIN_N or s[a].nunique() < 2 or s[b].nunique() < 2:
        return None, len(s)
    if not given:
        return float(stats.spearmanr(s[a], s[b]).statistic), len(s)
    z = s[given].to_numpy(float)
    keep = [i for i in range(z.shape[1]) if np.ptp(z[:, i]) > 0]
    if not keep:
        return float(stats.spearmanr(s[a], s[b]).statistic), len(s)
    X = np.column_stack([np.ones(len(s)), z[:, keep]])

    def resid(v):
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

    ra, rb = resid(s[a].to_numpy(float)), resid(s[b].to_numpy(float))
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None, len(s)
    return float(stats.spearmanr(ra, rb).statistic), len(s)


def main() -> int:
    d = prepare()
    cols = list(d.columns)
    others = [c for c in cols if c != TARGET]

    roles = {}
    for v in cols:
        pool = [c for c in cols if c != v]
        attenuations, flips, induced = [], 0, []
        for a, b in itertools.combinations(pool, 2):
            shared = d[[a, b, v]].dropna().index          # rows where all three exist
            if len(shared) < MIN_N:
                continue
            base, _ = rho(d, a, b, rows=shared)
            adj, _ = rho(d, a, b, given=[v], rows=shared)
            if base is None or adj is None:
                continue
            if abs(base) < INDEP and abs(adj) >= INDUCED:
                induced.append((a, b, round(base, 3), round(adj, 3)))
                continue
            if abs(base) < MIN_BASE:
                continue
            change = abs(adj) - abs(base)
            if change < 0:
                attenuations.append(-change / abs(base))
            if np.sign(adj) != np.sign(base) and abs(adj) > 0.05:
                flips += 1

        screening = float(np.mean(attenuations)) if attenuations else 0.0

        # does V lie between a root and the target?
        mediates = {}
        for r in ROOTS:
            if v == r or v == TARGET:
                continue
            for y in cols:
                if y in (r, v):
                    continue
                shared = d[[r, y, v]].dropna().index
                if len(shared) < MIN_N:
                    continue
                base, _ = rho(d, r, y, rows=shared)
                if base is None or abs(base) < ROOT_LINK:
                    continue
                adj, _ = rho(d, r, y, given=[v], rows=shared)
                if adj is None:
                    continue
                mediates[f"{r} -> {y}"] = round(1 - abs(adj) / abs(base), 3)
        # MEDIAN, not max: with ~20 root-pairs per variable the maximum is whichever pair
        # happened to attenuate most, which is cherry-picking on a 540-row table.
        med_max = round(float(np.median(list(mediates.values()))), 3) if mediates else None

        # what does V add to the target that nothing else does?
        if v == TARGET:
            uniq, marg = None, None
        else:
            cond = [c for c in others if c != v]
            shared = d[[v, TARGET] + cond].dropna().index
            marg, _ = rho(d, v, TARGET, rows=shared)
            u, _ = rho(d, v, TARGET, given=cond, rows=shared)
            uniq = None if u is None else abs(u)

        roles[v] = dict(
            screening=round(screening, 3),
            induced_dependence_pairs=len(induced), induced_examples=induced[:4],
            sign_flips_induced=flips,
            mediates_root_to_target=mediates,
            max_root_mediation=med_max,
            marginal_with_target=None if marg is None else round(marg, 3),
            unique_contribution_to_target=None if uniq is None else round(uniq, 3),
        )

    # ---- assign roles -----------------------------------------------------
    for v, r in roles.items():
        if v == TARGET:
            r["role"] = "target"
            r["why"] = "declared"
            continue
        if v in ROOTS:
            r["role"] = "root"
            r["why"] = "fixed before the accident; nothing in the claim can cause it"
            continue
        signals = []
        if r["induced_dependence_pairs"] >= N_INDUCED:
            signals.append("collider")
        if r["screening"] >= SCREEN:
            signals.append("on_path")
        if r["max_root_mediation"] is not None and r["max_root_mediation"] >= 0.30:
            signals.append("mediates_root_to_target")
        uniq = r["unique_contribution_to_target"]
        marg = abs(r["marginal_with_target"] or 0)
        if uniq is not None and marg >= 0.20 and uniq < UNIQUE:
            signals.append("recorder")

        if "collider" in signals:
            role = "collider"
        elif "mediates_root_to_target" in signals:
            role = "mediator"
        elif "recorder" in signals:
            role = "recorder"
        elif "on_path" in signals:
            role = "on_path_unresolved"
        else:
            role = "peripheral"
        if len([s for s in signals if s in ("collider", "recorder",
                                            "mediates_root_to_target")]) > 1:
            role = "ambiguous"
        r["role"] = role
        r["signals"] = signals

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        target=TARGET, roots=ROOTS,
        thresholds=dict(min_base=MIN_BASE, screen=SCREEN, unique=UNIQUE, indep=INDEP,
                        induced=INDUCED, n_induced=N_INDUCED, root_link=ROOT_LINK),
        anchor_note=("Roots are supplied from outside the data: age at injury and gender "
                     "are fixed before the accident. Without an anchor, a mediator and a "
                     "confounder are statistically indistinguishable -- conditioning on "
                     "either destroys dependence."),
        caveat=("`on_path_unresolved` means the column screens dependence but the data "
                "cannot say whether it sits between other variables or above them. "
                "Resolving it needs an ordering claim from statute, chronology or "
                "elicitation."),
        roles=roles,
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"wrote {OUT}\n")
    print(f"{'column':32}{'role':20}{'screen':>7}{'induc':>7}{'flips':>6}"
          f"{'rootmed':>9}{'uniq':>7}")
    order = {"target": 0, "collider": 1, "mediator": 2, "on_path_unresolved": 3,
             "recorder": 4, "peripheral": 5, "root": 6, "ambiguous": 7}
    for v, r in sorted(roles.items(), key=lambda kv: (order.get(kv[1]["role"], 9), kv[0])):
        rm = r["max_root_mediation"]
        uq = r["unique_contribution_to_target"]
        print(f"{v:32}{r['role']:20}{r['screening']:7.2f}"
              f"{r['induced_dependence_pairs']:7d}{r['sign_flips_induced']:6d}"
              f"{('  --  ' if rm is None else f'{rm:9.2f}')}"
              f"{('  --  ' if uq is None else f'{uq:7.2f}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
