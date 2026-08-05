"""Stage 15 — estimate the effects the DAG identifies, and refuse the ones it does not.

Stage 13 attached an estimator to the graph carelessly: it used "parents of the treatment"
as the adjustment set, fixed the outcome to Lump Sum, skipped exogenous treatments because
their adjustment set was empty, never checked whether an effect was identifiable at all,
and never used the graph's knowledge of which variables are mediators.

This does it properly. Everything here is what a DAG is FOR -- none of it is available to
an estimator working without one.

  MINIMAL BACKDOOR SETS.  Not parents-of-treatment, which is valid but rarely minimal.
      A set Z satisfies the backdoor criterion for (X, Y) when it contains no descendant of
      X and blocks every path into X. Smaller Z means less variance and fewer positivity
      problems, which on 540 rows is not a nicety.

  IDENTIFIABILITY.  If every set that would block a backdoor path contains an UNOBSERVED
      variable, the effect is not identifiable and no estimator can recover it. DoubleML
      asked for such an effect returns a confident number anyway. This refuses, and says
      which path it could not block.

  TOTAL vs CONTROLLED DIRECT.  The graph knows which variables are mediators, so the effect
      that flows around them can be separated from the effect that flows through them.
      Conditioning on a mediator is exactly the error that made stage 13's naive arm report
      contested causation as REDUCING an award; done deliberately and reported as a
      different estimand, it is informative rather than wrong.

  ANY OUTCOME.  Not just the award. Every directed path in the graph is an estimable
      quantity with its own adjustment set.

`indirect = total - controlled direct` is reported as a DESCRIPTIVE decomposition. It is not
a natural indirect effect: that needs cross-world assumptions this design does not support,
and on a nonlinear outcome scale the two do not coincide. Read it as "how much of the total
stops flowing when the mediators are held fixed".

Run:  python causal/stage15_dag_effects.py [--outcomes "Lump Sum,..."] [--learner TabPFN]
Out:  causal/provenance/dag_effects.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import re

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "ctp" / "ctp.csv"
GRAPH = HERE / "provenance" / "banded_graph.json"
OUT = HERE / "provenance" / "dag_effects.json"

SEED = 42
MIN_LEVEL_N = 25
MAX_LEVELS = 4
MAX_SET = 4        # largest adjustment set to search exhaustively for a minimal one


# ---------------------------------------------------------------- graph tools
class Graph:
    def __init__(self, edges, latent=(), nodes=None):
        self.edges = [(s, t) for s, t in edges]
        # Nodes must be carried explicitly. Deriving them from the edge list drops any node
        # left isolated -- which happens in the backdoor graph the moment a variable's only
        # arcs are the outgoing ones being removed, and then every lookup on it KeyErrors.
        self.nodes = sorted(set(nodes) if nodes else {n for e in self.edges for n in e})
        self.latent = set(latent)
        self.parents = {n: set() for n in self.nodes}
        self.children = {n: set() for n in self.nodes}
        for s, t in self.edges:
            self.parents[t].add(s)
            self.children[s].add(t)

    def descendants(self, x):
        seen, stack = set(), [x]
        while stack:
            n = stack.pop()
            for c in self.children.get(n, ()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def ancestors(self, nodes):
        seen, stack = set(nodes), list(nodes)
        while stack:
            n = stack.pop()
            for p in self.parents.get(n, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def d_separated(self, x, y, z) -> bool:
        """Ancestral-moral-graph test: X ⊥ Y | Z."""
        keep = self.ancestors({x, y} | set(z))
        adj = {n: set() for n in keep}
        for s, t in self.edges:
            if s in keep and t in keep:
                adj[s].add(t)
                adj[t].add(s)
        for n in keep:                                  # moralise: marry parents
            ps = [p for p in self.parents.get(n, ()) if p in keep]
            for a, b in combinations(ps, 2):
                adj[a].add(b)
                adj[b].add(a)
        blocked = set(z)
        if x in blocked or y in blocked:
            return True
        seen, stack = {x}, [x]
        while stack:
            n = stack.pop()
            if n == y:
                return False
            for m in adj.get(n, ()):
                if m not in seen and m not in blocked:
                    seen.add(m)
                    stack.append(m)
        return True

    def backdoor_graph(self, x):
        """G with edges OUT of x removed -- backdoor paths are the paths that survive."""
        return Graph([(s, t) for s, t in self.edges if s != x], self.latent,
                     nodes=self.nodes)

    def backdoor_sets(self, x, y, observed_only=True):
        """Every valid backdoor set up to MAX_SET, smallest first.

        Valid: contains no descendant of x, and d-separates x from y once edges out of x
        are removed.
        """
        gx = self.backdoor_graph(x)
        desc = self.descendants(x) | {x, y}
        pool = [n for n in self.nodes if n not in desc]
        if observed_only:
            pool = [n for n in pool if n not in self.latent]
        found = []
        for k in range(0, min(MAX_SET, len(pool)) + 1):
            for combo in combinations(pool, k):
                if gx.d_separated(x, y, set(combo)):
                    found.append(list(combo))
            if found:
                break                                    # smallest size wins
        return found

    def mediators(self, x, y):
        """Nodes on a directed path x -> ... -> y."""
        return sorted((self.descendants(x) & self.ancestors({y})) - {x, y})


# ---------------------------------------------------------------- estimation
def operative_thresholds():
    """Columns whose causal role is a threshold crossing, not their continuous value.

    s 4.11 MAIA bars non-economic loss unless assessed impairment exceeds 10%, and the data
    says the crossing is the whole of the effect: mean log non-economic loss 12.33 above the
    threshold against 0.70 at or below it, p = 2.7e-93, while future economic loss is
    unmoved at p = 0.99. Estimating a per-percent coefficient answers a question the scheme
    does not pose, so the column is binarised at its declared cut before estimation.
    """
    import yaml
    spec = yaml.safe_load((ROOT / "ctp" / "columns.yaml").read_text(encoding="utf-8"))
    out = {}
    for c in spec["columns"]:
        m = re.search(r"threshold at\s*([0-9.]+)", str(c.get("operative_form") or ""))
        if m:
            out[c["name"]] = float(m.group(1))
    return out


def load():
    d = pd.read_csv(CSV)
    d["Claimant Gender"] = (d["Claimant Gender"] == "Male").astype(float)
    d["Nature"] = (d["Nature"] == "Damages").astype(float)
    for c in ("Lump Sum", "Non-Economic Loss", "Future Economic Loss",
              "Claimant Weekly Income"):
        d[c] = np.log1p(d[c])
    for c in d.columns:
        if d[c].isna().any():
            d[c + "__missing"] = d[c].isna().astype(float)
            d[c] = d[c].fillna(d[c].median())
    # NOT applied here. The threshold is the operative form only where the column acts as
    # a CAUSE -- s 4.11 gates non-economic loss on the crossing. As an OUTCOME the
    # continuous value is the natural quantity: injury severity drives the assessed level,
    # not merely whether it clears 10%. Binarising globally turned `Injury Burden Intensity
    # -> WPI %` into a binary-outcome problem that DoubleML correctly refused. So the
    # transform is applied per estimate, to the treatment only.
    return d


def unadjusted(data, treatment, outcome):
    """The DAG says adjust for nothing, so the raw contrast IS the causal effect.

    DoubleML needs at least one covariate and errors on an empty design matrix, which is
    the wrong reason to lose an estimate: an exogenous treatment is the ONE case where no
    estimator is required at all. Welch interval on the difference in means.
    """
    from scipy import stats as st
    s = data[[treatment, outcome]].dropna()
    counts = s[treatment].value_counts()
    levels = [l for l in sorted(s[treatment].unique())
              if counts.get(l, 0) >= MIN_LEVEL_N][:MAX_LEVELS]
    if len(levels) < 2:
        return dict(error=f"fewer than two levels with >= {MIN_LEVEL_N} rows")
    base = s.loc[s[treatment] == levels[0], outcome]
    top = s.loc[s[treatment] == levels[-1], outcome]
    res = st.ttest_ind(top, base, equal_var=False)
    ci = res.confidence_interval(0.95)
    ate = float(top.mean() - base.mean())
    means = [float(s.loc[s[treatment] == l, outcome].mean()) for l in levels]
    return dict(levels=[float(x) for x in levels], n=int(len(s)), ate=ate,
                ci=[float(ci.low), float(ci.high)],
                se=float(ate / res.statistic) if res.statistic else None,
                significant=bool(ci.low > 0 or ci.high < 0),
                ate_all=[round(m - means[0], 4) for m in means],
                estimand="top-vs-base level contrast (exogenous, unadjusted)",
                seconds=0.0)


def as_treatment(data, treatment):
    """Binarise a column at its declared cut when it is the treatment, not the outcome."""
    cut = operative_thresholds().get(treatment)
    if cut is None:
        return data, None
    d = data.copy()
    d[treatment] = (d[treatment] > cut).astype(float)
    return d, cut


def estimate(data, treatment, outcome, adjust, learner_pair, n_rep=1):
    data, cut = as_treatment(data, treatment)
    if not list(adjust):
        r = unadjusted(data, treatment, outcome)
        if cut is not None and "estimand" in r:
            r["estimand"] = f"crossing the {cut:g} threshold (unadjusted, exogenous)"
        return r
    import doubleml as dml
    extra = [c + "__missing" for c in adjust if c + "__missing" in data.columns]
    cols = [outcome, treatment] + list(adjust) + extra
    df = data[list(dict.fromkeys(cols))].dropna()
    counts = df[treatment].value_counts()
    levels = [l for l in sorted(df[treatment].unique())
              if counts.get(l, 0) >= MIN_LEVEL_N][:MAX_LEVELS]
    if len(levels) < 2:
        return dict(error=f"fewer than two levels with >= {MIN_LEVEL_N} rows")
    t0 = time.time()
    try:
        obj = dml.DoubleMLData(df, outcome, treatment)
        m = dml.DoubleMLAPOS(obj, learner_pair["ml_g"], learner_pair["ml_m"],
                             treatment_levels=levels, n_rep=n_rep)
        m.fit()
        c = m.causal_contrast(reference_levels=levels[0])
        ci = c.confint(level=0.95)
        i = len(c.thetas) - 1
        return dict(levels=[float(x) for x in levels], n=int(len(df)),
                    ate=float(c.thetas[i]),
                    ci=[float(ci.iloc[i, 0]), float(ci.iloc[i, 1])],
                    se=float(c.ses[i]),
                    significant=bool(ci.iloc[i, 0] > 0 or ci.iloc[i, 1] < 0),
                    estimand=(f"crossing the {cut:g} threshold" if cut is not None
                              else "top-vs-base level contrast"),
                    ate_all=[float(x) for x in c.thetas],
                    seconds=round(time.time() - t0, 1))
    except Exception as exc:                              # noqa: BLE001
        return dict(error=str(exc)[:180])


def estimate_continuous(data, treatment, outcome, adjust, learner_pair, n_rep=1):
    """Partially linear model, for a treatment DoubleMLAPOS cannot handle.

    APOS contrasts discrete levels, so it cannot treat `Claimant Age`, `WPI %` or a dollar
    column -- which between them are the source of 12 of the graph's 36 edges, including
    both arithmetic edges into the award. DoubleMLPLR gives a single coefficient for a
    continuous treatment under the same backdoor set.

    The ESTIMAND DIFFERS: this is an effect per unit of the treatment, where APOS reports a
    contrast between the top and bottom level. Sign and significance are comparable across
    the two; the magnitudes are not, and anything displaying them together must say which
    is which.
    """
    import doubleml as dml
    extra = [c + "__missing" for c in adjust if c + "__missing" in data.columns]
    cols = [outcome, treatment] + list(adjust) + extra
    df = data[list(dict.fromkeys(cols))].dropna()
    if len(df) < 60 or df[treatment].nunique() < 3:
        return dict(error="too few rows or too few distinct treatment values")
    if not [c for c in cols if c not in (outcome, treatment)]:
        # no covariates: the partial correlation IS the adjusted association
        from scipy import stats as st
        r = st.spearmanr(df[treatment], df[outcome])
        return dict(n=int(len(df)), ate=float(r.statistic), ci=[float("nan")] * 2,
                    se=None, significant=bool(r.pvalue < 0.05),
                    estimand="rank correlation (exogenous, no adjustment)", seconds=0.0)
    t0 = time.time()
    try:
        obj = dml.DoubleMLData(df, outcome, treatment)
        m = dml.DoubleMLPLR(obj, learner_pair["ml_g"], learner_pair["ml_g"], n_rep=n_rep)
        m.fit()
        ci = m.confint(level=0.95)
        lo, hi = float(ci.iloc[0, 0]), float(ci.iloc[0, 1])
        return dict(n=int(len(df)), ate=float(m.coef[0]), ci=[lo, hi],
                    se=float(m.se[0]), significant=bool(lo > 0 or hi < 0),
                    estimand="per-unit coefficient (partially linear)",
                    seconds=round(time.time() - t0, 1))
    except Exception as exc:                              # noqa: BLE001
        return dict(error=str(exc)[:180])


def make_learner(name, device=None):
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "TabPFN":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from tabpfn import TabPFNRegressor, TabPFNClassifier
        return dict(ml_g=TabPFNRegressor(device=device, random_state=SEED),
                    ml_m=TabPFNClassifier(device=device, random_state=SEED)), device
    import lightgbm as lgbm
    a = dict(n_estimators=500, learning_rate=0.01, max_depth=3, min_data_in_leaf=10,
             lambda_l1=1, lambda_l2=2, random_state=SEED, verbose=-1)
    return dict(ml_g=lgbm.LGBMRegressor(**a), ml_m=lgbm.LGBMClassifier(**a)), "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes",
                    default="Lump Sum,Non-Economic Loss,Future Economic Loss")
    ap.add_argument("--learner", default="LightGBM", help="TabPFN | LightGBM")
    ap.add_argument("--device", default=None)
    ap.add_argument("--edges", action="store_true",
                    help="estimate every edge of the graph rather than treatment->outcome")
    args = ap.parse_args()

    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    graph = Graph([(e["source"], e["target"]) for e in g["edges"]], g.get("latent", []))
    data = load()
    learner, device = make_learner(args.learner, args.device)
    outcomes = [o.strip() for o in args.outcomes.split(",")]
    print(f"learner {args.learner} on {device}   nodes {len(graph.nodes)}   "
          f"edges {len(graph.edges)}   latent {sorted(graph.latent)}\n")

    treatments = [n for n in graph.nodes
                  if n in data.columns and data[n].nunique() <= 6]
    rows, refused = [], []

    if args.edges:
        # Every edge is an estimable quantity with its own backdoor set. Discrete sources
        # go through APOS, continuous through PLR, and the latent node through neither --
        # it has no data, which is what "latent" means.
        for src, dst in [(e["source"], e["target"]) for e in g["edges"]]:
            if src in graph.latent or dst in graph.latent:
                refused.append(dict(treatment=src, outcome=dst,
                                    reason="latent node: no values exist to estimate from"))
                continue
            sets_obs = graph.backdoor_sets(src, dst, observed_only=True)
            if not sets_obs:
                refused.append(dict(treatment=src, outcome=dst,
                                    reason="no backdoor set of observed variables"))
                continue
            z = sets_obs[0]
            # a thresholded treatment is discrete however many values the raw column has
            discrete = (data[src].nunique() <= 6
                        or src in operative_thresholds())
            est = (estimate(data, src, dst, z, learner) if discrete
                   else estimate_continuous(data, src, dst, z, learner))
            rows.append(dict(treatment=src, outcome=dst, backdoor_set=z,
                             exogenous=(len(z) == 0), edge=True,
                             treatment_kind="discrete" if discrete else "continuous",
                             mediators=[], total=est))
            mark = "ok " if "ate" in est else "XX "
            val = (f"{est['ate']:+.3f} [{est['ci'][0]:+.3f},{est['ci'][1]:+.3f}]"
                   if "ate" in est and est["ci"] == est["ci"] else est.get("error", "")[:40])
            print(f"{mark}{src} -> {dst:28} {val}")
        doc = dict(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            learner=args.learner, device=device, seed=SEED, mode="edges",
            graph=dict(nodes=len(graph.nodes), edges=len(graph.edges),
                       latent=sorted(graph.latent)),
            estimand_note=("Discrete treatments report a top-vs-base level contrast; "
                           "continuous treatments report a per-unit coefficient from a "
                           "partially linear model. Sign and significance are comparable "
                           "across the two; magnitudes are NOT."),
            n_estimated=len(rows), n_refused=len(refused),
            effects=rows, refused=refused)
        out = OUT.with_name("dag_edge_effects.json")
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
        print(f"estimated {len(rows)} of {len(g['edges'])} edges; refused {len(refused)}")
        return 0

    for outcome in outcomes:
        if outcome not in data.columns:
            continue
        for t in treatments:
            if t == outcome or outcome not in graph.descendants(t):
                continue           # no directed path: the graph asserts no causal effect

            sets_obs = graph.backdoor_sets(t, outcome, observed_only=True)
            sets_any = graph.backdoor_sets(t, outcome, observed_only=False)
            meds = [m for m in graph.mediators(t, outcome)
                    if m in data.columns and m not in graph.latent]

            if not sets_obs:
                need = sorted({n for s in sets_any for n in s} & graph.latent)
                refused.append(dict(
                    treatment=t, outcome=outcome,
                    reason=("no backdoor set of observed variables blocks every path; "
                            f"blocking requires the unobserved {need}" if need else
                            "no valid backdoor set exists within the searched size"),
                    latent_required=need))
                print(f"REFUSED  {t} -> {outcome}: not identifiable")
                continue

            z = sets_obs[0]
            row = dict(treatment=t, outcome=outcome, backdoor_set=z,
                       n_valid_sets=len(sets_obs), mediators=meds,
                       exogenous=(len(z) == 0))
            row["total"] = estimate(data, t, outcome, z, learner)
            if meds:
                row["controlled_direct"] = estimate(data, t, outcome,
                                                    list(dict.fromkeys(z + meds)), learner)
                tot, dr = row["total"].get("ate"), row["controlled_direct"].get("ate")
                if tot is not None and dr is not None:
                    row["implied_indirect"] = round(tot - dr, 4)
                    row["share_through_mediators"] = (
                        round(1 - dr / tot, 3) if abs(tot) > 1e-9 else None)
            rows.append(row)

            tot = row["total"]
            desc = "no adjustment (exogenous)" if not z else f"adjust {z}"
            if "ate" in tot:
                print(f"{t} -> {outcome}")
                print(f"    total  ATE {tot['ate']:+.3f} "
                      f"[{tot['ci'][0]:+.3f}, {tot['ci'][1]:+.3f}]  {desc}  n={tot['n']}")
                if "controlled_direct" in row and "ate" in row["controlled_direct"]:
                    cd = row["controlled_direct"]
                    print(f"    direct ATE {cd['ate']:+.3f} "
                          f"[{cd['ci'][0]:+.3f}, {cd['ci'][1]:+.3f}]  "
                          f"holding {len(meds)} mediator(s) fixed"
                          f"   -> {row.get('share_through_mediators')} via mediators")

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        learner=args.learner, device=device, seed=SEED,
        graph=dict(nodes=len(graph.nodes), edges=len(graph.edges),
                   latent=sorted(graph.latent)),
        method=("Minimal backdoor sets from the assembled DAG; effects estimated with "
                "DoubleMLAPOS. Outcomes on log scale, so an ATE of 0.10 is about a 10% "
                "change."),
        decomposition_caveat=(
            "implied_indirect = total - controlled direct. A DESCRIPTIVE decomposition, "
            "not a natural indirect effect: that needs cross-world assumptions this design "
            "does not support, and on a log outcome the two do not coincide."),
        identifiability_note=(
            "An effect is refused when no adjustment set of OBSERVED variables blocks every "
            "backdoor path. An estimator without a graph cannot refuse -- it returns a "
            "confident number regardless."),
        n_estimated=len(rows), n_refused=len(refused),
        effects=rows, refused=refused,
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"estimated {len(rows)}   refused as unidentifiable {len(refused)}")
    exo = [r for r in rows if r["exogenous"]]
    print(f"exogenous treatments needing NO adjustment: {len(exo)} "
          f"({', '.join(sorted({r['treatment'] for r in exo})) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
