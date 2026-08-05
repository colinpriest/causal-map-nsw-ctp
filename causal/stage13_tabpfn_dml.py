"""Stage 13 — a competing causal analysis: DoubleML + TabPFN, following the DoubleML docs.

The rest of this project builds structure: which variable causes which, and on what
evidence. This stage does the opposite thing. It assumes a structure and estimates an
EFFECT SIZE -- the Average Potential Outcome at each level of a treatment, and the Average
Treatment Effect against the reference level -- using double machine learning with TabPFN
as the nuisance learner, per
https://docs.doubleml.org/stable/examples/learners/py_tabpfn.html

TabPFN VERSION. The example pins ModelVersion.V2 (tabpfn 2.x). This runs TabPFN-3
(tabpfn 8.2.0), which is a materially different and much larger model -- worth stating,
because the example's headline claim is about model quality, and testing it against a
six-major-version-newer model is a stronger test than repeating it against 2.x.

8.x gates its weights behind a licence the user must accept. Its browser handoff is broken
on Windows (`select.select()` on stdin raises WinError 10038), so authentication goes
through TABPFN_TOKEN and TABPFN_NO_BROWSER=1 in .env instead. A script must not accept a
licence on a user's behalf; the token is read from the environment and never logged.

The two approaches answer different questions and neither subsumes the other, which is the
point of running both.

THE ADJUSTMENT SET IS THE WHOLE ARGUMENT
DoubleML estimates E[Y | do(D = d)] by adjusting for covariates X. It cannot tell you what
belongs in X -- that is an identification question, and identification comes from a causal
graph. So this stage runs every treatment TWICE:

  naive      X = every other column. What a practitioner does with no graph, and what the
             DoubleML documentation example implicitly does with its synthetic covariates.
             On this dataset it conditions on mediators and on a collider, so it is
             expected to be biased, and by how much is worth measuring.

  dag        X = the treatment's parents in the assembled graph
             (causal/provenance/banded_graph.json), which is the backdoor set the DAG
             licenses. Only available where the graph has an opinion.

The gap between the two is an estimate of what the graph is worth, in the units the
estimate is reported in.

WHAT THIS CANNOT DO
It cannot orient an edge, so it cannot build a graph. Reverse the treatment and the outcome
and DoubleML will happily estimate that too, with no complaint. Every number below is
conditional on a direction supplied from outside.

MISSINGNESS
DoubleML needs complete rows. `ctp` is 27.6% complete, so treatments are restricted to the
fully observed ordinals and covariates are median/mode-imputed with a missingness indicator
per imputed column -- which preserves the informative-missingness signal rather than
pretending it away. The complete-case alternative would leave 149 rows.

Run:  python causal/stage13_tabpfn_dml.py [--learners TabPFN,LightGBM,...] [--quick]
Out:  causal/provenance/tabpfn_dml.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV = ROOT / "ctp" / "ctp.csv"
GRAPH = HERE / "provenance" / "banded_graph.json"
OUT = HERE / "provenance" / "tabpfn_dml.json"

TARGET = "Lump Sum"
SEED = 42
MIN_LEVEL_N = 25          # a treatment level with fewer rows cannot support an APO
MAX_LEVELS = 4


def load() -> tuple[pd.DataFrame, list[str]]:
    d = pd.read_csv(CSV)
    d["Claimant Gender"] = (d["Claimant Gender"] == "Male").astype(float)
    d["Nature"] = (d["Nature"] == "Damages").astype(float)
    d[TARGET] = np.log(d[TARGET])
    imputed = []
    for c in d.columns:
        if c == TARGET or not d[c].isna().any():
            continue
        # An indicator keeps the informative-missingness signal that imputation destroys.
        d[c + "__missing"] = d[c].isna().astype(float)
        imputed.append(c + "__missing")
        d[c] = d[c].fillna(d[c].median())
    return d, imputed


def make_learners(names, device=None):
    # TabPFN is a transformer and wants CUDA; the DoubleML example notes CPU is a fallback.
    # An earlier run defaulted to CPU and was still going when the harness killed it.
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TabPFN device: {device}")
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    out = {}
    if "TabPFN" in names:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")          # TABPFN_TOKEN, never echoed
        from tabpfn import TabPFNRegressor, TabPFNClassifier
        out["TabPFN"] = dict(ml_g=TabPFNRegressor(device=device, random_state=SEED),
                             ml_m=TabPFNClassifier(device=device, random_state=SEED))
    if "LightGBM" in names:
        import lightgbm as lgbm
        a = dict(n_estimators=500, learning_rate=0.01, max_depth=3, min_data_in_leaf=10,
                 lambda_l1=1, lambda_l2=2, random_state=SEED, verbose=-1)
        out["LightGBM"] = dict(ml_g=lgbm.LGBMRegressor(**a), ml_m=lgbm.LGBMClassifier(**a))
    if "RandomForest" in names:
        a = dict(n_estimators=500, min_samples_leaf=10, random_state=SEED)
        out["RandomForest"] = dict(ml_g=RandomForestRegressor(**a),
                                   ml_m=RandomForestClassifier(**a))
    if "Linear" in names:
        out["Linear"] = dict(ml_g=LinearRegression(),
                             ml_m=LogisticRegression(max_iter=1000))
    return out


def dag_parents() -> dict[str, list[str]]:
    if not GRAPH.exists():
        return {}
    g = json.loads(GRAPH.read_text(encoding="utf-8"))
    par = {}
    for e in g["edges"]:
        par.setdefault(e["target"], []).append(e["source"])
    return par


def run(dml, data, treatment, covars, learners, n_rep):
    import doubleml as dml_pkg
    cols = [TARGET, treatment] + covars
    df = data[cols].dropna()
    levels = sorted(df[treatment].unique())
    counts = df[treatment].value_counts()
    levels = [l for l in levels if counts.get(l, 0) >= MIN_LEVEL_N][:MAX_LEVELS]
    if len(levels) < 2:
        return None, f"fewer than two levels with >= {MIN_LEVEL_N} rows"

    obj = dml_pkg.DoubleMLData(df, TARGET, treatment)
    res = {}
    for name, pair in learners.items():
        t0 = time.time()
        try:
            m = dml_pkg.DoubleMLAPOS(obj, pair["ml_g"], pair["ml_m"],
                                     treatment_levels=levels, n_rep=n_rep)
            m.fit()
            ci = m.confint(level=0.95)
            contrast = m.causal_contrast(reference_levels=levels[0])
            cci = contrast.confint(level=0.95)
            res[name] = dict(
                seconds=round(time.time() - t0, 1), levels=[float(x) for x in levels],
                apo=[float(x) for x in m.coef],
                apo_ci=[[float(a), float(b)] for a, b in ci.to_numpy()],
                ate=[float(x) for x in contrast.thetas],
                ate_ci=[[float(a), float(b)] for a, b in cci.to_numpy()],
                ate_se=[float(x) for x in contrast.ses],
            )
        except Exception as exc:                       # noqa: BLE001
            res[name] = dict(error=str(exc)[:200])
    return res, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default="TabPFN,LightGBM,RandomForest,Linear")
    ap.add_argument("--n-rep", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="TabPFN only, 3 treatments")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    import doubleml as dml_pkg
    names = ["TabPFN"] if args.quick else args.learners.split(",")
    learners = make_learners(names, args.device)
    data, imputed = load()
    parents = dag_parents()

    # Discrete columns are the only ones DoubleMLAPOS can treat.
    candidates = [c for c in data.columns
                  if c != TARGET and not c.endswith("__missing")
                  and data[c].nunique() <= 6]
    if args.quick:
        candidates = candidates[:3]
    all_covars = [c for c in data.columns if c != TARGET]
    print(f"treatments: {len(candidates)}   learners: {list(learners)}   "
          f"rows: {len(data)}   imputed-indicator columns: {len(imputed)}")

    results = []
    for t in candidates:
        naive_x = [c for c in all_covars if c != t and not c.startswith(t)]
        par = [p for p in parents.get(t, []) if p in data.columns]
        dag_x = par + [c for c in imputed if c.replace("__missing", "") in par]

        row = dict(treatment=t, n_levels=int(data[t].nunique()),
                   dag_parents=par, n_naive_covariates=len(naive_x))
        print(f"\n{t}  (naive X={len(naive_x)}, dag X={len(dag_x)})")

        r, err = run(dml_pkg, data, t, naive_x, learners, args.n_rep)
        row["naive"] = r or dict(skipped=err)
        if r:
            for k, v in r.items():
                if "ate" in v:
                    print(f"    naive {k:13} ATE(top vs base) = {v['ate'][-1]:+.3f} "
                          f"[{v['ate_ci'][-1][0]:+.3f}, {v['ate_ci'][-1][1]:+.3f}]"
                          f"  {v['seconds']}s")

        if dag_x:
            r2, err2 = run(dml_pkg, data, t, dag_x, learners, args.n_rep)
            row["dag"] = r2 or dict(skipped=err2)
            if r2:
                for k, v in r2.items():
                    if "ate" in v:
                        print(f"    dag   {k:13} ATE(top vs base) = {v['ate'][-1]:+.3f} "
                              f"[{v['ate_ci'][-1][0]:+.3f}, {v['ate_ci'][-1][1]:+.3f}]")
        else:
            row["dag"] = dict(skipped="the assembled graph gives this variable no parents")
        results.append(row)

    doc = dict(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        method=("DoubleMLAPOS per https://docs.doubleml.org/stable/examples/learners/"
                "py_tabpfn.html. Outcome is log(Lump Sum); ATE is reported in log-dollars, "
                "so 0.10 is roughly a 10% change in the award."),
        target=TARGET, outcome_transform="log", seed=SEED, n_rep=args.n_rep,
        learners=list(learners), n_rows=int(len(data)),
        tabpfn_version=(__import__("tabpfn").__version__ if "TabPFN" in learners else None),
        imputation=("median fill plus a per-column missingness indicator, because DoubleML "
                    "needs complete rows and complete-case would leave 149"),
        adjustment_sets=dict(
            naive="every other column -- conditions on mediators and on a collider",
            dag="the treatment's parents in the assembled graph, i.e. the backdoor set"),
        caveat=("DoubleML estimates effect SIZE given an assumed direction and adjustment "
                "set. It cannot orient an edge: swap treatment and outcome and it will "
                "estimate that too, without complaint."),
        results=results,
    )
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
