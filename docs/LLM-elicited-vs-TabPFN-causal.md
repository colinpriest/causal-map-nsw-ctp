# Two causal analyses of the same 540 decisions

This project now contains two causal analyses of the NSW CTP data, built on different
premises and answering different questions. This document sets them against each other.

> **Correction to an earlier framing.** This document originally set these up as
> "DAG: structure, no numbers" versus "DoubleML: numbers". That was wrong. A DAG plus an
> estimator is a complete method, and [`stage15`](../causal/stage15_dag_effects.py) now
> implements it: minimal backdoor sets by d-separation, identifiability refusals, and
> total-versus-direct decomposition. The honest contrast is **no graph versus your graph,
> same estimator both times** — see §3a.

| | **LLM-elicited DAG** | **DoubleML + TabPFN** |
|---|---|---|
| Question | *what causes what*, then *how large* | *how large is the effect* |
| Output | 17 nodes, 36 edges, **24 identified effects** | 10 treatment→outcome effect sizes |
| Shape | a graph | not a map at all — a contrast between two adjustment sets |
| Direction comes from | chronology, statute, domain claims, tested priors | **supplied by the analyst** |
| Effect sizes | none | ATE with 95% CI, in log-dollars |
| Artefact | [`ctp_reviewed_dag.html`](../causal/ctp_reviewed_dag.html) | [`ctp_identification_contrast.html`](../causal/ctp_identification_contrast.html) |
| Build | stages 0–11 | [`stage13`](../causal/stage13_tabpfn_dml.py), [`stage14`](../causal/stage14_render_dml_map.py) |

The headline is not that one wins. It is that **each supplies exactly what the other
cannot**, and that running DoubleML without a graph produces confident, significant,
sign-wrong answers on this dataset.

---

## 1. What each method can and cannot do

**DoubleML cannot orient an edge.** Give it a treatment and an outcome and it estimates the
effect. Swap them and it estimates that too, with equal confidence and no complaint. Every
arrow in its output is a direction the analyst supplied from outside. This is not a
criticism — it is a semi-parametric estimator, not a discovery algorithm.

It does mean there is no such thing as a "DoubleML causal map". An earlier version of this
project rendered one: a star with an arrow from each treatment to `Lump Sum`. Every one of
those directions was asserted, and every magnitude was conditional on the DAG's adjustment
set, so the picture displayed nothing the method had established. It has been replaced by
[`ctp_identification_contrast.html`](../causal/ctp_identification_contrast.html), which shows
the one thing these estimates do speak to: that the answer depends on what is conditioned on.
The superseded file is kept, banner-marked, as a record.

**The elicited DAG needs an estimator, but supplies everything else.** On its own it says
`WPI % → Non-Economic Loss` is a statutory gate quoting s 4.11 and gives no dollars. Attach
any estimator and it becomes a complete causal analysis — and one that does four things no
estimator can do alone (§3a).

**They fit together at exactly one point: the adjustment set.** DoubleML estimates
E[Y | do(D=d)] by adjusting for covariates X, and identification depends entirely on X being
a valid backdoor set. Nothing inside DoubleML determines X. A DAG does. So the graph the
rest of this project built is the thing that tells this analysis what to condition on — and
the consequences of getting it wrong are measurable, below.

---

## 2. Setup

Following <https://docs.doubleml.org/stable/examples/learners/py_tabpfn.html>, using
`DoubleMLAPOS` to estimate Average Potential Outcomes per treatment level and Average
Treatment Effects against the lowest level.

- **TabPFN 8.2.0 (TabPFN-3)**, six major versions past the example's `ModelVersion.V2`.
  Its weights are licence-gated and its browser handoff is broken on Windows
  (`select.select()` on stdin → `WinError 10038`), so authentication goes through
  `TABPFN_TOKEN` in `.env`. Run on an RTX 4070; CPU was tried first and was impractically
  slow, matching the example's note that CUDA is wanted.
- **Outcome** `log(Lump Sum)`, so an ATE of 0.10 ≈ a 10% change in the award.
- **Comparators** LightGBM and Linear, untuned, as in the example.
- **Missingness.** DoubleML needs complete rows; `ctp` is 27.6% complete. Covariates are
  median-filled **with a per-column missingness indicator**, preserving the informative
  missingness rather than erasing it. Complete-case would have left 149 rows.
- **Treatments** are the 10 discrete columns. `DoubleMLAPOS` cannot treat a continuous
  variable, so `WPI %` and the three dollar columns are structurally excluded — a real
  limitation on this dataset, not a choice.

Every treatment was run under **two adjustment sets**:

- `naive` — every other column. What a practitioner does without a graph.
- `dag` — the treatment's parents in the assembled graph: the backdoor set.

---

## 3. The result that matters: the adjustment set changes the answer

Same learner, same data, same treatment. Only X differs, so any gap is **identification,
not estimation**.

| Treatment | naive ATE | DAG ATE | change |
|---|---|---|---|
| `Causation Complexity` | **−0.332** [−0.53, −0.14] | **+0.600** | **sign flip** |
| `Nature` | −0.176 | +0.265 | **sign flip** |
| `Psychological Injury Emphasis` | +0.199 | **+0.748** | 3.8× larger |
| `Legal Procedural Complexity` | +0.253 | +0.663 | 2.6× larger |
| `Injury Burden Intensity` | +1.972 | +2.797 | 42% larger |
| `Treatment Burden` | +1.213 | +1.786 | 47% larger |
| `Work Impact Severity` | +1.236 | +1.518 | 23% larger |

### `Causation Complexity` is the clearest case

Naive: **−0.332, 95% CI [−0.525, −0.138]** — significant, and it says contested causation
*reduces* the award by roughly 28%. DAG-adjusted: **+0.600**.

The naive estimate is not merely imprecise. It is significant, tight, and pointing the wrong
way. It conditions on `Non-Economic Loss`, `Future Economic Loss` and `Lump Sum`'s other
determinants — that is, on mediators of the very effect being estimated, and on variables
that open a collider path through the target. A practitioner with no graph, following the
DoubleML documentation exactly, gets a publishable-looking negative result.

The elicited DAG's own annotation predicted this before any of it was run: seven edges were
flagged where the asserted direct effect and the marginal association disagree, and
`Causation Complexity → Future Economic Loss` was one of them, asserted negative against a
+0.20 marginal.

### Direction of the bias is systematic

Every non-flipping estimate is **larger** under DAG adjustment. Naive adjustment
conditions on mediators, which removes the indirect part of a total effect and shrinks it
toward zero. `Psychological Injury Emphasis` goes from +0.199 (not significant) to +0.748
(significant) — the naive analysis would have concluded that psychological injury does not
move an award.

---

## 3a. What the DAG does that the estimator cannot

[`stage15`](../causal/stage15_dag_effects.py) uses the graph properly rather than as a
source of a covariate list. **24 effects across three outcomes, 0 refused.**

**Minimal backdoor sets, by d-separation.** Not parents-of-treatment.
`Causation Complexity → Lump Sum` needs three covariates
(`Injury Burden Intensity`, `Pre-existing Condition Salience`,
`Psychological Injury Emphasis`) where the naive arm used nineteen. On 540 rows that is
variance and positivity, not tidiness.

**Knowing when to adjust for nothing.** Six treatment–outcome pairs are exogenous: the
graph gives the treatment no parents, so no adjustment is required and the raw contrast IS
the causal effect. DoubleML cannot express this — it errors on an empty design matrix — and
an analyst without a graph has no way to know it was allowed.

| exogenous pair | ATE | |
|---|---|---|
| `Liability Clarity → Non-Economic Loss` | −3.253 [−4.740, −1.767] | significant |
| `Pre-existing Condition Salience → Future Economic Loss` | −1.254 [−2.081, −0.428] | significant |
| `Liability Clarity → Future Economic Loss` | −0.916 [−1.669, −0.163] | significant |

**Total versus controlled direct.** The graph knows which variables mediate, so the effect
flowing *through* them separates from the effect flowing around them. **Five pairs reverse
sign** between total and direct:

| | total | direct | via mediators |
|---|---|---|---|
| `Causation Complexity → Lump Sum` | **+0.718** | **−0.137** | 109% |
| `Causation Complexity → Non-Economic Loss` | **+1.241** | **−1.135** | 293% |
| `Nature → Lump Sum` | +0.301 | −0.152 | 168% |
| `Pre-existing Condition Salience → Non-Economic Loss` | +0.327 | −1.042 | — |
| `Liability Clarity → Lump Sum` | −0.240 | +0.197 | — |

`Causation Complexity` closes a loop opened at the very start of this project. The original
graph asserted its effect on the money columns as **negative** — apportionment carves out
non-accident loss — against a **+0.20 marginal**, and flagged it as one of seven
sign-conflicting edges. The decomposition shows both are right: contested causation raises
awards *only* by travelling through injury severity and the heads of damage, and holding
those fixed it slightly reduces them. Total positive, direct negative.

Neither the naive run (−0.332, sign-wrong) nor the DAG-parents run (+0.600, total only)
could have shown that. It requires knowing which variables are mediators, which is
structure, not estimation.

**Identifiability refusals.** None triggered here, because the graph's only latent node
(`Psychological Injury`) does not sit on a backdoor path for any tested pair. The machinery
is in place: where every blocking set requires an unobserved variable, the effect is refused
and the offending path named, rather than estimated.

---

## 4. Learner comparison, and why it says less than the example's

The DoubleML example concludes that TabPFN wins: lowest nuisance RMSE and tightest
confidence intervals. **That conclusion is not reproducible here, and cannot be**, for a
reason that has nothing to do with TabPFN.

The example uses `make_irm_data_discrete_treatments` — **synthetic data with oracle ATEs**.
It knows the true answer, so "tightest interval" and "closest to truth" can be checked to be
the same thing. This dataset has no oracle. Nothing here can rank the learners on accuracy.

What can be measured:

| Learner | median CI width | total runtime (naive arm) |
|---|---|---|
| LightGBM | **0.437** | 109 s |
| TabPFN-3 | 0.722 | 671 s |
| Linear | 1.192 | 14 s |

On this dataset LightGBM produced the **narrower** intervals, at one sixth the runtime.
That reverses the example's ordering — but it is not evidence that LightGBM is more
accurate. A narrow interval around a biased estimate is the worst of both. With no ground
truth, interval width measures confidence, not correctness.

Where the learners genuinely disagree is more informative than which is tightest. On
`Causation Complexity` under naive adjustment, TabPFN gives −0.332 and LightGBM −0.259
(both significant, both negative) while **Linear gives +0.353, significant and positive**.
Three reasonable learners, three intervals, two of which exclude the truth as the other
sees it. When learners disagree in sign, the problem is rarely the learner.

---

## 5. Where the two analyses agree, and where they do not

**Agreement.** Both make `Injury Burden Intensity` the largest mover of the award — the
DAG gives it edges to both heads of damage, and DoubleML puts it at ATE +1.97 to +2.80,
several times any other treatment. Both make `Work Impact Severity` a genuine driver: the
DAG has it feeding `Future Economic Loss` with statutory backing from s 4.5(1)(a), and
DoubleML estimates +1.24 to +1.52.

**Disagreement worth noting.** The DAG's role analysis (stage 8) classified `Nature`,
`Causation Complexity` and `Legal Procedural Complexity` as **recorders** — strong marginal
association with the award, near-zero unique contribution once everything else is held
fixed. DoubleML under DAG adjustment gives all three non-trivial positive ATEs (+0.27,
+0.60, +0.66). These are not contradictory: "no unique contribution when everything is
conditioned on" and "a real total effect through mediators" are consistent, and the pair of
results together is more informative than either. A recorder can still carry a total effect
if it lies upstream of things that matter.

**Where DoubleML is silent.** It cannot treat `WPI %` — a continuous variable — so the one
edge with the strongest statutory backing in the entire project, the s 4.11 gate
`WPI % → Non-Economic Loss`, gets no effect estimate at all. The most secure structural
claim is invisible to the method that quantifies things.

---

## 6. What each approach cost

| | Elicited DAG | DoubleML + TabPFN |
|---|---|---|
| Wall clock | days of iteration across 14 stages | ~25 minutes on a GPU |
| API cost | ~1,100 cached LLM calls | none |
| Human input | substantial: definitions, 6 elicited edges, repeated correction | choice of treatment, outcome and adjustment set |
| Reproducible | yes — cache committed, all thresholds swept | yes — seeded |
| Fails how | silently, plausibly, in the direction of the prompt | loudly wrong only if you check the adjustment set |

The elicited pipeline's failure mode is a confident graph nobody has checked. This session
found four separate instances of a criterion that looked like rigour and was systematically
discarding fundamental relationships, plus a prompt instruction that produced 28 of 28
identical verdicts which vanished when removed.

DoubleML's failure mode is narrower and more dangerous: **one wrong adjustment set produces
one significant, tight, sign-wrong number**, with no internal signal that anything is amiss.
Nothing in the output of the naive `Causation Complexity` run indicates a problem.

---

## 7. Reproducing

```bash
python causal/stage13_tabpfn_dml.py --learners TabPFN,LightGBM,Linear
python causal/stage14_render_dml_map.py
```

Needs `doubleml`, `tabpfn==8.2.0`, `lightgbm`, and `TABPFN_TOKEN` plus
`TABPFN_NO_BROWSER=1` in `.env`. CUDA strongly recommended: TabPFN took 671 s on an RTX
4070 and was impractical on CPU.

Artefact: [`causal/provenance/tabpfn_dml.json`](../causal/provenance/tabpfn_dml.json) —
APOs, ATEs, confidence intervals and runtimes for every treatment × adjustment set ×
learner.
