# Two causal analyses of the same 540 decisions

This project now contains two causal analyses of the NSW CTP data, built on different
premises and answering different questions. This document sets them against each other.

> **Correction to an earlier framing.** This document originally set these up as
> "DAG: structure, no numbers" versus "DoubleML: numbers". That was wrong. A DAG plus an
> estimator is a complete method, and [`stage15`](../causal/stage15_dag_effects.py)
> implements it: minimal backdoor sets by d-separation, identifiability refusals,
> total-versus-direct decomposition, and an effect on **every edge of the graph** (§3a,
> §3b). The honest contrast is **no graph versus your graph, same estimator both
> times**.

| | **LLM-elicited DAG** | **DoubleML + TabPFN** |
|---|---|---|
| Question | *what causes what*, then *how large* | *how large is the effect* |
| Output | 17 nodes, 34 edges, **an effect on 33 of them** | 10 treatment→outcome effect sizes |
| Shape | a graph | not a map at all — a contrast between two adjustment sets |
| Direction comes from | chronology, statute, domain claims, tested priors | **supplied by the analyst** |
| Effect sizes | per edge, each under its own backdoor set | per treatment, one adjustment set at a time |
| Artefact | [`ctp_reviewed_dag.html`](../causal/ctp_reviewed_dag.html) | [`ctp_identification_contrast.html`](../causal/ctp_identification_contrast.html) |
| Build | stages 0–11, [`stage15`](../causal/stage15_dag_effects.py), [`elicited_edges.yaml`](../causal/elicited_edges.yaml) | [`stage13`](../causal/stage13_tabpfn_dml.py), [`stage14`](../causal/stage14_render_dml_map.py) |

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

**The elicited DAG needs an estimator, and supplies everything else.** On its own it says
`WPI % → Non-Economic Loss` is a statutory gate quoting s 4.11 and gives no dollars.
Attached to an estimator it becomes a complete causal analysis: every one of its edges is
an estimable quantity with its own adjustment set, and 33 of 34 now carry an effect with a
confidence interval (§3b). What the graph adds beyond the number is in §3a.

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

**`n_rep=5`.** An earlier version of this section reported `n_rep=1` figures. Re-running the
identical `naive` specification — which conditions on all 19 other columns and therefore
*cannot* be affected by the graph — moved three estimates by 38–50%. That is cross-fitting
Monte Carlo noise, and it was the same size as several gaps reported below as findings.
Every number in this section is now at `n_rep=5`; the run costs about 75 minutes.

---

## 3. The result that matters: the adjustment set changes the answer

Same learner, same data, same treatment. Only X differs, so any gap is **identification,
not estimation**.

| Treatment | naive ATE | DAG ATE | change |
|---|---|---|---|
| `Causation Complexity` | **−0.290** [−0.54, −0.07] | **+0.677** [+0.01, +1.44] | **sign flip** |
| `Nature` | **−0.202** [−0.38, −0.03] | +0.329 [−0.03, +0.69] | **sign flip** |
| `Work Impact Severity` | +0.999 [−0.33, **+2.39**] | **+1.324** [+1.00, +1.65] | 33% larger, and 4× tighter |
| `Psychological Injury Emphasis` | +0.244 [+0.03, +0.44] | **+0.760** [+0.56, +0.97] | 3.1× larger |
| `Treatment Burden` | +1.060 [+0.44, +1.67] | +1.803 [+1.21, +2.42] | 70% larger |
| `Injury Burden Intensity` | +1.935 [+1.25, +2.60] | +2.816 [+2.43, +3.20] | 46% larger |
| `Legal Procedural Complexity` | +0.248 [−0.08, +0.62] | +0.455 [−0.49, +1.41] | 1.8× larger, neither significant |

### `Causation Complexity` is the clearest case

Naive: **−0.290, 95% CI [−0.54, −0.07]** — significant, and it says contested causation
*reduces* the award by roughly 25%. DAG-adjusted: **+0.677**.

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

Every non-flipping estimate is **larger** under DAG adjustment — all five of them. Naive
adjustment conditions on mediators, which removes the indirect part of a total effect and
shrinks it toward zero. `Psychological Injury Emphasis` goes from +0.244 to +0.760: the
naive arm recovers under a third of the effect.

### Precision is *not* systematic, and the exception is instructive

`Work Impact Severity` is the clean case for a smaller adjustment set. Naive gives
+0.999 with a CI of [−0.33, +2.39] — nineteen covariates on 540 rows, and the interval
spans zero. The graph says one covariate suffices, and the estimate becomes
**+1.324 [+1.00, +1.65]**, four times tighter and clear of zero. Same data, same learner:
the eighteen covariates the graph did not ask for were buying nothing but variance.

But this does not generalise, and claiming it would be wrong. Only three of ten treatments
get a tighter interval under DAG adjustment. `Causation Complexity` goes the other way —
naive [−0.54, −0.07], DAG [+0.01, +1.44], three times *wider*.

That widening is the honest result, not an embarrassment. The narrow naive interval is
narrow around a number with the wrong sign: precision estimating the wrong estimand. The
wide DAG interval is what the data actually supports about the effect you asked for. **A
graph buys unbiasedness; what it does to variance depends on the graph.** An analyst
choosing an adjustment set by interval width will systematically choose the biased one.

---

## 3a. What the DAG does that the estimator cannot

[`stage15`](../causal/stage15_dag_effects.py) uses the graph properly rather than as a
source of a covariate list. **24 effects across three outcomes, 0 refused.**

**Minimal backdoor sets, by d-separation.** Not parents-of-treatment.
`Causation Complexity → Lump Sum` needs three covariates
(`Injury Burden Intensity`, `Pre-existing Condition Salience`,
`Psychological Injury Emphasis`) where the naive arm used nineteen. The gain here is
bias and positivity — sixteen fewer covariates to find overlap on. It is *not* variance:
this pair's interval widens under the smaller set (§3), and the naive arm's tighter
interval sits around a sign-wrong estimate.

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

Neither the naive run (−0.290, sign-wrong) nor the DAG-parents run (+0.677, total only)
could have shown that. It requires knowing which variables are mediators, which is
structure, not estimation.

**Identifiability refusals.** None triggered here, because the graph's only latent node
(`Psychological Injury`) does not sit on a backdoor path for any tested pair. The machinery
is in place: where every blocking set requires an unobserved variable, the effect is refused
and the offending path named, rather than estimated.

---

---

## 3b. Every edge of the DAG now carries an effect

The graph is no longer only a claim about direction. Each edge is a
(treatment, outcome) pair, so each has its own minimal backdoor set and its own estimate
([`stage15 --edges`](../causal/stage15_dag_effects.py)).

**33 of 34 edges estimated. 25 have a 95% interval excluding zero. One refused.**

The refusal is `Psychological Injury → Psychological Injury Emphasis`: the source is the
latent node, so no values exist to estimate from. That is the correct answer rather than a
gap, and it is the identifiability machinery of §3a doing its job on the one edge where it
bites.

### Two estimators, because one estimand does not cover the graph

`DoubleMLAPOS` contrasts discrete treatment levels and cannot treat a continuous variable.
That excludes `Claimant Age`, `Claimant Weekly Income`, `WPI %` and the two dollar columns —
between them the source of a third of the 34 edges, including both arithmetic edges into the
award. Those go through `DoubleMLPLR` instead, which gives a per-unit coefficient under the
same backdoor set.

**The magnitudes are therefore not comparable across edges**, and the map says so in its
legend and on every click:

| source | estimator | estimand |
|---|---|---|
| discrete | `DoubleMLAPOS` | top-vs-base level contrast |
| continuous | `DoubleMLPLR` | per-unit coefficient |
| exogenous (no parents) | none needed | unadjusted contrast |

`Non-Economic Loss → Lump Sum` reads **+0.021** not because the relationship is weak — it is
arithmetic — but because it is per log-dollar. Sign and significance mean the same thing on
every edge; the numbers do not.

### What the map now shows

[`ctp_reviewed_dag.html`](../causal/ctp_reviewed_dag.html) carries three encodings borrowed
from the DoubleML visualisations:

- **colour** — green where the effect is positive, red where negative
- **line style** — solid where the 95% interval excludes zero, dashed where it does not
- **the number**, on the edge

and one that is not borrowed: **grey and unlabelled where there is no estimate**. "Not
estimated" and "estimated at zero" are different claims, and a reader has to be able to tell
which one they are looking at.

Negative edges include `Liability Clarity → Legal Procedural Complexity` at **-0.566** — one
of the six elicited edges, clear fault removing a limb of argument — holding with the
interval excluding zero, and `Claimant Age → Future Economic Loss` at **-0.253**, the
retirement multiplier.

An earlier version of this section cited `WPI % → Work Impact Severity` at -0.004 as an
edge the estimate declined to support. That edge has since been removed altogether, for a
better reason than a weak number — see §3c.

---

---

## 3c. Two structural rules, neither of which needs evidence

The band constraint is one: an effect cannot precede its cause. A second was added after a
reviewer objected to `WPI % → Work Impact Severity` being in the graph at all.

**A measurement cannot cause what the measured state causes.** `WPI %` is a clinical
assessment of the permanent impairment remaining after stabilisation. The claimant's
impairment is what limits their capacity to work; the number an assessor later writes down
does not. The edge had rested on a single `reasoned_prior_path` asserting "WPI % affects
working capacity" — a model reading the assessment as the impairment, the same confusion
chronology had already caught in `WPI % → Injury Burden Intensity`.

The rule is declared in [`ctp/columns.yaml`](../ctp/columns.yaml) rather than applied by
hand, with one exception that has to be stated explicitly:

```yaml
measurement_of: the claimant's permanent impairment after stabilisation
operative_for: [Non-Economic Loss]     # s 4.11 MAIA — the recorded score is decisive
operative_form: threshold at 10% (s 4.11 MAIA)
```

`WPI % → Non-Economic Loss` survives because s 4.11 makes **the number on the certificate**
operative: no award below 10%, whatever the impairment behind it. Everywhere else the score
is a reading. Applying the rule removed two edges — the second, `WPI % → Lump Sum`, being
one stage 8 had already flagged with a unique contribution to the award of 0.07, its effect
running through the head of damage exactly as s 4.11 describes.

**And the score operates as a threshold, not a scale.** Estimating a per-percent coefficient
answers a question the scheme does not pose. The data is emphatic:

| | above 10% | at or below | p |
|---|---|---|---|
| Non-Economic Loss | 12.33 | 0.70 | 2.7×10⁻⁹³ |
| Future Economic Loss | 10.58 | 10.59 | **0.99** |

The crossing is essentially the whole of WPI's effect on non-economic loss, and does nothing
at all to future economic loss. Estimated as a threshold contrast,
`WPI % → Non-Economic Loss` is **+5.410 [+4.793, +6.027]** — a factor of roughly 220,
because below the gate the head is overwhelmingly zero. Under the continuous form that gate
is invisible.

The form is applied **only where the column causes**. As an outcome the continuous
percentage is the natural quantity — injury severity drives the assessed level, not merely
whether it clears a line — so `Injury Burden Intensity → WPI %` is estimated continuously at
**+8.479 [+6.397, +10.561]**. Each edge records which form was used.

### What this says about the two approaches

Neither rule is available to an estimator. DoubleML asked for the effect of `WPI %` on
`Work Impact Severity` returns a number; it has no concept of a variable being a measurement
of something else. Both rules come from knowing what the columns mean, which is what
[`ctp/columns.yaml`](../ctp/columns.yaml) exists to record — and both were found by a
domain reviewer looking at a drawn graph, which is what the graph is for.

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
| LightGBM | **0.410** | 519 s |
| TabPFN-3 | 0.636 | 3221 s |
| Linear | 1.053 | 74 s |

On this dataset LightGBM produced the **narrower** intervals, at one sixth the runtime.
That reverses the example's ordering — but it is not evidence that LightGBM is more
accurate. A narrow interval around a biased estimate is the worst of both. With no ground
truth, interval width measures confidence, not correctness.

Where the learners genuinely disagree is more informative than which is tightest. On
`Causation Complexity` under naive adjustment, TabPFN gives −0.290 and LightGBM −0.264
(both significant, both negative) while **Linear gives +0.397, significant and positive**.
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

**Where DoubleML was silent, and no longer is.** `DoubleMLAPOS` cannot treat `WPI %`, so
the edge with the strongest statutory backing in the project — the s 4.11 gate
`WPI % → Non-Economic Loss` — had no estimate at all: the most secure structural claim was
invisible to the method that quantifies things. Adding `DoubleMLPLR` for continuous
treatments (§3b) fixed that, and it was a limitation of how the estimator had been
attached rather than of the data.

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
