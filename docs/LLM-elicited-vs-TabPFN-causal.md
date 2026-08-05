# Two causal analyses of the same 540 decisions

This project now contains two causal analyses of the NSW CTP data, built on different
premises and answering different questions. This document sets them against each other.

| | **LLM-elicited DAG** | **DoubleML + TabPFN** |
|---|---|---|
| Question | *what causes what* | *how large is the effect* |
| Output | 17 nodes, 36 directed edges | 10 treatment→outcome effect sizes |
| Shape | a graph | a star: every edge ends at `Lump Sum` |
| Direction comes from | chronology, statute, domain claims, tested priors | **supplied by the analyst** |
| Effect sizes | none | ATE with 95% CI, in log-dollars |
| Artefact | [`ctp_reviewed_dag.html`](../causal/ctp_reviewed_dag.html) | [`ctp_tabpfn_dml_map.html`](../causal/ctp_tabpfn_dml_map.html) |
| Build | stages 0–11 | [`stage13`](../causal/stage13_tabpfn_dml.py), [`stage14`](../causal/stage14_render_dml_map.py) |

The headline is not that one wins. It is that **each supplies exactly what the other
cannot**, and that running DoubleML without a graph produces confident, significant,
sign-wrong answers on this dataset.

---

## 1. What each method can and cannot do

**DoubleML cannot orient an edge.** Give it a treatment and an outcome and it estimates the
effect. Swap them and it estimates that too, with equal confidence and no complaint. Every
arrow in its output is a direction the analyst supplied from outside. This is not a
criticism — it is a semi-parametric estimator, not a discovery algorithm — but it means a
"DoubleML causal map" is a map only in the sense that someone already drew it.

**The elicited DAG has no effect sizes.** It says `WPI % → Non-Economic Loss` is a statutory
gate quoting s 4.11, and says nothing about how many dollars. For a question like "how much
would an award move if working capacity were one level worse", the graph is silent.

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
