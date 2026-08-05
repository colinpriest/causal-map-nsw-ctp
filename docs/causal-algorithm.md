# The causal algorithm — how cause was determined

> ## ⚠ Superseded — this describes the ORIGINAL graph, not the current one
>
> This document explains the hand-authored 49-edge DAG produced by
> [`causal/build_ctp_causal_dag.py`](../causal/build_ctp_causal_dag.py). That graph has been
> **replaced** by the evidence-based pipeline in stages 0–15, whose output is
> [`causal/provenance/banded_graph.json`](../causal/provenance/banded_graph.json) and
> [`causal/ctp_reviewed_dag.html`](../causal/ctp_reviewed_dag.html) — currently 34 edges, each
> traceable to statute, a tested prior, a domain expert's claim, or a mediator path.
>
> It is kept because the before-and-after is the point of the project: the original graph was
> LLM-generated, described as "expert-elicited", and its 49 edges rested on nothing that could be
> checked. Read this to understand **what was replaced and why**. For how the graph is built now,
> start at [`README.md`](../README.md).


> ## ⚠ Provenance: the causal claims are LLM-generated and unreviewed
>
> Every arrow, sign, strength and mechanism sentence in this graph was **written by a large language
> model** (Claude) from its training knowledge of NSW CTP scheme mechanics. **No lawyer, actuary or
> claims practitioner has reviewed any edge.** There is no elicitation session, no interview record,
> and no section-level citation to the Acts the reasoning appeals to.
>
> Earlier versions of this repository described the graph as "expert-elicited". That was wrong, and
> the wording has been corrected throughout. It matters because "expert-elicited" implies a human
> domain expert stood behind each mechanism, and none did.
>
> **What this does not undermine.** The numbers are real. Every ρ, the collider measurement, and all
> sixteen descriptive statistics quoted in the node and edge prose are computed from `ctp/ctp.csv`
> and reproduce exactly — [`causal/verify_claims.py`](../causal/verify_claims.py) checks them and
> fails the build if any drifts.
>
> **What it does undermine.** The arrows. A language model producing plausible causal structure is
> exactly the failure mode where confident, well-written, internally consistent reasoning can still
> be wrong, and no amount of correlation on 540 observational rows can catch it. Treat every edge as
> a hypothesis pending review by someone who actually practises in the scheme.

This document explains the method behind [`causal/build_ctp_causal_dag.py`](../causal/build_ctp_causal_dag.py):
what counts as evidence of causation in this project, what does not, and how each of the 49 edges in
the graph was oriented and then checked.

The short answer: **the data never orients an edge.** Direction is asserted from the statutory and
temporal mechanics of the NSW CTP scheme, as represented in a language model's training data. The
data is then used to *audit* those assertions — to show which ones the observed associations are
consistent with, which ones they contradict, and why the contradictions are expected rather than
errors. An audit that a claim survives is not a claim that has been verified.

---

## 1. Why not causal discovery?

The obvious alternative is to run a discovery algorithm (PC, GES, LiNGAM, NOTEARS) over the 540 rows
and take whatever graph falls out. That was rejected for this dataset for four reasons:

1. **Discovery recovers a Markov equivalence class, not a DAG.** PC returns a CPDAG in which many
   edges stay undirected. Those are exactly the edges whose direction matters most here.
2. **The identifying assumptions do not hold.** Causal sufficiency (no unmeasured confounders) is
   plainly false: accident forces, claimant occupation, solicitor quality, and insurer reserving
   posture are all unobserved and all common causes of multiple recorded columns.
3. **n = 540 with 47% missingness in the key column.** Conditional-independence testing on 16
   variables with only 27.6% complete rows produces unstable orientations. Run PC twice on
   different resamples and the answer changes.
4. **Much of the truth is already written down.** NSW CTP quantum is governed by statute. The
   `WPI % → Non-Economic Loss` edge is not a hypothesis about the world — it is a threshold in the
   *Motor Accident Injuries Act 2017*. Discovering it from data would be strictly worse than reading
   it. **Caveat, and it is a real one:** nobody read it. The statute was recalled by a language
   model, not consulted, and no provision is cited. The argument for preferring statute over
   discovery is sound; this repository has not yet actually executed it.

Discovery is not discarded entirely; it is scoped as a **comparison arm** — a PC-stable CPDAG is
fitted separately so that its orientation accuracy can be scored against the asserted graph. But it
is a thing being measured, not the source of the graph.

---

## 2. The orientation rules

Every edge in the graph is justified by at least one of three rules, applied in this priority order.

### Rule 1 — Temporal precedence (the band constraint)

A CTP claim proceeds through a fixed chronology. Six **bands** encode it:

```
claimant (pre-accident)
    → accident & injury
        → clinical course & capacity
            → dispute & forum
                → heads of damage
                    → award
```

**Every edge must run from a band to the same band or a later one. No edge runs backwards.**
This is a hard structural constraint enforced by construction, and it does most of the orientation
work on its own. A claimant's age cannot be caused by their treatment burden; a settlement approval
cannot cause the collision.

The bands and their rationale live in the `BANDS` list in
[`build_ctp_causal_dag.py:34`](../causal/build_ctp_causal_dag.py#L34). Each node declares its band.

### Rule 2 — Statutory mechanics

Where the scheme prescribes an arithmetic or a gate, that prescription *is* the causal mechanism and
the direction is not open to argument. Four examples carry most of the weight:

| Mechanism | Edge | Why the direction is fixed |
|---|---|---|
| **>10% WPI gate** on general damages | `WPI % → Non-Economic Loss` | Non-economic loss is unavailable below the threshold. 77% of awards at WPI ≤ 10 are exactly $0. |
| **Pre-existing condition deduction** | `Pre-existing Condition Salience → WPI %` (negative) | The proportion of impairment attributable to a prior condition is statutorily subtracted from the assessed figure. |
| **Future economic loss formula** | `Claimant Weekly Income → Future Economic Loss`, `Work Impact Severity → …`, `Claimant Age → …` (negative) | FEL = lost weekly earnings × residual-capacity discount × years to retirement. Income is the multiplicand; age sets the multiplier. |
| **Additivity of the award** | `Non-Economic Loss → Lump Sum`, `Future Economic Loss → Lump Sum` | The total is composed of its heads. This is arithmetic, not inference. |

Each such edge is tagged `strength="high"` and its `mechanism` string states the statutory basis.

### Rule 3 — Exogeneity

Some variables are facts *of the accident* rather than facts of the claim, and therefore cannot be
effects of anything downstream. `Liability Clarity` is the clearest case: how clear fault is was
settled at the moment of the collision. It is a parent of the dispute layer and of every money
column (through contributory-negligence discounting), and a child of nothing.

The mirror-image application of this rule is spotting **leaves**. `Legal Procedural Complexity`
records how heavy the proceeding was; it does not set quantum. It therefore takes many parents and
has no children — asserting a `Legal Procedural Complexity → Lump Sum` edge would confuse a
correlate of severity for a cause of it.

### What the sign and strength annotations mean

Every edge carries:

- `sign` — `+`, `-`, or `~` (context-dependent). The **direction of the direct effect**, holding the
  other parents fixed. This is a claim about the mechanism, not about the correlation.
- `strength` — `high` / `medium` / `low`. **Stated confidence in the mechanism.** Statutory
  arithmetic is `high`; "occupation mix probably differs by gender" is `low`. It is *not* an effect
  size, and — since the same process wrote the signs and computed the correlations — there is no
  record establishing that it was fixed before the data was examined. Read it as a self-assessment,
  not as a blinded prior.
- `mechanism` — a one- or two-sentence prose statement of why the edge exists. This is the actual
  evidence for the edge, and it is what a reader should attack if they disagree with the graph.

---

## 3. The empirical audit

Once the graph is asserted, the build script annotates it with what the data actually shows. Four
checks run, and all four write their results into the output JSON.

### 3.1 Association per edge

For every edge, a pairwise-complete **Spearman ρ** is computed
([`build_ctp_causal_dag.py:315`](../causal/build_ctp_causal_dag.py#L315)):

- **Spearman, not Pearson** — the dollar columns are heavy-tailed and several relationships are
  monotone but not linear. Rank correlation catches those; Pearson does not.
- **`log1p` on the four dollar columns** (`Lump Sum`, `Non-Economic Loss`, `Future Economic Loss`,
  `Claimant Weekly Income`) before correlating. Rank correlation is invariant to this, but it keeps
  the residualisation in §3.3 on a sane scale.
- **Pairwise-complete deletion**, with `n` recorded alongside every ρ. With 47% missingness in
  `WPI %` the effective sample varies enormously between edges, and a reader needs to see that.
- **Guards**: fewer than 30 complete pairs, or a constant column, returns `None` rather than a
  spurious coefficient.
- The two string columns are binarised for this purpose only
  (`Claimant Gender = Male`, `Nature = Damages`); the contrasts are recorded in
  `categorical_contrasts` so the sign of ρ is interpretable.

**These ρ values are descriptive. They do not orient anything.** They exist so a sceptical reader can
see which asserted mechanisms the data supports.

### 3.2 Sign conflicts — the interesting failures

An edge is flagged `sign_conflicts_marginal` when the asserted direct effect points one way and the
observed marginal association points the other (with |ρ| > 0.03, to ignore noise around zero).

**Seven of the 49 edges are flagged**, and every one of them is a confounding story rather than a
mistake:

| Edge | Asserted | Observed ρ | Why they disagree |
|---|---|---|---|
| `Causation Complexity → Future Economic Loss` | `−` | **+0.20** | Apportionment carves out non-accident loss (negative direct effect), but contested causation arises in exactly the severe, psychological, high-value claims. Severity confounds. |
| `Liability Clarity → Non-Economic Loss` | `+` | −0.15 | Contributory negligence discounts the award (positive direct effect of clarity), but disputed-liability matters are the ones worth fighting, and they are the big ones. |
| `Liability Clarity → Future Economic Loss` | `+` | −0.11 | Same mechanism. |
| `Liability Clarity → Lump Sum` | `+` | −0.06 | Same mechanism. |
| `Causation Complexity → Non-Economic Loss` | `−` | +0.11 | Same as the FEL edge. |
| `Pre-existing Condition Salience → WPI %` | `−` | +0.07 | The statutory deduction is unambiguously negative, but prior conditions are documented in the claims where an impairment assessment happened at all — a selection effect on top of confounding by age. |
| `Claimant Age → Non-Economic Loss` | `−` | +0.06 | Shorter remaining life reduces the loss, but older claimants have more degenerative pathology and higher assessed impairment. |

This is the crux of the whole exercise. **A model fitted to the correlation structure reproduces the
marginal; a model given the graph reproduces the mechanism and lets the marginal fall out of it.**
These seven edges are the sharpest available test of whether the graph is buying anything.

### 3.3 The collider test

The target is a **collider**:

```
Non-Economic Loss  →  Lump Sum  ←  Future Economic Loss
```

Both heads are driven by injury severity, so they correlate positively. Conditioning on their sum
induces a *negative* dependence — the classic explaining-away pattern. The script measures it
directly ([`build_ctp_causal_dag.py:370`](../causal/build_ctp_causal_dag.py#L370)):

1. Take the 348 rows with both heads recorded.
2. Work on `log1p(NEL)`, `log1p(FEL)`, `log(Lump Sum)`.
3. Marginal Spearman: **ρ = +0.513**.
4. Residualise both heads on `log(Lump Sum)` by least squares, then correlate the residuals:
   **ρ = −0.459**.

A swing of just over one full correlation unit. This number is recorded in the JSON under
`colliders`, and it is the primary diagnostic for anything downstream that consumes the graph.

Why it matters practically: **`Lump Sum` is column 0 of `ctp.csv`**. Any autoregressive generator
that walks the file in column order produces the collider *first* and then conditions all fifteen
remaining columns on it — baking the −0.46 artefact into the synthetic data. That is the specific
failure the graph exists to prevent.

### 3.4 CSV-order violations

Each edge records `csv_order_violation`: whether its target appears *before* its source in the file.
**34 of 49 edges are violated.** The CSV column order is close to reverse-topological — the money
columns lead, the causes trail — which is the worst possible ordering for a sequential generative
model. This is measured rather than assumed, and the count is written to `stats.n_violations`.

### 3.5 Acyclicity

Before any of the above runs, the graph is topologically sorted with **Kahn's algorithm**
([`build_ctp_causal_dag.py:292`](../causal/build_ctp_causal_dag.py#L292)), ties broken by declaration
order for a stable, reproducible ordering. A cycle raises `ValueError`. Every edge is then asserted
to run forward in that order:

```python
for src, dst, *_ in EDGES:
    assert rank[src] < rank[dst], f"back-edge {src} -> {dst}"
```

The build **fails loudly** rather than emitting a graph with a cycle in it. The resulting order is
written to `topological_order` and is the generation order for anything consuming the artefact.

---

## 4. Missingness is part of the causal structure

Five columns are missing, and **none of them are missing at random**. Treating missingness as a
nuisance to impute away would destroy the most interesting structure in the dataset, so it is
modelled explicitly as an **m-graph**: for each partially observed variable `V`, a binary node
`R_V = 1` meaning "not recorded", with its own parents.

| Indicator | Target | Blank rate | Mechanism |
|---|---|---|---|
| `R_WPI` | `WPI %` | 46.9% | Recorded only when an impairment assessment was in evidence |
| `R_Income` | `Claimant Weekly Income` | 41.9% | Recorded only where earnings were in issue and proved |
| `R_FEL` | `Future Economic Loss` | 23.5% | Stated only when claimed and quantified |
| `R_NEL` | `Non-Economic Loss` | 15.4% | Stated only when the threshold question was reached |
| `R_Age` | `Claimant Age` | 8.1% | Occasionally not extractable from the decision |

Twelve edges connect them. Three deserve individual mention because they are qualitatively different
from ordinary MAR structure:

- **`WPI % → R_WPI` — self-masking (MNAR).** An assessment is *sought* when the >10% threshold is
  live. So whether WPI is recorded depends on WPI itself. This is a genuine self-loop in value space
  and cannot be handled by conditioning alone; it has to be declared as an assumption, not absorbed
  silently.
- **`R_Income → R_FEL` — a shared mechanism.** 50.4% of rows missing income are also missing future
  economic loss, against 4.1% of rows where income is present. The two blanks have a common cause
  (earnings were never in issue), and the indicator-to-indicator edge encodes it.
- **`Claimant Gender → R_Income` — an evidentiary artefact.** 47.7% blank for female claimants
  against 37.2% for male. This is a fact about how these decisions get written, and it is worth
  preserving rather than smoothing over.

Together the 16 substantive nodes and 5 indicators make a **21-node working graph**. Only 27.6% of
rows are complete, across 25 distinct missingness patterns.

---

## 5. What this method does and does not claim

**It does claim:**

- A reason is written down for every edge, in the `mechanism` field, rather than left implicit — so
  the graph can be argued with instead of merely accepted.
- The graph is acyclic, band-consistent, and machine-verified at build time.
- Every asserted edge has been checked against the data, and disagreements are reported rather than
  quietly reconciled.
- The collider at the target and the seven confounded edges are real, measured features of this
  dataset.
- Every descriptive statistic quoted in the prose reproduces from the CSV
  ([`verify_claims.py`](../causal/verify_claims.py)).

**It does not claim:**

- **That the graph is correct.** No ground truth exists, and no domain expert has reviewed it. It is
  a set of LLM-generated hypotheses with an audit trail — falsifiable, but not verified.
- **That any edge is authoritative on NSW law.** The statutory edges (the >10% WPI gate, the
  pre-existing-condition deduction, the FEL formula) appeal to the Motor Accident Injuries Act 2017
  and the Motor Accidents Compensation Act 1999 without citing a single provision. They are the
  load-bearing edges and the easiest to check — a practitioner should check them.
- **That any causal effect has been identified from data.** Nothing here estimates an ATE. The
  correlations are descriptive annotations.
- **That ρ is evidence for orientation.** A high ρ on an edge means the mechanism is at least
  visible in the data; it is not confirmation. A low or wrong-signed ρ is not refutation either —
  §3.2 exists precisely because confounding makes marginals unreliable.
- **Causal sufficiency.** Accident forces, occupation, legal representation and insurer behaviour
  are all unobserved common causes. Where an unobserved cause is known to matter, the `note` field
  on the node says so (e.g. `Injury Burden Intensity`: "the accident forces that produced it are
  unobserved — treated as exogenous noise").

Because the graph is asserted rather than discovered, **sensitivity to being wrong is itself a thing
to measure**. The intended check is a perturbed-graph arm: reverse *k* ∈ {2, 5, 10} edges (re-checking
acyclicity), rerun everything downstream, and report how far the results move. A method that only
works given a perfect hand-authored graph is not a usable method. That arm matters more here than
it would with a reviewed graph, because the authorship is weaker.

---

## 6. Reproducing the analysis

```bash
python causal/build_ctp_causal_dag.py
```

Prints the node/edge counts, the CSV-order violation count, the full topological order, and the
collider check. Writes [`causal/ctp_causal_dag.json`](../causal/ctp_causal_dag.json) — the single
source of truth consumed by both the dashboard and any downstream generator.

The build is deterministic: no sampling, no seeds, no network. Given the same `ctp/ctp.csv` it
produces byte-identical output.

---

## See also

- [`docs/dag-construction.md`](dag-construction.md) — how the artefact and dashboard are built, and
  how to add or change a node or edge.
- [`docs/replication-guide.md`](replication-guide.md) — running this method on a different dataset.
