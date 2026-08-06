# Review of the eight ordinal columns

Eight of the sixteen columns in `ctp.csv` are ordinal scores assigned by a language model
reading each decision. They are the columns statute cannot reach, so they carry most of the
graph's structure.

This document records their level distributions, **what separates the levels in the
underlying decisions**, and how each column is used downstream.

---

## 1. How the review was done

The instructions given to the original scoring pass were never recorded anywhere in this
project — only the observed ranges. So what the pass responded to was **recovered from the
codings themselves** ([`causal/stage12_derive_rubric.py`](../causal/stage12_derive_rubric.py)).

For each column, six decisions sitting at each level were sampled and shown to a model as
**unlabelled groups, in level order**. The model was told neither the column name nor what
the levels mean, and asked only what distinguishes the groups. It therefore describes what
separates the documents at each level, rather than rationalising a label it was handed.

**This does not recover the original instructions and does not claim to.** It describes the
output. If the original instructions were poor, this reproduces that faithfully.

**Limits.** Six documents per level is a small sample, and a model asked to distinguish
groups will find *something* — if levels are hard to separate from the text, it will latch
onto whatever incidentally differs between the sampled documents. A recovery that disagrees
with a column's name is a reason to look closer, not a conclusion.

---

## 2. What each column recovered as

| Column | What a blind reader said separates the levels |
|---|---|
| `Psychological Injury Emphasis` | "the presence and impact of psychological injuries and their role in the claims" |
| `Liability Clarity` | "the presence and degree of contributory negligence attributed to the claimant" |
| `Injury Burden Intensity` | "the type of damages being claimed or approved in the settlement" |
| `Treatment Burden` | "the type of loss or damages being claimed or approved" |
| `Work Impact Severity` | "the type of damages claimed and approved, specifically whether non-economic loss was included" |
| `Causation Complexity` | "the legislative framework under which the claims are assessed and settled" |
| `Legal Procedural Complexity` | "the complexity and severity of the injuries … particularly contributory negligence and the type of loss" |
| `Pre-existing Condition Salience` | "the presence or absence of entitlement to non-economic loss based on whole person impairment" |

**Read this as a description of the sampled documents, not a verdict on the columns.** Five
of the eight recover as something about the heads of damage. That is close to uninformative
here, because heads of damage are the most conspicuous thing in a CTP decision *and* they
move with severity: a claimant with a worse injury is more likely to have non-economic loss
awarded at all. A blind reader shown six decisions per level will name the visible
correlate, not the construct behind it. Recovering "type of damages" is therefore consistent
with a column that measures severity, and is not evidence against it.

What the exercise can support is narrower: the levels **do** separate the documents, in a way
a reader with no access to the column name can articulate. None of the eight came back as
noise.

---

## 3. Note: two columns' levels partly track which Act applied

`Causation Complexity` recovered as *"the legislative framework under which the claims are
assessed"*, and that holds against injury dates — Spearman **+0.291** with being a MACA 1999
claim (injury before 1 December 2017). `Legal Procedural Complexity` is stronger at
**+0.304**, with all 30 MACA cases at level 2 and no other level containing one. The other
six columns are clean.

**It does not affect any result.** Regime does not move the award (mean log award 11.998
under MACA against 12.149 under MAIA, p = 0.665), and dropping all 30 MACA cases shifts the
associations by 0.04–0.05 in the direction of strengthening them. Thirty cases are 5.6% of
the sample.

It is recorded as a fact about the coding and nothing more.

---

## 4. What should be done

**Code to a rubric that exists in advance.** The original scoring instructions were never
recorded, so nothing here can check the codings against what was asked for — only describe
what they came out as. That is a gap in the record, not a defect found in the columns, and
the only thing that closes it is writing the rubric down before the next pass rather than
inferring one afterwards.

---

## Reproducing

```bash
python causal/stage12_derive_rubric.py
```

Artefact: [`derived_rubric.json`](../causal/provenance/derived_rubric.json). Every model call
is cached under `causal/provenance/llm_cache/` with its prompt hash, so it re-runs at zero
cost and can be audited rather than taken on trust.
