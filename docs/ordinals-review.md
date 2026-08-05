# Review of the eight ordinal columns

Eight of the sixteen columns in `ctp.csv` are ordinal scores assigned by a language model
reading each decision. They are the columns statute cannot reach — the scheme has no
opinion about `Psychological Injury Emphasis` — so they carry most of the graph's structure.

This document records **what those columns turn out to be measuring**, which is not always
what they are named for.

---

## 1. How the review was done

The original coding rubric was never recorded anywhere in this project — only the observed
ranges. So the rubric was **recovered from the codings themselves**
([`causal/stage12_derive_rubric.py`](../causal/stage12_derive_rubric.py)).

For each column, six decisions the original coder placed at each level were sampled and
shown to a model as **unlabelled groups, in the coder's own order**. The model was told
neither the column name nor what the levels mean, and asked only what distinguishes the
groups. It is therefore describing the coder's behaviour, not rationalising a label it was
handed.

**Limits of the method, which bound everything below.** Six documents per level is a small
sample, and a model asked to distinguish groups will find *something* — if levels are hard
to separate from the text, it will latch onto whatever incidentally differs between the
sampled documents. A recovery that disagrees with a column's name is a reason to look
closer, not a conclusion.

---

## 2. What each column recovered as

| Column | What the coder appears to have responded to | Matches its name? |
|---|---|---|
| `Psychological Injury Emphasis` | "the presence and impact of psychological injuries and their role in the claims" | **yes** |
| `Liability Clarity` | "the presence and degree of contributory negligence attributed to the claimant" | **broadly** — contributory negligence is a fault concept |
| `Injury Burden Intensity` | "the type of damages being claimed or approved in the settlement" | **no** |
| `Treatment Burden` | "the type of loss or damages being claimed or approved" | **no** |
| `Work Impact Severity` | "the type of damages claimed and approved, specifically whether non-economic loss was included" | **no** |
| `Causation Complexity` | "the legislative framework under which the claims are assessed and settled" | **no** |
| `Legal Procedural Complexity` | "the complexity and severity of the injuries … particularly contributory negligence and the type of loss" | **partly** |
| `Pre-existing Condition Salience` | "the presence or absence of entitlement to non-economic loss based on whole person impairment" | **no** |

**Two of eight recover cleanly as the construct their name promises.**

The most striking pattern is that three columns — `Injury Burden Intensity`,
`Treatment Burden`, `Work Impact Severity` — all recover as variations on *what kind of
damages the settlement was for*, rather than as injury severity, treatment extent or lost
capacity. Whatever the coder was reading, it was closer to the shape of the award than to
the claimant's condition.

That is the finding worth acting on: **the names promise more than the codings deliver.**

---

## 3. Note: two columns' levels partly track which Act applied

`Causation Complexity` recovered as *"the legislative framework under which the claims are
assessed"*, and that holds against injury dates: Spearman **+0.291** with being a MACA 1999
claim (injury before 1 December 2017). `Legal Procedural Complexity` is stronger at
**+0.304**, and all 30 MACA cases sit at level 2 with no other level containing one. The
other six columns are clean.

**This is a note about coder behaviour, not a caveat on any result.** It was checked:

- Regime does not move the award — mean log award 11.998 under MACA against 12.149 under
  MAIA, **p = 0.665** — so it cannot confound anything.
- Dropping all 30 MACA cases barely shifts the associations: `Causation Complexity` with the
  award goes +0.242 → +0.294, `Legal Procedural Complexity` +0.377 → +0.415.

Thirty cases are 5.6% of the sample and do not differ on the outcome. It is recorded only
because it tells you something about what the coder was reading — a coder whose
`Causation Complexity` levels separate by procedural era was not reading difficulty of
attributing a condition to an accident.

---

## 4. Is an association between two coded columns real, or one reader's halo?

This is the question worth answering about reader-assigned columns, and it is answerable.

100 decisions were re-scored by a second, independent model
([`stage7a`](../causal/stage7a_recode_sample.py)). For each pair of columns, four
correlations were computed
([`stage7b`](../causal/stage7b_coder_reliability.py)):

```
within-coder    rho(A.injury, A.psych)   rho(B.injury, B.psych)
cross-coder     rho(A.injury, B.psych)   rho(B.injury, A.psych)
```

If two columns are related in the world, coder A's score on one predicts coder **B's** score
on the other, because both readers are tracking the same underlying case. If the
relationship is a halo — one reader forming an impression of severity and marking several
scales up together — it lives inside a single coder's columns and collapses when the coders
are crossed, because A's impression cannot reach B's scoring.

`survival = cross / within`.

| verdict | pairs |
|---|---|
| survives a change of coder | **19** |
| no association to test | 6 |
| attenuated, inconclusive | 2 |
| coder artefact | **1** |

The single artefact is `Injury Burden Intensity ~ Legal Procedural Complexity`: within-coder
0.176, cross-coder **0.025**, survival **0.14**. That association exists only inside one
reader's head, and [`stage9`](../causal/stage9_enforce_bands.py) drops any edge resting on
it.

`Injury Burden Intensity ~ Psychological Injury Emphasis` — physical injury travelling with
psychological injury — survives at **0.88**.

**Why this test survives criticism that per-column agreement scores do not.** It needs the
second reader only to be *independent*, not accurate. A halo cannot cross to another reader
however that reader was configured. If the second coder is noisier, cross-coder correlations
attenuate more than within-coder ones, so `survival` is understated — which biases toward
declaring an artefact, making a "survives" verdict conservative.

> **What this document does not contain.** An earlier version reported per-column
> agreement scores (Cohen's κ) as inter-coder *reliability*, and used them to argue that
> edges resting on `Treatment Burden` should be discounted. Those numbers have been removed.
> The second reader was given a median 4,128 characters of `Catchwords` + `Key Paragraphs`
> and never saw `Description`, `Narrative: Treatment History` or any other narrative field —
> so for `Treatment Burden` the treatment-history text was withheld from the reader asked to
> score treatment burden. Disagreement under those conditions measures the setup, not the
> column. And the columns **are** the data: the effect of `Treatment Burden` as recorded is a
> well-defined estimand whatever a second reader would have said.

---

## 5. What should be done

1. **Rename the columns, or restate their definitions**, so the name matches what is
   measured. Three recover as "the type of damages claimed", which is a different quantity
   from injury severity, treatment extent or lost capacity.
   [`ctp/columns.yaml`](../ctp/columns.yaml) is where a definition changes.
2. **Re-code with a written rubric.** The root problem is that the original instructions
   were never recorded. Everything here is reverse-engineering around that gap, and none of
   it substitutes for coding to a rubric that exists in advance.
3. **Drop the edge that fails the cross-coder test.** `Injury Burden Intensity ~ Legal
   Procedural Complexity` is a halo; stage 9 already excludes it.
4. **Both readers here are language models** — independent in context and prompt, not in
   kind. A human coding even 30 decisions would test something these methods structurally
   cannot.

---

## Reproducing

```bash
python causal/stage12_derive_rubric.py     # recover the rubric from the codings
python causal/stage7a_recode_sample.py     # re-score 100 decisions, second reader
python causal/stage7b_coder_reliability.py # cross-coder association test
```

Artefacts: [`derived_rubric.json`](../causal/provenance/derived_rubric.json),
[`recoded_sample.json`](../causal/provenance/recoded_sample.json),
[`coder_reliability.json`](../causal/provenance/coder_reliability.json). Every model call is
cached under `causal/provenance/llm_cache/` with its prompt hash, so all of it re-runs at
zero cost and can be audited rather than taken on trust.
