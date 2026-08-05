# Review of the eight ordinal columns

Eight of the sixteen columns in `ctp.csv` are ordinal scores assigned by a language model
reading each decision. They are the columns statute cannot reach, so they carry most of the
graph's structure.

This document records **what those columns turn out to be measuring**, which is not always
what they are named for.

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

| Column | What separates the levels | Matches its name? |
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

The clearest pattern: `Injury Burden Intensity`, `Treatment Burden` and `Work Impact
Severity` all recover as variations on *what kind of damages the settlement was for*, rather
than as injury severity, treatment extent or lost capacity. Whatever the scoring pass was
reading, it was closer to the shape of the award than to the claimant's condition.

**That is the finding: the names promise more than the codings deliver.**

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

It is recorded only as further evidence about what the scoring pass was reading: levels that
separate by procedural era were not tracking difficulty of attributing a condition to an
accident.

---

## 4. What should be done

1. **Rename the columns, or restate their definitions**, so the name matches what is
   measured. Three recover as "the type of damages claimed", which is a different quantity
   from injury severity, treatment extent or lost capacity.
   [`ctp/columns.yaml`](../ctp/columns.yaml) is where a definition changes.
2. **Re-code with a written rubric.** The root problem is that the original instructions
   were never recorded. Everything here is reverse-engineering around that gap, and none of
   it substitutes for coding to a rubric that exists in advance.

---

## Reproducing

```bash
python causal/stage12_derive_rubric.py
```

Artefact: [`derived_rubric.json`](../causal/provenance/derived_rubric.json). Every model call
is cached under `causal/provenance/llm_cache/` with its prompt hash, so it re-runs at zero
cost and can be audited rather than taken on trust.
