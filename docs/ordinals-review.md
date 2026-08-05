# Review of the eight ordinal columns

Eight of the sixteen columns in `ctp.csv` are ordinal scores assigned by a language model
reading each decision. They are the columns the statutory route cannot reach — the scheme
has no opinion about `Psychological Injury Emphasis` — and so they carry most of the
graph's structure. This document records what they were found to be measuring, which is
not always what they are named for.

Two of the findings are strong enough to change how the graph should be used. **Two
columns partly encode which Act applied to the claim**, and every one of the six
expert-elicited edges involves one of them.

---

## 1. How the review was done

The original coding rubric was never recorded anywhere in this project — only the observed
ranges. That was a problem in itself: it meant any re-coding exercise was testing whether
two readers guessed the same scale, not whether the scale was reliable.

So the rubric was **recovered from the codings**
([`causal/stage12_derive_rubric.py`](../causal/stage12_derive_rubric.py)). For each column,
six decisions the original coder placed at each level were sampled and shown to a model as
**unlabelled groups in the coder's own order**. The model was told neither the column name
nor what the levels mean, and asked only what distinguishes the groups. It is therefore
describing the coder's behaviour, not rationalising a label it was handed.

That recovered rubric was then fed back into the reliability study
([`stage7a`](../causal/stage7a_recode_sample.py)), so that a second independent reader
re-coded 100 decisions against what the original coder actually did.

Where the recovery suggested a column was tracking something other than its name, the
suggestion was **tested directly against the data** rather than accepted.

---

## 2. What each column recovered as

| Column | What the coder appears to have responded to | Recovers as its name? |
|---|---|---|
| `Psychological Injury Emphasis` | "the presence and impact of psychological injuries and their role in the claims" | **yes** |
| `Liability Clarity` | "the presence and degree of contributory negligence attributed to the claimant" | **broadly** — contributory negligence is a fault concept |
| `Causation Complexity` | "the legislative framework under which the claims are assessed and settled" | **no** |
| `Injury Burden Intensity` | "the type of damages being claimed or approved in the settlement" | **no** |
| `Treatment Burden` | "the type of loss or damages being claimed or approved" | **no** |
| `Work Impact Severity` | "the type of damages claimed and approved, specifically whether non-economic loss was included" | **no** |
| `Legal Procedural Complexity` | "the complexity and severity of the injuries … particularly the presence of contributory negligence and the type of loss" | **partly** |
| `Pre-existing Condition Salience` | "the presence or absence of entitlement to non-economic loss based on whole person impairment" | **no** |

Only two of eight recover cleanly as the construct their name promises. Three
(`Injury Burden Intensity`, `Treatment Burden`, `Work Impact Severity`) recover as
variations on *what kind of damages the settlement was for* rather than as injury severity,
treatment extent or lost capacity.

**Read this table with the method's limits in mind.** Six documents per level is a small
sample, and a model asked to distinguish groups will find *something* — if the levels are
genuinely hard to separate from the text, it will latch onto whatever incidentally differs
between the sampled documents. A recovery that disagrees with a column's name is a reason
to test, not a conclusion. Which is why the two most alarming ones were tested.

---

## 3. The two confirmed findings: regime leakage

`Causation Complexity` recovered as *"the legislative framework under which the claims are
assessed"* — that is, whether the claim fell under the Motor Accidents Compensation Act
1999 or the Motor Accident Injuries Act 2017, determined by whether the injury predates
1 December 2017.

Tested directly against injury dates, it holds.

### `Causation Complexity`

| level | MAIA 2017 | MACA 1999 | % MACA |
|---|---|---|---|
| 0 | 283 | 1 | 0.4% |
| 1 | 161 | 11 | 6.4% |
| 2 | 66 | 18 | **21.4%** |

Spearman(`Causation Complexity`, is-MACA) = **+0.291**, p = 5.3×10⁻¹²

### `Legal Procedural Complexity`

Worse, and structurally stranger:

| level | MAIA 2017 | MACA 1999 | % MACA |
|---|---|---|---|
| 0 | 306 | 0 | 0.0% |
| 1 | 40 | 0 | 0.0% |
| 2 | 162 | 30 | **15.6%** |
| 3 | 2 | 0 | 0.0% |

Spearman = **+0.304**, p = 4.8×10⁻¹³

**Every single MACA 1999 case sits at level 2.** Not a gradient — the entire old-regime
cohort lands in one level, and no other level contains any of it.

### The other six columns are clean

| Column | ρ with is-MACA | p |
|---|---|---|
| `Pre-existing Condition Salience` | +0.192 | 7×10⁻⁶ |
| `Psychological Injury Emphasis` | +0.133 | 0.002 |
| `Liability Clarity` | +0.072 | 0.09 |
| `Work Impact Severity` | +0.041 | 0.34 |
| `Injury Burden Intensity` | −0.017 | 0.69 |
| `Treatment Burden` | −0.043 | 0.31 |

So the leakage is specific, not general. It is concentrated in exactly the two columns the
rubric recovery flagged.

### Is this miscoding, or confounding?

Probably confounding, and it is worth being precise about why that still matters.

A claim injured before December 2017 and still being decided in 2021–2026 is an unusual
claim. It has run for years, which selects for claims that were contested — over causation,
over liability, over quantum. So old-regime cases genuinely *are* more procedurally complex
and more causally contested, on average, and a coder reading those decisions is responding
to something real.

But the consequence is the same either way: **these two columns are partly proxies for case
vintage**, and any causal estimate involving them absorbs the effect of everything else
that correlates with a claim being old and slow. That is not a defect a DAG can adjust away
unless injury date is in the graph, and it is not.

---

## 4. Consequence for the elicited edges

All six edges in [`causal/elicited_edges.yaml`](../causal/elicited_edges.yaml) involve one
of the two affected columns:

| Edge | Touches |
|---|---|
| `Liability Clarity → Legal Procedural Complexity` | LPC |
| `Pre-existing Condition Salience → Legal Procedural Complexity` | LPC |
| `Causation Complexity → Legal Procedural Complexity` | **both** |
| `Injury Burden Intensity → Causation Complexity` | CC |
| `Psychological Injury Emphasis → Causation Complexity` | CC |
| `Pre-existing Condition Salience → Causation Complexity` | CC |

This does **not** make those edges wrong. They are domain claims about how the scheme
works, and the scheme mechanics they describe are not in dispute — a prior condition to
apportion against really does generate argument and evidence.

What it means is narrower and still important: the *measured* association supporting any of
those edges is partly regime, so the edges cannot be validated against the data as cleanly
as their statistical support suggests. `Causation Complexity → Legal Procedural Complexity`
is the sharpest case, since both endpoints carry the confound and both are near-monotone in
regime.

---

## 5. Reliability, now measured against the recovered rubric

100 decisions re-coded by an independent reader
([`stage7b`](../causal/stage7b_coder_reliability.py)). Quadratic-weighted Cohen's κ:

| Column | κ | Spearman | exact |
|---|---|---|---|
| `Work Impact Severity` | **0.72** | 0.75 | 70% |
| `Psychological Injury Emphasis` | 0.59 | 0.71 | 59% |
| `Pre-existing Condition Salience` | 0.49 | 0.61 | 54% |
| `Causation Complexity` | 0.45 | 0.55 | 61% |
| `Liability Clarity` | 0.41 | 0.50 | **89%** |
| `Injury Burden Intensity` | 0.39 | 0.54 | 41% |
| `Legal Procedural Complexity` | 0.32 | 0.33 | 48% |
| `Treatment Burden` | **0.28** | 0.40 | 24% |

`Liability Clarity` is a kappa paradox, not a bad column: 89% exact agreement, but 92% of
its mass sits on one level so κ has almost no variance to reward. Read the agreement.

`Treatment Burden` at κ=0.28 with 24% exact agreement is the genuinely unreliable one — and
the recovered rubric matters here, because under the *reconstructed* rubric it scored 0.23.
The low agreement survives being measured against what the coder actually did, so it is
real unreliability rather than an artefact of my guessing the scale.

### Cross-coder association

The decisive test for whether an association between two coded columns is real: does coder
A's score on one predict coder **B's** score on the other? A relationship in the world
survives crossing the readers; a halo inside one reader's head does not.

| verdict | pairs |
|---|---|
| survives a change of coder | **19** |
| no association to test | 6 |
| attenuated, inconclusive | 2 |
| **coder artefact** | **1** |

The single artefact is `Injury Burden Intensity ~ Legal Procedural Complexity`:
within-coder 0.176, cross-coder **0.025**, survival ratio **0.14**. That association exists
only inside a single reader's head. [`stage9`](../causal/stage9_enforce_bands.py) drops any
edge resting on it.

---

## 6. What should be done

1. **Put injury date, or a regime indicator, into the analysis** if any estimate involving
   `Causation Complexity` or `Legal Procedural Complexity` is to be trusted. It is
   currently not a column in `ctp.csv` at all, though it is derivable from the raw
   workbook's `Injury Date`.
2. **Treat `Treatment Burden` as unreliable** (κ=0.28, 24% exact). Edges resting on it
   should be discounted regardless of what else supports them.
3. **Rename the columns, or restate their definitions**, so the name matches what is
   measured. Three of them recover as "the type of damages claimed", which is a different
   quantity from injury severity. [`ctp/columns.yaml`](../ctp/columns.yaml) is where a
   definition should change.
4. **Re-code with a written rubric.** The deepest problem is that the original instructions
   were never recorded. Everything in this document is reverse-engineering around that gap,
   and none of it substitutes for coding to a rubric that exists in advance.
5. **The recovery is a language model reading a language model's output.** Both readers here
   are LLMs — independent in context and prompt, not in kind. A human coding even 30
   decisions would test something these methods structurally cannot.

---

## Reproducing

```bash
python causal/stage12_derive_rubric.py     # recover the rubric from the codings
python causal/stage7a_recode_sample.py     # re-code 100 decisions, independent reader
python causal/stage7b_coder_reliability.py # reliability + cross-coder test
```

Artefacts: [`derived_rubric.json`](../causal/provenance/derived_rubric.json),
[`recoded_sample.json`](../causal/provenance/recoded_sample.json),
[`coder_reliability.json`](../causal/provenance/coder_reliability.json). Every model call is
cached under `causal/provenance/llm_cache/` with its prompt hash, so all of this re-runs at
zero cost and can be audited rather than taken on trust.
