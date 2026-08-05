# Data Dictionary — `ctp`

<!-- GENERATED FILE. Edit ctp/columns.yaml, then run `python causal/build_dictionary.py`. -->

**Target:** `Lump Sum` · **Rows:** 540 · **Columns:** 16

**Source.** NSW CTP impairment lump-sum awards from Personal Injury Commission decisions, scraped from [AustLII](https://www.austlii.edu.au/) and LLM-structured.

Each definition states **what the column measures** and nothing else. None says which phase of a claim a column belongs to, and none says what drives its level — those are conclusions the analysis derives, not inputs it is given. Where a definition implies an ordering it is because the term carries it: a *pre-existing* condition is by definition one the claimant had beforehand.

`Recorded` is how the value was produced. It is separate from what the column is about: the eight ordinals were assigned by a language model reading each decision, which is a fact about their derivation and does not make them measurements *of* the decision.

## Columns

| Column | Definition | Type | Missing | Recorded |
|---|---|---|---|---|
| `Lump Sum` | the total dollars awarded to the claimant | numeric | 0.0% | read directly off the decision |
| `WPI %` | whole person impairment as a percentage: a clinical assessment of the permanent impairment remaining once the claimant's condition has stabilised | numeric | 46.9% | read directly off the decision |
| `Non-Economic Loss` | dollars awarded for pain and suffering, as a component of the payout | numeric | 15.4% | read directly off the decision |
| `Future Economic Loss` | dollars awarded for loss of future earning capacity, as a component of the payout | numeric | 23.5% | read directly off the decision |
| `Claimant Weekly Income` | the claimant's weekly earnings before the accident | numeric | 41.9% | read directly off the decision |
| `Claimant Age` | the claimant's age in years at the date of injury | numeric | 8.2% | read directly off the decision |
| `Claimant Gender` | the claimant's recorded gender | categorical | 0.0% | read directly off the decision |
| `Nature` | which procedural route the claim took: approval of an agreed settlement, or a contested damages assessment | categorical | 0.0% | read directly off the decision |
| `Injury Burden Intensity` | the overall physical severity of the injuries the claimant sustained in the accident | ordinal (0–4) | 0.0% | assigned by a language model scoring the written decision |
| `Treatment Burden` | the extent of treatment, surgery and ongoing care the claimant underwent for those injuries | ordinal (0–3) | 0.0% | assigned by a language model scoring the written decision |
| `Work Impact Severity` | how much of the claimant's capacity to work was lost | ordinal (0–3) | 0.0% | assigned by a language model scoring the written decision |
| `Legal Procedural Complexity` | how straightforward or contested the components of the payout were to determine | ordinal (0–3) | 0.0% | assigned by a language model scoring the written decision |
| `Psychological Injury Emphasis` | how much the psychological damage was discussed in the court case **Indicator of the latent `Psychological Injury`.** | ordinal (0–2) | 0.0% | assigned by a language model scoring the written decision |
| `Liability Clarity` | how obvious it is who was at fault in the accident | ordinal (0–2) | 0.0% | assigned by a language model scoring the written decision |
| `Causation Complexity` | how straightforward it is to determine how much of the claimant's current physical health, psychological health, and ability to earn an income is due to the accident rather than to other causes | ordinal (0–2) | 0.0% | assigned by a language model scoring the written decision |
| `Pre-existing Condition Salience` | the physical and psychological health conditions the claimant had before the accident | ordinal (0–2) | 0.0% | assigned by a language model scoring the written decision |

## Unobserved variables

A latent node is carried in the causal graph but has no column. It exists because a measured column is an *indicator* of something rather than the thing itself, and conflating the two forces a false choice about when the variable arises. Nothing downstream may treat a latent node as data — it has no values and constrains structure only.

### `Psychological Injury` — unobserved

the psychological injury the claimant suffered as a result of the accident -- its actual severity, as distinct from how much it was discussed

- **Measured by:** `Psychological Injury Emphasis`
- **Why latent:** No column records it. The dataset holds only how prominently the decision discussed psychological damage, which is a noisy trace of the injury filtered through what was pleaded, what evidence was led, and what the decision-maker chose to set out.

