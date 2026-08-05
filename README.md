# causal-map-nsw-ctp

A pipeline that builds a causal DAG for NSW Compulsory Third Party impairment lump-sum
awards **from evidence with provenance**, and records what it could not establish.

Every edge in the output traces to something checkable: a provision quoted verbatim from
the Act, a reasoned prior that survived a prediction registered before the data was
consulted, a measured relationship that survived a change of human coder, or a domain
expert's stated claim. Edges without evidence are not drawn.

> ### ⚠ Read this before using the graph
>
> **No practitioner has reviewed it.** The statutory edges quote primary legislation, but
> the reading of those provisions was done by a language model, and most edges rest on
> model reasoning that was tested rather than verified. Treat the output as a structured,
> auditable set of hypotheses for domain review — not as a description of how NSW CTP
> awards are determined.
>
> The project began as a hand-authored 49-edge graph that turned out to be
> LLM-generated and mislabelled "expert-elicited". Rebuilding it with provenance is the
> point of everything here.

---

## What it produces

| Artefact | |
|---|---|
| [`causal/ctp_reviewed_dag.html`](causal/ctp_reviewed_dag.html) | Self-contained interactive graph. Click an edge for its provision quote or tested mechanism; click a node for parents, children and statistics. |
| [`causal/provenance/banded_graph.json`](causal/provenance/banded_graph.json) | The assembled graph: edges, evidence per edge, roles, violations, cycle breaks. |
| [`ctp/dictionary.md`](ctp/dictionary.md) | Data dictionary, generated from [`ctp/columns.yaml`](ctp/columns.yaml). |
| [`docs/ordinals-review.md`](docs/ordinals-review.md) | **What the eight ordinal columns actually measure** — two of them partly encode which Act applied, and all six elicited edges involve one of those two. |
| [`causal/ctp_tabpfn_dml_map.html`](causal/ctp_tabpfn_dml_map.html) | The competing analysis: DoubleML + TabPFN-3 effect sizes, naive vs DAG adjustment. |
| [`docs/LLM-elicited-vs-TabPFN-causal.md`](docs/LLM-elicited-vs-TabPFN-causal.md) | **The two approaches compared.** Changing only the adjustment set flips two treatments' sign. |
| `causal/provenance/*.json` | Every intermediate result, including the LLM request/response cache. |

Current state: **17 nodes** (1 latent), **37 edges**, acyclic, against 149 chronologically
permitted pairs. Sparse by construction — absence of an edge means no evidence was found,
not that no relationship exists.

---

## Evidence classes, strongest first

| Class | What it means | Edges |
|---|---|---|
| `elicited` | A domain expert stated it, with mechanism and verbatim quote. [`causal/elicited_edges.yaml`](causal/elicited_edges.yaml) | 6 |
| `statute` | A provision names the input, quoted verbatim from primary source, and cases citing it differ on the variable | 5 |
| `measurement` | An indicator and the latent quantity it traces | 1 |
| `reasoned_prior_tested` | Blind causal reasoning that passed a prediction fixed before the data was seen | 11 |
| `reasoned_prior_path` | An `indirect` verdict naming a mediator. **Weakest — and the count overstates support**, since one edge appearing as a leg in five paths is counted five times | 43 |

Read the class mix at *edge* level, not item level: 37 edges carry 66 evidence items.

---

## The pipeline

Each stage writes a JSON artefact and is independently re-runnable. LLM calls are cached
by prompt hash and the cache is committed, so the model steps are auditable rather than
merely repeatable in principle.

| Stage | | API |
|---|---|---|
| `stage0_measurement_ledger` | How each column was produced: 8 read off decisions, 8 scored by a model | |
| `stage1_associations` | All 120 pairs — marginal, partial, signed tail dependence, materiality | |
| `stage2_temporal` | Chronology from dated event logs, with the event→variable mapping *derived*, not asserted | |
| `stage3a_provision_links` | Which variables cases differ on when a provision is cited | |
| `stage3b_fetch_statute` | Provision text from locally saved primary source, identified by content | |
| `stage3c_read_provisions` | Blind reading of each provision + adversarial challenge | ✓ |
| `stage3d_statutory_edges` | Join the statutory and empirical legs | |
| `stage5a/b_priors` | Blind causal priors with pre-registered, executable predictions | ✓ |
| `stage6_sensitivity` | Sweep every threshold; report which ones decide the answer | |
| `stage7a/b_coder` | Re-code 100 decisions with an independent reader; cross-coder test | ✓ |
| `stage8_variable_roles` | Driver / pass-through / recorder from unique contribution to the target | |
| `stage9_enforce_bands` | Chronology enforced, evidence assembled, cycles broken | |
| `stage10_derive_bands` | Chronology *derived* blind, as a check on the asserted one | ✓ |
| `stage11_render_dag` | The reviewable page | |
| `stage12_derive_rubric` | Recover the ordinal coding rubric from the codings themselves | ✓ |
| `stage13_tabpfn_dml` | Competing analysis: DoubleML + TabPFN-3, naive vs DAG adjustment | |
| `stage14_render_dml_map` | The DoubleML effect map | |

Plus `check_calibration.py`, which validates the statutory reader against
[provisions whose correct reading is known](causal/calibration_provisions.yaml) — including
negative cases where a correct reader must find nothing. **Run it after any prompt change.**

```bash
python -m pip install -r requirements.txt
python causal/stage0_measurement_ledger.py
python causal/stage1_associations.py
# ... stages are ordered; 3c, 5a, 7a and 10 need OPENAI_API_KEY in .env
python causal/check_calibration.py
python causal/stage11_render_dag.py
```

---

## Design decisions worth knowing

**Chronology is the strongest constraint.** Six bands from pre-accident to award; no edge
may run backwards. That forbids 91 of 240 ordered pairs and does more orientation work
than every statistical signal combined. It caught a reasoned prior asserting
`WPI % → Injury Burden Intensity` at 100% sample agreement — backwards, since injury
burden is a fact of the crash and WPI is a later assessment of it.

**Definitions are separated from placement.** [`ctp/columns.yaml`](ctp/columns.yaml) says
what each column measures and nothing else. Deriving the chronology from those definitions
alone reproduces the asserted bands at ρ ≈ 0.91–0.95 — an independent check, since the
model never sees the band table.

**A latent node is carried for psychological injury.** `Psychological Injury Emphasis`
measures how much the damage was *discussed*; the injury itself arose at the accident.
One column cannot sit in both places, so the latent quantity is separated from its
indicator, with a measurement edge between them.

**Thresholds are swept, not trusted.** `stage6_sensitivity` reports which parameters decide
their own conclusions. Six of eleven are fragile — those results must be quoted with their
threshold attached.

**Prompt instructions are ablated.** A warning telling the model that two columns shared a
coder produced `measurement_artifact` for 28 of 28 pairs; removing it produced 0 of 28. The
verdict was the instruction, not the data. Both runs are committed. The question was then
settled by measurement instead — see below.

---

## What is known to be unresolved

- **Two ordinals partly encode which Act applied.** `Causation Complexity` recovers as
  "the legislative framework under which claims are assessed" (ρ=+0.291 with is-MACA-1999,
  p=5e-12); `Legal Procedural Complexity` is worse at +0.304, with all 30 old-regime cases
  landing in a single level. **All six elicited edges involve one of these two columns.**
  Injury date is not in `ctp.csv` and so cannot be adjusted for. See
  [`docs/ordinals-review.md`](docs/ordinals-review.md).
- **Only two of eight ordinals recover as the construct they are named for.** Three recover
  as "the type of damages being claimed" rather than injury severity, treatment extent or
  lost capacity.
- **Reliability varies widely.** `Work Impact Severity` κ=0.72; `Treatment Burden` κ=0.28
  with 24% exact agreement, which survives being measured against the recovered rubric and
  is therefore real unreliability. 19 of 28 pairs survive a change of coder; 1 is a
  confirmed coder artefact.
- **The statutory leg detects gates, limits and scope; it does not find everything.**
  Five links from 102 provisions read.
- **`reasoned_prior_path` evidence is weak** and some path labels are incoherent, where a
  backwards leg was reversed independently of the path it came from.
- **The original rubric for the ordinal columns is not recorded anywhere**, so stage 7's
  scale descriptions are reconstructed and its disagreement mixes unreliability with
  rubric drift.
- **Two instruments block automated retrieval** (`legislation.nsw.gov.au`, `austlii.edu.au`
  — Cloudflare, and AustLII's `robots.txt` names `ClaudeBot`). Statute is read from locally
  saved files, identified by content rather than filename.

---

## Data

540 NSW Personal Injury Commission CTP decisions scraped from
[AustLII](https://www.austlii.edu.au/) and LLM-structured. No rows dropped, no imputation —
missingness is preserved because it is informative (`WPI %` 46.9%, `Claimant Weekly Income`
41.9%). The raw workbook carries per-case AustLII URLs, pinpoint statutory citations,
verbatim key paragraphs and dated event logs; the modelling table keeps 16 columns.

`papers/` and `specs/` are not committed — see the READMEs in each.
