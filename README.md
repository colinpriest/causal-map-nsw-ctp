# causal-map-nsw-ctp

A pipeline that builds a causal DAG for NSW Compulsory Third Party impairment lump-sum
awards **from evidence with provenance**, and records what it could not establish.

Every edge in the output traces to something checkable: a provision quoted verbatim from
the Act, a reasoned prior that survived a prediction registered before the data was
consulted, or a domain expert's stated claim. Edges without evidence are not drawn.

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
| [`causal/ctp_reviewed_dag.html`](causal/ctp_reviewed_dag.html) | Self-contained interactive graph, with an effect estimate on 33 of 34 edges — green/red by sign, solid/dashed by whether the 95% interval excludes zero. Click an edge for its provision quote, tested mechanism, backdoor set and estimate. |
| [`causal/provenance/banded_graph.json`](causal/provenance/banded_graph.json) | The assembled graph: edges, evidence per edge, roles, violations, cycle breaks. |
| [`ctp/dictionary.md`](ctp/dictionary.md) | Data dictionary, generated from [`ctp/columns.yaml`](ctp/columns.yaml). |
| [`docs/ordinals-review.md`](docs/ordinals-review.md) | **What the eight ordinal columns actually measure** — only two recover as the construct they are named for. |
| [`causal/ctp_identification_contrast.html`](causal/ctp_identification_contrast.html) | The competing analysis: what changes when the adjustment set changes. Two treatments change sign. |
| [`docs/LLM-elicited-vs-TabPFN-causal.md`](docs/LLM-elicited-vs-TabPFN-causal.md) | **The two approaches compared.** Changing only the adjustment set flips two treatments' sign. |
| `causal/provenance/*.json` | Every intermediate result, including the LLM request/response cache. |

Current state: **17 nodes** (1 latent), **34 edges** with an effect estimate on 33 of them,
acyclic, against 141 chronologically permitted pairs. Sparse by construction — absence of an edge means no evidence was found,
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
| `stage8_variable_roles` | Driver / pass-through / recorder from unique contribution to the target | |
| `stage9_enforce_bands` | Chronology enforced, evidence assembled, cycles broken | |
| `stage10_derive_bands` | Chronology *derived* blind, as a check on the asserted one | ✓ |
| `stage11_render_dag` | The reviewable page | |
| `stage12_derive_rubric` | Recover the ordinal coding rubric from the codings themselves | ✓ |
| `stage13_tabpfn_dml` | Competing analysis: DoubleML + TabPFN-3, naive vs DAG adjustment | |
| `stage14_render_dml_map` | The identification contrast: naive vs DAG adjustment | |
| `stage15_dag_effects` | Minimal backdoor sets, identifiability refusals, an effect on every edge | |

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

**Two structural rules need no evidence.** Chronology — an effect cannot precede its cause.
And measurement semantics — a score cannot cause what the state it scores causes, so `WPI %`
may only reach `Non-Economic Loss`, where s 4.11 makes the recorded number itself operative.
Both are declared in [`ctp/columns.yaml`](ctp/columns.yaml) and enforced in stage 9.

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

**Prompt instructions are ablated.** A warning telling the model that two columns came from
the same scoring pass produced `measurement_artifact` for 28 of 28 pairs; removing it
produced 0 of 28. The verdict was the instruction, not the data. Both runs are committed,
and neither is used.

---

## What is known to be unresolved

- **Only two of eight ordinals recover as the construct they are named for.** Three recover
  as "the type of damages being claimed" rather than injury severity, treatment extent or
  lost capacity. The names promise more than the codings deliver — see
  [`docs/ordinals-review.md`](docs/ordinals-review.md).
- **The statutory leg detects gates, limits and scope; it does not find everything.**
  Five links from 102 provisions read.
- **`reasoned_prior_path` is the weakest evidence class and carries 17 of 36 edges.** It
  records that a model named a mediator, nothing more.
- **The instructions given to the ordinal scoring pass were never recorded.** Stage 12
  recovers what separates the levels, from the codings themselves — which describes the
  output, not the instructions behind it.
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
