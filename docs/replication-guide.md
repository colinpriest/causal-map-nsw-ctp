# Replicating this on a different dataset

This is the method that emerged from building the NSW CTP graph, written after the fact
rather than before. An earlier version of this guide was written before any of it existed
and got several things wrong; where a step here is unusual, it is because something went
wrong without it.

The method suits any tabular dataset produced by a process with a **knowable chronology**
and, ideally, a **written rulebook** — insurance claims, clinical pathways, loan
origination, regulatory decisions. It is not causal discovery: nothing here recovers a graph
from correlations.

---

## The shape of it

```
what each column IS          →  what the process's ORDER is  →  what EVIDENCE exists per edge
(dictionary, stage 0)           (chronology, stages 2/9/10)     (statute, priors, elicitation)
                                                              ↓
                                              structural rules that need no evidence
                                                              ↓
                                        estimation under the graph's own adjustment sets
```

Four evidence classes, strongest first. Everything downstream reports which one an edge
rests on, because they are not interchangeable:

| class | what it is |
|---|---|
| `elicited` | a human who knows the domain stated it, with mechanism and verbatim quote |
| `statute` | a provision quoted from primary source names the input |
| `reasoned_prior_tested` | blind model reasoning that passed a prediction fixed **before** the data was consulted |
| `reasoned_prior_path` | a model named a mediator. Weak — and it will end up carrying most of your edges |

---

## 1. Write the definitions down, separately from everything else

Put one definition per column in a machine-readable file that **every stage reads**
([`ctp/columns.yaml`](../ctp/columns.yaml)). Not a copy per script.

A definition says **what the column measures** and nothing else. Not which phase it belongs
to — that is the conclusion you are deriving. Not what drives its level — those are the
edges you are deriving.

Three fields that turned out to matter more than expected:

- **`recorded`** — how the value was produced, which is *separate from what it is about*. A
  score assigned by a model reading a document is a fact about derivation; it does not make
  the column a measurement *of the document*. Conflating these silently dragged three
  injury columns to the wrong phase.
- **`measurement_of`** — set when the column is a *score of a state* rather than the state.
  This has hard structural consequences (§3).
- **`operative_form`** — set when only part of the column's range does causal work. A
  statutory threshold at 10% means the crossing is the causal quantity, and estimating a
  per-unit coefficient answers a question the rulebook does not pose.

> **Why this file is the highest-leverage thing in the project.** Three separate failures
> traced to a stage holding its own private copy of a definition and drifting. Definitions
> are also the right place to put assertions: a band assignment is a conclusion, but "what
> does this column measure" is a checkable fact anyone in the domain can dispute.

---

## 2. Establish the chronology — assert it, then derive it independently

Divide the process into ordered phases and place every column. **Every edge runs forward.**
This single constraint forbade 131 of 272 ordered pairs here and did more orientation work
than every statistical signal combined.

Then **derive the same ordering blind** and compare. Show a model only the definitions —
never your phase table, never its size — and let it propose its own phases. Agreement is
corroboration; disagreement is a question about a definition.

That check reached ρ ≈ 0.91–0.95 against the hand-written bands, and each disagreement was
a genuinely ambiguous construct rather than a model error.

**Do not try to derive the chronology from data.** It was tried here from dated event logs
and largely failed: only 4 of 23 event types had an evidenced link to a column, covering 2
of 9 variables. An event log records when something was *done*, not when the state arose.

---

## 3. Add the structural rules that need no evidence

Two, and both caught real errors that no statistical signal could:

**Chronology.** An effect cannot precede its cause. This caught a reasoned prior asserting
`WPI % → Injury Burden Intensity` at 100% agreement across independent samples — backwards,
because injury burden is a fact of the crash and WPI is a later assessment of it. A
backwards edge is *reversed*, not discarded: the pair is related, only the direction was
refuted.

**Measurement semantics.** A score cannot cause what the state it scores causes. The
claimant's impairment limits their capacity to work; the number an assessor later writes
down does not. Declared via `measurement_of`, with an explicit exception list for the cases
where a rule makes the *recorded score itself* operative — a statutory threshold really does
turn on the number on the certificate.

Both rules were found by a domain reviewer looking at a drawn graph, which is the argument
for drawing it early and showing it to someone.

---

## 4. Mine the rulebook, if there is one

Where the process has written rules, they are the best non-human evidence available. The
pattern that worked:

1. **Find which provisions the corpus actually cites** — decisions here recorded them
   per case, 341 distinct.
2. **Retrieve the provision text from primary source**, identified **by content, not
   filename**. Two of six identifiers constructed by hand resolved to entirely the wrong
   instrument; the content guard caught both.
3. **Read each provision blind** — one provision, no correlations, no other provisions.
4. **Require a verbatim quote** for every claimed link, checked programmatically against
   the source. Allow elided quotes (`...`) with each fragment verbatim and in order — legal
   drafting is quoted that way and a strict substring test rejects honest quotes.
5. **Map concepts semantically.** Legislation will never use your column names. "Impairment
   of earning capacity" is the concept your work-capacity column measures. Demanding the
   column name appear tests vocabulary, not law.
6. **Run an adversarial pass** on every proposed link, with the grounds for refusal
   *enumerated* — otherwise the challenger invents stricter ones and refutes true claims.

> **Do not filter provisions by citation frequency.** Citation counts what is *disputed*,
> not what governs. The provision capping loss-of-earnings damages was cited 5 times in 540
> decisions because nobody argues about it; the evidentiary standard beside it was cited 111
> times. A `MIN_CITES` threshold discarded exactly the load-bearing rules.

**Keep a calibration set** — provisions whose correct reading you can verify by reading
them, *including negative cases where a correct reader must find nothing*. Run it after every
prompt change. Tuning a prompt until it correctly parses a sentence is calibration; tuning
until it emits your preferred graph is fitting. The negatives are what keep those apart.

---

## 5. Elicit priors blind, with a prediction attached

For relationships no rulebook covers, ask a model — but make the answer falsifiable:

- **Argue both directions before choosing.** A model asked "which way?" produces a fluent
  case for whatever it picks.
- **Widen the answer space** beyond A→B / B→A: `indirect` (mediator named), `common_cause`
  (cause named), `measurement_artifact`, `none`. Here `indirect` was the modal verdict — and
  each one asserts *two* edges, so discarding them threw away 42 of 77 verdicts.
- **Require a testable implication from a fixed menu** the next stage can execute, chosen
  before any data is seen. Include a non-monotone option: a real mechanism can show a
  near-zero rank correlation, as career earnings rising then falling at retirement do.
- **Sample independently several times** and record agreement as a confidence signal.

Then **run the predictions in a separate scripted pass**. A prior that survives its own
pre-registered prediction is worth more than any confidence score.

---

## 6. Treat the human as an evidence class

Domain claims from someone who knows the process are the strongest evidence in the project,
and the only kind that is not machine-generated. Record them as first-class data
([`causal/elicited_edges.yaml`](../causal/elicited_edges.yaml)) with attribution, date,
mechanism and **verbatim quote** — not as prose in a docstring.

Record `domain_notes` too: statements that are not edges but constrain which edges are
admissible. Include *why* each is recorded, because the reasoning that would delete a
correct edge tends to recur.

---

## 7. Sweep every threshold, ablate every instruction

**Numeric thresholds.** Re-apply each across a range and report which ones decide their own
conclusions. Six of eleven here were fragile — the temporal-link threshold moved the finding
from "4 links" to "17" across its plausible range, and I had reported the 4 as a result.

**Prompt instructions.** Re-run with each removed. A warning that two columns shared a
scoring pass produced the same verdict for 28 of 28 pairs; without it, 0 of 28. The verdict
was the instruction. **Finding this and not acting on it is worse than not finding it** —
the compromised output stayed wired in as a downstream input for several stages after the
ablation was documented.

---

## 8. Estimate under the graph's own adjustment sets

The graph is not only a claim about direction. Every edge is an estimable quantity:

- **Minimal backdoor sets by d-separation**, not parents-of-treatment. Three covariates
  instead of nineteen here, which on 540 rows is variance and positivity.
- **Refuse when identification fails.** If every blocking set needs an unobserved variable,
  no estimator can recover the effect — and an estimator asked anyway returns a confident
  number. Refusing is a capability nothing else has.
- **Know when to adjust for nothing.** An exogenous treatment needs no adjustment and the
  raw contrast *is* the effect. Estimators error on an empty design matrix; without a graph
  you cannot know none was needed.
- **Total versus controlled direct.** The graph knows which variables mediate. Five pairs
  here reverse sign between the two, which is the whole explanation for a contested-causation
  effect that looked negative and is positive.
- **Match the estimator to the treatment.** Discrete via APOS-style contrasts, continuous via
  a partially linear model. Using only one excluded a third of the edges here, including
  every arithmetic edge into the target.

Label every estimate with its **estimand**. A level contrast and a per-unit coefficient are
not comparable, and a picture showing both must say which is which.

---

## 9. Show it, with what is missing

Draw the graph with sign, significance and the number on each edge — and **grey where there
is no estimate**, because "not estimated" and "estimated at zero" are different claims.

State the coverage: how many of the chronologically permitted pairs have any evidence. A
sparse graph drawn without that context reads as a complete one.

---

## The failure mode to watch for

Four times here, a criterion that looked like rigour was systematically discarding the most
fundamental relationships:

| criterion | what it excluded |
|---|---|
| citation-frequency threshold | the provision capping loss-of-earnings damages |
| "the text must name the variable" | the earnings input to future economic loss |
| an adversarial pass with unstated grounds | the same provision, plus contributory negligence |
| "the statute does not list it" | two live litigation channels into quantum |

The common root: a formal specification is not a causal model. A rulebook says how a
calculation *ought* to proceed; your data records what a *process actually produced*, shaped
by costs, negotiation and compromise. Anything that moves the recorded number is in scope.

Every one of these was found by a person objecting to a drawn graph — not by any check in
the pipeline. Build the picture early and put it in front of someone who knows the domain.
