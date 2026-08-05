# specs/

Working specifications live here. Their contents are **not committed** (see
[`.gitignore`](../.gitignore)) — they are internal planning documents that change faster than the
code and describe work that is not yet in this repository.

The published account of what this repository actually contains is in [`docs/`](../docs/):

| Document | Covers |
|---|---|
| [`docs/causal-algorithm.md`](../docs/causal-algorithm.md) | How causal direction was determined, and how each edge is audited against the data |
| [`docs/dag-construction.md`](../docs/dag-construction.md) | How the DAG artefact and dashboard are built |
| [`docs/replication-guide.md`](../docs/replication-guide.md) | How to run this style of analysis on a different dataset |

Source files in [`causal/`](../causal/) contain docstring references to
`specs/nsw-ctp-causative.md`. That file is the design spec for the DAG-aware TabPFN generator
that consumes the graph; it is intentionally excluded here. The parts of it that matter for
understanding the graph itself are reproduced in `docs/`.
