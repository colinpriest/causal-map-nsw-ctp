# papers/

The PDFs in this directory are **not committed** (see [`.gitignore`](../.gitignore)) — they are
third-party arXiv preprints, and this repository does not redistribute them.

To restore the directory locally:

| File | Paper |
|---|---|
| `2603.10254v3.pdf` | Tugnoli, De Lorenzo, Virgolin & Cinà — *Improving TabPFN's Synthetic Data Generation by Integrating Causal Structure* — <https://arxiv.org/abs/2603.10254> |
| `2406.05216v1.pdf` | Ma, Dankar, Stein, Yu & Caterini — *TabPFGen: Tabular Data Generation with TabPFN* — <https://arxiv.org/abs/2406.05216> |

Neither paper is required to run any code in this repository. They are the methodological
background for the downstream generator that consumes
[`causal/ctp_causal_dag.json`](../causal/ctp_causal_dag.json); see
[`docs/causal-algorithm.md`](../docs/causal-algorithm.md) for how the graph is used.
