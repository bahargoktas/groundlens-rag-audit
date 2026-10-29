# GroundLens — Offline RAG Retrieval & Citation Audit

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB) ![MIT](https://img.shields.io/badge/License-MIT-green.svg)

**A fully offline evaluation framework for evidence retrieval, abstention, and source-grounded extractive answers.**

GroundLens compares BM25, word/character TF-IDF, and Reciprocal Rank Fusion (RRF) using a fictional policy-document corpus. It does **not** need API keys, GPUs, external services, real company documents or an LLM.

> **Research prototype:** The benchmark is intentionally synthetic and limited. It does not measure real-world RAG safety, semantic answer accuracy or production readiness.

## Research questions

- How often is the gold source returned at rank 1 and in the top 5?
- Does hybrid RRF improve on stand-alone lexical retrieval?
- Can a threshold chosen on a separate development set reduce unsupported citations?
- Do returned answer quotes occur verbatim in the cited source?
- How does the system handle superseded policies and unanswerable questions?

## Measured benchmark

The repository contains **55 fictional documents** (47 active, eight superseded) and **52 labelled queries** (18 development, 34 held-out test). Of the held-out questions, 24 are answerable and 10 intentionally unanswerable.

| Method | Recall@1 | Recall@5 | MRR@5 | Supported citation rate* | False citation rate** |
|---|---:|---:|---:|---:|---:|
| BM25 | 91.7% | 100.0% | 93.9% | 79.2% | 0.0% |
| TF-IDF | **95.8%** | **100.0%** | **97.9%** | 79.2% | 0.0% |
| RRF hybrid | 91.7% | 100.0% | 95.8% | 79.2% | 0.0% |

*Gold-source citation emitted on answerable test questions; **not** semantic correctness. **A false citation was not observed on these ten unanswerable questions**, which does not establish a general safety guarantee. These figures were produced in a local evaluation of this source tree, and GitHub Actions independently reproduces the experiment.

**Negative result:** Fusion did not outperform TF-IDF. Development-only threshold selection is strictly separated from the test set, and the system can abstain despite a high retrieval recall.

![Held-out benchmark](reports/retrieval_comparison.svg)

## Quick start

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
python -m groundlens evaluate --out-dir reports_local
python -m groundlens ask "How often should live API keys be rotated?"
```

Evaluation produces `metrics.json`, `audit.jsonl`, `REPORT.md`, and `retrieval_comparison.png`.

Sample extractive answer:

```json
{
  "abstained": false,
  "quote": "Production API keys are rotated every 60 days.",
  "citation_id": "kb_api_rotation"
}
```

## Architecture

```text
data/corpus.jsonl   (fictional policy documents)
        |
        +--> filter superseded documents
        +--> BM25 + TF-IDF ---> reciprocal rank fusion
        |                           |
        +--------------------------> ranked evidence
                                    |
                  separate dev threshold calibration
                                    |
                       extractive quote or abstain
                                    |
                       held-out test evaluation
                                    |
             metrics, audit trail and benchmark chart
```

## Layout

```text
.github/workflows/ci.yml     GitHub Actions (Python 3.10/3.12)
config/benchmark.json       Experiment configuration
data/corpus.jsonl           55 fictional documents
data/queries.jsonl          52 annotated questions
docs/                       Data/model cards, risks, Turkish tutorial
src/groundlens/             Retrieval, answering, evaluation and CLI
tests/                      Automated unit/integration tests
reports/                    Local benchmark summaries and preview chart
examples/                   Expected answer example
```

## Interpretation and limitations

- `Recall@k` and `MRR@k` evaluate *retrieval*, not generated-answer truthfulness.
- Gold-source citations evaluate source selection only, not semantic entailment.
- Confidence is a lexical heuristic, **not** a probability. A dev-set threshold is not certified calibration on shifted real-world data.
- An extractive sentence proves only that the cited document contained the quote.
- Document text is untrusted. A narrow regex filter rejects some instruction-like text but **is not a reliable prompt-injection defense**.
- The small synthetic corpus is lexically tidy; strong metrics should not be generalized to real document collections.

## Further reading

- [Reproducible experiment notes](docs/EXPERIMENTS.md)
- [Dataset card](docs/DATA_CARD.md)
- [Retrieval model card](docs/MODEL_CARD.md)
- [Security boundaries](docs/SECURITY.md)
- [Turkish learning guide](docs/TR_EGITIM.md)
- [Measured local benchmark report](reports/REPORT.md)

## Data provenance and license

The complete fictional corpus, queries and labels are original synthetic examples authored for this project. No real identities, patient data, external company policies, private text or third-party copyrighted corpus are used. The project code and original benchmark materials are released under the [MIT License](LICENSE). Dependencies retain their respective licenses.
