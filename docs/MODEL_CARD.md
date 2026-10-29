# Retrieval model and evaluation card

GroundLens implements word-level BM25, word/character TF-IDF and rank-based RRF. None of these is a generative LLM. Retrieval ranks active documents by lexical and subword evidence. A threshold applied to a heuristic confidence proxy is chosen using only development-set labels and evaluated on held-out test questions.

Metrics: Recall@1, Recall@5, MRR@5, nDCG@5, gold-source citation rate and false citation rate on unanswerable test questions. Exact quotes verify narrow source provenance, not logical entailment, factual accuracy or production safety.

The synthetic benchmark is not representative of real policy search, and the code must not be deployed in high-stakes applications.
