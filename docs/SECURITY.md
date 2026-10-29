# Security boundaries

Corpus content is untrusted data. A small regex heuristic demonstrates how instruction-like strings may be identified, but it is easy to bypass and **not** a prompt-injection security boundary.

The prototype does not provide authorization, authentication, private-document isolation, secure remote ingestion, PII redaction, cross-domain evaluation or guaranteed evidence entailment. Citation fidelity is narrower than factual correctness.
