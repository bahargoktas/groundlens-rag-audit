# Reproducible experiments

1. Install dependencies with `python -m pip install -e ".[dev]"`.
2. Run `python -m pytest -q` to validate data, ranking, citation logic and the complete pipeline.
3. Run `python -m groundlens evaluate --out-dir reports_local`.
4. Run a stricter dev false-positive constraint: `python -m groundlens evaluate --out-dir reports_strict --max-dev-fpr 0`.
5. Compare `metrics.json` and inspect `audit.jsonl`. The threshold uses only dev labels; never tune using the held-out test labels.

A production experiment would add genuinely independent domains, semantic relevance adjudication, prompt-injection adversarial testing and confidence intervals.
