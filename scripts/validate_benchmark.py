"""Validate the deterministic fictional benchmark source data."""
from pathlib import Path
from groundlens.data import load_benchmark

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    docs, queries = load_benchmark(root / "data")
    assert len(docs) == 55 and len(queries) == 52
    assert sum(d.status == "active" for d in docs) == 47
    assert sum(q.split == "dev" for q in queries) == 18
    assert sum(q.split == "test" for q in queries) == 34
    print("Benchmark validated: 55 documents, 52 questions")
