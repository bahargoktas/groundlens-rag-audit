"""JSONL benchmark schema, validation and file loading."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Document:
    id: str
    title: str
    body: str
    status: str = "active"

@dataclass(frozen=True)
class Query:
    id: str
    text: str
    split: str
    evidence_ids: tuple[str, ...]


def read_jsonl(path: str | Path):
    path = Path(path)
    with path.open(encoding='utf-8') as f:
        for n, row in enumerate(f, start=1):
            if row.strip():
                try:
                    yield json.loads(row)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {n}: {path}") from exc


def load_benchmark(folder: Path = ROOT / "data") -> tuple[list[Document], list[Query]]:
    docs = [Document(**r) for r in read_jsonl(folder / 'corpus.jsonl')]
    queries = [Query(id=r['id'], text=r['text'], split=r['split'], evidence_ids=tuple(r['evidence_ids']))
               for r in read_jsonl(folder / 'queries.jsonl')]
    if not docs or not queries:
        raise ValueError('Corpus and queries must be nonempty')
    dids = [d.id for d in docs]
    qids = [q.id for q in queries]
    if len(set(dids)) != len(dids) or len(set(qids)) != len(qids):
        raise ValueError('Duplicate document or query identifiers')
    active = {d.id for d in docs if d.status == 'active'}
    if any(d.status not in {'active','superseded'} for d in docs):
        raise ValueError('Invalid document status')
    if any(not d.title or not d.body for d in docs):
        raise ValueError('Empty document content')
    for q in queries:
        if q.split not in {'dev', 'test'} or not q.text.strip():
            raise ValueError(f'Invalid query split/text: {q.id}')
        if not set(q.evidence_ids).issubset(active):
            raise ValueError(f'Invalid/non-active evidence reference: {q.id}')
    return docs, queries
