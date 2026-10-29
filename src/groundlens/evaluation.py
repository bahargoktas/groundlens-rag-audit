"""Strict dev-set threshold selection and held-out retrieval/abstention evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .answering import answer
from .data import load_benchmark
from .retrieval import Retriever

METHODS = ('bm25', 'tfidf', 'hybrid')


def rank_metrics(queries, ranked_cache: dict, k: int=5) -> dict:
    relevant = [q for q in queries if q.evidence_ids]
    if not relevant:
        raise ValueError('No answerable examples')
    found1 = found5 = reciprocal = ndcg = 0.0
    for q in relevant:
        ids = [d['id'] for d in ranked_cache[q.id][:k]]
        pos = next((i+1 for i, x in enumerate(ids) if x in q.evidence_ids), None)
        found1 += bool(pos == 1)
        found5 += bool(pos)
        reciprocal += (1/pos if pos else 0)
        ndcg += (1/np.log2(pos+1) if pos else 0)
    n = len(relevant)
    return {'recall@1': found1/n, 'recall@5': found5/n, 'mrr@5': reciprocal/n, 'ndcg@5': ndcg/n}


def answer_metrics(queries, ranks: dict, threshold: float) -> dict:
    good = [q for q in queries if q.evidence_ids]
    empty = [q for q in queries if not q.evidence_ids]
    audits = []
    for q in queries:
        result = answer(q.text, ranks[q.id], threshold)
        matched = result['citation_id'] in q.evidence_ids if result['citation_id'] else False
        audits.append({'id': q.id, 'split': q.split, 'question': q.text,
                       'evidence_ids': list(q.evidence_ids), **result,
                       'gold_evidence_cited': matched})
    good_ans = [a for a in audits if a['evidence_ids']]
    empty_ans = [a for a in audits if not a['evidence_ids']]
    supported = sum(a['gold_evidence_cited'] for a in good_ans)/len(good) if good else 0
    false = sum(not a['abstained'] for a in empty_ans)/len(empty) if empty else 0
    covered = [a for a in audits if not a['abstained']]
    return {'supported_citation_rate': supported, 'unanswerable_false_citation_rate': false,
            'answered_count': len(covered), 'abstained_count': len(audits)-len(covered)}, audits


def choose_threshold(queries, ranks: dict, max_dev_fpr: float=0.1) -> dict:
    if not 0 <= max_dev_fpr <= 1:
        raise ValueError('Invalid maximum dev false positive rate')
    # Candidate scores computed from dev queries only. No test label dependency.
    scores = sorted({x['confidence'] for q in queries for x in ranks[q.id]} | {0.0, 1.0})
    candidates = sorted(set([0, 1] + [min(1, x+1e-7) for x in scores] + scores))
    eligible = []
    for t in candidates:
        m, _ = answer_metrics(queries, ranks, t)
        if m['unanswerable_false_citation_rate'] <= max_dev_fpr + 1e-9:
            eligible.append((m['supported_citation_rate'], -m['unanswerable_false_citation_rate'], t))
    if not eligible:
        return {'threshold': 1.0, 'dev_supported': 0.0, 'dev_false': 0.0}
    best = max(eligible)
    m, _ = answer_metrics(queries, ranks, best[2])
    return {'threshold': best[2], 'dev_supported': m['supported_citation_rate'],
            'dev_false': m['unanswerable_false_citation_rate']}


def evaluate(out_dir: str | Path, data_dir=None, max_dev_fpr: float=0.1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    docs, queries = load_benchmark() if data_dir is None else load_benchmark(Path(data_dir))
    dev = [q for q in queries if q.split == 'dev']
    test = [q for q in queries if q.split == 'test']
    if not dev or not test:
        raise ValueError('Dev and test splits are required')
    retriever = Retriever(docs)
    result = {'dataset': {'documents': len(docs), 'active_documents': len(retriever.docs),
                          'dev_queries': len(dev), 'test_queries': len(test),
                          'test_answerable': sum(bool(q.evidence_ids) for q in test),
                          'test_unanswerable': sum(not q.evidence_ids for q in test)},
              'methods': {}}
    audits = []
    for method in METHODS:
        dev_ranks = {q.id: retriever.rank(q.text, method) for q in dev}
        cal = choose_threshold(dev, dev_ranks, max_dev_fpr)
        test_ranks = {q.id: retriever.rank(q.text, method) for q in test}
        retrieval = rank_metrics(test, test_ranks)
        ans_metrics, records = answer_metrics(test, test_ranks, cal['threshold'])
        result['methods'][method] = {**retrieval, **ans_metrics, 'dev_calibration': cal}
        for record in records:
            audits.append({'method': method, **record})
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (output/'audit.jsonl').open('w', encoding='utf-8') as f:
        for a in audits:
            f.write(json.dumps(a, ensure_ascii=False)+'\n')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [m.upper() for m in METHODS]
    x = np.arange(len(labels))
    ax.bar(x-.17, [result['methods'][m]['recall@5'] for m in METHODS], width=.34, label='Recall@5')
    ax.bar(x+.17, [result['methods'][m]['mrr@5'] for m in METHODS], width=.34, label='MRR@5')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Held-out retrieval score')
    ax.set_title('GroundLens: evidence retrieval on fictional documents')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'retrieval_comparison.png', dpi=140)
    plt.close(fig)
    lines=['# GroundLens benchmark report','',
           'Evaluation on original fictional documents. No LLM used.','',
           '| Retriever | Recall@1 | Recall@5 | MRR@5 | Supported citations | False citations on unanswerable |',
           '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        d=result['methods'][m]
        lines.append(f'| {m} | {d["recall@1"]:.3f} | {d["recall@5"]:.3f} | {d["mrr@5"]:.3f} | {d["supported_citation_rate"]:.3f} | {d["unanswerable_false_citation_rate"]:.3f} |')
    lines+=['','## Calibration', 'Thresholds are chosen from the 18 development examples only. Test labels are never used for threshold selection.','']
    for m in METHODS:
        c=result['methods'][m]['dev_calibration']
        lines.append(f'- **{m}**: threshold {c["threshold"]:.5f}; dev false citation rate {c["dev_false"]:.3f}.')
    lines+=['','## Limitations','',
            '- These policies are synthetic and not representative of a real organization.',
            '- Correct citation IDs do not guarantee semantically correct answers.',
            '- Dev calibration does not guarantee test-time false answer rate.',
            '- Regex filtering is not a prompt-injection security boundary.',
            '', '![Metrics](retrieval_comparison.png)']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
