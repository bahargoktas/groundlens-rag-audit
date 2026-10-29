import pytest
from groundlens.data import Query,load_benchmark
from groundlens.evaluation import rank_metrics, choose_threshold, evaluate
from groundlens.retrieval import Retriever

def test_recall_perfect():
    queries=[Query('q','hello','test',('a',))]
    ranks={'q':[{'id':'a'}]}
    assert rank_metrics(queries,ranks)['mrr@5']==1

def test_recall_missing():
    queries=[Query('q','hello','test',('a',))]
    ranks={'q':[{'id':'other'}]}
    assert rank_metrics(queries,ranks)['recall@5']==0

def test_recall_requires_answerable():
    with pytest.raises(ValueError):rank_metrics([Query('q','x','test',())], {'q':[]})

def test_choose_threshold_invalid_fpr():
    with pytest.raises(ValueError):choose_threshold([],{},-1)

def test_calibration_dev_only():
    docs,queries=load_benchmark()
    retriever=Retriever(docs)
    dev=[q for q in queries if q.split=='dev']
    ranks={q.id:retriever.rank(q.text) for q in dev}
    c=choose_threshold(dev,ranks)
    assert 0<=c['threshold']<=1

def test_end_to_end(tmp_path):
    result=evaluate(tmp_path)
    assert result['dataset']['test_queries']==34
    assert (tmp_path/'metrics.json').is_file()
    assert (tmp_path/'audit.jsonl').is_file()
    assert (tmp_path/'retrieval_comparison.png').is_file()
    assert (tmp_path/'REPORT.md').is_file()

def test_metrics_bounds(tmp_path):
    d=evaluate(tmp_path)
    for x in d['methods'].values():
        assert all(0<=x[k]<=1 for k in ['recall@1','recall@5','mrr@5','supported_citation_rate','unanswerable_false_citation_rate'])

def test_audit_row_count(tmp_path):
    evaluate(tmp_path)
    assert sum(1 for _ in (tmp_path/'audit.jsonl').open())==102
