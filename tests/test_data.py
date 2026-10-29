import json
from pathlib import Path
import pytest
from groundlens.data import Document, Query, load_benchmark, read_jsonl

def test_dataset_size():
    docs, qs = load_benchmark()
    assert len(docs) == 55 and len(qs) == 52

def test_dev_split_size():
    _, qs=load_benchmark()
    assert sum(q.split=='dev' for q in qs)==18

def test_test_split_size():
    _, qs=load_benchmark()
    assert sum(q.split=='test' for q in qs)==34

def test_doc_status_counts():
    docs,_=load_benchmark()
    assert sum(d.status=='active' for d in docs)==47

def test_evidence_is_valid():
    docs,qs=load_benchmark()
    ids={d.id for d in docs if d.status=='active'}
    assert all(set(q.evidence_ids)<=ids for q in qs)

def test_duplicates_fail(tmp_path):
    (tmp_path/'corpus.jsonl').write_text(''.join(json.dumps({'id':'same','title':'hi','body':'a','status':'active'})+'\n' for _ in range(2)))
    (tmp_path/'queries.jsonl').write_text(json.dumps({'id':'q','text':'X','split':'dev','evidence_ids':[]})+'\n')
    with pytest.raises(ValueError,match='Duplicate'):
        load_benchmark(tmp_path)

def test_invalid_gold_fail(tmp_path):
    (tmp_path/'corpus.jsonl').write_text(json.dumps({'id':'old','title':'hi','body':'a','status':'superseded'})+'\n')
    (tmp_path/'queries.jsonl').write_text(json.dumps({'id':'q','text':'X','split':'dev','evidence_ids':['old']})+'\n')
    with pytest.raises(ValueError,match='Invalid/non-active'):
        load_benchmark(tmp_path)

def test_invalid_json_fail(tmp_path):
    p=tmp_path/'bad.jsonl'
    p.write_text('!')
    with pytest.raises(ValueError,match='Invalid JSON'):
        list(read_jsonl(p))
