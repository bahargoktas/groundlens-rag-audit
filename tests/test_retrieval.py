import numpy as np
import pytest
from groundlens.data import Document, load_benchmark
from groundlens.retrieval import BM25, WordCharacterTFIDF, Retriever, tokenize, ranked_indices, reciprocal_rank_fusion

def test_tokenize_lowercase():
    assert tokenize('API Key! 60 days')==['api','key','60','days']

def test_bm25_empty_fail():
    with pytest.raises(ValueError): BM25([])

def test_tfidf_empty_fail():
    with pytest.raises(ValueError): WordCharacterTFIDF([])

def test_bm25_relevance():
    docs=[Document('a','Key rotation','Rotate API keys'), Document('b','Kittens','Cats sleep')]
    assert BM25(docs).score('API key rotation')[0] > BM25(docs).score('API key rotation')[1]

def test_tfidf_relevance():
    docs=[Document('a','Key rotation','Rotate API keys'),Document('b','Kittens','Cats sleep')]
    assert WordCharacterTFIDF(docs).score('API key rotation')[0] > WordCharacterTFIDF(docs).score('API key rotation')[1]

def test_tie_stability():
    assert ranked_indices(np.array([1.,1.,2.]))==[2,0,1]

def test_rrf_top():
    scores=reciprocal_rank_fusion([1,0],[1,0],2)
    assert scores[1]>scores[0]

def test_rrf_constant_validation():
    with pytest.raises(ValueError): reciprocal_rank_fusion([1,0],[0,1],2,0)

def test_superseded_filtered():
    docs,_=load_benchmark()
    retriever=Retriever(docs)
    assert len(retriever.docs)==47

def test_unknown_method():
    docs,_=load_benchmark()
    with pytest.raises(ValueError): Retriever(docs).rank('hello','foobar')

def test_rank_repeatable():
    docs,_=load_benchmark()
    r=Retriever(docs)
    assert r.rank('API keys rotation')[0]['id']==r.rank('API keys rotation')[0]['id']
