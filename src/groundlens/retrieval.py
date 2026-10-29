"""Lexical retrievers and rank fusion; no learned sentence embeddings or network calls."""
from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .data import Document

TOKEN = re.compile(r"[a-z0-9]+", re.I)

def tokenize(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


class BM25:
    def __init__(self, docs: list[Document], k1=1.5, b=.75):
        if not docs:
            raise ValueError('Empty document collection')
        self.docs = docs
        self.counts = [Counter(tokenize(d.title + ' ' + d.body)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avgdl = sum(self.lengths) / len(docs)
        self.df = Counter()
        for counts in self.counts:
            self.df.update(counts.keys())
        self.k1, self.b = k1, b

    def score(self, query: str) -> np.ndarray:
        terms = set(tokenize(query))
        n = len(self.docs)
        result = []
        for counts, length in zip(self.counts, self.lengths):
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if frequency == 0:
                    continue
                idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                score += idf * frequency * (self.k1 + 1) / (frequency + self.k1 * (1-self.b+self.b*length/self.avgdl))
            result.append(score)
        return np.asarray(result, dtype=float)


class WordCharacterTFIDF:
    def __init__(self, docs: list[Document]):
        if not docs:
            raise ValueError('Empty documents')
        texts = [d.title + ' ' + d.body for d in docs]
        self.words = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True)
        self.word_mat = self.words.fit_transform(texts)
        self.char_mat = self.chars.fit_transform(texts)

    def score(self, query: str) -> np.ndarray:
        a = (self.word_mat @ self.words.transform([query]).T).toarray().ravel()
        b = (self.char_mat @ self.chars.transform([query]).T).toarray().ravel()
        return .65 * a + .35 * b


def ranked_indices(scores: np.ndarray) -> list[int]:
    # Stable deterministic ties. Zero scores cannot be interpreted as retrieved evidence.
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def reciprocal_rank_fusion(left: list[int], right: list[int], n: int, constant: int = 60) -> np.ndarray:
    if constant <= 0:
        raise ValueError('RRF constant must be positive')
    scores = np.zeros(n, dtype=float)
    for ranks in (left, right):
        for pos, i in enumerate(ranks, start=1):
            scores[i] += 1 / (constant + pos)
    return scores


class Retriever:
    def __init__(self, docs: list[Document]):
        self.docs = [d for d in docs if d.status == 'active']
        if not self.docs:
            raise ValueError('No active documents')
        self.bm25 = BM25(self.docs)
        self.tfidf = WordCharacterTFIDF(self.docs)

    def rank(self, query: str, method: str = 'hybrid') -> list[dict]:
        bm = self.bm25.score(query)
        tf = self.tfidf.score(query)
        if method == 'bm25':
            scores = bm
        elif method == 'tfidf':
            scores = tf
        elif method == 'hybrid':
            scores = reciprocal_rank_fusion(ranked_indices(bm), ranked_indices(tf), len(self.docs))
        else:
            raise ValueError('Unknown retrieval method')
        # The same method-independent similarity proxy makes abstention comparison interpretable.
        bm_scale = max(1e-9, float(np.max(bm)))
        confidence = np.clip(.55 * tf + .45 * (1 - np.exp(-bm / bm_scale)), 0, 1)
        return [{'id': self.docs[i].id, 'title': self.docs[i].title,
                 'body': self.docs[i].body, 'score': float(scores[i]),
                 'confidence': float(confidence[i])} for i in ranked_indices(scores)]
