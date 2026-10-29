"""Extractive, source-labeled responses with explicitly limited safety heuristics."""
from __future__ import annotations
import re
from .retrieval import tokenize

# Demonstrative *heuristic*, NOT a prompt injection security boundary.
INSTRUCTION_RE = re.compile(r'ignore (all )?(previous|prior) instructions|reveal your system prompt|override safety rules', re.I)


def has_instruction_like_text(text: str) -> bool:
    return bool(INSTRUCTION_RE.search(text))


def extract_quote(body: str, question: str) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', body.strip()) if s.strip()]
    query_words = set(tokenize(question)) - {'a','the','is','are','be','how','much','many','what','when','which','do','does','should','can','in','and','to','of','for','on'}
    return max(sentences, key=lambda s: (len(query_words & set(tokenize(s))), -len(s)), default='')


def answer(question: str, ranked: list[dict], threshold: float) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError('threshold outside [0, 1]')
    for item in ranked:
        if has_instruction_like_text(item['body']):
            continue
        if item['confidence'] < threshold:
            continue
        quote = extract_quote(item['body'], question)
        if not quote or quote not in item['body']:
            continue
        return {'abstained': False, 'quote': quote, 'citation_id': item['id'],
                'confidence': round(item['confidence'], 6)}
    return {'abstained': True, 'quote': None, 'citation_id': None, 'confidence': None}
