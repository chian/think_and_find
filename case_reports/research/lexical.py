"""Local BM25/context ranking adapted from nano-graphrag extraction utilities.

Copyright (c) 2024 Gustavo Ye. MIT license: see ../../method_loop/LICENSE.
No runtime dependency on the originating repository.
"""

from collections import Counter
import math
import re


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CONTEXT_RADII = (0, 1, 4)
_CONTEXT_WEIGHTS = (3.0, 1.0, 1.0)
_RANK_FUSION_OFFSET = 60.0


def rank_spans(spans, query):
    """Rank all spans; only the episode may stop reading the ranked sequence."""
    spans = tuple(spans)
    terms = tuple(_TOKEN_RE.findall(str(query).lower()))
    if not spans or not terms:
        return spans
    tokens = [tuple(_TOKEN_RE.findall(span.text.lower())) for span in spans]
    scores = [_bm25_scores(_context_documents(tokens, radius), terms)
              for radius in _CONTEXT_RADII]
    ranks = [_positive_ranks(scale) for scale in scores]

    def order(index):
        fused = sum(weight / (_RANK_FUSION_OFFSET + scale[index])
                    for weight, scale in zip(_CONTEXT_WEIGHTS, ranks) if index in scale)
        return -fused, -scores[0][index], index

    return tuple(spans[index] for index in sorted(range(len(spans)), key=order))


def _context_documents(tokens, radius):
    return [tuple(token for neighbor in tokens[max(0, index - radius):index + radius + 1]
                  for token in neighbor) for index in range(len(tokens))]


def _bm25_scores(documents, terms):
    average_length = sum(map(len, documents)) / len(documents)
    if not average_length:
        return [0.0] * len(documents)
    frequencies = [Counter(document) for document in documents]
    document_frequency = Counter(term for document in documents for term in set(document))
    count = len(documents)

    def score(index):
        total = 0.0
        for term in terms:
            frequency = frequencies[index].get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(1.0 + (count - document_frequency[term] + 0.5)
                                         / (document_frequency[term] + 0.5))
            # Standard BM25 constants change ordering, never admission or stopping.
            saturation, length_normalization = 1.5, 0.75
            denominator = frequency + saturation * (1.0 - length_normalization
                + length_normalization * len(documents[index]) / average_length)
            total += inverse_frequency * frequency * (saturation + 1.0) / denominator
        return total

    return [score(index) for index in range(count)]


def _positive_ranks(scores):
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return {index: rank for rank, index in enumerate(ordered, 1) if scores[index] > 0.0}
