"""NDCG@K computation for retrieval quality assertions."""

from __future__ import annotations

import math


def _dcg(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    total = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        grade = relevance.get(doc_id, 0.0)
        total += grade / math.log2(i + 1)
    return total


def ndcg(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Normalised DCG@K. Returns 0.0 if ideal DCG is 0."""
    actual = _dcg(ranked_ids, relevance, k)
    ideal_ids = sorted(relevance, key=lambda d: relevance[d], reverse=True)
    ideal = _dcg(ideal_ids, relevance, k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal
