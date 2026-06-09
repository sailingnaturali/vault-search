"""Reciprocal Rank Fusion + the hybrid search entry point."""

from __future__ import annotations

from vault_search.embed import Embedder
from vault_search.index import Index
from vault_search.models import SearchHit


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, rowid in enumerate(ranking, start=1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank)
    # sort by score desc, ties broken by smaller rowid for determinism
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def search(index: Index, embedder: Embedder, query: str, limit: int = 5,
           k: int = 60, pool: int = 20, mode: str = "hybrid") -> list[SearchHit]:
    """Hybrid search: BM25 + vector KNN, fused with Reciprocal Rank Fusion.

    k: RRF constant (higher = flatter rank weighting).
    pool: candidates pulled from each retriever before fusion (affects recall).
    mode: "hybrid" (both), "vector" (KNN only), or "keyword" (BM25 only).
    """
    if mode not in ("hybrid", "vector", "keyword"):
        raise ValueError(f"unknown mode: {mode}")
    rankings: list[list[int]] = []
    if mode in ("hybrid", "keyword"):
        rankings.append(index.bm25(query, n=pool))
    if mode in ("hybrid", "vector"):
        rankings.append(index.knn(embedder.encode([query])[0], n=pool))
    fused = rrf_fuse(rankings, k=k)
    return [SearchHit(chunk=index.get_chunk(rowid), score=score, retriever=mode)
            for rowid, score in fused[:limit]]
