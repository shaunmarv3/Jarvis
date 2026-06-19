"""Local embeddings via Ollama (nomic-embed-text) + cosine helpers."""

from __future__ import annotations

import math

from .config import settings

_EMB = None


def get_embeddings():
    """Cached OllamaEmbeddings instance."""
    global _EMB
    if _EMB is None:
        from langchain_ollama import OllamaEmbeddings

        _EMB = OllamaEmbeddings(
            model=settings.embed_model, base_url=settings.ollama_base_url
        )
    return _EMB


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_by_relevance(query: str, papers: list[dict], top_k: int, min_score: float) -> list[dict]:
    """Keep the papers whose title+abstract best match `query` (cosine on embeddings).

    Degrades gracefully: if embeddings are unavailable, returns papers unchanged.
    """
    if not papers:
        return papers
    try:
        emb = get_embeddings()
        qv = emb.embed_query(query)
        docs = [
            f"{p.get('title', '')}. {(p.get('abstract') or '')[:500]}" for p in papers
        ]
        dvs = emb.embed_documents(docs)
    except Exception:
        return papers[:top_k]

    scored = []
    for p, dv in zip(papers, dvs):
        p = {**p, "_score": round(cosine(qv, dv), 3)}
        scored.append(p)
    scored.sort(key=lambda x: x["_score"], reverse=True)
    kept = [p for p in scored if p["_score"] >= min_score] or scored
    return kept[:top_k]
