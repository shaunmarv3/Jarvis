"""Local embeddings via Ollama (nomic-embed-text), used by the per-paper vector stores."""

from __future__ import annotations

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
