"""Semantic Scholar Graph API (free, no key, ~1 req/s)."""

from __future__ import annotations

import requests

from ..config import settings
from ..utils import truncate

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,year,authors,externalIds,openAccessPdf,url"


def _to_paper(item: dict) -> dict:
    ext = item.get("externalIds") or {}
    pdf = (item.get("openAccessPdf") or {}).get("url")
    return {
        "source": "semantic_scholar",
        "id": ext.get("ArXiv") or ext.get("DOI") or item.get("paperId"),
        "title": (item.get("title") or "").strip(),
        "authors": [a.get("name") for a in (item.get("authors") or [])][:8],
        "year": item.get("year"),
        "abstract": (item.get("abstract") or "").strip(),
        "url": item.get("url"),
        "pdf_url": pdf,
        "doi": ext.get("DOI"),
    }


def s2_search(query: str, max_results: int = 5) -> dict:
    """Keyword search on Semantic Scholar."""
    try:
        resp = requests.get(
            f"{_BASE}/paper/search",
            params={"query": query, "limit": max_results, "fields": _FIELDS},
            timeout=settings.request_timeout,
            headers={"User-Agent": "jarvis-research-agent"},
        )
        resp.raise_for_status()
        items = resp.json().get("data", []) or []
    except Exception as exc:
        return {"papers": [], "text": f"Semantic Scholar search failed: {exc}"}

    papers = [_to_paper(i) for i in items]
    if not papers:
        return {"papers": [], "text": f"No Semantic Scholar results for '{query}'."}

    lines = [
        f"- {p['title']} ({p['year']}) — {truncate(p['abstract'], 180)}" for p in papers
    ]
    return {
        "papers": papers,
        "text": f"Semantic Scholar results for '{query}':\n" + "\n".join(lines),
    }


def s2_resolve_title(title: str) -> dict:
    """Resolve a (possibly fuzzy) title to the single best-matching paper."""
    try:
        resp = requests.get(
            f"{_BASE}/paper/search/match",
            params={"query": title, "fields": _FIELDS},
            timeout=settings.request_timeout,
            headers={"User-Agent": "jarvis-research-agent"},
        )
        resp.raise_for_status()
        items = resp.json().get("data", []) or []
    except Exception as exc:
        return {"papers": [], "text": f"Title resolve failed: {exc}"}

    if not items:
        return {"papers": [], "text": f"No match found for title '{title}'."}
    paper = _to_paper(items[0])
    return {"papers": [paper], "text": f"Best match: {paper['title']} ({paper['year']})"}
