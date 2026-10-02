"""OpenAlex — 250M+ scholarly works, free, no key (polite pool via email)."""

from __future__ import annotations

from ..config import settings
from ..utils import truncate
from . import _http

_BASE = "https://api.openalex.org/works"


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def _to_paper(item: dict) -> dict:
    pl = item.get("primary_location") or {}
    pdf = pl.get("pdf_url") or (item.get("open_access") or {}).get("oa_url")
    return {
        "source": "openalex",
        "id": (item.get("ids") or {}).get("doi") or item.get("id"),
        "title": (item.get("title") or item.get("display_name") or "").strip(),
        "authors": [
            (a.get("author") or {}).get("display_name")
            for a in (item.get("authorships") or [])
        ][:8],
        "year": item.get("publication_year"),
        "abstract": _reconstruct_abstract(item.get("abstract_inverted_index")),
        "url": (item.get("ids") or {}).get("doi") or item.get("id"),
        "pdf_url": pdf,
        "cited_by": item.get("cited_by_count"),
    }


def openalex_search(query: str, max_results: int = 5) -> dict:
    """Broad scholarly search with citation counts.

    Matches title+abstract only: the default `search` also hits full text, which
    surfaces off-topic papers that merely mention a query word somewhere.
    """
    try:
        data = _http.get(
            _BASE,
            params={
                "filter": f"title_and_abstract.search:{query.replace(',', ' ')}",
                "per_page": max(1, min(int(max_results), 25)),
                "sort": "relevance_score:desc",
                "mailto": settings.contact_email,
            },
        )
        items = data.get("results", []) or []
    except Exception as exc:
        return {"papers": [], "text": f"OpenAlex search failed: {exc}"}

    papers = [_to_paper(i) for i in items if (i.get("title") or i.get("display_name"))]
    if not papers:
        return {"papers": [], "text": f"No OpenAlex results for '{query}'."}
    lines = [
        f"- {p['title']} ({p['year']}, cited {p.get('cited_by') or 0}×) — {truncate(p['abstract'], 160)}"
        for p in papers
    ]
    return {"papers": papers, "text": f"OpenAlex results for '{query}':\n" + "\n".join(lines)}
