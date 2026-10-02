"""Crossref REST API — resolve fuzzy titles / DOIs to canonical metadata."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..config import settings
from . import _http

_WORKS = "https://api.crossref.org/works"
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_MIN_TITLE_SIMILARITY = 0.93  # "Is Attention All You Need?" scores 0.88 vs the real title


def _to_paper(item: dict) -> dict:
    title = (item.get("title") or [""])[0]
    authors = [
        " ".join(filter(None, [a.get("given"), a.get("family")]))
        for a in (item.get("author") or [])
    ][:8]
    year = None
    issued = (item.get("issued") or {}).get("date-parts") or [[None]]
    if issued and issued[0]:
        year = issued[0][0]
    return {
        "source": "crossref",
        "id": item.get("DOI"),
        "doi": item.get("DOI"),
        "title": title.strip(),
        "authors": authors,
        "year": year,
        "abstract": re.sub(r"<[^>]+>", "", item.get("abstract", "")).strip(),
        "url": item.get("URL"),
        "pdf_url": None,
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower()).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def crossref_resolve(title_or_doi: str) -> dict:
    """Look up a DOI directly, or find the work whose title best matches.

    Title lookups only return a match when the title is genuinely close —
    Crossref's top hit for "Attention is all you need" is a different paper.
    """
    doi_match = _DOI.search(title_or_doi or "")
    try:
        if doi_match:
            data = _http.get(f"{_WORKS}/{doi_match.group(0)}", params={"mailto": settings.contact_email})
            paper = _to_paper(data.get("message", {}))
            return {"papers": [paper], "text": f"DOI {paper['doi']} — {paper['title']}"}

        data = _http.get(
            _WORKS,
            params={"query.bibliographic": title_or_doi, "rows": 5, "mailto": settings.contact_email},
        )
        items = (data.get("message") or {}).get("items") or []
    except Exception as exc:
        return {"papers": [], "text": f"Crossref lookup failed: {exc}"}

    candidates = [_to_paper(i) for i in items if i.get("title")]
    if not candidates:
        return {"papers": [], "text": f"No Crossref match for '{title_or_doi}'."}
    best = max(candidates, key=lambda p: title_similarity(p["title"], title_or_doi))
    if title_similarity(best["title"], title_or_doi) < _MIN_TITLE_SIMILARITY:
        near = "; ".join(f"{p['title']} ({p['year']})" for p in candidates[:3])
        return {"papers": [], "text": f"No confident Crossref match for '{title_or_doi}'. Closest: {near}"}
    return {"papers": [best], "text": f"Crossref match: {best['title']} (DOI {best['doi']})"}
