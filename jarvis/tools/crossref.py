"""Crossref REST API — resolve fuzzy titles / DOIs to canonical metadata."""

from __future__ import annotations

import re

import requests

from ..config import settings

_WORKS = "https://api.crossref.org/works"
_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


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


def crossref_resolve(title_or_doi: str) -> dict:
    """Look up a DOI directly, or find the best work matching a title."""
    doi_match = _DOI.search(title_or_doi or "")
    try:
        if doi_match:
            resp = requests.get(
                f"{_WORKS}/{doi_match.group(0)}",
                params={"mailto": settings.contact_email},
                timeout=settings.request_timeout,
            )
            resp.raise_for_status()
            paper = _to_paper(resp.json().get("message", {}))
            return {"papers": [paper], "text": f"DOI {paper['doi']} — {paper['title']}"}

        resp = requests.get(
            _WORKS,
            params={
                "query.bibliographic": title_or_doi,
                "rows": 1,
                "mailto": settings.contact_email,
            },
            timeout=settings.request_timeout,
        )
        resp.raise_for_status()
        items = (resp.json().get("message") or {}).get("items") or []
    except Exception as exc:
        return {"papers": [], "text": f"Crossref lookup failed: {exc}"}

    if not items:
        return {"papers": [], "text": f"No Crossref match for '{title_or_doi}'."}
    paper = _to_paper(items[0])
    return {"papers": [paper], "text": f"Crossref match: {paper['title']} (DOI {paper['doi']})"}
