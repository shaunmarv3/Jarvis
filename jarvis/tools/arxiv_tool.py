"""arXiv search + exact-paper fetch (free, no key).

Talks to the Atom API directly through the shared rate-limited/cached HTTP layer
(arXiv asks for <= 1 request every 3 s; the old per-call client ignored that across
parallel subagents and got HTTP 429s).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from ..config import PAPERS_DIR
from ..utils import truncate
from . import _http

_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _clean_id(id_or_url: str) -> str | None:
    if not id_or_url:
        return None
    m = _ARXIV_ID.search(id_or_url)
    return m.group(1) if m else None


def _text(el, path: str) -> str:
    node = el.find(path, _NS)
    return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""


def _parse(feed: str) -> list[dict]:
    root = ET.fromstring(feed)
    papers = []
    for e in root.findall("a:entry", _NS):
        entry_id = _text(e, "a:id")
        aid = _clean_id(entry_id)
        if not aid:
            continue
        published = _text(e, "a:published")
        papers.append(
            {
                "source": "arxiv",
                "id": aid,
                "title": _text(e, "a:title"),
                "authors": [_text(a, "a:name") for a in e.findall("a:author", _NS)][:8],
                "year": int(published[:4]) if published[:4].isdigit() else None,
                "abstract": _text(e, "a:summary"),
                "url": f"https://arxiv.org/abs/{aid}",
                "pdf_url": f"https://arxiv.org/pdf/{aid}",
            }
        )
    return papers


def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Search arXiv; returns {'papers': [...], 'text': ...}."""
    try:
        feed = _http.get(
            _API,
            params={
                "search_query": f"all:{query}",
                "max_results": max(1, min(int(max_results), 20)),
                "sortBy": "relevance",
            },
            as_json=False,
        )
        papers = _parse(feed)
    except Exception as exc:  # network / parse issues degrade gracefully
        return {"papers": [], "text": f"arXiv search failed: {exc}"}

    if not papers:
        return {"papers": [], "text": f"No arXiv results for '{query}'. Try broader keywords."}
    lines = [
        f"- {p['title']} ({p['year']}) [arXiv:{p['id']}] — {truncate(p['abstract'], 200)}"
        for p in papers
    ]
    return {"papers": papers, "text": f"arXiv results for '{query}':\n" + "\n".join(lines)}


def arxiv_fetch(id_or_url: str, download: bool = True) -> dict:
    """Fetch a specific arXiv paper by id/url; optionally download the PDF."""
    aid = _clean_id(id_or_url)
    if not aid:
        return {"papers": [], "text": f"Could not parse an arXiv id from '{id_or_url}'."}
    try:
        papers = _parse(_http.get(_API, params={"id_list": aid}, as_json=False))
        if not papers:
            return {"papers": [], "text": f"arXiv has no paper {aid}."}
        paper = papers[0]
        if download:
            from .reader import save_pdf

            path = PAPERS_DIR / f"{aid}.pdf"
            if not save_pdf(paper["pdf_url"], path):
                return {"papers": [paper], "text": f"arXiv:{aid} metadata fetched, but its PDF link did not return a PDF."}
            paper["local_path"] = str(path)
    except Exception as exc:
        return {"papers": [], "text": f"arXiv fetch failed for {aid}: {exc}"}

    return {"papers": [paper], "text": f"Fetched arXiv:{aid} — {paper['title']}"}
