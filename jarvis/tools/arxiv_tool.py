"""arXiv search + exact-paper fetch (free, no key)."""

from __future__ import annotations

import re

from ..config import PAPERS_DIR, settings
from ..utils import truncate

_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def _clean_id(id_or_url: str) -> str | None:
    if not id_or_url:
        return None
    m = _ARXIV_ID.search(id_or_url)
    return m.group(1) if m else None


def _to_paper(r) -> dict:
    aid = r.get_short_id().split("v")[0]
    return {
        "source": "arxiv",
        "id": aid,
        "title": (r.title or "").strip().replace("\n", " "),
        "authors": [a.name for a in r.authors][:8],
        "year": r.published.year if r.published else None,
        "abstract": (r.summary or "").strip().replace("\n", " "),
        "url": r.entry_id,
        "pdf_url": r.pdf_url,
    }


def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Search arXiv; returns {'papers': [...], 'text': ...}."""
    try:
        import arxiv

        client = arxiv.Client(page_size=max_results, delay_seconds=1, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = [_to_paper(r) for r in client.results(search)]
    except Exception as exc:  # network / parse issues degrade gracefully
        return {"papers": [], "text": f"arXiv search failed: {exc}"}

    if not papers:
        return {"papers": [], "text": f"No arXiv results for '{query}'."}

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
        import arxiv

        client = arxiv.Client()
        r = next(client.results(arxiv.Search(id_list=[aid])))
        paper = _to_paper(r)
        if download:
            path = PAPERS_DIR / f"{aid}.pdf"
            if not path.exists():
                r.download_pdf(dirpath=str(PAPERS_DIR), filename=f"{aid}.pdf")
            paper["local_path"] = str(path)
    except Exception as exc:
        return {"papers": [], "text": f"arXiv fetch failed for {aid}: {exc}"}

    return {"papers": [paper], "text": f"Fetched arXiv:{aid} — {paper['title']}"}


# Avoid an unused-import warning for settings while keeping it importable here.
_ = settings.request_timeout
