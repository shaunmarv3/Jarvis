"""Download + parse + map-reduce summarize a paper PDF.

Map-reduce (chunk -> summarize -> combine) is deliberate: local models like
qwen3.5:9b have a limited context window, so we never stuff a whole paper in
one prompt.
"""

from __future__ import annotations

import re

import requests

from ..config import PAPERS_DIR, settings
from ..llm import get_llm
from ..utils import chunk_text

_CHUNK_PROMPT = (
    "You are summarizing one section of the research paper titled '{title}'.\n"
    "Capture the key technical content (problem, method, results) concisely.\n\n"
    "SECTION:\n{chunk}\n\nCONCISE SUMMARY:"
)

_COMBINE_PROMPT = (
    "Combine these section summaries of the paper '{title}' into a clear briefing "
    "for a researcher. Use this structure:\n"
    "**Problem** · **Approach/Method** · **Key Results** · **Why it matters** · "
    "**Limitations / open questions**.\n\n"
    "SECTION SUMMARIES:\n{parts}\n\nBRIEFING:"
)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80] or "paper"


def _ensure_pdf(paper: dict) -> str | None:
    """Return a local PDF path, downloading if needed. None if unavailable."""
    if paper.get("local_path"):
        return paper["local_path"]

    # arXiv papers: use the dedicated fetcher (handles ids cleanly).
    if paper.get("source") == "arxiv" and paper.get("id"):
        from .arxiv_tool import arxiv_fetch

        res = arxiv_fetch(paper["id"], download=True)
        if res["papers"] and res["papers"][0].get("local_path"):
            return res["papers"][0]["local_path"]

    url = paper.get("pdf_url")
    if not url:
        return None
    name = _safe_name(paper.get("id") or paper.get("title") or "paper") + ".pdf"
    path = PAPERS_DIR / name
    if path.exists():
        return str(path)
    try:
        resp = requests.get(
            url,
            timeout=settings.request_timeout,
            headers={"User-Agent": "jarvis-research-agent"},
        )
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return str(path)
    except Exception:
        return None


def summarize_paper(paper: dict, backend: str | None = None, max_chunks: int = 8) -> dict:
    """Download (if needed), parse, and map-reduce summarize a paper."""
    title = paper.get("title", "(untitled)")
    path = _ensure_pdf(paper)
    if not path:
        return {
            "text": (
                f"Could not obtain a PDF for '{title}'. "
                "It may be paywalled or have no open-access copy.\n\n"
                f"Abstract on file:\n{paper.get('abstract', '(none)')}"
            )
        }

    try:
        import pymupdf4llm

        markdown = pymupdf4llm.to_markdown(path)
    except Exception as exc:
        return {"text": f"Failed to parse PDF '{title}': {exc}"}

    llm = get_llm(backend, temperature=0.2, max_tokens=1500)
    chunks = chunk_text(markdown, size=6000)[:max_chunks]
    if not chunks:
        return {"text": f"PDF for '{title}' had no extractable text."}

    parts: list[str] = []
    for ch in chunks:
        try:
            parts.append(llm.invoke(_CHUNK_PROMPT.format(title=title, chunk=ch)).content)
        except Exception as exc:
            parts.append(f"(chunk summary failed: {exc})")

    if len(parts) == 1:
        briefing = parts[0]
    else:
        try:
            briefing = llm.invoke(
                _COMBINE_PROMPT.format(title=title, parts="\n\n".join(parts))
            ).content
        except Exception as exc:
            briefing = "\n\n".join(parts) + f"\n\n(combine step failed: {exc})"

    header = f"# {title}\n" + (f"_Local file: {path}_\n\n" if path else "\n")
    return {"text": header + briefing, "local_path": path}
