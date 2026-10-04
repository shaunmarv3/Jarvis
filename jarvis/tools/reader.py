"""Full-text reading for subagents: open a paper (PDF) and return the passages most
relevant to a focus question, so findings come from the paper body — not just the
abstract — without flooding a small model's context window.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

import requests

from ..config import PAPERS_DIR, settings
from . import _http

_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = set(
    "the and for with that this from are was were been have has its their our into than "
    "which these those also using used use can not but such via more most other each".split()
)


def _terms(s: str) -> list[str]:
    return [w for w in _WORD.findall((s or "").lower()) if w not in _STOP]


def focused_passages(text: str, focus: str, max_chars: int = 5000, chunk: int = 1200) -> str:
    """Pick the chunks of `text` that best match `focus` (BM25-lite), in document order.

    The opening chunk (title/abstract) is always kept for orientation.
    """
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if len(text) <= max_chars:
        return text
    chunks = [text[i : i + chunk] for i in range(0, len(text), chunk)]
    q = set(_terms(focus))
    if not q:
        return text[:max_chars]

    docs = [Counter(_terms(c)) for c in chunks]
    n = len(docs)
    df = Counter(t for d in docs for t in set(d) if t in q)
    avg = sum(sum(d.values()) for d in docs) / n or 1.0

    def score(d: Counter) -> float:
        length = sum(d.values()) or 1
        s = 0.0
        for t in q:
            f = d.get(t, 0)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * length / avg))
        return s

    ranked = sorted(range(1, n), key=lambda i: score(docs[i]), reverse=True)
    keep, used = {0}, len(chunks[0])
    for i in ranked:
        if used + len(chunks[i]) > max_chars:
            break
        if score(docs[i]) <= 0:
            break
        keep.add(i)
        used += len(chunks[i])
    return "\n[…]\n".join(chunks[i].strip() for i in sorted(keep))


def _pdf_text(path: str) -> str:
    cache = PAPERS_DIR / (re.sub(r"[^A-Za-z0-9._-]+", "_", path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]) + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    import pymupdf

    with pymupdf.open(path) as doc:
        text = "\n".join(page.get_text() for page in doc)
    try:
        cache.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text


def is_pdf(path) -> bool:
    """True if the file starts like a PDF (a paywall or error page saved as .pdf does not)."""
    try:
        with open(path, "rb") as f:
            return b"%PDF" in f.read(1024)
    except OSError:
        return False


def save_pdf(url: str, path) -> bool:
    """Download `url` to `path` only if the response really is a PDF.

    An existing valid file is reused; an existing non-PDF file (saved by older versions)
    is replaced. Returns False for HTML / paywall responses. Network errors raise.
    """
    if os.path.exists(path):
        if is_pdf(path):
            return True
        os.remove(path)
    resp = requests.get(url, timeout=settings.request_timeout * 2, headers={"User-Agent": _http.UA})
    resp.raise_for_status()
    if b"%PDF" not in resp.content[:1024]:
        return False
    with open(path, "wb") as f:
        f.write(resp.content)
    return True


def _download_pdf(url: str) -> str | None:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("://", 1)[-1])[:100]
    if not name.endswith(".pdf"):
        name += ".pdf"
    path = PAPERS_DIR / name
    return str(path) if save_pdf(url, path) else None


def _s2_open_pdf(doi: str) -> tuple[str | None, str]:
    data = _http.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
        params={"fields": "title,openAccessPdf,externalIds"},
    )
    ext = data.get("externalIds") or {}
    if ext.get("ArXiv"):
        return f"https://arxiv.org/pdf/{ext['ArXiv']}", data.get("title", "")
    return (data.get("openAccessPdf") or {}).get("url") or None, data.get("title", "")


def paper_read(paper: str, focus: str = "", max_chars: int = 5000) -> dict:
    """Read a paper's full text. `paper` = arXiv id/URL, DOI, or direct PDF URL."""
    from .arxiv_tool import _clean_id

    ref = (paper or "").strip()
    title = ref
    meta: dict = {}
    try:
        aid = _clean_id(ref) if ("arxiv" in ref.lower() or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", ref)) else None
        if aid:
            url = f"https://arxiv.org/pdf/{aid}"
            from .arxiv_tool import arxiv_fetch

            found = arxiv_fetch(aid, download=False).get("papers") or []
            meta = found[0] if found else {"source": "arxiv", "id": aid, "title": f"arXiv:{aid}",
                                           "url": f"https://arxiv.org/abs/{aid}"}
            title = meta.get("title") or title
        elif ref.lower().startswith("http") and "doi.org/" not in ref.lower():
            url = ref
        else:
            doi = re.search(r"10\.\d{4,9}/\S+", ref)
            if not doi:
                return {"text": f"Can't read '{ref}': give an arXiv id, a DOI, or a PDF URL (or a source id like S3)."}
            url, title = _s2_open_pdf(doi.group(0))
            meta = {"source": "semantic_scholar", "id": doi.group(0), "doi": doi.group(0),
                    "title": title or doi.group(0), "url": f"https://doi.org/{doi.group(0)}"}
            if not url:
                return {"text": f"No open-access PDF found for {ref}. Use its abstract instead."}
        path = _download_pdf(url)
        if not path:
            return {"text": f"{url} did not return a PDF (paywalled or an HTML page — try read_web)."}
        text = _pdf_text(path)
    except Exception as exc:
        return {"text": f"Could not read paper '{ref}': {exc}"}
    if not text.strip():
        return {"text": f"PDF for '{ref}' has no extractable text."}
    body = focused_passages(text, focus or title, max_chars=max_chars)
    meta = meta or {"title": ref, "url": ref}
    return {"text": f"Full text of {title} (passages relevant to: {focus or 'overview'}):\n\n{body}",
            "content": text, "meta": meta}
