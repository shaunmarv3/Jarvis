"""General web research: search (ddgs) + read/extract a page (scrapling → fallbacks)."""

from __future__ import annotations

import re

import requests

from ..config import settings
from ..utils import truncate

_UA = "Mozilla/5.0 (compatible; jarvis-research-agent/0.1)"


def web_search(query: str, max_results: int = None) -> dict:
    """Search the open web via DuckDuckGo (free, no key)."""
    n = max_results or settings.web_max_results
    try:
        from ddgs import DDGS

        with DDGS() as d:
            raw = list(d.text(query, max_results=n))
    except Exception as exc:
        return {"web": [], "text": f"Web search failed: {exc}"}

    hits = []
    for r in raw:
        hits.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body") or r.get("snippet", ""),
            }
        )
    if not hits:
        return {"web": [], "text": f"No web results for '{query}'."}
    lines = [f"- {h['title']} — {truncate(h['snippet'], 160)} ({h['url']})" for h in hits]
    return {"web": hits, "text": f"Web results for '{query}':\n" + "\n".join(lines)}


def _fetch_scrapling(url: str) -> str | None:
    """Optional: Scrapling stealth fetch. Only if enabled AND Playwright present."""
    if not settings.use_scrapling:
        return None
    try:
        from scrapling.fetchers import Fetcher

        page = Fetcher.get(url, stealthy_headers=True)
        for attr in ("get_all_text", "text"):
            fn = getattr(page, attr, None)
            if callable(fn):
                txt = fn()
                if txt and len(txt) > 200:
                    return "TEXT::" + txt
        body = getattr(page, "html_content", None) or getattr(page, "body", None)
        return str(body) if body else None
    except Exception:
        return None


def _fetch_html(url: str) -> str | None:
    scr = _fetch_scrapling(url)
    if scr:
        return scr
    # Primary: trafilatura's fetcher (handles encoding/redirects well).
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return downloaded
    except Exception:
        pass
    # Fallback: plain requests.
    try:
        resp = requests.get(url, timeout=settings.request_timeout, headers={"User-Agent": _UA})
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _extract_text(raw: str, url: str) -> str:
    if raw.startswith("TEXT::"):
        return raw[6:]
    # trafilatura gives clean main-content text (purpose-built for this).
    try:
        import trafilatura

        extracted = trafilatura.extract(raw, url=url, include_comments=False)
        if extracted:
            return extracted
    except Exception:
        pass
    # crude tag strip as last resort.
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def web_read(url: str, max_chars: int = 6000) -> dict:
    """Fetch a web page and return its main text content."""
    raw = _fetch_html(url)
    if not raw:
        return {"text": f"Could not fetch {url}."}
    content = _extract_text(raw, url)
    if not content:
        return {"text": f"No readable text extracted from {url}."}
    return {"text": f"Content of {url}:\n\n{truncate(content, max_chars)}", "content": content}
