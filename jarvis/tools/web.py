"""General web research: search (Tavily → Exa → DuckDuckGo) + read/extract a page."""

from __future__ import annotations

import re

import requests

from ..config import settings
from ..utils import truncate
from . import _http

_UA = "Mozilla/5.0 (compatible; jarvis-research-agent/0.1)"


def _tavily(query: str, n: int) -> list[dict]:
    # Through the shared layer: retries, rate limiting and the disk cache (saves paid credits).
    data = _http.post(
        "https://api.tavily.com/search",
        {"query": query, "max_results": n, "search_depth": "basic"},
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
    )
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in data.get("results", [])
    ]


def _exa(query: str, n: int) -> list[dict]:
    data = _http.post(
        "https://api.exa.ai/search",
        {"query": query, "numResults": n, "contents": {"text": {"maxCharacters": 800}}},
        headers={"x-api-key": settings.exa_api_key},
    )
    return [
        {"title": r.get("title") or r.get("url", ""), "url": r.get("url", ""), "snippet": (r.get("text") or "")[:800]}
        for r in data.get("results", [])
    ]


def _ddgs(query: str, n: int) -> list[dict]:
    from ddgs import DDGS

    try:
        client = DDGS(timeout=settings.request_timeout)
    except TypeError:  # older ddgs without a timeout argument
        client = DDGS()
    with client as d:
        raw = list(d.text(query, max_results=n))
    return [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body") or r.get("snippet", "")}
        for r in raw
    ]


def web_providers() -> list[tuple[str, object]]:
    """Provider chain by available keys: Tavily -> Exa -> DuckDuckGo (always, keyless)."""
    chain = []
    if settings.tavily_api_key:
        chain.append(("tavily", _tavily))
    if settings.exa_api_key:
        chain.append(("exa", _exa))
    chain.append(("duckduckgo", _ddgs))
    return chain


def web_search(query: str, max_results: int = None) -> dict:
    """Search the open web; falls through the provider chain until one returns results."""
    n = max(1, min(int(max_results or settings.web_max_results), 10))
    errors = []
    hits: list[dict] = []
    provider = ""
    for provider, fn in web_providers():
        try:
            hits = [h for h in fn(query, n) if h.get("url")]
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            continue
        if hits:
            break
    if not hits:
        why = f" ({'; '.join(errors)})" if errors else ""
        return {"web": [], "text": f"No web results for '{query}'{why}."}
    for h in hits:
        h["source"] = provider
    lines = [f"- {h['title']} — {truncate(h['snippet'], 160)} ({h['url']})" for h in hits]
    return {"web": hits, "text": f"Web results for '{query}' (via {provider}):\n" + "\n".join(lines)}


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


def web_read(url: str, focus: str = "", max_chars: int = 5000) -> dict:
    """Fetch a web page and return its main text (the passages most relevant to `focus`)."""
    from .reader import focused_passages

    raw = _fetch_html(url)
    if not raw:
        return {"text": f"Could not fetch {url} (blocked or offline) — try another source."}
    content = _extract_text(raw, url)
    if not content:
        return {"text": f"No readable text extracted from {url}."}
    body = focused_passages(content, focus, max_chars=max_chars) if focus else truncate(content, max_chars)
    return {"text": f"Content of {url}:\n\n{body}", "content": content}
