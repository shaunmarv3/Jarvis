"""Shared HTTP layer for every research tool.

* Per-host rate limiting (thread-safe), so parallel subagents don't stampede an API
  into HTTP 429 — arXiv asks for 1 request / 3 s, Semantic Scholar ~1 req/s.
* Retry with exponential backoff on 429 / 5xx, honoring Retry-After.
* On-disk response cache (CACHE_DIR), so repeated queries cost nothing and survive
  rate-limit windows.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from urllib.parse import urlparse

import requests

from ..config import CACHE_DIR, settings

UA = "jarvis-research-agent/0.3 (+https://github.com)"

# Minimum seconds between requests to the same host.
_MIN_INTERVAL = {
    "export.arxiv.org": 3.0,
    "api.semanticscholar.org": 1.1,
    "api.openalex.org": 0.15,
    "api.crossref.org": 0.1,
    "api.github.com": 2.0,
    "www.reddit.com": 2.0,
}
_DEFAULT_INTERVAL = 0.0

_locks: dict[str, threading.Lock] = {}
_last: dict[str, float] = {}
_registry_lock = threading.Lock()


class HTTPError(RuntimeError):
    pass


def _host_lock(host: str) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(host, threading.Lock())


def _wait_turn(host: str) -> None:
    interval = _MIN_INTERVAL.get(host, _DEFAULT_INTERVAL)
    if interval <= 0:
        return
    with _host_lock(host):
        wait = _last.get(host, 0.0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.monotonic()


def _cache_path(url: str, params: dict | None):
    key = json.dumps([url, sorted((params or {}).items())], default=str)
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")


def _cache_get(url: str, params: dict | None):
    if settings.tool_cache_hours <= 0:
        return None
    p = _cache_path(url, params)
    try:
        if p.exists() and time.time() - p.stat().st_mtime < settings.tool_cache_hours * 3600:
            return json.loads(p.read_text(encoding="utf-8"))["body"]
    except Exception:
        return None
    return None


def _cache_put(url: str, params: dict | None, body) -> None:
    if settings.tool_cache_hours <= 0:
        return
    try:
        _cache_path(url, params).write_text(json.dumps({"url": url, "body": body}), encoding="utf-8")
    except Exception:
        pass


def get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    as_json: bool = True,
    retries: int = 3,
    cache: bool = True,
    timeout: int | None = None,
):
    """GET with rate limiting, retries and caching. Returns parsed JSON or text.

    Raises HTTPError with an actionable message when all attempts fail.
    """
    if cache:
        hit = _cache_get(url, params)
        if hit is not None:
            return hit

    host = urlparse(url).netloc
    hdrs = {"User-Agent": UA, **(headers or {})}
    last_err = ""
    for attempt in range(retries + 1):
        _wait_turn(host)
        try:
            resp = requests.get(
                url, params=params, headers=hdrs, timeout=timeout or settings.request_timeout
            )
        except requests.RequestException as exc:
            last_err = f"network error: {exc}"
        else:
            if resp.status_code == 200:
                body = resp.json() if as_json else resp.text
                if cache:
                    _cache_put(url, params, body)
                return body
            last_err = f"HTTP {resp.status_code}"
            if resp.status_code not in (429, 500, 502, 503, 504):
                break  # client error — retrying won't help
            if attempt < retries:
                retry_after = resp.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2.0 * (2**attempt)
                time.sleep(min(delay, 20.0))
                continue
        if attempt < retries:
            time.sleep(1.5 * (2**attempt))

    hint = " (rate-limited — try again shortly or use a different source)" if "429" in last_err else ""
    raise HTTPError(f"{host}: {last_err}{hint}")
