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

from .. import __version__
from ..config import CACHE_DIR, settings

UA = f"jarvis-research-agent/{__version__} (+https://github.com/shaunmarv3/Jarvis)"

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


def _cache_path(url: str, params: dict | None, body: dict | None = None):
    # GET keys stay [url, params] so existing cache files remain valid; POST adds its JSON body.
    parts = [url, sorted((params or {}).items())] + ([body] if body is not None else [])
    key = json.dumps(parts, default=str, sort_keys=True)
    return CACHE_DIR / (hashlib.sha256(key.encode()).hexdigest()[:32] + ".json")


def _cache_get(url: str, params: dict | None, body: dict | None = None):
    if settings.tool_cache_hours <= 0:
        return None
    p = _cache_path(url, params, body)
    try:
        if p.exists() and time.time() - p.stat().st_mtime < settings.tool_cache_hours * 3600:
            return json.loads(p.read_text(encoding="utf-8"))["body"]
    except Exception:
        return None
    return None


def _cache_put(url: str, params: dict | None, body, json_body: dict | None = None) -> None:
    if settings.tool_cache_hours <= 0:
        return
    try:
        _cache_path(url, params, json_body).write_text(json.dumps({"url": url, "body": body}), encoding="utf-8")
    except Exception:
        pass


def _request(
    method: str,
    url: str,
    params: dict | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    as_json: bool = True,
    retries: int = 3,
    cache: bool = True,
    timeout: int | None = None,
):
    """One rate-limited, retried, cached request. Returns parsed JSON or text.

    Raises HTTPError with an actionable message when all attempts fail.
    """
    if cache:
        hit = _cache_get(url, params, json_body)
        if hit is not None:
            return hit

    host = urlparse(url).netloc
    hdrs = {"User-Agent": UA, **(headers or {})}
    send = requests.post if method == "POST" else requests.get
    last_err = ""
    for attempt in range(retries + 1):
        _wait_turn(host)
        try:
            kwargs = {"params": params, "headers": hdrs, "timeout": timeout or settings.request_timeout}
            if json_body is not None:
                kwargs["json"] = json_body
            resp = send(url, **kwargs)
        except requests.RequestException as exc:
            last_err = f"network error: {exc}"
        else:
            if resp.status_code == 200:
                body = resp.json() if as_json else resp.text
                if cache:
                    _cache_put(url, params, body, json_body)
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


def get(url: str, params: dict | None = None, headers: dict | None = None, as_json: bool = True,
        retries: int = 3, cache: bool = True, timeout: int | None = None):
    """GET with rate limiting, retries and caching."""
    return _request("GET", url, params=params, headers=headers, as_json=as_json,
                    retries=retries, cache=cache, timeout=timeout)


def post(url: str, json_body: dict, headers: dict | None = None, as_json: bool = True,
         retries: int = 3, cache: bool = True, timeout: int | None = None):
    """POST a JSON body with the same rate limiting, retries and caching (search APIs)."""
    return _request("POST", url, json_body=json_body, headers=headers, as_json=as_json,
                    retries=retries, cache=cache, timeout=timeout)
