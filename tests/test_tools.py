"""Tool layer: HTTP retry/cache/rate-limit, graceful failures, matching logic."""

import time

import pytest
import requests

import jarvis.tools._http as http
from jarvis import config
from jarvis.tools import TOOL_FUNCS, TOOL_SCHEMAS


class Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code = code
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


def test_retries_on_429_then_succeeds(monkeypatch):
    seq = [Resp(429, headers={"Retry-After": "0"}), Resp(200, {"ok": 1})]
    monkeypatch.setattr(requests, "get", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get("https://api.example.org/x") == {"ok": 1}


def test_client_error_fails_fast_with_actionable_message(monkeypatch):
    n = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: n.append(1) or Resp(404))
    with pytest.raises(http.HTTPError, match="HTTP 404"):
        http.get("https://api.example.org/x")
    assert len(n) == 1


def test_429_exhaustion_says_rate_limited(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(429))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    with pytest.raises(http.HTTPError, match="rate-limited"):
        http.get("https://api.example.org/x", retries=1)


def test_disk_cache_avoids_second_request(monkeypatch, tmp_path):
    monkeypatch.setattr(config.settings, "tool_cache_hours", 1)
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path)
    n = []
    monkeypatch.setattr(requests, "get", lambda *a, **k: n.append(1) or Resp(200, {"v": 2}))
    assert http.get("https://api.example.org/c", {"q": "a"}) == {"v": 2}
    assert http.get("https://api.example.org/c", {"q": "a"}) == {"v": 2}
    assert len(n) == 1


def test_rate_limiter_spaces_requests_per_host(monkeypatch):
    monkeypatch.setitem(http._MIN_INTERVAL, "slow.example.org", 0.15)
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(200, {}))
    t0 = time.monotonic()
    for _ in range(3):
        http.get("https://slow.example.org/x", cache=False)
    assert time.monotonic() - t0 >= 0.28  # 2 enforced gaps of 0.15s


@pytest.mark.parametrize("name", sorted(TOOL_FUNCS))
def test_every_tool_degrades_gracefully_offline(monkeypatch, name):
    def offline(*a, **k):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", offline)
    monkeypatch.setattr(requests, "post", offline)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub.HfApi, "list_datasets", offline)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_models", offline)
    import ddgs

    monkeypatch.setattr(ddgs.DDGS, "text", offline)
    args = {"read_paper": {"paper": "2401.00001"}, "read_web": {"url": "https://e.com"},
            "inspect_dataset": {"dataset": "squad"}, "fetch_arxiv": {"id_or_url": "2401.00001"},
            "resolve_title": {"title": "x"}, "resolve_doi": {"title_or_doi": "x"}}.get(name, {"query": "x"})
    out = TOOL_FUNCS[name](**args)
    assert isinstance(out, dict) and isinstance(out.get("text"), str) and out["text"]


def test_schemas_match_funcs():
    offered = {t.name for t in TOOL_SCHEMAS}
    assert offered <= set(TOOL_FUNCS)
    assert "search_reddit" not in offered  # Reddit blocks anonymous clients


def test_crossref_rejects_near_miss_titles(monkeypatch):
    from jarvis.tools import crossref

    items = [{"title": ["Is Attention All You Need?"], "DOI": "10.1/x"},
             {"title": ["Attention Is All You Need: valuation of tokens"], "DOI": "10.1/y"}]
    monkeypatch.setattr(crossref._http, "get", lambda *a, **k: {"message": {"items": items}})
    out = crossref.crossref_resolve("Attention is all you need")
    assert out["papers"] == [] and "No confident" in out["text"]
    items.append({"title": ["Attention Is All You Need"], "DOI": "10.1/z"})
    assert crossref.crossref_resolve("Attention is all you need")["papers"][0]["doi"] == "10.1/z"


def test_web_provider_chain_follows_keys(monkeypatch):
    from jarvis.tools import web

    assert [n for n, _ in web.web_providers()] == ["duckduckgo"]
    monkeypatch.setattr(config.settings, "exa_api_key", "e")
    monkeypatch.setattr(config.settings, "tavily_api_key", "t")
    assert [n for n, _ in web.web_providers()] == ["tavily", "exa", "duckduckgo"]


def test_web_search_falls_through_failing_provider(monkeypatch):
    from jarvis.tools import web

    def bad(q, n):
        raise RuntimeError("quota")

    monkeypatch.setattr(web, "web_providers", lambda: [("tavily", bad), ("duckduckgo", lambda q, n: [
        {"title": "T", "url": "https://u", "snippet": "s"}])])
    out = web.web_search("q")
    assert out["web"][0]["source"] == "duckduckgo" and "via duckduckgo" in out["text"]


def test_focused_passages_picks_relevant_chunk():
    from jarvis.tools.reader import focused_passages

    text = "Title and abstract. " * 30 + "filler words here. " * 400 + "The faithfulness metric is computed by claim verification. " * 5 + "more filler. " * 400
    out = focused_passages(text, "faithfulness metric", max_chars=2500)
    assert "faithfulness metric" in out and out.startswith("Title and abstract")
    assert len(out) <= 2600


def test_arxiv_atom_parsing():
    from jarvis.tools.arxiv_tool import _parse

    feed = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2309.15217v2</id>
    <published>2023-09-26T00:00:00Z</published><title>Ragas:
      Automated Evaluation</title><summary>We introduce Ragas.</summary>
    <author><name>Shahul Es</name></author></entry></feed>"""
    p = _parse(feed)[0]
    assert p["id"] == "2309.15217" and p["title"] == "Ragas: Automated Evaluation" and p["year"] == 2023
    assert p["pdf_url"] == "https://arxiv.org/pdf/2309.15217"
