"""Failure handling: health checks, timeouts, per-subagent resume, isolation, context fitting."""

import json
import time

import pytest
import requests
from langchain_core.messages import AIMessage

import jarvis.nodes as nodes
import jarvis.subagent as sa
from conftest import REAL_CHECK_BACKEND, FakeLLM, fake_paper, tool_call
from jarvis import config, llm
from jarvis.sources import SourceRegistry

SPEC = {"objective": "Find RAG eval methods", "sub_query": "rag evaluation", "sources": ["academic", "web"],
        "tool_budget": 4, "key_questions": ["Which metrics?"]}


# --------------------------------------------------------------------------- health check


class R:
    def __init__(self, code, body=None):
        self.status_code = code
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_check_backend_says_how_to_fix_ollama(monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", down)
    assert "isn't running" in REAL_CHECK_BACKEND("ollama")

    monkeypatch.setattr(requests, "get", lambda *a, **k: R(200, {"models": [{"name": "llama3:8b"}]}))
    msg = REAL_CHECK_BACKEND("ollama")
    assert "aren't pulled" in msg and f"ollama pull {config.settings.ollama_model}" in msg

    monkeypatch.setattr(requests, "get", lambda *a, **k: R(200, {"models": [{"name": config.settings.ollama_model}]}))
    assert REAL_CHECK_BACKEND("ollama") is None


def test_check_backend_reports_rejected_deepseek_key(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(401))
    assert "rejected the API key" in REAL_CHECK_BACKEND("deepseek")
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(200))
    assert REAL_CHECK_BACKEND("deepseek") is None


def test_headless_refuses_to_run_on_a_dead_backend(monkeypatch):
    from jarvis import headless

    monkeypatch.setattr(llm, "check_backend", lambda *a, **k: "Ollama isn't running")
    with pytest.raises(headless.BackendUnavailable, match="isn't running"):
        headless.run_research("q", backend="ollama")


# --------------------------------------------------------------------------- timeouts


def test_hung_tool_is_abandoned_and_the_subagent_moves_on(monkeypatch):
    monkeypatch.setattr(sa.settings, "tool_timeout", 0.3)

    def hang(query, max_results=5):
        time.sleep(3)
        return {"text": "too late"}

    llm_ = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_web", {"query": "x"}, 0),
                                          tool_call("search_arxiv", {"query": "x"}, 1)]),
        AIMessage(content="report [S1]"),
    ])
    monkeypatch.setattr(sa, "get_llm", lambda *a, **k: llm_)
    monkeypatch.setattr(sa, "TOOL_FUNCS", {
        "search_web": hang,
        "search_arxiv": lambda query, max_results=5: {"papers": [fake_paper(1)], "text": "arXiv"},
    })
    t0 = time.monotonic()
    out = sa.run_subagent(SPEC, "deepseek", SourceRegistry())
    assert time.monotonic() - t0 < 2  # did not wait for the 3 s tool
    texts = [m.content for m in llm_.calls[1] if m.type == "tool"]
    assert "timed out" in texts[0] and texts[1].startswith("arXiv")  # the healthy tool still answered
    assert out["sources"] == ["S1"]


def test_wave_time_limit_stops_research_and_still_reports(monkeypatch):
    llm_ = FakeLLM(subagent_script=[AIMessage(content="", tool_calls=[tool_call("search_web", {"query": "x"})])])
    monkeypatch.setattr(sa, "get_llm", lambda *a, **k: llm_)
    monkeypatch.setattr(sa, "TOOL_FUNCS", {})
    out = sa.run_subagent(SPEC, "deepseek", SourceRegistry(), deadline=time.monotonic() - 1)
    assert len(llm_.calls) == 1 and out["turns"] == 0  # only the closer ran
    assert out["findings"]


# --------------------------------------------------------------------------- per-subagent resume


class Crash(BaseException):
    """Stands in for the process dying (not an Exception, so nothing swallows it)."""


def _state(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "RUNS_DIR", tmp_path)
    return {"backend": "ollama", "run_id": "run1", "brief": "b", "sources": {}}


def test_resume_reuses_finished_subagents_and_their_source_ids(tmp_path, monkeypatch):
    specs = [{"objective": "one"}, {"objective": "two"}, {"objective": "three"}]
    state = _state(tmp_path, monkeypatch)

    def first_attempt(spec, backend, registry, i, total, brief, run_id, deadline):
        if i == 2:
            raise Crash()
        sid = registry.add("paper", fake_paper(i))
        return {"objective": spec["objective"], "findings": f"found [{sid}]", "sources": [sid],
                "tools": [], "tool_calls": 1, "turns": 1}

    monkeypatch.setattr(nodes, "run_subagent", first_attempt)
    with pytest.raises(Crash):
        nodes.dispatch(specs, state)
    saved = json.loads((tmp_path / "run1" / "wave_1.json").read_text(encoding="utf-8"))
    assert list(saved["done"]) == ["1"] and "S1" in saved["registry"]

    ran = []

    def second_attempt(spec, backend, registry, i, total, brief, run_id, deadline):
        ran.append(i)
        sid = registry.add("paper", fake_paper(10 + i))
        return {"objective": spec["objective"], "findings": f"found [{sid}]", "sources": [sid],
                "tools": [], "tool_calls": 1, "turns": 1}

    monkeypatch.setattr(nodes, "run_subagent", second_attempt)
    reports, sources = nodes.dispatch(specs, state)
    assert ran == [2, 3]  # subagent 1 was not run again
    assert reports[0]["findings"] == "found [S1]" and sources["S1"]["item"]["title"] == "Paper number 1"
    assert {reports[1]["sources"][0], reports[2]["sources"][0]} == {"S2", "S3"}  # numbering continued
    assert not (tmp_path / "run1" / "wave_1.json").exists()  # cleaned up once the wave finished


def test_saved_progress_is_ignored_for_a_different_plan(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch)
    (tmp_path / "run1").mkdir()
    (tmp_path / "run1" / "wave_1.json").write_text(json.dumps(
        {"objectives": ["old plan"], "registry": {}, "done": {"1": {"objective": "old plan"}}}), encoding="utf-8")
    ran = []
    monkeypatch.setattr(nodes, "run_subagent", lambda spec, *a: ran.append(spec["objective"]) or {
        "objective": spec["objective"], "findings": "f", "sources": [], "tools": [], "tool_calls": 0, "turns": 1})
    nodes.dispatch([{"objective": "new plan"}], state)
    assert ran == ["new plan"]


def test_one_failing_subagent_does_not_sink_the_wave(tmp_path, monkeypatch):
    state = _state(tmp_path, monkeypatch)

    def flaky(spec, backend, registry, i, *a):
        if i == 1:
            raise RuntimeError("boom")
        return {"objective": spec["objective"], "findings": "ok", "sources": [], "tools": [], "tool_calls": 1,
                "turns": 1}

    monkeypatch.setattr(nodes, "run_subagent", flaky)
    reports, _ = nodes.dispatch([{"objective": "a"}, {"objective": "b"}], {**state, "backend": "deepseek"})
    assert "failed: boom" in reports[0]["findings"] and reports[1]["findings"] == "ok"
    saved = json.loads((tmp_path / "run1" / "wave_1.json").read_text(encoding="utf-8"))
    assert list(saved["done"]) == ["2"]  # the failed one stays pending, so /resume retries it


# --------------------------------------------------------------------------- review + plan


@pytest.mark.parametrize("verdict, expected", [(True, 0), ("true", 0), (False, 1), ("false", 1), (None, 1)])
def test_review_respects_the_leads_verdict(monkeypatch, verdict, expected):
    review = {"gaps": ["g"], "followups": [{"objective": "dig deeper"}]}
    if verdict is not None:
        review["sufficient"] = verdict
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: FakeLLM(review=review))
    out = nodes.review_node({"backend": "deepseek", "brief": "b", "subagent_reports": []})
    assert len(out["followups"]) == expected and out["gaps"] == ["g"]


def test_single_agent_ablation_merges_the_plan(monkeypatch):
    plan = {"query_type": "breadth_first", "subagents": [
        {"objective": "a", "key_questions": ["qa"], "sources": ["academic"]},
        {"objective": "b", "key_questions": ["qb"], "sources": ["web", "code"]}]}
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: FakeLLM(plan=plan))
    monkeypatch.setattr(nodes.settings, "single_agent", True)
    out = nodes.plan_node({"query": "q", "brief": "the brief", "backend": "deepseek"})
    (spec,) = out["plan"]
    assert spec["objective"] == "the brief" and spec["key_questions"] == ["qa", "qb"]
    assert spec["sources"] == ["academic", "web", "code"] and spec["tool_budget"] == nodes.settings.subagent_hard_cap


# --------------------------------------------------------------------------- context fitting


def test_report_prompt_is_trimmed_to_fit_the_ollama_window(monkeypatch):
    llm_ = FakeLLM()
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm_)
    reg = SourceRegistry()
    sids = [reg.add("paper", fake_paper(i, abstract="long abstract " * 40)) for i in range(40)]
    reports = [{"objective": f"o{i}", "findings": ("finding " * 700) + " ".join(f"[{s}]" for s in sids),
                "sources": sids} for i in range(4)]
    state = {"backend": "ollama", "query": "q", "brief": "b", "subagent_reports": reports, "sources": reg.to_dict()}
    nodes.report_node(state)
    prompt = llm_.calls[0]
    assert len(prompt) <= llm.input_char_budget("ollama", config.settings.report_max_tokens)
    assert "Citation rules (strict)" in prompt  # the instructions survived

    nodes.report_node({**state, "backend": "deepseek"})  # no trimming on the big window
    assert len(llm_.calls[1]) > len(prompt)


# --------------------------------------------------------------------------- storage + http


def test_post_requests_are_cached_by_body(monkeypatch, tmp_path):
    import jarvis.tools._http as http

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"results": [1]}

    monkeypatch.setattr(config.settings, "tool_cache_hours", 1)
    monkeypatch.setattr(http, "CACHE_DIR", tmp_path)
    sent = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(k["json"]) or Resp())
    for body in ({"query": "a"}, {"query": "a"}, {"query": "b"}):
        assert http.post("https://api.example.org/search", body) == {"results": [1]}
    assert sent == [{"query": "a"}, {"query": "b"}]


def test_save_pdf_rejects_html_and_replaces_old_fake_pdfs(monkeypatch, tmp_path):
    from jarvis.tools import reader

    class Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    path = tmp_path / "p.pdf"
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(b"<html>Sign in to read</html>"))
    assert reader.save_pdf("https://x/p.pdf", path) is False and not path.exists()

    path.write_bytes(b"<html>saved by an old version</html>")
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(b"%PDF-1.7 real"))
    assert reader.save_pdf("https://x/p.pdf", path) is True and reader.is_pdf(path)


def test_corrupt_vector_registry_is_rebuilt_from_folders(monkeypatch, tmp_path):
    from jarvis import store

    tmp_path = tmp_path / "vectorstore"
    tmp_path.mkdir()
    (tmp_path / "not-a-store").mkdir()  # ignored: not named <slug>__<9 digits>
    monkeypatch.setattr(store, "VECTOR_DIR", tmp_path)
    monkeypatch.setattr(store, "_REGISTRY", tmp_path / "registry.json")
    (tmp_path / "ragas__120000000").mkdir()
    (tmp_path / "ragas__120000000" / "jarvis_meta.json").write_text(json.dumps(
        {"folder": "ragas__120000000", "paper_id": "2309.15217", "title": "Ragas", "created_at": "", "chunks": 9}),
        encoding="utf-8")
    (tmp_path / "old-paper__130000000").mkdir()  # indexed before folders kept their own entry
    (tmp_path / "registry.json").write_text('[{"folder": "ragas__1', encoding="utf-8")  # half-written

    papers = store.list_papers()
    assert [p["title"] for p in papers] == ["old paper", "Ragas"]
    assert store.find_folder({"id": "2309.15217"}) == "ragas__120000000"
    assert list(tmp_path.glob("registry.corrupt-*.json"))  # bad file kept for inspection


def test_usage_survives_a_restart_for_the_same_run(tmp_path):
    path = tmp_path / "usage.json"
    t = llm.UsageTracker()
    t.attach(path, "thread-1")
    t.record("lead", {"input_tokens": 100, "output_tokens": 50})
    t.record("worker", {"input_tokens": 1000, "output_tokens": 10})

    fresh = llm.UsageTracker()  # a new process
    assert fresh.restore(path, "thread-1") is True
    assert fresh.snapshot()["lead"] == {"calls": 1, "in": 100, "cached": 0, "out": 50}
    assert fresh.snapshot()["worker"]["in"] == 1000
    assert llm.UsageTracker().restore(path, "another-run") is False
