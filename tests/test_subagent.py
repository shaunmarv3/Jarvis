import threading
import time

from langchain_core.messages import AIMessage, ToolMessage

import jarvis.subagent as sa
from conftest import FakeLLM, fake_paper, tool_call
from jarvis.sources import SourceRegistry

SPEC = {"objective": "Find RAG eval methods", "sub_query": "rag evaluation", "sources": ["academic", "web"],
        "tool_budget": 4, "key_questions": ["Which metrics?"]}


def _install(monkeypatch, llm, funcs):
    monkeypatch.setattr(sa, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "TOOL_FUNCS", funcs)


def test_model_sees_its_tool_results_and_cites_them(monkeypatch):
    llm = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_arxiv", {"query": "rag eval"})]),
        AIMessage(content="", tool_calls=[tool_call("complete_task", {"report": "RAGAS measures faithfulness [S1]."})]),
    ])
    _install(monkeypatch, llm, {"search_arxiv": lambda query, max_results=5: {
        "papers": [fake_paper(1)], "text": "arXiv results for 'rag eval':\n- ..."}})
    reg = SourceRegistry()
    out = sa.run_subagent(SPEC, "deepseek", reg)

    second_call = llm.calls[1]
    tool_msgs = [m for m in second_call if isinstance(m, ToolMessage)]
    assert tool_msgs and "[S1] Paper number 1" in tool_msgs[0].content  # the loop feeds results back
    assert out["findings"] == "RAGAS measures faithfulness [S1]."
    assert out["sources"] == ["S1"] and out["tool_calls"] == 1
    assert "complete_task" in llm.bound_tools


def test_tools_in_one_turn_run_in_parallel(monkeypatch):
    active, peak, lock = [0], [0], threading.Lock()

    def slow(query, max_results=5):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return {"web": [{"title": query, "url": f"https://e.com/{query}", "snippet": "s"}], "text": "Web results"}

    llm = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_web", {"query": q}, i) for i, q in enumerate("abc")]),
        AIMessage(content="done [S1][S2][S3]"),
    ])
    _install(monkeypatch, llm, {"search_web": slow})
    t0 = time.time()
    out = sa.run_subagent(SPEC, "deepseek", SourceRegistry())
    assert peak[0] == 3 and time.time() - t0 < 0.5
    assert len(out["sources"]) == 3


def test_budget_is_enforced_and_every_call_is_answered(monkeypatch):
    calls = [tool_call("search_web", {"query": f"q{i}"}, i) for i in range(6)]
    llm = FakeLLM(subagent_script=[AIMessage(content="", tool_calls=calls)])
    ran = []
    _install(monkeypatch, llm, {"search_web": lambda query, max_results=5: ran.append(query) or {"text": "none"}})
    out = sa.run_subagent({**SPEC, "tool_budget": 4}, "deepseek", SourceRegistry())
    assert len(ran) == 4 and out["tool_calls"] == 4
    # closer call: history holds one ToolMessage per tool call (2 of them "Skipped")
    closer_msgs = llm.calls[-1]
    tms = [m for m in closer_msgs if isinstance(m, ToolMessage)]
    assert len(tms) == 6 and sum("budget exhausted" in m.content for m in tms) == 2
    assert out["findings"]  # always returns a report


def test_answer_from_memory_triggers_starter_searches_from_sources(monkeypatch):
    llm = FakeLLM(subagent_script=[AIMessage(content="I already know the answer."), AIMessage(content="report [S1]")])
    hit = []
    funcs = {
        "search_arxiv": lambda query, max_results=5: hit.append("arxiv") or {"papers": [fake_paper(1)], "text": "arXiv"},
        "search_web": lambda query, max_results=5: hit.append("web") or {"text": "No web results"},
        "search_github": lambda query, max_results=5: hit.append("github") or {"text": "none"},
    }
    _install(monkeypatch, llm, funcs)
    out = sa.run_subagent({**SPEC, "sources": ["academic", "web"]}, "deepseek", SourceRegistry())
    assert sorted(hit) == ["arxiv", "web"]  # chosen from the delegation's sources, not hardwired
    assert out["findings"] == "report [S1]"


def test_source_id_is_resolved_for_read_paper(monkeypatch):
    seen = {}

    def read(paper, focus=""):
        seen["paper"] = paper
        return {"text": "Full text …", "content": "x" * 50}

    llm = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("read_paper", {"paper": "S1", "focus": "metric"})]),
        AIMessage(content="report [S1]"),
    ])
    _install(monkeypatch, llm, {"read_paper": read})
    reg = SourceRegistry()
    reg.add("paper", fake_paper(1))
    out = sa.run_subagent(SPEC, "deepseek", reg)
    assert seen["paper"] == "https://arxiv.org/abs/2401.00001"
    tool_msg = [m for m in llm.calls[1] if isinstance(m, ToolMessage)][0]
    assert tool_msg.content.startswith("[S1] Full text")
    assert out["sources"] == ["S1"]


def test_failing_tool_and_bad_args_never_crash(monkeypatch):
    def boom(query, max_results=5):
        raise RuntimeError("HTTP 500")

    llm = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_web", {"query": "x", "bogus": 1}, 0),
                                          tool_call("no_such_tool", {}, 1)]),
        AIMessage(content="report"),
    ])
    _install(monkeypatch, llm, {"search_web": boom})
    sa.run_subagent(SPEC, "deepseek", SourceRegistry())
    texts = [m.content for m in llm.calls[1] if isinstance(m, ToolMessage)]
    assert "failed: HTTP 500" in texts[0] and "Unknown tool" in texts[1]


def test_compact_trims_old_tool_results_only():
    from langchain_core.messages import HumanMessage

    msgs = [HumanMessage("task"), AIMessage(content="", tool_calls=[tool_call("search_web", {}, 0)]),
            ToolMessage(content="A" * 5000, tool_call_id="call_search_web_0"),
            AIMessage(content="", tool_calls=[tool_call("search_web", {}, 1)]),
            ToolMessage(content="B" * 5000, tool_call_id="call_search_web_1")]
    out = sa._compact(msgs, 3000)
    assert len(out[2].content) < 800 and out[4].content == "B" * 5000


def test_full_text_reads_have_their_own_allowance(monkeypatch):
    searches = [tool_call("search_web", {"query": f"q{i}"}, i) for i in range(4)]
    reads = [tool_call("read_web", {"url": f"https://e.com/{i}"}, 10 + i) for i in range(5)]
    llm = FakeLLM(subagent_script=[AIMessage(content="", tool_calls=searches),
                                   AIMessage(content="", tool_calls=reads),
                                   AIMessage(content="report")])
    ran = {"search": 0, "read": 0}

    def search(query, max_results=5):
        ran["search"] += 1
        return {"text": "none"}

    def read(url, focus=""):
        ran["read"] += 1
        return {"text": "page", "content": "page text"}

    _install(monkeypatch, llm, {"search_web": search, "read_web": read})
    out = sa.run_subagent({**SPEC, "tool_budget": 4}, "deepseek", SourceRegistry())
    # search budget spent in turn 1 doesn't block reading: up to subagent_max_reads reads still run
    assert ran == {"search": 4, "read": sa.settings.subagent_max_reads}
    assert out["tool_calls"] == 4 + sa.settings.subagent_max_reads


def test_read_by_bare_id_registers_real_metadata(monkeypatch):
    llm = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_arxiv", {"query": "x"}, 0)]),
        AIMessage(content="", tool_calls=[tool_call("read_paper", {"paper": "2401.00001", "focus": "f"}, 1)]),
        AIMessage(content="report"),
    ])
    _install(monkeypatch, llm, {
        "search_arxiv": lambda query, max_results=5: {"papers": [fake_paper(1)], "text": "arXiv results"},
        "read_paper": lambda paper, focus="": {"text": "Full text", "content": "body", "meta": fake_paper(1)},
    })
    reg = SourceRegistry()
    out = sa.run_subagent(SPEC, "deepseek", reg)
    assert out["sources"] == ["S1"] and len(reg.items) == 1  # the read deduped with the search hit
    assert reg.get("S1")["item"]["title"] == "Paper number 1"
