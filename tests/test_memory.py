"""Conversation memory: a follow-up question builds on the previous run's report and sources."""

import json

from langchain_core.messages import AIMessage

import jarvis.nodes as nodes
import jarvis.subagent as sa
from conftest import FakeLLM, fake_paper, tool_call
from jarvis.headless import run_research


class FollowUpLLM(FakeLLM):
    """FakeLLM whose clarify step says whether the request follows up on the previous run."""

    def __init__(self, follow_up: bool, **kw):
        super().__init__(**kw)
        self.follow_up = follow_up

    def invoke(self, msgs):
        text = msgs if isinstance(msgs, str) else ""
        if "Classify intent" in text:
            self.calls.append(msgs)
            return AIMessage(content=json.dumps({"intent": "general", "follow_up": self.follow_up,
                                                 "brief": "Compare RAGAS with ARES.", "query": "ares", "questions": []}))
        return super().invoke(msgs)


def _prior() -> dict:
    return {"query": "how is RAG evaluated?", "brief": "Survey RAG evaluation.",
            "findings": "# RAG evaluation\n**TL;DR** RAGAS scores faithfulness without references [S7].",
            "sources": {"S7": {"kind": "paper", "item": fake_paper(7, title="Ragas paper")}}}


def _wire(monkeypatch, llm):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "TOOL_FUNCS", {"search_arxiv": lambda query, max_results=5: {
        "papers": [fake_paper(8, title="ARES paper")], "text": "arXiv results"}})


def test_build_prior_maps_numbered_citations_back_to_source_ids():
    state = {"query": "q", "brief": "b", "cited_sids": ["S4", "S9"],
             "sources": {"S4": {"kind": "paper", "item": {"title": "A"}}, "S9": {"kind": "web", "item": {"title": "B"}},
                         "S11": {"kind": "web", "item": {"title": "never cited"}}},
             "report": "Claim [1]. Other [2][1]. A year [2023].\n\n## Sources\n- [1] A\n- [2] B\n\n_(saved to x.md)_"}
    prior = nodes.build_prior(state)
    assert prior["findings"] == "Claim [S4]. Other [S9][S4]. A year [2023]."
    assert set(prior["sources"]) == {"S4", "S9"}  # only what the report cited
    assert nodes.build_prior({"report": ""}) == {}


def test_follow_up_builds_on_the_previous_report(monkeypatch):
    llm = FollowUpLLM(True, subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_arxiv", {"query": "ares"})]),
        AIMessage(content="", tool_calls=[tool_call("complete_task", {"report": "ARES trains judges [S8]."})]),
    ], report="# RAGAS vs ARES\nRAGAS is reference-free [S7]; ARES trains judges [S8].")
    _wire(monkeypatch, llm)
    final = run_research("compare that with ARES", backend="ollama", prior=_prior())

    assert final["follow_up"] is True
    prompts = [c for c in llm.calls if isinstance(c, str)]
    clarify, plan = prompts[0], next(p for p in prompts if "orchestrating a team" in p)
    report = next(p for p in prompts if "writing the final research report" in p)
    assert 'Request: "how is RAG evaluated?"' in clarify  # the clarify step saw the last run
    assert "FOLLOW-UP" in plan and "faithfulness without references [S7]" in plan  # plan only what's new
    assert "[S7] Ragas paper" in report  # the carried-over source is citable
    assert final["citation_stats"]["invalid_tags"] == 0  # [S7] is valid in the new run
    assert final["cited_sids"] == ["S7", "S8"]  # new sources continue numbering after S7
    assert "Ragas paper" in final["report"] and "ARES paper" in final["report"]


def test_new_topic_ignores_the_previous_run(monkeypatch):
    llm = FollowUpLLM(False)
    _wire(monkeypatch, llm)
    final = run_research("best vector databases?", backend="ollama", prior=_prior())

    assert final["follow_up"] is False and final["prior"] == {}
    plan = next(c for c in llm.calls if isinstance(c, str) and "orchestrating a team" in c)
    assert "FOLLOW-UP" not in plan
    assert "S7" not in final["sources"]  # nothing carried over


def test_no_previous_run_means_no_memory_section(monkeypatch):
    llm = FollowUpLLM(True)  # even if the model claims a follow-up, there is nothing to follow
    _wire(monkeypatch, llm)
    final = run_research("how is RAG evaluated?", backend="ollama")
    assert final["follow_up"] is False
    assert "PREVIOUS RESEARCH in this session" not in llm.calls[0]
