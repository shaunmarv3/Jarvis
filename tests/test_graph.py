"""End-to-end graph run with a fake LLM and fake tools (no network)."""

from langchain_core.messages import AIMessage

import jarvis.nodes as nodes
import jarvis.subagent as sa
from conftest import FakeLLM, fake_paper, tool_call
from jarvis.headless import run_research


def _wire(monkeypatch, llm):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "TOOL_FUNCS", {
        "search_arxiv": lambda query, max_results=5: {"papers": [fake_paper(1), fake_paper(2)], "text": "arXiv results"},
        "search_web": lambda query, max_results=6: {"web": [{"title": "Blog", "url": "https://b.com", "snippet": "s"}],
                                                    "text": "Web results"},
    })


def test_full_run_produces_grounded_cited_report(monkeypatch):
    llm = FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_arxiv", {"query": "rag"}, 0),
                                          tool_call("search_web", {"query": "rag"}, 1)]),
        AIMessage(content="", tool_calls=[tool_call("complete_task", {"report": "Faithfulness metric [S1]; blog [S3]."})]),
    ])
    _wire(monkeypatch, llm)
    final = run_research("how is RAG evaluated?", backend="ollama")

    report = final["report"]
    assert "[S" not in report.split("## Sources")[0]  # all tags converted
    assert "- [1] Paper number 1 (2024)" in report
    assert final["citation_stats"]["invalid_tags"] == 1  # the fake report cites a non-existent [S99]
    assert final["query_type"] == "straightforward"
    assert final["papers"][0]["title"] == "Paper number 1"  # cited papers listed first for /read
    assert len(final["subagent_reports"]) == 1


def test_plan_ceiling_and_spec_defaults(monkeypatch):
    many = {"query_type": "breadth_first", "subagents": [
        {"objective": f"facet {i}", "sources": "papers, github, junk", "tool_budget": 99} for i in range(12)]}
    llm = FakeLLM(plan=many)
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    out = nodes.plan_node({"query": "q", "brief": "b", "backend": "ollama"})
    assert len(out["plan"]) == nodes.subagent_ceiling("ollama")
    spec = out["plan"][0]
    assert spec["sources"] == ["academic", "code"]  # aliases mapped, junk dropped
    assert spec["tool_budget"] == nodes.settings.subagent_hard_cap  # clamped
    assert spec["key_questions"] == ["facet 0"]


def test_empty_plan_falls_back_to_one_subagent(monkeypatch):
    llm = FakeLLM(plan={"subagents": []})
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    out = nodes.plan_node({"query": "q", "brief": "the brief", "backend": "ollama"})
    assert len(out["plan"]) == 1 and out["plan"][0]["objective"] == "the brief"


def test_review_triggers_exactly_one_followup_wave(monkeypatch):
    review = {"sufficient": False, "gaps": ["no production view"], "followups": [
        {"objective": "production practice", "sources": ["web", "community"]},
        {"objective": "extra 1"}, {"objective": "extra 2"}]}
    llm = FakeLLM(review=review)
    _wire(monkeypatch, llm)
    monkeypatch.setattr(nodes.settings, "lead_iteration", True)
    final = run_research("rag eval", backend="ollama")
    # 1 planned + max_followup_subagents (2) follow-ups, and no second follow-up round
    assert len(final["subagent_reports"]) == 1 + nodes.settings.max_followup_subagents
    assert final["followup_done"] is True
    assert final["gaps"] == ["no production view"]


def test_lead_iteration_off_by_default_on_ollama(monkeypatch):
    llm = FakeLLM(review={"sufficient": False, "followups": [{"objective": "x"}]})
    _wire(monkeypatch, llm)
    final = run_research("rag eval", backend="ollama")
    assert len(final["subagent_reports"]) == 1
