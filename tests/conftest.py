"""Shared test fixtures: no network, no real LLM, isolated data dirs."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every data dir at a temp folder and disable the HTTP cache."""
    for name in ("CACHE_DIR", "REPORTS_DIR", "RUNS_DIR", "PAPERS_DIR"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setattr(config, name, d)
    import jarvis.nodes as nodes
    import jarvis.subagent as subagent
    import jarvis.tools._http as http

    monkeypatch.setattr(http, "CACHE_DIR", tmp_path / "cache_dir")
    monkeypatch.setattr(nodes, "REPORTS_DIR", tmp_path / "reports_dir")
    monkeypatch.setattr(nodes, "RUNS_DIR", tmp_path / "runs_dir")
    monkeypatch.setattr(subagent, "RUNS_DIR", tmp_path / "runs_dir")
    monkeypatch.setattr(config.settings, "tool_cache_hours", 0)
    monkeypatch.setattr(config.settings, "deepseek_api_key", "")
    yield


class FakeLLM:
    """Stands in for a chat model. Routes by prompt content; records every call.

    `subagent_script` is a list of AIMessages returned in order to subagent calls.
    """

    def __init__(self, subagent_script=None, plan=None, review=None, report=None):
        self.calls: list = []
        self.subagent_script = list(subagent_script or [])
        self.plan = plan or {"query_type": "straightforward", "subagents": [
            {"objective": "Find RAG eval methods", "sub_query": "rag evaluation", "sources": ["academic", "web"],
             "tool_budget": 4}]}
        self.review = review or {"sufficient": True, "gaps": [], "followups": []}
        self.report = report or "# Report\n\n**TL;DR** RAGAS measures faithfulness [S1]. Also see [S2][S99]."
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [getattr(t, "name", str(t)) for t in tools]
        return self

    def invoke(self, msgs):
        self.calls.append(msgs)
        text = msgs if isinstance(msgs, str) else "\n".join(str(getattr(m, "content", m)) for m in msgs)
        if "You are a research SUBAGENT" in text:
            if self.subagent_script:
                return self.subagent_script.pop(0)
            return AIMessage(content="Final report: RAGAS defines faithfulness [S1].")
        if "Classify intent" in text:
            return AIMessage(content=json.dumps({"intent": "find_papers", "brief": "Survey RAG evaluation.",
                                                 "query": "rag evaluation", "questions": []}))
        if "orchestrating a team" in text:
            return AIMessage(content=json.dumps(self.plan))
        if "subagents have reported back" in text:
            return AIMessage(content=json.dumps(self.review))
        if "writing the final research report" in text:
            return AIMessage(content=self.report)
        return AIMessage(content="{}")


def tool_call(name: str, args: dict, idx: int = 0) -> dict:
    return {"name": name, "args": args, "id": f"call_{name}_{idx}", "type": "tool_call"}


def fake_paper(i: int, **kw) -> dict:
    base = {"source": "arxiv", "id": f"2401.{i:05d}", "title": f"Paper number {i}", "year": 2024,
            "abstract": f"Abstract {i} about RAG evaluation faithfulness.", "url": f"https://arxiv.org/abs/2401.{i:05d}",
            "pdf_url": f"https://arxiv.org/pdf/2401.{i:05d}"}
    base.update(kw)
    return base


def strip_markup(s: str) -> str:
    return re.sub(r"\[/?[a-z ]+\]", "", s)
