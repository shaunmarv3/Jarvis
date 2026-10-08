"""Shared graph state."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    run_id: str  # folder under data/runs/ holding this run's artifacts
    query: str  # the user's original request
    title: str  # 2-5 word topic name from clarify (the terminal tab title)
    backend: str  # "ollama" | "deepseek"
    memory: str  # the user's standing preferences (data/memory.md), set by the CLI only
    intent: str  # find_papers | pull_exact | read | find_datasets | general
    brief: str  # short research brief (confirmed by the user)
    clarify_questions: list  # lead's clarifying questions: [{question, options}] (may be empty)
    clarify_answers: str  # the user's answers, formatted "Q: ... | A: ..." per line
    current_query: str  # first search query suggested by clarify/brief
    query_type: str  # straightforward | depth_first | breadth_first (lead's classification)
    plan: list  # lead's delegation specs: [{objective, key_questions, sub_query, sources, ...}]
    max_subagents: int  # ceiling the lead was allowed (backend-dependent)
    replan: bool  # confirm step: user edited the brief -> re-plan before fan-out
    subagent_reports: list  # each subagent's {objective, findings, sources, tools, tool_calls, turns}
    sources: dict  # source registry: {"S1": {"kind": "paper"|"web"|..., "item": {...}}}
    followups: list  # lead's gap-filling delegations after review (at most one wave)
    gaps: list  # gaps the lead identified in review
    followup_done: bool
    report: str  # final report text (with numbered citations)
    citation_stats: dict  # {"cited", "invalid_tags", "retrieved"}
    cited_sids: list  # source ids in citation order: [1] -> cited_sids[0]
    papers: list  # papers for /read and /papers (cited first)
    # Conversation memory: the previous run in this session, given by the CLI.
    prior: dict  # {"query", "brief", "findings" (report body with [S#] tags), "sources": {sid: entry}}
    follow_up: bool  # clarify decided this request builds on `prior`
