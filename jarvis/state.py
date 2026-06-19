"""Shared graph state."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str  # the user's original request
    backend: str  # "ollama" | "deepseek"
    intent: str  # find_papers | pull_exact | read | find_datasets | general
    brief: str  # short research brief (confirmed by the user)
    current_query: str  # the query for the current loop iteration
    queries: list  # all queries issued so far
    plan: list  # lead agent's subagent specs: [{objective, sub_query, tools}]
    max_subagents: int  # ceiling the lead was allowed (backend-dependent)
    replan: bool  # confirm step: user edited the brief -> re-plan before fan-out
    subagent_reports: list  # each subagent's {objective, findings, papers, datasets, web, tools}
    papers: list  # accumulated paper dicts
    datasets: list  # accumulated dataset dicts
    web: list  # accumulated web hits
    last_evidence: str  # raw tool output text from the most recent act step
    last_tools: list  # "name(args)" strings the agent called this step (for live view)
    findings: str  # running synthesis
    gaps: str  # open knowledge gaps from the last reflect
    complete: bool  # reflect decided we're done
    loop_count: int
    max_loops: int
    report: str  # final report text
