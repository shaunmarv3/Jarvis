"""LangGraph node functions: clarify, confirm, act, synthesize, reflect, finalize."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .config import settings
from .embeddings import rank_by_relevance
from .llm import get_llm
from .prompts import (
    ACT_SYSTEM,
    ACT_USER,
    CLARIFY_PROMPT,
    FINALIZE_PROMPT,
    REFLECT_PROMPT,
    SYNTH_PROMPT,
)
from .state import AgentState
from .tools import TOOL_FUNCS, TOOL_SCHEMAS
from .utils import dedup_papers, extract_json, truncate

# Default tool to run per intent when the model declines to emit a tool call.
_FALLBACK = {
    "find_datasets": ["search_datasets"],
    "pull_exact": ["resolve_title", "search_arxiv"],
    "read": ["resolve_title", "search_arxiv"],
    "general": ["search_web", "search_arxiv"],
}
_DEFAULT_FALLBACK = ["search_arxiv", "search_semantic_scholar", "search_openalex"]


def clarify_node(state: AgentState) -> dict:
    query = state.get("query") or ""
    llm = get_llm(state.get("backend"), temperature=0.2)
    raw = llm.invoke(CLARIFY_PROMPT.format(query=query)).content
    data = extract_json(raw)

    intent = data.get("intent") or "find_papers"
    brief = data.get("brief") or f"Research the user's request: {query}"
    first_query = data.get("query") or query
    return {
        "intent": intent,
        "brief": brief,
        "current_query": first_query,
        "queries": [first_query],
        "papers": [],
        "datasets": [],
        "findings": "",
        "gaps": "",
        "complete": False,
        "loop_count": 0,
        "max_loops": state.get("max_loops") or settings.max_loops,
    }


def confirm_node(state: AgentState) -> dict:
    """Human-in-the-loop: pause for the user to approve or edit the brief.

    The CLI resumes with `Command(resume=<text>)`. A plain yes/go proceeds;
    any other text is treated as an edited brief.
    """
    decision = interrupt(
        {"type": "confirm_brief", "intent": state.get("intent"), "brief": state.get("brief")}
    )
    if isinstance(decision, dict):
        decision = decision.get("text", "")
    text = (decision or "").strip()
    if text and text.lower() not in {"y", "yes", "go", "ok", "okay", "proceed", "sure"}:
        return {
            "brief": text,
            "messages": [HumanMessage(content=f"(brief refined) {text}")],
        }
    return {}


def _run_tool(name: str, args: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"text": f"(unknown tool {name})"}
    try:
        return fn(**args)
    except Exception as exc:  # never let a tool crash the graph
        return {"text": f"(tool {name} failed: {exc})"}


def act_node(state: AgentState) -> dict:
    """One round of tool-calling toward the current goal."""
    backend = state.get("backend")
    llm = get_llm(backend, temperature=0.2).bind_tools(TOOL_SCHEMAS)

    msgs = [
        ("system", ACT_SYSTEM),
        (
            "user",
            ACT_USER.format(
                brief=state.get("brief", ""),
                current_query=state.get("current_query", ""),
                paper_count=len(state.get("papers", [])),
                gaps=state.get("gaps") or "none yet",
            ),
        ),
    ]
    try:
        ai = llm.invoke(msgs)
        tool_calls = getattr(ai, "tool_calls", None) or []
    except Exception:
        tool_calls = []

    # Fallback: if the model didn't pick tools, run sensible defaults.
    if not tool_calls:
        q = state.get("current_query", "")
        names = _FALLBACK.get(state.get("intent", ""), _DEFAULT_FALLBACK)
        arg = {"title": q} if names[0] in {"resolve_title", "resolve_doi"} else {"query": q}
        tool_calls = [{"name": n, "args": arg} for n in names]

    new_papers: list[dict] = []
    new_datasets: list[dict] = []
    new_web: list[dict] = []
    summaries: list[str] = []
    called: list[str] = []
    for tc in tool_calls:
        args = tc.get("args") or {}
        called.append(f"{tc['name']}({', '.join(f'{k}={truncate(str(v), 40)}' for k, v in args.items())})")
        out = _run_tool(tc["name"], args)
        new_papers.extend(out.get("papers", []) or [])
        new_datasets.extend(out.get("datasets", []) or [])
        new_web.extend(out.get("web", []) or [])
        summaries.append(out.get("text", ""))

    papers = dedup_papers(list(state.get("papers", [])) + new_papers)
    # Relevance filter: rank against the focused user query (not the verbose brief,
    # which can inflate scores of tangential papers) and drop off-topic noise.
    ranked = rank_by_relevance(
        f"{state.get('query', '')} {state.get('current_query', '')}".strip(),
        papers,
        top_k=settings.keep_top_papers,
        min_score=settings.relevance_min,
    )
    # Never let the filter wipe everything out if scores run low.
    if not ranked and papers:
        ranked = papers[: settings.keep_top_papers]
    datasets = state.get("datasets", []) + [
        d for d in new_datasets if d not in state.get("datasets", [])
    ]
    evidence = "\n\n".join(s for s in summaries if s)[:6000]
    return {
        "papers": ranked,
        "datasets": datasets,
        "web": state.get("web", []) + new_web,
        "last_evidence": evidence,
        "last_tools": called,
        "messages": [AIMessage(content=evidence)],
    }


def synthesize_node(state: AgentState) -> dict:
    papers = state.get("papers", [])
    material = "\n".join(
        f"- {p.get('title')} ({p.get('year')}): {truncate(p.get('abstract', ''), 240)}"
        for p in papers[-10:]
    ) or "(no papers yet)"
    # Include non-paper evidence (web pages, dataset inspections) from this step.
    evidence = state.get("last_evidence", "")
    if evidence:
        material += "\n\nOther evidence gathered:\n" + truncate(evidence, 2500)

    llm = get_llm(state.get("backend"), temperature=0.3)
    try:
        findings = llm.invoke(
            SYNTH_PROMPT.format(
                brief=state.get("brief", ""),
                findings=state.get("findings") or "(none yet)",
                new_material=material,
            )
        ).content
    except Exception:
        findings = state.get("findings") or material
    return {"findings": findings}


def reflect_node(state: AgentState) -> dict:
    loop = state.get("loop_count", 0) + 1
    max_loops = state.get("max_loops", settings.max_loops)

    # Hard stop at the loop budget.
    if loop >= max_loops:
        return {"loop_count": loop, "complete": True, "gaps": ""}

    llm = get_llm(state.get("backend"), temperature=0.2)
    try:
        data = extract_json(
            llm.invoke(
                REFLECT_PROMPT.format(
                    brief=state.get("brief", ""),
                    loop=loop,
                    max_loops=max_loops,
                    findings=state.get("findings") or "(none)",
                )
            ).content
        )
    except Exception:
        data = {}

    complete = bool(data.get("complete"))
    next_query = (data.get("next_query") or "").strip()
    gap = (data.get("gap") or "").strip()

    update: dict = {"loop_count": loop, "complete": complete, "gaps": gap}
    if not complete and next_query:
        update["current_query"] = next_query
        update["queries"] = state.get("queries", []) + [next_query]
    elif not complete and not next_query:
        update["complete"] = True  # nothing more to ask
    return update


def _format_papers(papers: list[dict]) -> str:
    if not papers:
        return "(none)"
    return "\n".join(
        f"- {p.get('title')} ({p.get('year')}) — {p.get('url') or p.get('pdf_url') or ''}"
        for p in papers[:15]
    )


def _format_datasets(datasets: list[dict]) -> str:
    if not datasets:
        return "(none)"
    return "\n".join(f"- {d.get('title')} — {d.get('url')}" for d in datasets[:12])


def finalize_node(state: AgentState) -> dict:
    llm = get_llm(state.get("backend"), temperature=0.4)
    try:
        report = llm.invoke(
            FINALIZE_PROMPT.format(
                brief=state.get("brief", ""),
                findings=state.get("findings") or "(no synthesis)",
                papers=_format_papers(state.get("papers", [])),
                datasets=_format_datasets(state.get("datasets", [])),
            )
        ).content
    except Exception as exc:
        report = (
            f"(report generation failed: {exc})\n\n"
            f"Findings:\n{state.get('findings', '')}\n\n"
            f"Papers:\n{_format_papers(state.get('papers', []))}"
        )

    # Persist the report to disk.
    try:
        import re
        from datetime import datetime

        from .config import REPORTS_DIR

        slug = re.sub(r"[^a-z0-9]+", "-", (state.get("query", "report")).lower())[:50].strip("-")
        path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
        path.write_text(report, encoding="utf-8")
        report += f"\n\n_(saved to {path})_"
    except Exception:
        pass

    return {"report": report, "messages": [AIMessage(content=report)]}


def should_continue(state: AgentState) -> str:
    if state.get("complete") or state.get("loop_count", 0) >= state.get(
        "max_loops", settings.max_loops
    ):
        return "finalize"
    return "act"
