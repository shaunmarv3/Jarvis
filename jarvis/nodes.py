"""LangGraph node functions for the orchestrator-worker research flow.

Flow: clarify -> plan(lead) -> confirm(human) -> fanout(subagents) -> synthesize
      -> finalize -> cite.

The reflective search loop lives *inside* each subagent (see subagent.py); the lead
agent decomposes the brief into independent subagents, fans them out (sequentially on
Ollama, in parallel on DeepSeek), then merges and cites their findings.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .config import settings, subagent_ceiling
from .embeddings import rank_by_relevance
from .events import emit
from .llm import get_llm
from .prompts import (
    BRIEF_PROMPT,
    CITE_PROMPT,
    CLARIFY_PROMPT,
    FINALIZE_PROMPT,
    MERGE_PROMPT,
    PLAN_PROMPT,
)
from .state import AgentState
from .subagent import run_subagent
from .utils import dedup_papers, extract_json, truncate


def clarify_node(state: AgentState) -> dict:
    """Classify intent, draft a PROVISIONAL brief, and (only if vague) propose clarifying questions."""
    query = state.get("query") or ""
    llm = get_llm(state.get("backend"), temperature=0.2)
    raw = llm.invoke(CLARIFY_PROMPT.format(query=query)).content
    data = extract_json(raw)

    intent = data.get("intent") or "find_papers"
    brief = data.get("brief") or f"Research the user's request: {query}"
    first_query = data.get("query") or query

    # Keep only well-formed questions (text + up to 4 options), cap at 3.
    questions: list[dict] = []
    for q in (data.get("questions") or [])[:3]:
        if not isinstance(q, dict):
            continue
        text = (q.get("question") or "").strip()
        if not text:
            continue
        opts = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()][:4]
        questions.append({"question": text, "options": opts})

    return {
        "intent": intent,
        "brief": brief,
        "current_query": first_query,
        "queries": [first_query],
        "clarify_questions": questions,
        "clarify_answers": "",
        "plan": [],
        "subagent_reports": [],
        "papers": [],
        "datasets": [],
        "web": [],
        "findings": "",
        "complete": False,
    }


def after_clarify(state: AgentState) -> str:
    """Route: ask the user the clarifying questions if there are any, else plan straight away."""
    return "ask" if state.get("clarify_questions") else "plan"


def ask_node(state: AgentState) -> dict:
    """Human-in-the-loop: pause so the CLI can collect answers to the clarifying questions.

    The CLI resumes with `Command(resume=<answers>)`, where answers is a formatted
    "Q: ... | A: ..." string (or a dict carrying one). Empty answers = user skipped.
    """
    decision = interrupt(
        {
            "type": "clarify_questions",
            "questions": state.get("clarify_questions", []),
        }
    )
    answers = decision.get("answers", "") if isinstance(decision, dict) else (decision or "")
    return {"clarify_answers": (answers or "").strip()}


def brief_node(state: AgentState) -> dict:
    """Refine the provisional brief using the user's answers (skipped-everything keeps it as-is)."""
    answers = (state.get("clarify_answers") or "").strip()
    if not answers:
        return {}  # nothing clarified — provisional brief stands
    llm = get_llm(state.get("backend"), temperature=0.2)
    try:
        data = extract_json(
            llm.invoke(
                BRIEF_PROMPT.format(
                    query=state.get("query", ""),
                    brief=state.get("brief", ""),
                    answers=answers,
                )
            ).content
        )
    except Exception:
        data = {}
    brief = (data.get("brief") or "").strip() or state.get("brief", "")
    first_query = (data.get("query") or "").strip() or state.get("current_query", "")
    return {"brief": brief, "current_query": first_query, "queries": [first_query]}


def plan_node(state: AgentState) -> dict:
    """LEAD agent: decompose the brief into independent subagents (count scales to complexity)."""
    backend = state.get("backend")
    ceiling = subagent_ceiling(backend)
    llm = get_llm(backend, temperature=0.2)
    try:
        data = extract_json(
            llm.invoke(
                PLAN_PROMPT.format(
                    query=state.get("query", ""),
                    brief=state.get("brief", ""),
                    max_subagents=ceiling,
                )
            ).content
        )
    except Exception:
        data = {}

    plan: list[dict] = []
    for s in (data.get("subagents") or [])[:ceiling]:
        objective = (s.get("objective") or "").strip()
        if not objective:
            continue
        plan.append(
            {
                "objective": objective,
                "sub_query": (s.get("sub_query") or objective).strip(),
                "tools": (s.get("tools") or "").strip(),
            }
        )

    # Fallback: never leave the lead with no plan — research the brief with one subagent.
    if not plan:
        plan = [
            {
                "objective": state.get("brief") or state.get("query", ""),
                "sub_query": state.get("current_query") or state.get("query", ""),
                "tools": "",
            }
        ]
    return {"plan": plan, "max_subagents": ceiling, "replan": False}


def confirm_node(state: AgentState) -> dict:
    """Human-in-the-loop: show the brief AND the subagent split, wait for approval.

    The CLI resumes with `Command(resume=<text>)`. A plain yes/go proceeds to fan-out;
    any other text is treated as an edited brief and triggers a re-plan.
    """
    decision = interrupt(
        {
            "type": "confirm_plan",
            "intent": state.get("intent"),
            "brief": state.get("brief"),
            "plan": state.get("plan", []),
        }
    )
    if isinstance(decision, dict):
        decision = decision.get("text", "")
    text = (decision or "").strip()
    if text and text.lower() not in {"y", "yes", "go", "ok", "okay", "proceed", "sure"}:
        return {
            "brief": text,
            "replan": True,
            "messages": [HumanMessage(content=f"(brief refined) {text}")],
        }
    return {"replan": False}


def after_confirm(state: AgentState) -> str:
    """Route: re-plan if the user edited the brief, otherwise fan out the subagents."""
    return "plan" if state.get("replan") else "fanout"


def fanout_node(state: AgentState) -> dict:
    """Dispatch the subagents — sequential on Ollama, parallel (threads) on DeepSeek."""
    plan = state.get("plan", [])
    backend = state.get("backend")
    total = len(plan)
    parallel = backend == "deepseek" and settings.parallel_subagents and total > 1
    emit(f"[dim]· dispatching {total} subagent(s) [{'parallel' if parallel else 'sequential'}][/]")

    if parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(total, 8)) as ex:
            reports = list(
                ex.map(
                    lambda it: run_subagent(it[1], backend, it[0], total),
                    list(enumerate(plan, 1)),
                )
            )
    else:
        reports = [run_subagent(s, backend, i, total) for i, s in enumerate(plan, 1)]

    # Aggregate every subagent's haul.
    all_papers: list[dict] = []
    all_datasets: list[dict] = []
    all_web: list[dict] = []
    tools: list[str] = []
    for rep in reports:
        all_papers.extend(rep.get("papers", []) or [])
        all_datasets.extend(rep.get("datasets", []) or [])
        all_web.extend(rep.get("web", []) or [])
        tools.extend(rep.get("tools", []) or [])

    papers = dedup_papers(all_papers)
    # Relevance filter against the original request (drop off-topic noise).
    ranked = rank_by_relevance(
        state.get("query", ""),
        papers,
        top_k=settings.keep_top_papers,
        min_score=settings.relevance_min,
    )
    if not ranked and papers:
        ranked = papers[: settings.keep_top_papers]

    seen: set[str] = set()
    datasets: list[dict] = []
    for d in all_datasets:
        key = (d.get("url") or d.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            datasets.append(d)

    return {
        "subagent_reports": reports,
        "papers": ranked,
        "datasets": datasets,
        "web": all_web,
        "last_tools": tools,
    }


def synthesize_node(state: AgentState) -> dict:
    """LEAD agent: merge the subagent briefings into one synthesis."""
    reports = state.get("subagent_reports", [])
    briefings = "\n\n".join(
        f"### Subagent {i}: {r.get('objective', '')}\n{r.get('findings', '')}"
        for i, r in enumerate(reports, 1)
    ) or "(no subagent findings)"

    llm = get_llm(state.get("backend"), temperature=0.3)
    try:
        findings = llm.invoke(
            MERGE_PROMPT.format(brief=state.get("brief", ""), briefings=briefings)
        ).content
    except Exception:
        findings = briefings
    return {"findings": findings}


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
    """Draft the report from the merged synthesis (citations added by cite_node)."""
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
    return {"report": report}


def cite_node(state: AgentState) -> dict:
    """CITATION agent: pin claims to the real sources retrieved, append a Sources list, persist."""
    sources: list[dict] = []
    for p in state.get("papers", [])[:15]:
        sources.append(
            {"title": p.get("title", ""), "url": p.get("url") or p.get("pdf_url") or ""}
        )
    for w in state.get("web", [])[:8]:
        sources.append({"title": w.get("title", ""), "url": w.get("url") or ""})

    draft = state.get("report", "")
    final = draft
    if sources and draft:
        numbered = "\n".join(
            f"[{i}] {s['title']} — {s['url']}" for i, s in enumerate(sources, 1)
        )
        llm = get_llm(state.get("backend"), temperature=0.2)
        try:
            final = llm.invoke(
                CITE_PROMPT.format(report=draft, sources=numbered)
            ).content
        except Exception:
            final = draft

    # Persist the final, cited report to disk.
    try:
        import re
        from datetime import datetime

        from .config import REPORTS_DIR

        slug = re.sub(r"[^a-z0-9]+", "-", (state.get("query", "report")).lower())[:50].strip("-")
        path = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"
        path.write_text(final, encoding="utf-8")
        final += f"\n\n_(saved to {path})_"
    except Exception:
        pass

    return {"report": final, "messages": [AIMessage(content=final)]}
