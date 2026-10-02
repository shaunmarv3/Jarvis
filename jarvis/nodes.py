"""LangGraph node functions for the orchestrator-worker research flow.

clarify -> (ask -> brief)? -> plan(lead) -> confirm(human) -> fanout(subagents)
        -> review(lead) -> (followup wave)? -> report(lead) -> cite(deterministic)

The lead (a stronger model on DeepSeek) classifies the query and writes rich
delegations; subagents (a cheaper model) run real tool loops in isolated contexts,
registering every result in a shared source registry; the lead reviews for gaps,
optionally dispatches ONE follow-up wave, then writes the report with [S#] tags that
`cite_node` turns into numbered citations straight from the registry.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .config import REPORTS_DIR, RUNS_DIR, default_budget, lead_iteration_enabled, settings, subagent_ceiling
from .events import emit
from .llm import get_llm
from .prompts import BRIEF_PROMPT, CLARIFY_PROMPT, PLAN_PROMPT, REPORT_PROMPT, REVIEW_PROMPT
from .sources import SourceRegistry, finalize_citations, render_item
from .state import AgentState
from .subagent import run_subagent
from .utils import extract_json, truncate

SOURCE_KINDS = ("academic", "web", "datasets", "code", "community")
_TAG = re.compile(r"S\d+")


def _today() -> str:
    return date.today().isoformat()


def _slug(text: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "report").lower()).strip("-")[:n] or "report"


# --------------------------------------------------------------------------- clarify


def clarify_node(state: AgentState) -> dict:
    """Classify intent, draft a PROVISIONAL brief, and (only if vague) propose clarifying questions."""
    query = state.get("query") or ""
    llm = get_llm(state.get("backend"), temperature=0.2, role="lead", max_tokens=2048, thinking=False)
    try:
        data = extract_json(llm.invoke(CLARIFY_PROMPT.format(query=query, date=_today())).content)
    except Exception as exc:
        emit(f"[yellow]  ! clarify failed ({truncate(str(exc), 100)}) — using your request as the brief[/]")
        data = {}

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
        "run_id": f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(query, 40)}",
        "intent": data.get("intent") or "find_papers",
        "brief": data.get("brief") or f"Research the user's request: {query}",
        "current_query": data.get("query") or query,
        "clarify_questions": questions,
        "clarify_answers": "",
        "plan": [],
        "subagent_reports": [],
        "sources": {},
        "followups": [],
        "gaps": [],
        "followup_done": False,
        "papers": [],
        "report": "",
    }


def after_clarify(state: AgentState) -> str:
    return "ask" if state.get("clarify_questions") else "plan"


def ask_node(state: AgentState) -> dict:
    """Human-in-the-loop: pause so the CLI can collect answers to the clarifying questions."""
    decision = interrupt({"type": "clarify_questions", "questions": state.get("clarify_questions", [])})
    answers = decision.get("answers", "") if isinstance(decision, dict) else (decision or "")
    return {"clarify_answers": (answers or "").strip()}


def brief_node(state: AgentState) -> dict:
    """Refine the provisional brief using the user's answers (skipped-everything keeps it as-is)."""
    answers = (state.get("clarify_answers") or "").strip()
    if not answers:
        return {}
    llm = get_llm(state.get("backend"), temperature=0.2, role="lead", max_tokens=2048, thinking=False)
    try:
        data = extract_json(llm.invoke(BRIEF_PROMPT.format(
            query=state.get("query", ""), brief=state.get("brief", ""), answers=answers)).content)
    except Exception:
        data = {}
    return {
        "brief": (data.get("brief") or "").strip() or state.get("brief", ""),
        "current_query": (data.get("query") or "").strip() or state.get("current_query", ""),
    }


# --------------------------------------------------------------------------- plan


def normalize_spec(s: dict, backend: str | None, fallback_objective: str = "") -> dict | None:
    """Validate/default one delegation spec from the lead. Returns None if unusable."""
    if not isinstance(s, dict):
        return None
    objective = str(s.get("objective") or fallback_objective).strip()
    if not objective:
        return None
    kq = s.get("key_questions") or []
    if isinstance(kq, str):
        kq = [kq]
    sources = s.get("sources") or []
    if isinstance(sources, str):
        sources = [x.strip() for x in sources.split(",")]
    aliases = {"papers": "academic", "paper": "academic", "models": "datasets", "models-data": "datasets",
               "dataset": "datasets", "github": "code", "repos": "code", "hn": "community", "reddit": "community"}
    sources = [aliases.get(str(x).lower().strip(), str(x).lower().strip()) for x in sources]
    sources = [x for i, x in enumerate(sources) if x in SOURCE_KINDS and x not in sources[:i]] or ["academic", "web"]
    try:
        budget = int(s.get("tool_budget") or default_budget(backend))
    except (TypeError, ValueError):
        budget = default_budget(backend)
    return {
        "objective": objective,
        "key_questions": [str(q).strip() for q in kq if str(q).strip()][:3] or [objective],
        "sub_query": str(s.get("sub_query") or objective).strip(),
        "sources": sources,
        "tool_budget": max(3, min(budget, settings.subagent_hard_cap)),
        "output_format": str(s.get("output_format") or "concrete findings with [S#] citations").strip(),
        "boundaries": str(s.get("boundaries") or "").strip(),
        "tools": str(s.get("tools") or "").strip(),
    }


def plan_node(state: AgentState) -> dict:
    """LEAD: classify the query, then decompose it into delegations (count scales to complexity)."""
    backend = state.get("backend")
    ceiling = subagent_ceiling(backend)
    llm = get_llm(backend, temperature=0.2, role="lead", max_tokens=6144)
    try:
        data = extract_json(llm.invoke(PLAN_PROMPT.format(
            query=state.get("query", ""), brief=state.get("brief", ""),
            max_subagents=ceiling, date=_today())).content)
    except Exception as exc:
        emit(f"[yellow]  ! planning failed ({truncate(str(exc), 100)}) — using one subagent[/]")
        data = {}

    plan = [p for p in (normalize_spec(s, backend) for s in (data.get("subagents") or [])) if p][:ceiling]
    if not plan:  # never leave the lead without a plan
        plan = [normalize_spec({"objective": state.get("brief") or state.get("query", ""),
                                "sub_query": state.get("current_query") or state.get("query", ""),
                                "sources": ["academic", "web"]}, backend)]
    qtype = data.get("query_type") if data.get("query_type") in {"straightforward", "depth_first", "breadth_first"} else ""
    return {"plan": plan, "query_type": qtype, "max_subagents": ceiling, "replan": False}


def confirm_node(state: AgentState) -> dict:
    """Human-in-the-loop: show the brief AND the delegation plan, wait for approval."""
    decision = interrupt({
        "type": "confirm_plan",
        "intent": state.get("intent"),
        "brief": state.get("brief"),
        "plan": state.get("plan", []),
        "query_type": state.get("query_type", ""),
    })
    if isinstance(decision, dict):
        decision = decision.get("text", "")
    text = (decision or "").strip()
    if text and text.lower() not in {"y", "yes", "go", "ok", "okay", "proceed", "sure"}:
        return {"brief": text, "replan": True, "messages": [HumanMessage(content=f"(brief refined) {text}")]}
    return {"replan": False}


def after_confirm(state: AgentState) -> str:
    return "plan" if state.get("replan") else "fanout"


# --------------------------------------------------------------------------- research


def dispatch(specs: list[dict], state: AgentState, start_idx: int = 1) -> tuple[list[dict], dict]:
    """Run subagents (parallel on DeepSeek, sequential on one local GPU)."""
    backend = state.get("backend")
    registry = SourceRegistry(state.get("sources") or {})
    total = start_idx - 1 + len(specs)
    parallel = backend == "deepseek" and settings.parallel_subagents and len(specs) > 1
    emit(f"[dim]· dispatching {len(specs)} subagent(s) [{'parallel' if parallel else 'sequential'}][/]")

    def go(item):
        i, spec = item
        return run_subagent(spec, backend, registry, i, total, state.get("brief", ""), state.get("run_id", ""))

    items = list(enumerate(specs, start_idx))
    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(specs), 8)) as ex:
            reports = list(ex.map(go, items))
    else:
        reports = [go(it) for it in items]
    return reports, registry.to_dict()


def fanout_node(state: AgentState) -> dict:
    reports, sources = dispatch(state.get("plan", []), state)
    return {"subagent_reports": reports, "sources": sources}


def _briefings(reports: list[dict], limit: int = 2600) -> str:
    return "\n\n".join(
        f"### Subagent {i}: {r.get('objective', '')}\n{truncate(r.get('findings', ''), limit)}"
        for i, r in enumerate(reports, 1)
    ) or "(no subagent findings)"


def review_node(state: AgentState) -> dict:
    """LEAD: check the findings against the brief; maybe request ONE follow-up wave."""
    backend = state.get("backend")
    if state.get("followup_done") or not lead_iteration_enabled(backend) or settings.max_followup_subagents <= 0:
        return {"followups": []}
    llm = get_llm(backend, temperature=0.2, role="lead", max_tokens=6144)
    try:
        data = extract_json(llm.invoke(REVIEW_PROMPT.format(
            brief=state.get("brief", ""), briefings=_briefings(state.get("subagent_reports", []), 1800),
            max_followups=settings.max_followup_subagents)).content)
    except Exception:
        return {"followups": []}
    if data.get("sufficient", True) and not data.get("followups"):
        return {"followups": [], "gaps": data.get("gaps") or []}
    specs = [p for p in (normalize_spec(s, backend) for s in (data.get("followups") or [])) if p]
    specs = specs[: settings.max_followup_subagents]
    if specs:
        emit(f"[dim]· lead found gaps → {len(specs)} follow-up subagent(s): "
             f"{'; '.join(truncate(s['objective'], 50) for s in specs)}[/]")
    return {"followups": specs, "gaps": data.get("gaps") or []}


def after_review(state: AgentState) -> str:
    return "followup" if state.get("followups") else "report"


def followup_node(state: AgentState) -> dict:
    prior = state.get("subagent_reports", [])
    reports, sources = dispatch(state.get("followups", []), state, start_idx=len(prior) + 1)
    return {"subagent_reports": prior + reports, "sources": sources, "followup_done": True, "followups": []}


# --------------------------------------------------------------------------- report


def _catalog(state: AgentState, cap: int) -> str:
    """Sources the subagents actually cited first, then their other finds, up to `cap`."""
    sources = state.get("sources") or {}
    reports = state.get("subagent_reports", [])
    cited = []
    for r in reports:
        for sid in _TAG.findall(r.get("findings", "")):
            if sid in sources and sid not in cited:
                cited.append(sid)
    extra = [s for r in reports for s in r.get("sources", []) if s in sources and s not in cited]
    ordered = list(dict.fromkeys(cited + extra))[:cap]
    return "\n".join(render_item(s, sources[s]["kind"], sources[s]["item"]) for s in ordered) or "(none)"


def report_node(state: AgentState) -> dict:
    """LEAD: write the final report from the subagent reports, keeping [S#] tags."""
    backend = state.get("backend")
    cap = 80 if backend == "deepseek" else 25
    limit = 3000 if backend == "deepseek" else 1400
    prompt = REPORT_PROMPT.format(
        date=_today(), query=state.get("query", ""), brief=state.get("brief", ""),
        briefings=_briefings(state.get("subagent_reports", []), limit), catalog=_catalog(state, cap))
    report, error = "", ""
    # If thinking consumes the whole output budget the answer comes back empty — retry without it.
    for thinking in (None, False):
        try:
            llm = get_llm(backend, temperature=0.3, role="lead", max_tokens=settings.report_max_tokens,
                          thinking=thinking)
            report = (llm.invoke(prompt).content or "").strip()
        except Exception as exc:
            error = str(exc)
        if report:
            break
        emit("[yellow]  ! report came back empty — retrying without extended thinking[/]")
    if not report:
        report = (f"# Research findings\n\n(report writing failed: {error or 'empty response'})\n\n"
                  + _briefings(state.get("subagent_reports", [])))
    return {"report": report}


def cite_node(state: AgentState) -> dict:
    """Turn [S#] tags into numbered citations from the registry; save report + artifacts."""
    sources = state.get("sources") or {}
    final, cited, stats = finalize_citations(state.get("report", ""), sources)

    cited_papers = [c["item"] for c in cited if c["kind"] == "paper"]
    others = [e["item"] for e in sources.values() if e["kind"] == "paper" and e["item"] not in cited_papers]
    papers = (cited_papers + others)[: max(settings.keep_top_papers, len(cited_papers))]

    run_id = state.get("run_id") or f"{datetime.now():%Y%m%d-%H%M%S}"
    try:
        path = REPORTS_DIR / f"{run_id}.md"
        path.write_text(final, encoding="utf-8")
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.md").write_text(final, encoding="utf-8")
        (run_dir / "sources.json").write_text(json.dumps(sources, indent=1, default=str), encoding="utf-8")
        (run_dir / "plan.json").write_text(json.dumps({
            "query": state.get("query"), "brief": state.get("brief"), "query_type": state.get("query_type"),
            "plan": state.get("plan"), "gaps": state.get("gaps"), "citation_stats": stats,
            "subagents": [{k: r.get(k) for k in ("objective", "tool_calls", "turns", "tools")}
                          for r in state.get("subagent_reports", [])],
        }, indent=1, default=str), encoding="utf-8")
        final += f"\n\n_(saved to {path})_"
    except Exception:
        pass
    return {"report": final, "papers": papers, "citation_stats": stats, "messages": [AIMessage(content=final)]}
