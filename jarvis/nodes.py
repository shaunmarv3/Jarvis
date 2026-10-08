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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from .config import REPORTS_DIR, RUNS_DIR, default_budget, lead_iteration_enabled, settings, subagent_ceiling
from .events import emit, note
from .llm import get_llm, input_char_budget
from .memory import prompt_block as _memory_block
from .prompts import (
    BRIEF_PROMPT,
    CLARIFY_PROMPT,
    PLAN_PROMPT,
    PRIOR_FOR_CLARIFY,
    PRIOR_FOR_PLAN,
    PRIOR_FOR_REPORT,
    REPORT_PROMPT,
    REVIEW_PROMPT,
)
from .sources import SourceRegistry, finalize_citations, render_item
from .state import AgentState
from .subagent import run_subagent
from .utils import extract_json, truncate, write_json_atomic

SOURCE_KINDS = ("academic", "web", "datasets", "code", "community")
_TAG = re.compile(r"S\d+")


def _today() -> str:
    return date.today().isoformat()


def _slug(text: str, n: int = 50) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "report").lower()).strip("-")[:n] or "report"


def _as_bool(value, default: bool) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return default if value is None else bool(value)


def _fit(build, levels: list[tuple], budget: int | None, what: str) -> str:
    """Build a prompt at the richest evidence level that fits `budget` characters.

    Small local models get a fixed context window, and an over-long prompt is trimmed
    silently by the server. Trimming our own evidence first keeps the instructions intact.
    """
    prompt = ""
    for i, level in enumerate(levels):
        prompt = build(*level)
        if budget is None or len(prompt) <= budget:
            if i:
                note(f"Trimmed {what} to fit the {settings.ollama_num_ctx}-token context window")
            return prompt
    note(f"{what[:1].upper()}{what[1:]} is still longer than the context window after trimming; "
         "the model may miss some of it", "warn")
    return prompt


def _memory(state: AgentState) -> str:
    """The user's standing preferences (data/memory.md) as a prompt section, or ''."""
    return _memory_block(state.get("memory") or "", state.get("backend"))


def short_title(text: str, n: int = 44) -> str:
    """A label-sized version of a long objective, cut at a word boundary."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) <= n:
        return t
    return (t[: n - 1].rsplit(" ", 1)[0] or t[: n - 1]).rstrip(" ,.;:-") + "…"


# --------------------------------------------------------------------------- conversation memory


def build_prior(state: dict) -> dict:
    """Turn a finished run into the context a follow-up question needs: the request, the
    report body with its [n] citations mapped back to [S#] ids, and the cited sources."""
    report = state.get("report") or ""
    if not report.strip():
        return {}
    sids = state.get("cited_sids") or []
    sources = state.get("sources") or {}
    body = report.split("\n## Sources\n", 1)[0]
    body = re.sub(r"\s*_\(saved to [^)]*\)_\s*$", "", body)

    def back(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{sids[n - 1]}]" if 1 <= n <= len(sids) else m.group(0)

    return {
        "query": state.get("query", ""),
        "brief": state.get("brief", ""),
        "findings": re.sub(r"\[(\d+)\]", back, body).strip(),
        "sources": {s: sources[s] for s in sids if s in sources},
    }


def _prior_block(state: AgentState, template: str, deepseek_chars: int, ollama_chars: int) -> str:
    prior = state.get("prior") or {}
    if not prior.get("findings") or not state.get("follow_up", True):
        return ""
    n = deepseek_chars if state.get("backend") == "deepseek" else ollama_chars
    return template.format(query=prior.get("query", ""), findings=truncate(prior["findings"], n))


# --------------------------------------------------------------------------- clarify


def clarify_node(state: AgentState) -> dict:
    """Classify intent, draft a PROVISIONAL brief, decide whether this follows up on the
    previous run, and (only if vague) propose clarifying questions."""
    query = state.get("query") or ""
    prior = state.get("prior") or {}
    llm = get_llm(state.get("backend"), temperature=0.2, role="lead", max_tokens=2048, thinking=False)
    try:
        data = extract_json(llm.invoke(CLARIFY_PROMPT.format(
            query=query, date=_today(), prior=_prior_block(state, PRIOR_FOR_CLARIFY, 1500, 700),
            memory=_memory(state))).content)
    except Exception as exc:
        note(f"Couldn't analyse the request ({truncate(str(exc), 100)}); using it as the brief", "warn")
        data = {}
    follow_up = bool(prior.get("findings")) and _as_bool(data.get("follow_up"), False)
    if follow_up:
        note(f"Building on your last question: \"{truncate(prior.get('query', ''), 70)}\" "
             f"({len(prior.get('sources') or {})} sources carried over)")

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
        "title": short_title(str(data.get("title") or "").strip(), 40),
        "intent": data.get("intent") or "find_papers",
        "brief": data.get("brief") or f"Research the user's request: {query}",
        "current_query": data.get("query") or query,
        "clarify_questions": questions,
        "clarify_answers": "",
        "plan": [],
        "subagent_reports": [],
        # A follow-up starts from the previous run's cited sources (same ids), so the new
        # report can cite them and subagents that re-find them get the existing id.
        "sources": dict(prior.get("sources") or {}) if follow_up else {},
        "follow_up": follow_up,
        "prior": prior if follow_up else {},
        "followups": [],
        "gaps": [],
        "followup_done": False,
        "papers": [],
        "report": "",
        "cited_sids": [],
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
        "title": short_title(str(s.get("title") or "").strip() or objective),
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
            prior=_prior_block(state, PRIOR_FOR_PLAN, 2500, 900), memory=_memory(state),
            max_subagents=ceiling, date=_today())).content)
    except Exception as exc:
        note(f"Planning failed ({truncate(str(exc), 100)}); using one research agent", "warn")
        data = {}

    plan = [p for p in (normalize_spec(s, backend) for s in (data.get("subagents") or [])) if p][:ceiling]
    if not plan:  # never leave the lead without a plan
        plan = [normalize_spec({"objective": state.get("brief") or state.get("query", ""),
                                "sub_query": state.get("current_query") or state.get("query", ""),
                                "sources": ["academic", "web"]}, backend)]
    if settings.single_agent and len(plan) > 1:
        plan = [merge_specs(plan, state.get("brief") or state.get("query", ""))]
    qtype = data.get("query_type") if data.get("query_type") in {"straightforward", "depth_first", "breadth_first"} else ""
    return {"plan": plan, "query_type": qtype, "max_subagents": ceiling, "replan": False}


def merge_specs(plan: list[dict], brief: str) -> dict:
    """Eval ablation: one subagent that owns the whole plan — every key question and source
    kind, with the largest budget allowed — so "several subagents vs one" is a fair test."""
    kq = list(dict.fromkeys(q for s in plan for q in s["key_questions"]))
    return {
        "title": "Whole research brief",
        "objective": brief,
        "key_questions": kq[:8],
        "sub_query": plan[0]["sub_query"],
        "sources": list(dict.fromkeys(k for s in plan for k in s["sources"])),
        "tool_budget": settings.subagent_hard_cap,
        "output_format": "a complete report that answers every key question with [S#] citations",
        "boundaries": "",
        "tools": "",
    }


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


def _wave_path(state: AgentState, start_idx: int):
    run_id = state.get("run_id")
    return RUNS_DIR / run_id / f"wave_{start_idx}.json" if run_id else None


def _load_wave(path, specs: list[dict]) -> dict | None:
    """Progress saved by an interrupted run of this same wave, or None."""
    if not path or not path.exists():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if saved.get("objectives") != [s.get("objective") for s in specs]:
        return None  # a different plan: start over
    return saved


def _failed_report(spec: dict, exc: Exception) -> dict:
    return {"objective": spec.get("objective", ""), "findings": f"(this subagent failed: {truncate(str(exc), 200)})",
            "sources": [], "tools": [], "tool_calls": 0, "turns": 0}


def dispatch(specs: list[dict], state: AgentState, start_idx: int = 1) -> tuple[list[dict], dict]:
    """Run subagents (parallel on DeepSeek, sequential on one local GPU).

    Each finished subagent is saved to runs/<id>/wave_<n>.json together with the source
    registry, so if the process dies mid-wave, re-running this node (/resume) reuses the
    finished subagents — same results, same source ids — and only runs the rest.
    """
    backend = state.get("backend")
    path = _wave_path(state, start_idx)
    saved = _load_wave(path, specs)
    registry = SourceRegistry(saved["registry"] if saved else (state.get("sources") or {}))
    done: dict[int, dict] = {int(k): v for k, v in ((saved or {}).get("done") or {}).items()}
    total = start_idx - 1 + len(specs)
    pending = [(i, s) for i, s in enumerate(specs, start_idx) if i not in done]
    parallel = backend == "deepseek" and settings.parallel_subagents and len(pending) > 1
    followup = start_idx > 1
    before = len(state.get("sources") or {})
    emit("wave_start", agents=[(i, s.get("title") or short_title(s.get("objective", ""))) for i, s in pending],
         parallel=parallel, followup=followup, reused=len(done))
    limit = settings.wave_time_limit_min
    deadline = time.monotonic() + limit * 60 if limit and limit > 0 else None
    lock = threading.Lock()
    failed: dict[int, dict] = {}

    def save() -> None:
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, {"objectives": [s.get("objective") for s in specs],
                                     "registry": registry.to_dict(), "done": {str(k): v for k, v in done.items()}},
                              indent=None)
        except Exception:
            pass  # progress saving is best-effort; the run itself must not fail on it

    def go(item):
        i, spec = item
        try:
            result = run_subagent(spec, backend, registry, i, total, state.get("brief", ""),
                                  state.get("run_id", ""), deadline, memory=_memory(state))
        except Exception as exc:  # one broken subagent must not sink its siblings
            emit("agent_warn", idx=i, text=f"failed: {truncate(str(exc), 120)}")
            emit("agent_done", idx=i, searches=0, reads=0, sources=0, failed=True)
            with lock:
                failed[i] = _failed_report(spec, exc)  # not saved as done, so /resume retries it
            return
        with lock:
            done[i] = result
            save()

    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(pending), 8)) as ex:
            list(ex.map(go, pending))
    else:
        for item in pending:
            go(item)
    reports = [done.get(i) or failed[i] for i in range(start_idx, start_idx + len(specs))]
    sources = registry.to_dict()
    emit("wave_done", agents=len(reports), followup=followup,
         searches=sum(r.get("searches", 0) for r in reports), reads=sum(r.get("reads", 0) for r in reports),
         sources=len(sources), new_sources=max(0, len(sources) - before))
    if path and not failed:
        try:
            path.unlink(missing_ok=True)  # the graph checkpoint now holds the results
        except Exception:
            pass
    return reports, sources


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
    reports = state.get("subagent_reports", [])
    prompt = _fit(
        lambda limit: REVIEW_PROMPT.format(brief=state.get("brief", ""), briefings=_briefings(reports, limit),
                                           max_followups=settings.max_followup_subagents),
        [(1800,), (1200,), (800,), (500,), (300,)], input_char_budget(backend, 1536), "the review prompt")
    try:
        data = extract_json(llm.invoke(prompt).content)
    except Exception:
        return {"followups": []}
    gaps = data.get("gaps") or []
    raw = data.get("followups") or []
    # The lead's verdict wins: "sufficient" means no follow-up wave, even if it also listed
    # some (nice-to-haves). If it gave no verdict, listing follow-ups means it wants them.
    if _as_bool(data.get("sufficient"), default=not raw) or not raw:
        return {"followups": [], "gaps": gaps}
    specs = [p for p in (normalize_spec(s, backend) for s in raw) if p]
    specs = specs[: settings.max_followup_subagents]
    if specs:
        emit("gaps", titles=[s["title"] for s in specs])
    return {"followups": specs, "gaps": gaps}


def after_review(state: AgentState) -> str:
    return "followup" if state.get("followups") else "report"


def followup_node(state: AgentState) -> dict:
    prior = state.get("subagent_reports", [])
    reports, sources = dispatch(state.get("followups", []), state, start_idx=len(prior) + 1)
    return {"subagent_reports": prior + reports, "sources": sources, "followup_done": True, "followups": []}


# --------------------------------------------------------------------------- report


def _catalog(state: AgentState, cap: int) -> str:
    """Sources the subagents actually cited first, then (on a follow-up) the sources the
    previous report cited, then the subagents' other finds, up to `cap`."""
    sources = state.get("sources") or {}
    reports = state.get("subagent_reports", [])
    cited = []
    for r in reports:
        for sid in _TAG.findall(r.get("findings", "")):
            if sid in sources and sid not in cited:
                cited.append(sid)
    carried = [s for s in ((state.get("prior") or {}).get("sources") or {}) if s in sources] \
        if state.get("follow_up") else []
    extra = [s for r in reports for s in r.get("sources", []) if s in sources and s not in cited]
    ordered = list(dict.fromkeys(cited + carried + extra))[:cap]
    return "\n".join(render_item(s, sources[s]["kind"], sources[s]["item"]) for s in ordered) or "(none)"


def report_node(state: AgentState) -> dict:
    """LEAD: write the final report from the subagent reports, keeping [S#] tags."""
    backend = state.get("backend")
    reports = state.get("subagent_reports", [])

    def build(cap: int, limit: int, prior_chars: int) -> str:
        prior = _prior_block(state, PRIOR_FOR_REPORT, prior_chars, prior_chars)
        return REPORT_PROMPT.format(
            date=_today(), query=state.get("query", ""), brief=state.get("brief", ""), prior=prior,
            memory=_memory(state), briefings=_briefings(reports, limit), catalog=_catalog(state, cap))

    if backend == "deepseek":
        levels = [(80, 3000, 3000)]
    else:  # shrink evidence step by step until it fits next to a full-length report
        levels = [(25, 1400, 1200), (18, 1100, 900), (12, 800, 600), (8, 500, 400), (5, 300, 250)]
    prompt = _fit(build, levels, input_char_budget(backend, settings.report_max_tokens), "the report evidence")
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
        note("The report came back empty; retrying without extended thinking", "warn")
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
        write_json_atomic(run_dir / "sources.json", sources, indent=1)
        write_json_atomic(run_dir / "plan.json", {
            "query": state.get("query"), "brief": state.get("brief"), "query_type": state.get("query_type"),
            "follow_up_of": (state.get("prior") or {}).get("query") if state.get("follow_up") else None,
            "plan": state.get("plan"), "gaps": state.get("gaps"), "citation_stats": stats,
            "subagents": [{k: r.get(k) for k in ("objective", "tool_calls", "turns", "tools")}
                          for r in state.get("subagent_reports", [])],
        }, indent=1)
        final += f"\n\n_(saved to {path})_"
    except Exception:
        pass
    return {"report": final, "papers": papers, "citation_stats": stats, "cited_sids": [c["sid"] for c in cited],
            "messages": [AIMessage(content=final)]}
