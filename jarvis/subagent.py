"""Subagent runner — a genuine agentic research loop for ONE delegated task.

Each subagent keeps its own conversation (its own context window) and loops:
model picks tools -> tools run IN PARALLEL -> results (tagged with source ids) go
back into the conversation -> model reads them, decides the next step ... until it
calls `complete_task`, its tool budget runs out, or searches stop finding anything
new. That observe -> orient -> decide -> act loop is the core of Anthropic's research
subagents; the previous version never showed the model its own results.
"""

from __future__ import annotations

import inspect
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from .config import RUNS_DIR, default_budget, settings
from .events import emit
from .llm import get_llm, input_char_budget
from .prompts import SUBAGENT_FINISH, SUBAGENT_SYSTEM, SUBAGENT_TASK
from .sources import RESULT_KINDS, SourceRegistry, render_item
from .tools import TOOL_FUNCS, TOOL_SCHEMAS
from .utils import truncate


@tool
def complete_task(report: str) -> str:
    """Finish your task: send your final report (with [S#] source ids after claims) to the lead."""
    return ""


# If the model answers from memory instead of searching, kick things off ourselves,
# choosing tools from the delegation's source types (not a hardwired academic set).
_FALLBACK = {
    "academic": ["search_arxiv", "search_openalex", "search_semantic_scholar"],
    "web": ["search_web"],
    "datasets": ["search_datasets", "search_hf_models"],
    "code": ["search_github"],
    "community": ["search_hn"],
}
# Full-text reads get their own allowance so a subagent that spends its search budget in a
# burst of parallel queries can still open the key sources (abstracts are often not enough).
_READ_TOOLS = {"read_paper", "read_web", "inspect_dataset"}
_SID = re.compile(r"^\s*\[?\s*(S\d+)\s*\]?\s*$", re.IGNORECASE)
_MAX_TOOL_TEXT = 6000


def _call(name: str, args: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"text": f"Unknown tool '{name}'. Available: {', '.join(TOOL_FUNCS)}."}
    try:
        params = inspect.signature(fn).parameters
        clean = {k: v for k, v in (args or {}).items() if k in params}
        return fn(**clean)
    except Exception as exc:  # a tool must never crash a subagent
        return {"text": f"Tool {name} failed: {exc}. Try different arguments or another tool."}


def _resolve_source_ids(args: dict, registry: SourceRegistry) -> tuple[dict, str | None]:
    """Let the model pass a source id ("S4") wherever a paper / URL is expected."""
    args = dict(args or {})
    for key in ("paper", "url", "id_or_url"):
        v = args.get(key)
        m = _SID.match(v) if isinstance(v, str) else None
        entry = registry.get(m.group(1)) if m else None
        if not entry:
            continue
        item = entry["item"]
        if key == "paper" or entry["kind"] == "paper":
            ref = ""
            if item.get("source") == "arxiv" or "arxiv.org" in str(item.get("url", "")):
                ref = item.get("url", "")
            ref = ref or item.get("pdf_url") or item.get("doi") or item.get("url") or ""
            args[key] = ref
        else:
            args[key] = item.get("url", "")
        return args, m.group(1).upper()
    return args, None


def execute_tool(name: str, args: dict, registry: SourceRegistry) -> tuple[str, list[str]]:
    """Run one tool, register its results as sources, return (text for the model, sids)."""
    args, sid_ref = _resolve_source_ids(args, registry)
    out = _call(name, args)
    sids: list[str] = []
    lines: list[str] = []
    for kind, key in RESULT_KINDS:
        for item in out.get(key) or []:
            sid = registry.add(kind, item)
            sids.append(sid)
            lines.append(render_item(sid, kind, registry.get(sid)["item"]))

    if lines:
        header = (out.get("text") or "").split("\n", 1)[0]
        text = header + "\n" + "\n".join(lines)
    elif out.get("content"):  # a full-text read: tag it so claims from it are citable
        if not sid_ref:
            ref = args.get("url") or args.get("paper") or ""
            kind = "web" if name == "read_web" else "paper"
            # paper reads carry real metadata, so a bare arXiv id / DOI dedups with search hits
            sid_ref = registry.add(kind, out.get("meta") or {"title": ref, "url": ref})
        sids.append(sid_ref)
        text = f"[{sid_ref}] {out.get('text', '')}"
    else:
        text = out.get("text") or "(no output)"
    return truncate(text, _MAX_TOOL_TEXT), sids


def _fmt_call(name: str, args: dict) -> str:
    return f"{name}({', '.join(f'{k}={truncate(str(v), 40)}' for k, v in (args or {}).items())})"


def _size(msgs: list) -> int:
    return sum(len(str(m.content)) for m in msgs)


def _trim(m: ToolMessage, n: int, note: str) -> ToolMessage:
    text = str(m.content)
    return m if len(text) <= n else ToolMessage(content=text[:n] + f" …[{note}]", tool_call_id=m.tool_call_id)


def _compact(msgs: list, limit_chars: int) -> list:
    """Fit the conversation into `limit_chars`, trimming deliberately instead of letting a
    small local model's server cut the prompt (which can drop the instructions).

    Stage 1 shrinks tool results older than the latest model turn to 700 chars; stage 2 to
    150 chars; stage 3 shares what is left among the latest round's results. The system
    prompt, the task and the model's own messages are never cut.
    """
    if _size(msgs) <= limit_chars:
        return msgs
    ai_idx = [i for i, m in enumerate(msgs) if m.type == "ai"]
    last_ai = ai_idx[-1] if ai_idx else len(msgs)
    out = list(msgs)
    for keep in (700, 150):
        out = [_trim(m, keep, "older result trimmed") if isinstance(m, ToolMessage) and i < last_ai else m
               for i, m in enumerate(out)]
        if _size(out) <= limit_chars:
            return out
    latest = [i for i, m in enumerate(out) if isinstance(m, ToolMessage) and i > last_ai]
    if latest:
        fixed = _size([m for i, m in enumerate(out) if i not in latest])
        share = max(200, (limit_chars - fixed) // len(latest))
        out = [_trim(m, share, "trimmed to fit the context window") if i in latest else m
               for i, m in enumerate(out)]
    return out


def _tools_overhead(tools: list) -> int:
    """Characters the bound tool schemas add to every request (they share the context window)."""
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return len(json.dumps([convert_to_openai_tool(t) for t in tools]))
    except Exception:
        return 6000


def _run_parallel(calls: list, registry: SourceRegistry, deadline: float | None = None) -> dict:
    """Run one turn's tool calls concurrently. A call still running after the timeout is
    abandoned (its thread finishes on its own) and answered with an error the model can
    act on, so one hung API can't stall the subagent."""
    limit = float(settings.tool_timeout)
    if deadline:
        limit = max(5.0, min(limit, deadline - time.monotonic()))
    ex = ThreadPoolExecutor(max_workers=min(4, len(calls)))
    futs = {c["id"]: ex.submit(execute_tool, c["name"], c.get("args") or {}, registry) for c in calls}
    end = time.monotonic() + limit
    out = {}
    for c in calls:
        try:
            out[c["id"]] = futs[c["id"]].result(timeout=max(0.0, end - time.monotonic()))
        except FuturesTimeout:
            out[c["id"]] = (f"Tool {c['name']} timed out after {limit:.0f}s and was abandoned. "
                            "Try another source or different arguments.", [])
        except Exception as exc:  # execute_tool already guards tools; this is a last line
            out[c["id"]] = (f"Tool {c['name']} failed: {exc}.", [])
    ex.shutdown(wait=False, cancel_futures=True)
    return out


def _clamp_budget(spec: dict, backend: str | None) -> int:
    try:
        b = int(spec.get("tool_budget") or default_budget(backend))
    except (TypeError, ValueError):
        b = default_budget(backend)
    return max(2, min(b, settings.subagent_hard_cap))


def run_subagent(
    spec: dict,
    backend: str | None,
    registry: SourceRegistry,
    idx: int = 1,
    total: int = 1,
    brief: str = "",
    run_id: str = "",
    deadline: float | None = None,
) -> dict:
    """Research one delegated task end-to-end. Returns a structured report dict.

    `deadline` (a time.monotonic() value) is the wave's time limit: once it passes, the
    subagent stops researching and writes its report from what it already has.
    """
    objective = (spec.get("objective") or "").strip()
    budget = _clamp_budget(spec, backend)
    sources = [s for s in (spec.get("sources") or ["academic", "web"]) if s in _FALLBACK] or ["academic", "web"]
    tag = f"[{idx}/{total}]"
    tools = TOOL_SCHEMAS + [complete_task]
    window = input_char_budget(backend)  # None on DeepSeek (window far larger than 8 turns)
    ctx_limit = 400_000 if window is None else max(4000, window - _tools_overhead(tools))

    llm = get_llm(backend, temperature=0.2, role="worker").bind_tools(tools)
    msgs: list = [
        SystemMessage(SUBAGENT_SYSTEM.format(date=date.today().isoformat(), budget=budget,
                                             max_reads=settings.subagent_max_reads)),
        HumanMessage(SUBAGENT_TASK.format(
            objective=objective,
            key_questions="\n".join(f"- {q}" for q in (spec.get("key_questions") or [objective])),
            sub_query=spec.get("sub_query") or objective,
            sources=", ".join(sources),
            tools=spec.get("tools") or ", ".join(t for s in sources for t in _FALLBACK[s]),
            output_format=spec.get("output_format") or "concrete findings with [S#] citations",
            boundaries=spec.get("boundaries") or "(none stated)",
            brief=brief or objective,
        )),
    ]

    used, reads, stale, turns = 0, 0, 0, 0
    max_reads = settings.subagent_max_reads
    called: list[str] = []
    seen: set[str] = set()
    report: str | None = None
    emit(f"[dim]  → subagent {tag} [cyan]{truncate(objective, 70)}[/] · budget {budget}[/]")

    for turn in range(settings.subagent_max_turns):
        if deadline and time.monotonic() >= deadline:
            emit(f"[yellow]  ! subagent {tag} hit the wave time limit — writing its report now[/]")
            break
        turns = turn + 1
        try:
            ai = llm.invoke(_compact(msgs, ctx_limit))
        except Exception as exc:
            emit(f"[yellow]  ! subagent {tag} LLM error: {truncate(str(exc), 120)}[/]")
            break
        calls = list(getattr(ai, "tool_calls", None) or [])

        if not calls:
            if used == 0 and turn == 0:
                # Answered from memory — run starting searches ourselves, then let it continue.
                q = spec.get("sub_query") or objective
                texts = []
                for s in sources:
                    for name in _FALLBACK[s][:1]:
                        text, sids = execute_tool(name, {"query": q}, registry)
                        used += 1
                        seen.update(sids)
                        called.append(_fmt_call(name, {"query": q}))
                        texts.append(text)
                emit(f"[dim]    {tag} (model skipped tools — ran starter searches)[/]")
                share = max(800, ctx_limit // (3 * max(1, len(texts))))  # compaction never trims this message
                msgs.append(HumanMessage("Do not answer from memory. Starting search results:\n\n"
                                         + "\n\n".join(truncate(t, share) for t in texts)
                                         + "\n\nContinue researching with tools, then call complete_task."))
                continue
            report = (ai.content or "").strip() or None
            break

        msgs.append(ai)
        finish = next((c for c in calls if c["name"] == "complete_task"), None)
        work = [c for c in calls if c["name"] != "complete_task"]
        searches_ok = [c for c in work if c["name"] not in _READ_TOOLS][: max(0, budget - used)]
        reads_ok = [c for c in work if c["name"] in _READ_TOOLS][: max(0, max_reads - reads)]
        allowed = [c for c in work if c in searches_ok or c in reads_ok]

        results = {}
        if allowed:
            for c in allowed:
                emit(f"[dim]    {tag} {_fmt_call(c['name'], c.get('args'))}[/]")
            results = _run_parallel(allowed, registry, deadline)
        used += len(searches_ok)
        reads += len(reads_ok)

        new = 0
        for c in calls:  # every tool call must be answered, in order
            if c["id"] in results:
                text, sids = results[c["id"]]
                new += len(set(sids) - seen)
                seen.update(sids)
                called.append(_fmt_call(c["name"], c.get("args")))
            elif c["name"] == "complete_task":
                text = "Report received."
            else:
                text = ("Skipped: full-text read allowance used up." if c["name"] in _READ_TOOLS
                        else "Skipped: search budget exhausted — read key sources or call complete_task.")
            msgs.append(ToolMessage(content=text, tool_call_id=c["id"]))

        if finish:  # tools called in the same turn have already run; accept the report
            report = str((finish.get("args") or {}).get("report") or "").strip() or None
            break
        if used >= budget and reads >= max_reads:
            break
        if used >= budget:
            msgs.append(HumanMessage(f"Search budget used. You may open up to {max_reads - reads} key source(s) "
                                     "with read_paper / read_web to verify details, then call complete_task."))
        stale = stale + 1 if new == 0 else 0
        if stale >= 2:
            msgs.append(HumanMessage("Your last searches found nothing new. Unless a key question is still "
                                     "unanswered, call complete_task with your report now."))

    if not report:
        try:
            closer = get_llm(backend, temperature=0.2, role="worker").bind_tools(tools, tool_choice="none")
            ai = closer.invoke(_compact(msgs + [HumanMessage(SUBAGENT_FINISH)], ctx_limit))
            report = (ai.content or "").strip()
            if not report and getattr(ai, "tool_calls", None):
                report = str(ai.tool_calls[0].get("args", {}).get("report", "")).strip()
        except Exception as exc:
            emit(f"[yellow]  ! subagent {tag} could not write report: {truncate(str(exc), 120)}[/]")
        if not report:  # last resort: hand the lead the raw source list
            report = "Raw sources found (no written report):\n" + "\n".join(
                render_item(s, registry.get(s)["kind"], registry.get(s)["item"]) for s in list(seen)[:12]
            )

    emit(f"[dim]  ↳ subagent {tag} done · {used} searches + {reads} reads · {len(seen)} sources[/]")
    result = {
        "objective": objective,
        "findings": report,
        "sources": sorted(seen, key=lambda s: int(s[1:])),
        "tools": called,
        "tool_calls": used + reads,
        "turns": turns,
    }
    _save_transcript(run_id, idx, spec, result)
    return result


def _save_transcript(run_id: str, idx: int, spec: dict, result: dict) -> None:
    """Persist each subagent's work as an artifact (Anthropic: subagent outputs go to a
    filesystem, not only through the lead) — handy for debugging and evals."""
    if not run_id:
        return
    try:
        d = RUNS_DIR / run_id
        d.mkdir(parents=True, exist_ok=True)
        body = (
            f"# Subagent {idx}: {spec.get('objective', '')}\n\n"
            f"Budget {spec.get('tool_budget')} · used {result['tool_calls']} · turns {result['turns']}\n\n"
            "## Tool calls\n" + "\n".join(f"- {c}" for c in result["tools"]) +
            f"\n\n## Report\n{result['findings']}\n"
        )
        (d / f"subagent_{idx}.md").write_text(body, encoding="utf-8")
    except Exception:
        pass
