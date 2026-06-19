"""Subagent runner — an isolated research loop for ONE objective.

Each subagent gets its own context window (its own message list), runs a couple of
broad->narrow search rounds with the tools, then writes a short briefing back to the
lead. It's a pure function of its spec, so it's safe to run either sequentially
(local Ollama, one GPU) or inside a thread pool (DeepSeek, true parallelism).
"""

from __future__ import annotations

from .config import settings
from .events import emit
from .llm import get_llm
from .prompts import (
    SUBAGENT_REFLECT,
    SUBAGENT_SYNTH,
    SUBAGENT_SYSTEM,
    SUBAGENT_USER,
)
from .tools import TOOL_FUNCS, TOOL_SCHEMAS
from .utils import extract_json, truncate

# If the model declines to pick tools, run a sensible academic default.
_FALLBACK_TOOLS = ["search_arxiv", "search_semantic_scholar", "search_openalex"]


def _run_tool(name: str, args: dict | None) -> dict:
    fn = TOOL_FUNCS.get(name)
    if not fn:
        return {"text": f"(unknown tool {name})"}
    try:
        return fn(**(args or {}))
    except Exception as exc:  # a tool must never crash a subagent
        return {"text": f"(tool {name} failed: {exc})"}


def run_subagent(spec: dict, backend: str | None, idx: int = 1, total: int = 1) -> dict:
    """Research one objective end-to-end and return a structured report.

    Returns: {objective, findings, papers, datasets, web, tools}.
    """
    objective = (spec.get("objective") or "").strip()
    cur = (spec.get("sub_query") or objective).strip()
    rounds = max(1, settings.subagent_rounds)

    llm = get_llm(backend, temperature=0.2).bind_tools(TOOL_SCHEMAS)
    papers: list[dict] = []
    datasets: list[dict] = []
    web: list[dict] = []
    called: list[str] = []
    evidence: list[str] = []

    for r in range(1, rounds + 1):
        emit(f"[dim]  → subagent {idx}/{total} · round {r}: [cyan]{truncate(cur, 52)}[/][/]")
        msgs = [
            ("system", SUBAGENT_SYSTEM),
            (
                "user",
                SUBAGENT_USER.format(
                    objective=objective, query=cur, round=r, rounds=rounds,
                    have=f"{len(papers)} papers",
                ),
            ),
        ]
        try:
            ai = llm.invoke(msgs)
            tool_calls = getattr(ai, "tool_calls", None) or []
        except Exception:
            tool_calls = []
        if not tool_calls:
            tool_calls = [{"name": n, "args": {"query": cur}} for n in _FALLBACK_TOOLS]

        for tc in tool_calls:
            args = tc.get("args") or {}
            called.append(
                f"{tc['name']}({', '.join(f'{k}={truncate(str(v), 30)}' for k, v in args.items())})"
            )
            out = _run_tool(tc["name"], args)
            papers.extend(out.get("papers", []) or [])
            datasets.extend(out.get("datasets", []) or [])
            web.extend(out.get("web", []) or [])
            if out.get("text"):
                evidence.append(out["text"])

        # Broad -> narrow: ask for a sharper follow-up query for the next round.
        if r < rounds:
            try:
                data = extract_json(
                    llm.invoke(
                        SUBAGENT_REFLECT.format(
                            objective=objective,
                            evidence=truncate("\n".join(evidence), 2000),
                        )
                    ).content
                )
                nq = (data.get("next_query") or "").strip()
                if nq:
                    cur = nq
            except Exception:
                pass

    raw = truncate("\n\n".join(e for e in evidence if e), 4000) or "(no evidence gathered)"
    try:
        findings = llm.invoke(
            SUBAGENT_SYNTH.format(objective=objective, evidence=raw)
        ).content
    except Exception:
        findings = raw
    emit(f"[dim]  ↳ subagent {idx}/{total} done — {len(papers)} papers[/]")

    return {
        "objective": objective,
        "findings": findings,
        "papers": papers,
        "datasets": datasets,
        "web": web,
        "tools": called,
    }
