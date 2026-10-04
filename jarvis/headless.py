"""Run a full research request without a human: auto-skips clarifying questions and
auto-approves the plan. Used by the smoke test and the eval harness."""

from __future__ import annotations

import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from . import llm as _llm
from .llm import resolve_backend, set_active_backend, usage


class BackendUnavailable(RuntimeError):
    pass


def run_research(query: str, backend: str | None = None, graph=None, answers: str = "",
                 prior: dict | None = None, check: bool = True) -> dict:
    """Returns the final graph state plus {'usage', 'cost', 'seconds', 'backend'}.

    `prior` is the previous run's context (nodes.build_prior) for follow-up questions.
    With `check`, a dead backend raises BackendUnavailable instead of producing a hollow report.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from .graph import build_graph

    backend, notice = resolve_backend(backend)
    if notice:
        print(f"[note] {notice}")
    if check and (problem := _llm.check_backend(backend)):
        raise BackendUnavailable(problem)
    set_active_backend(backend)
    graph = graph or build_graph(MemorySaver())
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 60}
    usage.reset()
    t0 = time.time()

    payload = {"messages": [HumanMessage(content=query)], "query": query, "backend": backend,
               "prior": prior or {}}
    for _ in range(6):  # clarify -> confirm (-> re-plan) interrupts
        interrupted = None
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                intr = chunk["__interrupt__"]
                interrupted = intr[0].value if isinstance(intr, (list, tuple)) else intr
        if not interrupted:
            break
        if interrupted.get("type") == "clarify_questions":
            payload = Command(resume={"answers": answers})
        else:
            payload = Command(resume="yes")

    final = dict(graph.get_state(config).values)
    final["usage"] = usage.snapshot()
    final["cost"] = usage.cost(backend)
    final["seconds"] = round(time.time() - t0, 1)
    final["backend"] = backend
    return final
