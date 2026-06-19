"""Verbose, unbuffered demo: watch the agent actually reason & decide."""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from jarvis.config import settings
from jarvis.graph import build_graph
from jarvis.llm import resolve_backend

settings.subagent_rounds = 1  # keep the demo short


def show(node, update):
    print(f"\n===== node: {node} =====", flush=True)
    if not isinstance(update, dict):
        print(update, flush=True)
        return
    if node == "clarify":
        print("intent :", update.get("intent"), flush=True)
        print("brief  :", update.get("brief"), flush=True)
        print("query  :", update.get("current_query"), flush=True)
    elif node == "plan":
        plan = update.get("plan", [])
        print(f"lead spawned {len(plan)} subagent(s):", flush=True)
        for i, s in enumerate(plan, 1):
            print(f"   {i}. {s.get('objective')}  [q: {s.get('sub_query')}]", flush=True)
    elif node == "fanout":
        papers = update.get("papers", [])
        print(f"papers gathered: {len(papers)}", flush=True)
        for p in papers[:6]:
            print(f"   - [{p.get('source')}] {p.get('title')} ({p.get('year')})", flush=True)
    elif node == "synthesize":
        print("findings:\n", (update.get("findings") or "")[:600], flush=True)
    elif node == "finalize":
        print("DRAFT REPORT:\n", (update.get("report") or "")[:600], flush=True)
    elif node == "cite":
        print("FINAL (cited) REPORT:\n", update.get("report", ""), flush=True)


def main():
    backend, notice = resolve_backend()
    if notice:
        print("[note]", notice, flush=True)
    print(f"[backend={backend} model={settings.ollama_model}]", flush=True)

    graph = build_graph()
    cfg = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 60}
    q = "find recent papers on retrieval-augmented generation evaluation"
    print(f"\nUSER QUERY: {q}", flush=True)

    def drive(payload):
        for chunk in graph.stream(payload, cfg, stream_mode="updates"):
            if "__interrupt__" in chunk:
                print("\n--- INTERRUPT: agent is asking to confirm the brief ---", flush=True)
                print("payload:", chunk["__interrupt__"][0].value, flush=True)
                return "interrupt"
            for node, update in chunk.items():
                show(node, update)
        return "done"

    init = {"messages": [HumanMessage(content=q)], "query": q, "backend": backend,
            "max_loops": settings.max_loops}
    if drive(init) == "interrupt":
        print("\n>>> (auto-answering 'yes' to proceed)", flush=True)
        drive(Command(resume="yes"))

    print("\n[demo complete]", flush=True)


if __name__ == "__main__":
    main()
