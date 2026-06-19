"""Headless smoke test: build the graph, auto-confirm, run one query.

Usage:
  python scripts/smoke.py            # uses default backend (ollama)
  python scripts/smoke.py deepseek   # force a backend

Network and a running LLM are required for a *full* pass. The script is lenient:
it always checks the graph runs end-to-end and prints what it found.
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from jarvis.config import settings
from jarvis.graph import build_graph
from jarvis.llm import resolve_backend


def main() -> int:
    requested = sys.argv[1] if len(sys.argv) > 1 else None
    backend, notice = resolve_backend(requested)
    if notice:
        print(f"[note] {notice}")
    print(f"[smoke] backend={backend} model={settings.ollama_model} max_loops={settings.max_loops}")

    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 60}
    query = "find recent papers on retrieval-augmented generation evaluation"

    def drive(payload):
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                return "interrupt"
            for node in chunk:
                print(f"  · {node}")
        return "done"

    init = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "backend": backend,
        "max_loops": settings.max_loops,
    }
    status = drive(init)
    if status == "interrupt":
        print("  · (auto-confirming brief)")
        drive(Command(resume="yes"))

    final = graph.get_state(config).values
    papers = final.get("papers", [])
    report = final.get("report", "")

    print("\n========== RESULT ==========")
    print(f"papers found: {len(papers)}")
    for p in papers[:5]:
        print(f"  - {p.get('title')} ({p.get('year')}) [{p.get('source')}]")
    print(f"\nreport length: {len(report)} chars")
    print(report[:800])

    assert isinstance(report, str), "no report produced"
    if not report.strip():
        print("\n[WARN] empty report — check the LLM backend is reachable.")
        return 1
    if not papers:
        print("\n[WARN] no papers found — likely offline or rate-limited.")
    print("\n[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
