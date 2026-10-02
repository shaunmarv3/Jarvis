"""Headless end-to-end smoke test: one real research run, auto-confirmed.

Usage:
  python scripts/smoke.py                       # default backend
  python scripts/smoke.py deepseek              # force a backend
  python scripts/smoke.py deepseek "your query" # custom query
"""

from __future__ import annotations

import sys

from jarvis.events import set_sink
from jarvis.headless import run_research
from jarvis.llm import usage


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    backend = sys.argv[1] if len(sys.argv) > 1 else None
    query = sys.argv[2] if len(sys.argv) > 2 else "how are RAG systems evaluated in research and in production?"
    from rich.console import Console

    set_sink(Console(highlight=False).print)

    final = run_research(query, backend)
    report = final.get("report", "")
    print("\n" + "=" * 70)
    print(f"query_type: {final.get('query_type')} · subagents: {len(final.get('subagent_reports', []))} "
          f"· sources: {len(final.get('sources', {}))} · citations: {final.get('citation_stats')}")
    print(f"time: {final['seconds']}s · {usage.summary(final['backend'])}")
    print("=" * 70 + "\n")
    print(report)
    if not report.strip():
        print("\n[FAIL] empty report")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
