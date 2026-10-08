"""Headless end-to-end smoke test: one real research run, auto-confirmed.

Usage:
  python scripts/smoke.py                       # default backend
  python scripts/smoke.py deepseek              # force a backend
  python scripts/smoke.py deepseek "your query" # custom query
  python scripts/smoke.py ollama "query" --follow-up "compare that with X"   # also test memory
"""

from __future__ import annotations

import argparse
import sys

from jarvis.events import set_sink
from jarvis.headless import BackendUnavailable, run_research
from jarvis.llm import usage
from jarvis.nodes import build_prior
from jarvis.progress import plain_line


def _show(final: dict) -> str:
    report = final.get("report", "")
    print("\n" + "=" * 70)
    print(f"query_type: {final.get('query_type')} · follow-up: {final.get('follow_up')} · "
          f"subagents: {len(final.get('subagent_reports', []))} · sources: {len(final.get('sources', {}))} · "
          f"citations: {final.get('citation_stats')}")
    print(f"time: {final['seconds']}s · {usage.summary(final['backend'])}")
    print("=" * 70 + "\n")
    print(report)
    return report


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", nargs="?", default=None)
    ap.add_argument("query", nargs="?", default="how are RAG systems evaluated in research and in production?")
    ap.add_argument("--follow-up", default="", help="a second question that builds on the first report")
    args = ap.parse_args()
    def show_event(event: str, data: dict) -> None:
        if (line := plain_line(event, data)) is not None:
            print(line, flush=True)

    set_sink(show_event)

    try:
        final = run_research(args.query, args.backend)
    except BackendUnavailable as exc:
        print(f"[FAIL] {exc}")
        return 1
    if not _show(final).strip():
        print("\n[FAIL] empty report")
        return 1
    if args.follow_up:
        second = run_research(args.follow_up, args.backend, prior=build_prior(final))
        if not _show(second).strip():
            print("\n[FAIL] empty follow-up report")
            return 1
        if not second.get("follow_up"):
            print("\n[WARN] the lead treated the follow-up as a new topic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
