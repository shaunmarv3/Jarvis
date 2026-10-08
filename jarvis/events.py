"""A tiny progress sink so deep nodes (e.g. the subagent fan-out) can report live
status to the CLI without importing Rich or the console.

Nodes emit structured events — a name plus plain data — and never format them:

  note        text, level ("info" | "warn")          a one-off message
  wave_start  agents [(idx, title)], parallel, followup, reused
  agent_start idx, total, title, budget
  agent_tool  idx, tool, target                       one tool call (target: query / title / url)
  agent_warn  idx, text
  agent_done  idx, searches, reads, sources
  wave_done   agents, searches, reads, sources, new_sources, followup
  gaps        titles                                  the lead asked for a follow-up wave

The CLI registers a sink at startup (`set_sink(view.handle)`) that draws them; headless
scripts leave it unset and `emit()` becomes a no-op. Keeps nodes UI-agnostic.
"""

from __future__ import annotations

from typing import Callable

Sink = Callable[[str, dict], None]
_sink: Sink | None = None


def set_sink(fn: Sink | None) -> None:
    global _sink
    _sink = fn


def emit(event: str, **data) -> None:
    if _sink is None:
        return
    try:
        _sink(event, data)
    except Exception:
        pass


def note(text: str, level: str = "info") -> None:
    emit("note", text=text, level=level)
