"""A tiny progress sink so deep nodes (e.g. the subagent fan-out) can report live
status to the CLI without importing Rich or the console.

The CLI registers a sink at startup (`set_sink(console.print)`); headless scripts
leave it unset and `emit()` becomes a no-op. Keeps nodes UI-agnostic.
"""

from __future__ import annotations

from typing import Callable

_sink: Callable[[str], None] | None = None


def set_sink(fn: Callable[[str], None] | None) -> None:
    global _sink
    _sink = fn


def emit(message: str) -> None:
    if _sink is None:
        return
    try:
        _sink(message)
    except Exception:
        pass
