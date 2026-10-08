"""User memory — a plain markdown file of standing preferences every agent keeps in mind
(like Claude Code's CLAUDE.md / MEMORY.md).

The file lives at data/memory.md. Users read and change it with /memory and /remember,
or edit it by hand: everything outside <!-- comments --> is shown to the agents. Notes
are the "- " bullet lines; that's what /memory numbers and /memory remove deletes.

Only the interactive CLI loads it (into the run's state), so headless runs and evals are
never influenced by one person's preferences.
"""

from __future__ import annotations

import re

from . import config
from .utils import truncate, write_text_atomic

TEMPLATE = """# Jarvis memory

<!--
Standing preferences that every Jarvis agent keeps in mind: the planner, every research
subagent, the report writer, /ask and /read. One note per "- " line. Edit freely.
Examples:
- I'm new to ML: explain jargon briefly and keep reports short.
- Prefer papers from 2023 onwards; skip Medium and SEO blog posts.
- Always include GitHub implementations (PyTorch if possible).
-->

"""

# How much of the memory each backend sees (the local 8k-token window is shared with everything else).
MAX_CHARS = {"deepseek": 2000, "ollama": 800}
_BULLET = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def path():
    return config.DATA_DIR / "memory.md"


def raw() -> str:
    """The file exactly as written ('' if it doesn't exist yet)."""
    try:
        return path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def ensure() -> None:
    """Create the file with its how-to comment if it doesn't exist (for /memory edit)."""
    if not path().exists():
        write_text_atomic(path(), TEMPLATE)


def text() -> str:
    """What the agents see: the file without comments, the template's title and blank runs."""
    body = _COMMENT.sub("", raw())
    lines = [ln.rstrip() for ln in body.splitlines()]
    if lines and lines[0].strip() == "# Jarvis memory":
        lines = lines[1:]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def notes() -> list[str]:
    """The bullet notes, in order (comments excluded)."""
    return [m.group(1) for ln in _COMMENT.sub("", raw()).splitlines() if (m := _BULLET.match(ln))]


def add(note: str) -> str:
    """Append one note; returns it as stored (one line)."""
    note = re.sub(r"\s+", " ", note or "").strip().lstrip("-* ").strip()
    if not note:
        raise ValueError("empty note")
    current = raw() or TEMPLATE
    if not current.endswith("\n"):
        current += "\n"
    write_text_atomic(path(), current + f"- {note}\n")
    return note


def remove(n: int) -> str | None:
    """Delete note #n (1-based, as /memory numbers them). Returns its text, or None."""
    body = raw()
    # Number only bullets outside comments, so the examples in the template never count.
    spans = [(m.start(), m.end()) for m in _COMMENT.finditer(body)]
    pos, count, out, removed = 0, 0, [], None
    for line in body.splitlines(keepends=True):
        in_comment = any(a <= pos < b for a, b in spans)
        m = None if in_comment else _BULLET.match(line)
        pos += len(line)
        if m:
            count += 1
            if count == n:
                removed = m.group(1)
                continue
        out.append(line)
    if removed is not None:
        write_text_atomic(path(), "".join(out))
    return removed


def clear() -> None:
    write_text_atomic(path(), TEMPLATE)


def limit(backend: str | None) -> int:
    return MAX_CHARS.get((backend or "ollama").lower(), MAX_CHARS["ollama"])


def prompt_block(memory: str, backend: str | None) -> str:
    """The memory as a prompt section ('' when empty), cut to the backend's share."""
    from .prompts import MEMORY_BLOCK

    memory = (memory or "").strip()
    if not memory:
        return ""
    return MEMORY_BLOCK.format(memory=truncate(memory, limit(backend)))
