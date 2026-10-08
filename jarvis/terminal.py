"""The terminal tab title — "✦ <topic>" like Claude Code's "✳ <topic>" tabs.

Windows uses SetConsoleTitleW, which works in both the classic console (cmd.exe) and
Windows Terminal; other platforms get the standard OSC escape. Nothing is written
when stdout isn't a real terminal (pipes, tests, CI).
"""

from __future__ import annotations

import re
import sys

IDLE = "✦"
SPINNER = "◐◓◑◒"
APP = "Jarvis"
_MAX = 40


def topic_from(text: str, n: int = _MAX) -> str:
    """A tab-sized topic: one line, no markup-ish clutter, cut at a word boundary."""
    t = re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip().strip("\"'`")
    if len(t) <= n:
        return t
    cut = t[: n - 1].rsplit(" ", 1)[0] or t[: n - 1]
    return cut.rstrip(" ,.;:-") + "…"


def title_text(topic: str = "", frame: int | None = None) -> str:
    glyph = IDLE if frame is None else SPINNER[frame % len(SPINNER)]
    return f"{glyph} {topic_from(topic) or APP}"


_original: str | None = None  # the title before Jarvis started (Windows only; restored on exit)


def set_title(text: str) -> None:
    """Set the tab/window title."""
    global _original
    try:
        if not sys.stdout.isatty():
            return
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.windll.kernel32
            if _original is None:
                buf = ctypes.create_unicode_buffer(1024)
                _original = buf.value if k32.GetConsoleTitleW(buf, 1024) else ""
            k32.SetConsoleTitleW(text)
        else:
            sys.stdout.write(f"\x1b]0;{text}\x07")
            sys.stdout.flush()
    except Exception:
        pass  # a title is cosmetic; never let it break the REPL


def restore_title() -> None:
    """Put back the title the terminal had before Jarvis (an empty one elsewhere = the default)."""
    set_title(_original or "")
