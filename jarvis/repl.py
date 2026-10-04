"""Interactive input for the REPL.

A `prompt_toolkit` session with a live slash-command dropdown (Claude-Code style):
as you type `/`, matching commands appear in a menu below the cursor with their
descriptions. Falls back to a plain line read when stdin isn't a terminal (e.g. piped
input in tests/scripts), so non-interactive use still works.
"""

from __future__ import annotations

import sys

# (command, arg-hint, description) — drives BOTH the completion menu and /help,
# so they can never drift out of sync.
COMMANDS: list[tuple[str, str, str]] = [
    ("/read", "<N>", "download, summarize & index paper N into its own vector store"),
    ("/db", "", "list papers in the vector DB (each stored separately)"),
    ("/ask", "[N] <question>", "ask a question about indexed paper N (pick N from the dropdown)"),
    ("/use", "<N>", "select DB paper N as the target for /ask"),
    ("/forget", "<N>", "delete DB paper N's vectors"),
    ("/save", "<N>", "download paper N's PDF to data/papers"),
    ("/papers", "", "list the last run's papers (cited first)"),
    ("/sources", "", "list every source the last run retrieved (papers, web, code, data, community)"),
    ("/new", "", "start a fresh topic (the next question won't build on the last report)"),
    ("/resume", "", "resume the last research run if it was interrupted or crashed"),
    ("/cost", "", "token usage & estimated cost of the last run"),
    ("/dataset", "<query>", "quick HuggingFace dataset search"),
    ("/inspect", "<hub_id>", "inspect a dataset (cols, rows, sample, README)"),
    ("/web", "<query>", "quick web search (DuckDuckGo)"),
    ("/backend", "ollama|deepseek", "switch the LLM brain"),
    ("/model", "<name>", "switch the Ollama model"),
    ("/help", "", "show commands"),
    ("/quit", "", "exit"),
]

_session = None  # lazily built prompt_toolkit PromptSession
_session_papers = lambda: []  # getter the CLI registers so /save & /read can list results


def set_session_papers_getter(fn) -> None:
    """Register a callable returning the current session's search-result papers,
    so the completer can offer them as a dropdown for /save and /read."""
    global _session_papers
    _session_papers = fn


def _strip_bom(s: str) -> str:
    """Drop a leading BOM that Windows PowerShell prepends when piping stdin
    (U+FEFF if stdin is UTF-8, or its cp1252 mojibake form)."""
    for bom in (chr(0xFEFF), chr(0xEF) + chr(0xBB) + chr(0xBF)):
        if s.startswith(bom):
            return s[len(bom):]
    return s


def _short(s: str, n: int = 56) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _db_items() -> list[tuple[str, str]]:
    """(number, title) for each indexed paper in the vector DB."""
    try:
        from .store import list_papers

        return [(str(i), e.get("title") or e["folder"]) for i, e in enumerate(list_papers(), 1)]
    except Exception:
        return []


def _session_items() -> list[tuple[str, str]]:
    """(number, title) for each paper in the current session's search results."""
    try:
        return [(str(i), p.get("title", "")) for i, p in enumerate(_session_papers(), 1)]
    except Exception:
        return []


# Which commands complete their FIRST argument from a live list, and where from.
_ARG_SOURCES = {
    "/use": _db_items,
    "/forget": _db_items,
    "/ask": _db_items,  # pick the paper number, then type the question
    "/save": _session_items,
    "/read": _session_items,
    "/backend": lambda: [("ollama", "local · sequential subagents"),
                         ("deepseek", "cloud · parallel subagents")],
}


def _make_completer():
    """A Completer that suggests slash-commands, then context-aware arguments. No console needed."""
    from prompt_toolkit.completion import Completer, Completion

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return

            sp = text.find(" ")
            if sp == -1:
                # Still typing the command name -> suggest commands.
                for name, hint, desc in COMMANDS:
                    if name.startswith(text):
                        meta = f"{hint}   {desc}".strip() if hint else desc
                        yield Completion(name, start_position=-len(text), display=name, display_meta=meta)
                return

            # Typing the first argument -> offer that command's live items (if any).
            cmd = text[:sp].lower()
            arg = text[sp + 1:]
            source = _ARG_SOURCES.get(cmd)
            if source is None or " " in arg:  # only the first arg
                return
            items = source()
            if not items:
                return
            for value, label in items:
                if value.startswith(arg):
                    yield Completion(
                        value,
                        start_position=-len(arg),
                        display=f"{value}  {_short(label)}" if label else value,
                        display_meta=_short(label, 60),
                    )

    return SlashCompleter()


def _build_session():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style

    style = Style.from_dict(
        {
            "ag": "bold #5eeafc",
            "completion-menu": "bg:#0d1b24",
            "completion-menu.completion": "bg:#0d1b24 #8aa0ad",
            "completion-menu.completion.current": "bg:#2563eb #ffffff bold",
            "completion-menu.meta.completion": "bg:#091319 #5a6b78",
            "completion-menu.meta.completion.current": "bg:#1d4ed8 #d7e8ff",
            "scrollbar.background": "bg:#0d1b24",
            "scrollbar.button": "bg:#2563eb",
        }
    )

    return PromptSession(
        completer=_make_completer(),
        complete_while_typing=True,
        reserve_space_for_menu=6,
        style=style,
    )


def read_line() -> str:
    """Read one REPL line. Live slash-menu when interactive; plain read when piped.

    Propagates EOFError / KeyboardInterrupt (the caller turns them into a clean exit).
    """
    global _session
    if not sys.stdin.isatty():
        return _strip_bom(input("jarvis > ")).strip()
    try:
        from prompt_toolkit.formatted_text import HTML

        if _session is None:
            _session = _build_session()
        return _strip_bom(_session.prompt(HTML("\n<ag>jarvis ❯ </ag>"))).strip()
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        # Any prompt_toolkit/terminal failure -> degrade gracefully to a plain read.
        return _strip_bom(input("jarvis > ")).strip()
