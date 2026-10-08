"""Live progress for a research run, in the spirit of Claude Code's display.

While research agents work, each gets ONE line that updates in place: what it is doing
right now, in plain words ("Searching arXiv: "…"", "Reading: <paper title>"). When a
wave finishes, the live block collapses into a short summary that stays on screen.
`/verbose` brings back the old one-line-per-tool-call log for debugging.

The graph emits structured events (jarvis.events); this module is the only place that
turns them into text. `plain_line` does the same for scripts without a live terminal.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable

from rich.console import Console, Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from . import terminal

_READS = {"read_paper", "read_web", "inspect_dataset"}
_TOOL_LABELS = {
    "search_arxiv": "Searching arXiv",
    "fetch_arxiv": "Fetching from arXiv",
    "search_semantic_scholar": "Searching Semantic Scholar",
    "resolve_title": "Looking up the paper",
    "search_openalex": "Searching OpenAlex",
    "resolve_doi": "Looking up the DOI",
    "read_paper": "Reading",
    "search_datasets": "Searching HF datasets",
    "inspect_dataset": "Inspecting dataset",
    "search_web": "Searching the web",
    "read_web": "Reading",
    "search_github": "Searching GitHub",
    "search_hf_models": "Searching HF models",
    "search_hn": "Searching Hacker News",
    "search_reddit": "Searching Reddit",
}
_SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# What runs *after* each node returns, so the spinner always names the real stage.
BUSY_AFTER = {
    "clarify": "Planning the research",
    "brief": "Planning the research",
    "fanout": "Checking the findings for gaps",
    "review": "Writing the report",
    "followup": "Writing the report",
    "report": "Attaching citations",
}


def fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _plural(n: int, word: str, many: str = "") -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def describe_tool(tool: str, target: str = "", width: int = 60) -> str:
    """'search_arxiv', 'LLM unlearning' -> 'Searching arXiv: "LLM unlearning"'."""
    label = _TOOL_LABELS.get(tool, tool.replace("_", " ").capitalize())
    target = " ".join((target or "").split())
    if len(target) > width:
        target = target[: width - 1].rstrip() + "…"
    if not target:
        return label
    return f"{label}: {target}" if tool in _READS or not tool.startswith("search") else f'{label}: "{target}"'


def _counts(searches: int, reads: int) -> str:
    parts = [_plural(searches, "search", "searches")]
    if reads:
        parts.append(_plural(reads, "read"))
    return " · ".join(parts)


def plain_line(event: str, data: dict) -> str | None:
    """One plain-text line per event, for headless scripts (smoke tests, logs)."""
    d = data
    if event == "note":
        return ("! " if d.get("level") == "warn" else "· ") + d.get("text", "")
    if event == "wave_start":
        kind = "follow-up" if d.get("followup") else "research"
        mode = "in parallel" if d.get("parallel") else "one at a time"
        reused = f" (reusing {d['reused']} finished before the interruption)" if d.get("reused") else ""
        return f"· {kind}: {_plural(len(d.get('agents') or []), 'agent')} {mode}{reused}"
    if event == "agent_start":
        return f"  → [{d.get('idx')}] {d.get('title', '')}"
    if event == "agent_tool":
        return f"    [{d.get('idx')}] {describe_tool(d.get('tool', ''), d.get('target', ''))}"
    if event == "agent_warn":
        return f"  ! [{d.get('idx')}] {d.get('text', '')}"
    if event == "agent_done":
        if d.get("failed"):
            return f"  ✗ [{d.get('idx')}] failed"
        return (f"  ✓ [{d.get('idx')}] done · {_counts(d.get('searches', 0), d.get('reads', 0))} · "
                f"{_plural(d.get('sources', 0), 'source')}")
    if event == "wave_done":
        return (f"✓ {_plural(d.get('agents', 0), 'agent')} finished · {_counts(d.get('searches', 0), d.get('reads', 0))}"
                f" · {d.get('sources', 0)} sources so far")
    if event == "gaps":
        return f"● found a gap → {_plural(len(d.get('titles') or []), 'more agent')}: {', '.join(d.get('titles') or [])}"
    return None


class ProgressView:
    """Draws a run's live status. Thread-safe: subagents report from worker threads.

    Lock order: never print to the console while holding `_lock` — Rich's refresh thread
    holds the console lock while it calls `__rich__`, which takes `_lock`.
    """

    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self.topic = ""
        self._lock = threading.RLock()
        self._live = None
        self._stage = ""
        self._t0 = time.time()
        self._cost: Callable[[], str] = lambda: ""
        self._wave: dict | None = None
        self._agents: dict[int, dict] = {}

    # ------------------------------------------------------------------ lifecycle

    @contextmanager
    def running(self, stage: str, cost: Callable[[], str] | None = None, t0: float | None = None):
        """Show the live display (spinner, stage, agents) while the body runs."""
        from rich.live import Live

        with self._lock:
            self._stage = stage
            self._cost = cost or (lambda: "")
            self._t0 = t0 or time.time()
        stop = threading.Event()

        def animate_title():  # the tab title spins like Claude Code's while Jarvis works
            frame = 0
            while not stop.wait(0.3):
                terminal.set_title(terminal.title_text(self.topic, frame))
                frame += 1

        ticker = threading.Thread(target=animate_title, daemon=True)
        live = Live(self, console=self.console, refresh_per_second=8, transient=True)
        try:
            with live:
                self._live = live
                ticker.start()
                yield self
        finally:
            self._live = None
            stop.set()
            terminal.set_title(terminal.title_text(self.topic))
            with self._lock:
                self._wave, self._agents = None, {}

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._stage = stage

    def _print(self, *lines) -> None:
        for line in lines:
            if line is not None:
                self.console.print(line, highlight=False)

    # ------------------------------------------------------------------ graph progress

    def node_done(self, node: str, update) -> None:
        """A graph node finished: print its one-line outcome and move the spinner on."""
        update = update if isinstance(update, dict) else {}
        line = None
        if node == "clarify":
            if update.get("title"):
                self.topic = update["title"]
                terminal.set_title(terminal.title_text(self.topic))
            line = "[green]✓[/] Understood the request"
        elif node == "brief":
            line = "[green]✓[/] Refined the brief with your answers"
        elif node == "plan":
            n = len(update.get("plan") or [])
            line = f"[green]✓[/] Planned {_plural(n, 'research agent')}"
        elif node == "review" and "gaps" in update and not update.get("followups"):
            line = "[green]✓[/] Checked the findings · no gaps worth another round"
        elif node == "report":
            line = "[green]✓[/] Wrote the report"
        elif node == "cite":
            st = update.get("citation_stats") or {}
            line = (f"[green]✓[/] Linked {_plural(st.get('cited', 0), 'citation')} "
                    f"[dim]({st.get('retrieved', 0)} sources retrieved)[/]")
        if node in BUSY_AFTER:
            self.set_stage(BUSY_AFTER[node])
        self._print(line)

    def handle(self, event: str, data: dict) -> None:
        """The events.emit sink."""
        out: list = []
        with self._lock:
            out = self._apply(event, data)
        self._print(*out)

    def _apply(self, event: str, d: dict) -> list:
        """Update state for one event; return lines to print (printed outside the lock)."""
        if event == "note":
            warn = d.get("level") == "warn"
            return [f"[yellow]! {escape(d.get('text', ''))}[/]" if warn else f"[dim]· {escape(d.get('text', ''))}[/]"]
        if event == "wave_start":
            self._wave = {"followup": d.get("followup"), "parallel": d.get("parallel"), "t0": time.time()}
            self._agents = {i: {"title": t, "state": "queued", "activity": "", "searches": 0, "reads": 0,
                                "sources": 0} for i, t in d.get("agents") or []}
            if d.get("reused"):
                return [f"[dim]· Reusing {_plural(d['reused'], 'agent')} that finished before the interruption[/]"]
            return []
        a = self._agents.setdefault(d.get("idx", 0), {"title": d.get("title", ""), "state": "queued",
                                                       "activity": "", "searches": 0, "reads": 0, "sources": 0}) \
            if event.startswith("agent_") else None
        if event == "agent_start":
            a.update(state="running", activity="Thinking about where to look", title=d.get("title") or a["title"])
            return [f"[dim]  → [{d.get('idx')}] {escape(a['title'])}[/]"] if self.verbose else []
        if event == "agent_tool":
            tool = d.get("tool", "")
            a["reads" if tool in _READS else "searches"] += 1
            a["activity"] = describe_tool(tool, d.get("target", ""))
            return [f"[dim]    [{d.get('idx')}] {escape(a['activity'])}[/]"] if self.verbose else []
        if event == "agent_warn":
            a["warn"] = d.get("text", "")
            return [f"[yellow]  ! {escape(a['title'])}: {escape(d.get('text', ''))}[/]"]
        if event == "agent_done":
            a.update(state="failed" if d.get("failed") else "done", searches=d.get("searches", a["searches"]),
                     reads=d.get("reads", a["reads"]), sources=d.get("sources", 0))
            return []
        if event == "wave_done":
            return self._wave_summary(d)
        if event == "gaps":
            titles = d.get("titles") or []
            return [f"[cyan]●[/] Found a gap → {_plural(len(titles), 'more agent')}: "
                    f"{escape(', '.join(titles))}"]
        return []

    def _wave_summary(self, d: dict) -> list:
        took = fmt_secs(time.time() - self._wave["t0"]) if self._wave else ""
        n = d.get("agents", 0)
        verb = "Followed up with" if d.get("followup") else "Researched with"
        found = (f"+{d.get('new_sources', 0)} new sources" if d.get("followup")
                 else _plural(d.get("sources", 0), "source"))
        lines = [f"[green]✓[/] {verb} {_plural(n, 'agent')} · {_counts(d.get('searches', 0), d.get('reads', 0))} · "
                 f"{found} [dim]({took})[/]"]
        for _, a in sorted(self._agents.items()):
            if a["state"] == "failed":
                lines.append(f"    [red]✗[/] [dim]{escape(a['title'])} · failed[/]")
            else:
                lines.append(f"    [dim]✓ {escape(a['title'])} · {_plural(a['sources'], 'source')}[/]")
        self._wave, self._agents = None, {}
        return lines

    # ------------------------------------------------------------------ rendering

    def _header(self, label: str) -> Table:
        spin = _SPIN[int(time.time() * 10) % len(_SPIN)]
        bits = [fmt_secs(time.time() - self._t0)]
        if cost := self._cost():
            bits.append(cost)
        bits.append("ctrl+c to pause")
        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
        grid.add_column(no_wrap=True, justify="right")
        grid.add_row(Text.from_markup(f"[cyan]{spin}[/] [bold]{escape(label)}…[/]"),
                     Text(" · ".join(bits), style="dim"))
        return grid

    def __rich__(self):
        with self._lock:
            if not self._wave or not self._agents:
                return self._header(self._stage)
            agents = sorted(self._agents.items())
            n = len(agents)
            done = sum(1 for _, a in agents if a["state"] in ("done", "failed"))
            if self._wave.get("followup"):
                label = f"Following up on a gap · {_plural(n, 'agent')}"
            else:
                mode = "in parallel" if self._wave.get("parallel") else "one at a time"
                label = f"Researching · {_plural(n, 'agent')} {mode}"
            if done:
                label += f" · {done} done"
            rows = Table.grid(padding=(0, 1), expand=True)
            rows.add_column(width=3, no_wrap=True)
            rows.add_column(no_wrap=True, overflow="ellipsis", max_width=38)
            rows.add_column(no_wrap=True, overflow="ellipsis", ratio=1)
            rows.add_column(no_wrap=True, justify="right")
            spin = _SPIN[int(time.time() * 10) % len(_SPIN)]
            for _, a in agents:
                count = _counts(a["searches"], a["reads"]) if (a["searches"] or a["reads"]) else ""
                if a["state"] == "done":
                    rows.add_row(Text("  ✓", "green"), Text(a["title"]),
                                 Text(f"done · {_plural(a['sources'], 'source')}", "dim"), Text(count, "dim"))
                elif a["state"] == "failed":
                    rows.add_row(Text("  ✗", "red"), Text(a["title"]), Text(a.get("warn") or "failed", "red"),
                                 Text(count, "dim"))
                elif a["state"] == "queued":
                    rows.add_row(Text("  ○", "dim"), Text(a["title"], "dim"), Text("waiting its turn", "dim"),
                                 Text(""))
                else:
                    activity = Text(a.get("warn") or a["activity"], "yellow" if a.get("warn") else "")
                    rows.add_row(Text(f"  {spin}", "cyan"), Text(a["title"], "bold"), activity, Text(count, "dim"))
            return Group(self._header(label), rows)
