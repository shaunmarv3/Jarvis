"""Jarvis CLI — a Rich REPL over the LangGraph research agent."""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .banner import render_banner
from .config import settings
from .graph import build_graph
from .llm import active_backend, resolve_backend, set_active_backend
from .tools.pdf_reader import summarize_paper
from .utils import truncate

console = Console()

_NODE_LABELS = {
    "clarify": "🧭 understanding your request",
    "act": "🔎 searching & gathering",
    "synthesize": "📝 synthesizing findings",
    "reflect": "🤔 reflecting on gaps",
    "finalize": "📄 compiling report",
}

_HELP = """[bold cyan]Commands[/]
  [green]<free text>[/]              ask the agent (it plans, confirms, then researches)
  [green]/read <N>[/]                download & summarize result N
  [green]/save <N>[/]                download paper N's PDF to data/papers
  [green]/papers[/]                  list this session's papers
  [green]/backend ollama|deepseek[/] switch the LLM brain
  [green]/model <name>[/]            switch the Ollama model
  [green]/help[/]                    show this help
  [green]/quit[/]                    exit
"""


def _drive(graph, payload, config):
    """Stream the graph, printing progress. Returns ('interrupt', value) or ('done', None)."""
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            intr = chunk["__interrupt__"]
            value = intr[0].value if isinstance(intr, (list, tuple)) else intr
            return ("interrupt", value)
        for node, update in chunk.items():
            label = _NODE_LABELS.get(node)
            if label:
                console.print(f"[dim]· {label}[/]")
            if node == "act" and isinstance(update, dict):
                console.print(f"[dim]  ↳ {len(update.get('papers', []))} papers so far[/]")
            if node == "reflect" and isinstance(update, dict):
                if update.get("complete"):
                    console.print("[dim]  ↳ enough gathered[/]")
                elif update.get("gaps"):
                    console.print(f"[dim]  ↳ gap: {update['gaps']}[/]")
    return ("done", None)


def _run_query(graph, query: str) -> list[dict]:
    """Run one research request through the graph; returns the session papers."""
    backend, notice = resolve_backend()
    if notice:
        console.print(f"[yellow]{notice}[/]")

    config = {
        "configurable": {"thread_id": uuid.uuid4().hex},
        "recursion_limit": 60,
    }
    init = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "backend": backend,
        "max_loops": settings.max_loops,
    }

    status, value = _drive(graph, init, config)
    while status == "interrupt":
        brief = value.get("brief", "") if isinstance(value, dict) else str(value)
        intent = value.get("intent", "") if isinstance(value, dict) else ""
        console.print(
            Panel(
                brief,
                title=f"[bold]research brief[/] [dim]({intent})[/]",
                border_style="yellow",
            )
        )
        ans = Prompt.ask(
            "[bold]Proceed?[/] [dim]Y = go · type to refine the brief · n = cancel[/]",
            default="y",
        )
        if ans.strip().lower() in {"n", "no", "q", "quit", "cancel"}:
            console.print("[red]cancelled[/]")
            return []
        status, value = _drive(graph, Command(resume=ans), config)

    final = graph.get_state(config).values
    report = final.get("report") or final.get("findings") or "(no results)"
    console.print(Panel(Markdown(report), title="[bold green]report[/]", border_style="green"))
    return final.get("papers", [])


def _show_papers(papers: list[dict]) -> None:
    if not papers:
        console.print("[dim]no papers yet — run a search first[/]")
        return
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", justify="right")
    table.add_column("title")
    table.add_column("year", justify="right")
    table.add_column("src")
    for i, p in enumerate(papers, 1):
        table.add_row(str(i), truncate(p.get("title", ""), 70), str(p.get("year") or "—"), p.get("source", ""))
    console.print(table)


def _read_paper(papers: list[dict], idx: int) -> None:
    if not (1 <= idx <= len(papers)):
        console.print(f"[red]no paper #{idx}[/] (have {len(papers)})")
        return
    paper = papers[idx - 1]
    console.print(f"[dim]reading: {paper.get('title')} …[/]")
    with console.status("[cyan]downloading & summarizing…[/]"):
        out = summarize_paper(paper, backend=active_backend())
    console.print(Panel(Markdown(out["text"]), border_style="blue"))


def _save_paper(papers: list[dict], idx: int) -> None:
    if not (1 <= idx <= len(papers)):
        console.print(f"[red]no paper #{idx}[/]")
        return
    from .tools.pdf_reader import _ensure_pdf

    path = _ensure_pdf(papers[idx - 1])
    if path:
        console.print(f"[green]saved[/] → {path}")
    else:
        console.print("[yellow]no downloadable PDF for that paper[/]")


def main() -> None:
    set_active_backend(settings.default_backend)
    console.clear()
    render_banner(console, active_backend(), settings.ollama_model)

    graph = build_graph()
    session_papers: list[dict] = []

    while True:
        try:
            line = Prompt.ask("\n[bold cyan]jarvis[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye 👋[/]")
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in {"/quit", "/exit", "/q"}:
                console.print("[dim]bye 👋[/]")
                break
            elif cmd == "/help":
                console.print(_HELP)
            elif cmd == "/papers":
                _show_papers(session_papers)
            elif cmd == "/read":
                if arg.isdigit():
                    _read_paper(session_papers, int(arg))
                else:
                    console.print("[red]usage: /read <N>[/]")
            elif cmd == "/save":
                if arg.isdigit():
                    _save_paper(session_papers, int(arg))
                else:
                    console.print("[red]usage: /save <N>[/]")
            elif cmd == "/backend":
                if arg.lower() in {"ollama", "deepseek"}:
                    set_active_backend(arg.lower())
                    _, notice = resolve_backend()
                    console.print(f"[green]backend → {active_backend()}[/]")
                    if notice:
                        console.print(f"[yellow]{notice}[/]")
                else:
                    console.print("[red]usage: /backend ollama|deepseek[/]")
            elif cmd == "/model":
                if arg:
                    settings.ollama_model = arg
                    console.print(f"[green]ollama model → {arg}[/]")
                else:
                    console.print("[red]usage: /model <name>[/]")
            else:
                console.print(f"[red]unknown command {cmd}[/] — try /help")
            continue

        # Free text → run the research graph.
        try:
            papers = _run_query(graph, line)
            if papers:
                session_papers = papers
                console.print(f"[dim]· {len(papers)} papers available — /papers, /read N[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted[/]")
        except Exception as exc:  # keep the REPL alive
            console.print(f"[red]error:[/] {exc}")


if __name__ == "__main__":
    main()
