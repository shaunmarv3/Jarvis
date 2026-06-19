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
    "plan": "🧠 lead agent planning subagents",
    "fanout": "🔎 subagents researching",
    "synthesize": "📝 merging subagent findings",
    "finalize": "📄 compiling report",
    "cite": "🔗 attaching citations",
}

_HELP = """[bold cyan]Commands[/]
  [green]<free text>[/]              research anything (plans, confirms, then searches)
  [green]/read <N>[/]                download, summarize & index paper N into its own vector store
  [green]/db[/]                      list papers in the vector DB (each stored separately)
  [green]/ask <question>[/]          ask a question; pick which indexed paper to query (no mixing)
  [green]/use <N>[/]                 select DB paper N as the target for /ask
  [green]/forget <N>[/]              delete DB paper N's vectors
  [green]/save <N>[/]                download paper N's PDF to data/papers
  [green]/papers[/]                  list this session's search results
  [green]/dataset <query>[/]         search HuggingFace + Papers with Code datasets
  [green]/inspect <hub_id>[/]        inspect a dataset (cols, rows, sample, README) — e.g. /inspect squad
  [green]/web <query>[/]             quick web search (DuckDuckGo)
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
            if node == "plan" and isinstance(update, dict):
                for i, sub in enumerate(update.get("plan", []) or [], 1):
                    console.print(f"[dim]  {i}. {truncate(sub.get('objective', ''), 64)}[/]")
            if node == "fanout" and isinstance(update, dict):
                console.print(f"[dim]  ↳ {len(update.get('papers', []))} papers kept[/]")
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
        plan = value.get("plan", []) if isinstance(value, dict) else []
        body = brief
        if plan:
            lines = "\n".join(
                f"  [cyan]{i}.[/] {sub.get('objective', '')}" for i, sub in enumerate(plan, 1)
            )
            body += f"\n\n[bold]Plan — {len(plan)} subagent(s):[/]\n{lines}"
        console.print(
            Panel(
                body,
                title=f"[bold]research brief[/] [dim]({intent})[/]",
                border_style="yellow",
            )
        )
        ans = Prompt.ask(
            "[bold]Proceed?[/] [dim]Y = go · type to refine the brief & re-plan · n = cancel[/]",
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


def _read_paper(papers: list[dict], idx: int) -> tuple[dict, str] | None:
    """Summarize paper #idx, persist summary, index into its OWN store. Returns (paper, folder)."""
    if not (1 <= idx <= len(papers)):
        console.print(f"[red]no paper #{idx}[/] (have {len(papers)})")
        return None
    paper = papers[idx - 1]
    console.print(f"[dim]reading: {paper.get('title')} …[/]")
    with console.status("[cyan]downloading & summarizing…[/]"):
        out = summarize_paper(paper, backend=active_backend())
    console.print(Panel(Markdown(out["text"]), border_style="blue"))

    # Persist summary + index full text into a dedicated vector folder.
    from .qa import ensure_indexed, save_summary

    try:
        spath = save_summary(paper, out["text"])
        console.print(f"[dim]summary saved → {spath}[/]")
    except Exception as exc:
        console.print(f"[yellow]could not save summary: {exc}[/]")
    with console.status("[cyan]indexing for /ask…[/]"):
        ok, note, folder = ensure_indexed(paper)
    if ok:
        console.print(f"[dim]· {note} → folder [cyan]{folder}[/] — now [green]/ask <question>[/][/]")
        return paper, folder
    console.print(f"[yellow]{note}[/]")
    return None


def _list_db(select: bool = False) -> str | None:
    """Show the vectorized-paper library. If select, prompt for one and return its folder."""
    from .store import list_papers

    papers = list_papers()
    if not papers:
        console.print("[dim]no papers in the vector DB yet — /read one first[/]")
        return None
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("#", justify="right")
    table.add_column("title")
    table.add_column("indexed")
    table.add_column("chunks", justify="right")
    table.add_column("folder")
    for i, e in enumerate(papers, 1):
        table.add_row(
            str(i), truncate(e.get("title", "") or e["folder"], 48),
            (e.get("created_at", "") or "").replace("T", " ")[5:], str(e.get("chunks", "")),
            truncate(e["folder"], 34),
        )
    console.print(table)
    if not select:
        return None
    ans = Prompt.ask("[bold]ask which paper #?[/] [dim](enter to cancel)[/]", default="")
    if ans.isdigit() and 1 <= int(ans) <= len(papers):
        return papers[int(ans) - 1]["folder"]
    return None


def _ask(folder: str | None, question: str) -> str | None:
    """Answer a question against ONE paper's store. Returns the folder used."""
    from .store import list_papers
    from .qa import ask_folder

    # If no paper selected, let the user pick from the library (no mixing).
    if not folder:
        papers = list_papers()
        if not papers:
            console.print("[yellow]no papers indexed — /read one first[/]")
            return None
        if len(papers) == 1:
            folder = papers[0]["folder"]
        else:
            folder = _list_db(select=True)
            if not folder:
                return None

    title = next((e["title"] for e in list_papers() if e["folder"] == folder), folder)
    with console.status("[cyan]thinking…[/]"):
        answer, docs = ask_folder(question, folder, backend=active_backend())
    console.print(Panel(Markdown(answer), title=f"[bold]Q&A · {truncate(title, 50)}[/]", border_style="magenta"))
    return folder


def _forget(idx: int) -> None:
    from .store import delete_paper, list_papers

    papers = list_papers()
    if not (1 <= idx <= len(papers)):
        console.print(f"[red]no DB paper #{idx}[/]")
        return
    e = papers[idx - 1]
    ok = delete_paper(e["folder"])
    console.print(f"[green]deleted[/] {truncate(e.get('title', ''), 50)}" if ok else "[yellow]nothing removed[/]")


def _search_datasets_cmd(query: str) -> None:
    from .tools.datasets import dataset_search

    with console.status("[cyan]searching datasets…[/]"):
        out = dataset_search(query)
    console.print(Panel(out["text"], title="[bold]datasets[/]", border_style="cyan"))


def _inspect_dataset_cmd(hub_id: str) -> None:
    from .tools.hf_inspect import dataset_inspect

    with console.status(f"[cyan]inspecting {hub_id}…[/]"):
        out = dataset_inspect(hub_id)
    console.print(Panel(out["text"], title=f"[bold]{hub_id}[/]", border_style="cyan"))


def _web_cmd(query: str) -> None:
    from .tools.web import web_search

    with console.status("[cyan]searching the web…[/]"):
        out = web_search(query)
    console.print(Panel(out["text"], title="[bold]web[/]", border_style="cyan"))


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


def _read_line(prompt: str) -> str:
    """Read a REPL line, stripping a leading BOM that Windows PowerShell prepends
    when piping stdin. The BOM arrives as U+FEFF if stdin is UTF-8, or as its
    cp1252 mojibake (\\xef\\xbb\\xbf) if stdin fell back to cp1252 — handle both.
    `chr()` keeps the markers out of the source as fragile literal bytes.
    """
    raw = Prompt.ask(prompt)
    for bom in (chr(0xFEFF), chr(0xEF) + chr(0xBB) + chr(0xBF)):
        if raw.startswith(bom):
            raw = raw[len(bom):]
            break
    return raw.strip()


def main() -> None:
    # Force UTF-8 so the ASCII banner / box-drawing chars never hit a cp1252 crash on Windows.
    import sys

    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Let deep nodes (subagent fan-out) stream live progress to this console.
    from .events import set_sink

    set_sink(lambda m: console.print(m))

    set_active_backend(settings.default_backend)
    try:
        console.clear()
        render_banner(console, active_backend(), settings.ollama_model)
    except Exception:
        console.print("[bold cyan]J A R V I S[/] — personal research agent")

    graph = build_graph()
    session_papers: list[dict] = []
    current_folder: str | None = None  # active paper's vector folder, target of /ask

    while True:
        try:
            line = _read_line("\n[bold cyan]jarvis[/]")
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
                    res = _read_paper(session_papers, int(arg))
                    if res:
                        current_folder = res[1]
                else:
                    console.print("[red]usage: /read <N>[/]")
            elif cmd == "/db":
                _list_db()
            elif cmd == "/ask":
                if arg:
                    used = _ask(current_folder, arg)
                    if used:
                        current_folder = used
                else:
                    console.print("[red]usage: /ask <question>[/]")
            elif cmd == "/use":
                if arg.isdigit():
                    from .store import list_papers

                    papers_db = list_papers()
                    if 1 <= int(arg) <= len(papers_db):
                        current_folder = papers_db[int(arg) - 1]["folder"]
                        console.print(f"[green]active paper → {truncate(papers_db[int(arg) - 1]['title'], 50)}[/]")
                    else:
                        console.print(f"[red]no DB paper #{arg}[/]")
                else:
                    console.print("[red]usage: /use <N>[/]")
            elif cmd == "/forget":
                if arg.isdigit():
                    _forget(int(arg))
                    current_folder = None
                else:
                    console.print("[red]usage: /forget <N>[/]")
            elif cmd == "/dataset":
                if arg:
                    _search_datasets_cmd(arg)
                else:
                    console.print("[red]usage: /dataset <query>[/]")
            elif cmd == "/inspect":
                if arg:
                    _inspect_dataset_cmd(arg)
                else:
                    console.print("[red]usage: /inspect <hub_id>[/]")
            elif cmd == "/web":
                if arg:
                    _web_cmd(arg)
                else:
                    console.print("[red]usage: /web <query>[/]")
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
