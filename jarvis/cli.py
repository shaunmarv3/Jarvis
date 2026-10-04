"""Jarvis CLI — a Rich REPL over the LangGraph research agent."""

from __future__ import annotations

import json
import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import llm as _llm
from .banner import render_banner
from .config import DATA_DIR, settings, subagent_ceiling
from .graph import build_graph
from .llm import active_backend, model_label, resolve_backend, set_active_backend, usage
from .nodes import build_prior
from .repl import COMMANDS, read_line, set_session_papers_getter
from .tools.pdf_reader import summarize_paper
from .utils import truncate, write_json_atomic

console = Console()

_NODE_LABELS = {
    "clarify": " understanding your request",
    "brief": " refining the brief from your answers",
    "plan": " lead agent planned the research",
    "fanout": " subagents finished researching",
    "review": " lead reviewed the findings",
    "followup": " follow-up subagents finished",
    "report": " lead wrote the report",
    "cite": " citations attached",
}
_LAST_RUN = DATA_DIR / "last_run.json"
_LAST_USAGE = DATA_DIR / "last_run_usage.json"  # token counts, saved after every LLM call


def _build_help() -> str:
    """Build the /help text from the same COMMANDS list that powers the live dropdown."""
    rows = [
        "[bold cyan]Commands[/]  [dim](type / for live suggestions)[/]",
        f"  [green]{'<free text>'.ljust(22)}[/] research anything (plans, confirms, then searches)",
    ]
    for name, hint, desc in COMMANDS:
        left = (f"{name} {hint}".strip()).ljust(22)
        rows.append(f"  [green]{left}[/] {desc}")
    return "\n".join(rows)


_HELP = _build_help()


# What the agent is busy doing *after* each node returns (the next blocking stage),
# so the live spinner always names what's actually running — never blank.
_BUSY_AFTER = {
    "clarify": "planning the research",
    "brief": "planning the research",
    "fanout": "lead reviewing findings for gaps",
    "review": "lead writing the report",
    "followup": "lead writing the report",
    "report": "attaching citations",
}


def _fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


def _drive(graph, payload, config, busy: str = "working"):
    """Stream the graph behind a live spinner. Returns ('interrupt', value) or ('done', None).

    The spinner names the stage that's running and ticks every second with elapsed
    time and the cost so far, so a long research run never looks frozen.
    """
    import threading
    import time

    t0 = time.time()
    stage = [busy]
    stop = threading.Event()
    backend = active_backend()

    with console.status(f"[cyan]{busy}…[/]", spinner="dots") as status:

        def tick():
            while not stop.wait(1.0):
                cost = usage.cost(backend)
                money = f" · ${cost:.3f} so far" if backend == "deepseek" else ""
                status.update(f"[cyan]{stage[0]}…[/] [dim]{_fmt_secs(time.time() - t0)}{money}[/]")

        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()
        try:
            for chunk in graph.stream(payload, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    intr = chunk["__interrupt__"]
                    value = intr[0].value if isinstance(intr, (list, tuple)) else intr
                    return ("interrupt", value)
                for node, update in chunk.items():
                    label = _NODE_LABELS.get(node)
                    if label:
                        console.print(f"[dim]· {label}[/]")
                    if node in ("fanout", "followup") and isinstance(update, dict):
                        console.print(f"[dim]  ↳ {len(update.get('sources', {}) or {})} unique sources so far[/]")
                    nxt = _BUSY_AFTER.get(node)
                    if nxt:
                        stage[0] = nxt
        finally:
            stop.set()
    return ("done", None)


def _run_summary(final: dict, backend: str, seconds: float | None) -> None:
    """The end-of-run panel: what the research cost and what it did."""
    reports = final.get("subagent_reports") or []
    calls = sum(r.get("tool_calls") or 0 for r in reports)
    stats = final.get("citation_stats") or {}
    snap = usage.snapshot()
    by_role = usage.cost_by_role(backend)
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    if backend == "deepseek":
        table.add_row("cost", f"[bold green]${usage.cost(backend):.3f}[/]  [dim](estimated at peak rates; off-peak is half)[/]")
        table.add_row("  lead", f"${by_role.get('lead', 0):.3f} · {snap['lead']['calls']} calls · "
                                f"{snap['lead']['in'] / 1000:.1f}k in / {snap['lead']['out'] / 1000:.1f}k out "
                                f"[dim]({settings.deepseek_lead_model})[/]")
        table.add_row("  subagents", f"${by_role.get('worker', 0):.3f} · {snap['worker']['calls']} calls · "
                                     f"{snap['worker']['in'] / 1000:.1f}k in / {snap['worker']['out'] / 1000:.1f}k out "
                                     f"[dim]({settings.deepseek_worker_model})[/]")
    else:
        table.add_row("cost", "[bold green]$0.00[/] [dim](local Ollama)[/]")
    if seconds is not None:
        table.add_row("time", _fmt_secs(seconds))
    table.add_row("research", f"{len(reports)} subagents · {calls} tool calls"
                  + (f" · {final.get('query_type')}" if final.get("query_type") else ""))
    if stats:
        table.add_row("sources", f"{stats.get('cited', 0)} cited of {stats.get('retrieved', 0)} retrieved"
                      + (f" · {stats['invalid_tags']} invalid tags dropped" if stats.get("invalid_tags") else ""))
    console.print(Panel(table, title="[bold]run summary[/]", border_style="cyan", expand=False))


def _ask_clarifying(questions: list[dict]) -> str:
    """Render each clarifying question and collect one answer apiece.

    A number picks an option, free text becomes a custom answer, Enter/0 skips that
    question. Returns a "Q: … | A: …" block (empty if everything was skipped).
    """
    if not questions:
        return ""
    console.print(
        "[dim]a couple of quick questions to sharpen the search "
        "(number to pick · type your own · Enter to skip):[/]"
    )
    lines: list[str] = []
    for q in questions:
        text = q.get("question", "")
        opts = q.get("options", []) or []
        console.print(f"\n[bold cyan]? {text}[/]")
        for i, opt in enumerate(opts, 1):
            console.print(f"   [green]{i}[/]) {opt}")
        console.print("   [dim](or type your own · Enter/0 to skip)[/]")
        raw = Prompt.ask("[bold]>[/]", default="").strip()
        if not raw or raw == "0":
            continue
        answer = opts[int(raw) - 1] if (raw.isdigit() and 1 <= int(raw) <= len(opts)) else raw
        lines.append(f"Q: {text} | A: {answer}")
    return "\n".join(lines)


def _plan_panel(value: dict) -> None:
    plan = value.get("plan", []) or []
    qtype = value.get("query_type") or value.get("intent", "")
    body = value.get("brief", "")
    if plan:
        rows = [
            f"  [cyan]{i}.[/] {sub.get('objective', '')}\n"
            f"     [dim]sources: {', '.join(sub.get('sources', []))} · budget {sub.get('tool_budget', '?')} tool calls[/]"
            for i, sub in enumerate(plan, 1)
        ]
        body += f"\n\n[bold]Plan — {len(plan)} subagent(s):[/]\n" + "\n".join(rows)
    console.print(Panel(body, title=f"[bold]research brief[/] [dim]({qtype})[/]", border_style="yellow"))


def _drive_until_done(graph, payload, config, busy: str) -> bool:
    """Drive the graph through its human-in-the-loop interrupts. False if cancelled."""
    status, value = _drive(graph, payload, config, busy=busy)
    while status == "interrupt":
        itype = value.get("type") if isinstance(value, dict) else None
        if itype == "clarify_questions":
            answers = _ask_clarifying(value.get("questions", []))
            status, value = _drive(graph, Command(resume={"answers": answers}), config, busy="drafting the brief")
            continue
        _plan_panel(value if isinstance(value, dict) else {"brief": str(value)})
        ans = Prompt.ask(
            "[bold]Proceed?[/] [dim]Y = go · type to refine the brief & re-plan · n = cancel[/]",
            default="y",
        )
        if ans.strip().lower() in {"n", "no", "q", "quit", "cancel"}:
            console.print("[red]cancelled[/]")
            return False
        status, value = _drive(graph, Command(resume=ans), config, busy="subagents researching")
    return True


def _show_report(graph, config, backend: str, seconds: float | None = None) -> dict:
    final = dict(graph.get_state(config).values)
    report = final.get("report") or "(no results)"
    console.print(Panel(Markdown(report), title="[bold green]report[/]", border_style="green"))
    _run_summary(final, backend, seconds)
    return final


def _backend_ready(backend: str) -> bool:
    """Fail fast with a fix-it message instead of letting a dead backend produce a hollow report."""
    problem = _llm.check_backend(backend)
    if problem:
        console.print(f"[red]can't start:[/] {problem}")
        return False
    return True


def _run_query(graph, query: str, prior: dict | None = None) -> dict | None:
    """Run one research request through the graph; returns the final state (None if it
    didn't finish). `prior` is the previous run in this session, for follow-up questions."""
    import time

    backend, notice = resolve_backend()
    if notice:
        console.print(f"[yellow]{notice}[/]")
    if not _backend_ready(backend):
        return None

    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    try:
        write_json_atomic(_LAST_RUN, {"thread_id": thread_id, "query": query, "backend": backend})
    except Exception:
        pass
    usage.reset()
    usage.attach(_LAST_USAGE, thread_id)  # so /resume after a crash still knows what was spent
    t0 = time.time()
    init = {"messages": [HumanMessage(content=query)], "query": query, "backend": backend, "prior": prior or {}}
    if not _drive_until_done(graph, init, config, busy="understanding your request"):
        return None
    return _show_report(graph, config, backend, time.time() - t0)


def _last_run() -> dict | None:
    try:
        return json.loads(_LAST_RUN.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resume(graph) -> dict | None:
    """Continue the last run from its latest checkpoint (after a crash or Ctrl+C)."""
    import time

    last = _last_run()
    if not last:
        console.print("[dim]no previous run to resume[/]")
        return None
    thread_id = last["thread_id"]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    snap = graph.get_state(config)
    if not snap or not snap.values:
        console.print("[dim]the last run has no saved checkpoint[/]")
        return None
    backend = last.get("backend", "ollama")
    # Restore what the run already spent (saved after every LLM call), even after a restart.
    usage.reset()
    restored = usage.restore(_LAST_USAGE, thread_id)
    usage.attach(_LAST_USAGE, thread_id)
    if not snap.next:
        console.print(f"[dim]the last run ({truncate(last.get('query', ''), 50)}) already finished — showing it[/]")
        return _show_report(graph, config, backend)
    if not _backend_ready(backend):
        return None
    console.print(f"[cyan]resuming '{truncate(last.get('query', ''), 60)}' at: {', '.join(snap.next)}[/]"
                  + ("" if restored else " [dim](earlier token usage unknown)[/]"))
    set_active_backend(backend)
    t0 = time.time()
    if not _drive_until_done(graph, None, config, busy="resuming"):
        return None
    return _show_report(graph, config, backend, time.time() - t0)


def _show_sources(graph) -> None:
    last = _last_run()
    values = {}
    if last:
        try:
            values = graph.get_state({"configurable": {"thread_id": last["thread_id"]}}).values
        except Exception:
            values = {}
    sources = values.get("sources") or {}
    if not sources:
        console.print("[dim]no sources yet — run a research query first[/]")
        return
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("id", justify="right")
    table.add_column("kind")
    table.add_column("title")
    table.add_column("link")
    for sid, e in sorted(sources.items(), key=lambda kv: int(kv[0][1:])):
        it = e["item"]
        table.add_row(sid, e["kind"], truncate(it.get("title", ""), 60),
                      truncate(it.get("url") or it.get("pdf_url") or "", 50))
    console.print(table)


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
    if not _backend_ready(resolve_backend()[0]):
        return None
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
    if problem := _llm.check_embeddings():
        console.print(f"[yellow]can't index this paper for /ask: {problem}[/]")
        return None
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
    from .qa import ask_folder
    from .store import list_papers

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
    if problem := (_llm.check_embeddings() or _llm.check_backend(resolve_backend()[0])):
        console.print(f"[red]can't answer:[/] {problem}")
        return None
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
        eff = resolve_backend()[0]  # what will really run (DeepSeek without a key runs Ollama)
        render_banner(console, eff, model_label(eff))
    except Exception:
        console.print("[bold cyan]J A R V I S[/] — personal research agent")
    if resolve_backend()[0] == "deepseek":
        console.print(f"[dim]lead: {settings.deepseek_lead_model} · subagents: {settings.deepseek_worker_model}[/]")

    graph = build_graph()
    session_papers: list[dict] = []
    current_folder: str | None = None  # active paper's vector folder, target of /ask
    session_prior: dict = {}  # the last finished run; the next question may follow up on it
    # Let the /save & /read dropdowns list the current search results (closure sees reassignments).
    set_session_papers_getter(lambda: session_papers)

    while True:
        try:
            line = read_line()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye 👋[/]")
            break
        if not line:
            continue

        if line.startswith("/"):
            try:
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
                elif cmd == "/sources":
                    _show_sources(graph)
                elif cmd == "/cost":
                    console.print(f"[dim]{usage.summary(active_backend())}[/]")
                elif cmd == "/new":
                    session_prior = {}
                    console.print("[green]new topic[/] [dim]· the next question starts fresh instead of "
                                  "building on the last report[/]")
                elif cmd == "/resume":
                    try:
                        final = _resume(graph)
                        if final:
                            session_papers = final.get("papers") or session_papers
                            session_prior = build_prior(final)
                    except KeyboardInterrupt:
                        console.print("\n[yellow]interrupted — /resume continues from the last checkpoint[/]")
                    except Exception as exc:
                        console.print(f"[red]error:[/] {exc}")
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
                        # Optional leading paper number (from the dropdown): "/ask 2 <question>".
                        folder, question = current_folder, arg
                        toks = arg.split(maxsplit=1)
                        if toks[0].isdigit():
                            from .store import list_papers

                            dbp = list_papers()
                            n = int(toks[0])
                            if 1 <= n <= len(dbp):
                                folder = dbp[n - 1]["folder"]
                                question = toks[1] if len(toks) > 1 else ""
                        if not question.strip():
                            console.print("[yellow]add a question: /ask <N> <question>[/]")
                        else:
                            used = _ask(folder, question)
                            if used:
                                current_folder = used
                    else:
                        console.print("[red]usage: /ask <question>  (or /ask <N> <question>)[/]")
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
                        eff, notice = resolve_backend()  # effective backend (may fall back)
                        mode = "parallel" if eff == "deepseek" else "sequential"
                        console.print(
                            f"[green]backend → {eff}[/] "
                            f"[dim]· up to {subagent_ceiling(eff)} subagents ({mode})[/]"
                        )
                        if eff == "deepseek":
                            console.print(f"[dim]lead: {settings.deepseek_lead_model} · subagents: {settings.deepseek_worker_model}[/]")
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
            except KeyboardInterrupt:
                console.print("\n[yellow]interrupted[/]")
            except Exception as exc:  # a failing command must not kill the REPL
                console.print(f"[red]error:[/] {exc}")
            continue

        # Free text → run the research graph (it may follow up on the previous run).
        try:
            final = _run_query(graph, line, session_prior)
            if final:
                papers = final.get("papers") or []
                if papers:
                    session_papers = papers
                    console.print(f"[dim]· {len(papers)} papers available — /papers, /read N · /sources for everything[/]")
                session_prior = build_prior(final)
                console.print("[dim]· ask a follow-up (it builds on this report) · /new to start a fresh topic[/]")
        except KeyboardInterrupt:
            console.print("\n[yellow]interrupted — /resume continues from the last checkpoint[/]")
        except Exception as exc:  # keep the REPL alive
            console.print(f"[red]error:[/] {exc} [dim](/resume retries from the last checkpoint)[/]")


if __name__ == "__main__":
    main()
