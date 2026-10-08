"""The live progress view, plain-language tool descriptions, titles and the tab title."""

import io

from rich.console import Console

import jarvis.nodes as nodes
import jarvis.terminal as terminal
from jarvis.progress import ProgressView, describe_tool, plain_line
from jarvis.sources import SourceRegistry
from jarvis.subagent import _target


def _view(verbose=False):
    console = Console(file=io.StringIO(), width=120, force_terminal=False, color_system=None, record=True)
    return ProgressView(console, verbose=verbose), console


def _wave(view):
    view.handle("wave_start", {"agents": [(1, "Gradient-based unlearning"), (2, "Model editing")],
                               "parallel": True, "followup": False, "reused": 0})
    view.handle("agent_start", {"idx": 1, "total": 2, "title": "Gradient-based unlearning", "budget": 8})
    view.handle("agent_tool", {"idx": 1, "tool": "search_arxiv", "target": "LLM unlearning gradient ascent"})
    view.handle("agent_tool", {"idx": 1, "tool": "read_paper", "target": "Large Language Model Unlearning"})


def test_tools_are_described_in_plain_words():
    assert describe_tool("search_semantic_scholar", "influence functions") == \
        'Searching Semantic Scholar: "influence functions"'
    assert describe_tool("read_paper", "A Survey of Knowledge Editing") == "Reading: A Survey of Knowledge Editing"
    assert describe_tool("search_web") == "Searching the web"
    assert describe_tool("new_tool", "x") == "New tool: x"  # unknown tools still read like words
    assert describe_tool("search_web", "y" * 200, width=20).endswith('…"')


def test_live_frame_shows_one_line_per_agent():
    view, console = _view()
    _wave(view)
    console.print(view)  # render one live frame
    frame = console.export_text()
    assert "Researching · 2 agents in parallel" in frame
    assert "Gradient-based unlearning" in frame and "Reading: Large Language Model Unlearning" in frame
    assert "1 search · 1 read" in frame
    assert "Model editing" in frame and "waiting its turn" in frame
    assert "search_arxiv(" not in frame  # no raw function-call syntax


def test_wave_collapses_into_a_short_summary():
    view, console = _view()
    _wave(view)
    view.handle("agent_done", {"idx": 1, "searches": 6, "reads": 2, "sources": 32})
    view.handle("agent_done", {"idx": 2, "searches": 0, "reads": 0, "sources": 0, "failed": True})
    view.handle("wave_done", {"agents": 2, "searches": 6, "reads": 2, "sources": 32, "new_sources": 32,
                              "followup": False})
    out = console.export_text()
    assert "✓ Researched with 2 agents · 6 searches · 2 reads · 32 sources" in out
    assert "✓ Gradient-based unlearning · 32 sources" in out and "✗ Model editing · failed" in out
    assert "Searching arXiv" not in out  # tool calls are not logged unless /verbose
    console.print(view)
    assert "Researching" not in console.export_text()  # the live agent rows are gone


def test_verbose_logs_every_tool_call():
    view, console = _view(verbose=True)
    _wave(view)
    out = console.export_text()
    assert '[1] Searching arXiv: "LLM unlearning gradient ascent"' in out


def test_node_outcomes_and_gaps_read_as_sentences():
    view, console = _view()
    view.node_done("clarify", {"title": "LLM unlearning methods"})
    view.node_done("plan", {"plan": [{}, {}, {}]})
    view.node_done("review", {"followups": []})  # review skipped (no lead iteration): print nothing
    view.handle("gaps", {"titles": ["Recent surveys"]})
    view.node_done("cite", {"citation_stats": {"cited": 41, "retrieved": 108}})
    view.handle("note", {"text": "Trimmed the report evidence", "level": "warn"})
    out = console.export_text()
    assert view.topic == "LLM unlearning methods"
    assert "✓ Understood the request" in out and "✓ Planned 3 research agents" in out
    assert "Checked the findings" not in out
    assert "● Found a gap → 1 more agent: Recent surveys" in out
    assert "✓ Linked 41 citations (108 sources retrieved)" in out
    assert "! Trimmed the report evidence" in out


def test_markup_in_titles_is_not_interpreted():
    view, console = _view()
    view.handle("note", {"text": "odd [bold]title[/bold]"})
    assert "odd [bold]title[/bold]" in console.export_text()


def test_plain_lines_for_headless_scripts():
    assert plain_line("agent_tool", {"idx": 2, "tool": "search_github", "target": "unlearning"}) == \
        '    [2] Searching GitHub: "unlearning"'
    assert plain_line("wave_done", {"agents": 3, "searches": 27, "reads": 9, "sources": 95}) == \
        "✓ 3 agents finished · 27 searches · 9 reads · 95 sources so far"
    assert plain_line("unknown", {}) is None


def test_tool_target_resolves_source_ids_to_titles():
    reg = SourceRegistry()
    sid = reg.add("paper", {"title": "A Survey of Knowledge Editing", "url": "https://arxiv.org/abs/2310.16218"})
    assert _target({"paper": sid, "focus": "taxonomy"}, reg) == "A Survey of Knowledge Editing"
    assert _target({"url": "https://www.example.com/post"}, reg) == "example.com/post"
    assert _target({"query": "unlearning", "max_results": 8}, reg) == "unlearning"
    assert _target({}, reg) == ""


def test_plan_specs_get_short_titles():
    spec = nodes.normalize_spec({"objective": "Identify and summarize recent gradient-based and "
                                              "fine-tuning-style machine unlearning methods for LLMs"}, "ollama")
    assert len(spec["title"]) <= 44 and spec["title"].endswith("…")
    assert nodes.normalize_spec({"objective": "x", "title": "Model editing"}, "ollama")["title"] == "Model editing"


def test_tab_title_text_and_no_output_when_not_a_terminal(capsys):
    assert terminal.title_text() == "✦ Jarvis"
    assert terminal.title_text("LLM unlearning", frame=1) == "◓ LLM unlearning"
    long = terminal.topic_from("how do large language models forget training data after unlearning?")
    assert len(long) <= 40 and long.endswith("…")
    terminal.set_title("x")  # pytest's stdout is not a TTY
    assert capsys.readouterr().out == ""
