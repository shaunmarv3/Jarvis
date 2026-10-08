"""User memory (data/memory.md): file operations, and that every agent — but no headless run — sees it."""

import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from prompt_toolkit.document import Document

import jarvis.memory as memory
import jarvis.nodes as nodes
import jarvis.subagent as sa
from conftest import FakeLLM, fake_paper, tool_call
from jarvis.graph import build_graph
from jarvis.headless import run_research
from jarvis.repl import _make_completer

NOTE = "Explain jargon briefly; I am new to ML."


def test_add_list_remove_and_clear(tmp_path):
    assert memory.path().parent == tmp_path / "data_dir"  # isolated by conftest, never the real file
    assert memory.notes() == [] and memory.text() == ""
    memory.add("  - prefer   papers from 2023+  ")
    memory.add("I use\nPyTorch")
    assert memory.notes() == ["prefer papers from 2023+", "I use PyTorch"]
    raw = memory.raw()
    assert raw.startswith("# Jarvis memory") and "<!--" in raw  # the how-to comment stays in the file
    # The template's example bullets live in a comment: agents never see them and they are not numbered.
    assert memory.text() == "- prefer papers from 2023+\n- I use PyTorch"
    assert memory.remove(3) is None
    assert memory.remove(1) == "prefer papers from 2023+"
    assert memory.notes() == ["I use PyTorch"] and "Always include GitHub" in memory.raw()
    memory.clear()
    assert memory.notes() == [] and memory.text() == ""


def test_hand_edited_file_keeps_free_text():
    memory.path().write_text("# My prefs\n\nI work on medical imaging.\n* skip blogs\n", encoding="utf-8")
    assert memory.notes() == ["skip blogs"]
    assert memory.text() == "# My prefs\n\nI work on medical imaging.\n* skip blogs"


def test_prompt_block_is_capped_per_backend():
    assert memory.prompt_block("", "deepseek") == ""
    long = "x" * 5000
    assert len(memory.prompt_block(long, "ollama")) < len(memory.prompt_block(long, "deepseek")) < 2600
    block = memory.prompt_block(NOTE, "ollama")
    assert "USER MEMORY" in block and NOTE in block and "never cite" in block


def _llm():
    return FakeLLM(subagent_script=[
        AIMessage(content="", tool_calls=[tool_call("search_arxiv", {"query": "rag"})]),
        AIMessage(content="", tool_calls=[tool_call("complete_task", {"report": "Faithfulness [S1]."})]),
    ])


def _wire(monkeypatch, llm):
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(sa, "TOOL_FUNCS", {"search_arxiv": lambda query, max_results=5: {
        "papers": [fake_paper(1)], "text": "arXiv results"}})


def _texts(llm) -> list[str]:
    return [m if isinstance(m, str) else "\n".join(str(getattr(x, "content", x)) for x in m) for m in llm.calls]


def test_cli_state_memory_reaches_every_agent(monkeypatch):
    """The CLI puts memory.text() in the initial state; clarify, plan, subagents and the report all see it."""
    llm = _llm()
    _wire(monkeypatch, llm)
    graph = build_graph(MemorySaver())
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 60}
    payload = {"messages": [HumanMessage(content="q")], "query": "how is RAG evaluated?", "backend": "ollama",
               "memory": NOTE}
    for _ in range(4):  # clarify -> confirm interrupts
        interrupted = [c for c in graph.stream(payload, config, stream_mode="updates") if "__interrupt__" in c]
        if not interrupted:
            break
        payload = Command(resume="yes")
    texts = _texts(llm)
    for marker in ("Classify intent", "orchestrating a team", "You are a research SUBAGENT",
                   "writing the final research report"):
        prompt = next(t for t in texts if marker in t)
        assert NOTE in prompt and "USER MEMORY" in prompt, marker


def test_headless_runs_and_evals_never_see_the_memory_file(monkeypatch):
    memory.add(NOTE)
    llm = _llm()
    _wire(monkeypatch, llm)
    final = run_research("how is RAG evaluated?", backend="ollama")
    assert final["report"]
    assert not any("USER MEMORY" in t or NOTE in t for t in _texts(llm))


def test_completer_offers_memory_subcommands_and_notes():
    memory.add("prefer papers from 2023+")
    memory.add("skip Medium posts")
    comp = _make_completer()

    def texts(t):
        return [c.text for c in comp.get_completions(Document(t, len(t)), None)]

    assert "/memory" in texts("/mem") and "/remember" in texts("/rem")
    assert set(texts("/memory ")) == {"edit", "remove", "clear", "add"}
    assert texts("/memory remove ") == ["1", "2"]
    assert texts("/memory remove 2") == ["2"]
    assert texts("/memory edit ") == []  # no list for other sub-commands
