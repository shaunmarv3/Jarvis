"""Assemble the LangGraph research loop."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    act_node,
    clarify_node,
    confirm_node,
    finalize_node,
    reflect_node,
    should_continue,
    synthesize_node,
)
from .state import AgentState


def build_graph(checkpointer=None):
    """Build & compile the agent graph.

    A checkpointer is required for the human-in-the-loop `interrupt()` in the
    confirm node to pause/resume; defaults to an in-memory saver.
    """
    g = StateGraph(AgentState)
    g.add_node("clarify", clarify_node)
    g.add_node("confirm", confirm_node)
    g.add_node("act", act_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("reflect", reflect_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "clarify")
    g.add_edge("clarify", "confirm")
    g.add_edge("confirm", "act")
    g.add_edge("act", "synthesize")
    g.add_edge("synthesize", "reflect")
    g.add_conditional_edges(
        "reflect", should_continue, {"act": "act", "finalize": "finalize"}
    )
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
