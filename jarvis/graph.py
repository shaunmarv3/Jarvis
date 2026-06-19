r"""Assemble the LangGraph orchestrator-worker research flow.

clarify -> plan(lead) -> confirm(human) --edit--> plan
                                         \--go--> fanout(subagents) -> synthesize
                                                  -> finalize -> cite -> END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    after_confirm,
    cite_node,
    clarify_node,
    confirm_node,
    fanout_node,
    finalize_node,
    plan_node,
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
    g.add_node("plan", plan_node)
    g.add_node("confirm", confirm_node)
    g.add_node("fanout", fanout_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("finalize", finalize_node)
    g.add_node("cite", cite_node)

    g.add_edge(START, "clarify")
    g.add_edge("clarify", "plan")
    g.add_edge("plan", "confirm")
    g.add_conditional_edges("confirm", after_confirm, {"plan": "plan", "fanout": "fanout"})
    g.add_edge("fanout", "synthesize")
    g.add_edge("synthesize", "finalize")
    g.add_edge("finalize", "cite")
    g.add_edge("cite", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
