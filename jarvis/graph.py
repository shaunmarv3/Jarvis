r"""Assemble the LangGraph orchestrator-worker research flow.

clarify --questions?--> ask(human) -> brief -> plan
        \--none-------------------------------> plan
plan -> confirm(human) --edit--> plan
                        \--go--> fanout(subagents) -> review(lead)
review --gaps--> followup(one extra wave) -> report
       \--ok--------------------------------> report -> cite -> END
"""

from __future__ import annotations

import sqlite3

from langgraph.graph import END, START, StateGraph

from .config import DATA_DIR
from .nodes import (
    after_clarify,
    after_confirm,
    after_review,
    ask_node,
    brief_node,
    cite_node,
    clarify_node,
    confirm_node,
    fanout_node,
    followup_node,
    plan_node,
    report_node,
    review_node,
)
from .state import AgentState

CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"


def default_checkpointer():
    """Durable checkpoints so an interrupted run can be resumed (/resume)."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver(sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False))
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def build_graph(checkpointer=None):
    """Build & compile the agent graph (a checkpointer is needed for interrupts/resume)."""
    g = StateGraph(AgentState)
    g.add_node("clarify", clarify_node)
    g.add_node("ask", ask_node)
    g.add_node("brief", brief_node)
    g.add_node("plan", plan_node)
    g.add_node("confirm", confirm_node)
    g.add_node("fanout", fanout_node)
    g.add_node("review", review_node)
    g.add_node("followup", followup_node)
    g.add_node("report", report_node)
    g.add_node("cite", cite_node)

    g.add_edge(START, "clarify")
    g.add_conditional_edges("clarify", after_clarify, {"ask": "ask", "plan": "plan"})
    g.add_edge("ask", "brief")
    g.add_edge("brief", "plan")
    g.add_edge("plan", "confirm")
    g.add_conditional_edges("confirm", after_confirm, {"plan": "plan", "fanout": "fanout"})
    g.add_edge("fanout", "review")
    g.add_conditional_edges("review", after_review, {"followup": "followup", "report": "report"})
    g.add_edge("followup", "report")
    g.add_edge("report", "cite")
    g.add_edge("cite", END)

    return g.compile(checkpointer=checkpointer or default_checkpointer())
