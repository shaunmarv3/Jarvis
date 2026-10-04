"""The simple system Jarvis must beat to justify its cost: ONE web search + ONE arXiv search,
then ONE LLM call (the same lead model) writes the report. Citations go through the same
source registry and `finalize_citations`, so both systems are graded on equal terms."""

from __future__ import annotations

BASELINE_PROMPT = """You are a research assistant. Answer the user's question using ONLY the search
results below. Today is {date}.

Question: "{query}"

Search results (each tagged with a source id):
{results}

Write a well-structured markdown answer: a title line (# ...), a **TL;DR** of 2-4 sentences,
then the key findings with concrete details. Put the source id (e.g. [S3]) right after every
factual claim. Use only ids shown above, never invent sources, and do NOT write a Sources
section — it is generated automatically. If the results are thin, say so."""

SEARCHES = ("search_web", "search_arxiv")


def run_baseline(query: str, backend: str) -> dict:
    """Returns a state-like dict (report, sources, citation_stats, subagent_reports)."""
    from datetime import date

    from jarvis.config import settings
    from jarvis.llm import get_llm
    from jarvis.sources import SourceRegistry, finalize_citations
    from jarvis.subagent import execute_tool

    registry = SourceRegistry()
    texts = [execute_tool(name, {"query": query}, registry)[0] for name in SEARCHES]
    prompt = BASELINE_PROMPT.format(date=date.today().isoformat(), query=query, results="\n\n".join(texts))
    report = (get_llm(backend, temperature=0.3, role="lead", max_tokens=settings.report_max_tokens)
              .invoke(prompt).content or "").strip()
    sources = registry.to_dict()
    final, cited, stats = finalize_citations(report, sources)
    return {"report": final, "sources": sources, "citation_stats": stats,
            "subagent_reports": [{"tool_calls": len(SEARCHES), "tools": list(SEARCHES)}]}
