"""Tool registry.

Two views of the same tools:
  * `TOOL_SCHEMAS` — LangChain `@tool` objects, bound to the LLM so it can choose
    tools (the docstrings/signatures become the tool schema).
  * `TOOL_FUNCS`   — the raw callables that return *structured* results
    ({'papers'|'datasets': [...], 'text': ...}). The graph executes these so it
    can populate state, not just feed strings back to the model.

Logic lives once in the raw functions; the @tool wrappers just call them.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .arxiv_tool import arxiv_fetch, arxiv_search
from .crossref import crossref_resolve
from .datasets import dataset_search
from .openalex import openalex_search
from .semantic_scholar import s2_resolve_title, s2_search

# name -> raw structured callable (executed by the graph's act node)
TOOL_FUNCS = {
    "search_arxiv": arxiv_search,
    "fetch_arxiv": arxiv_fetch,
    "search_semantic_scholar": s2_search,
    "resolve_title": s2_resolve_title,
    "search_openalex": openalex_search,
    "resolve_doi": crossref_resolve,
    "search_datasets": dataset_search,
}


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv for preprints by topic/keywords. Best for recent ML/CS/physics work."""
    return arxiv_search(query, max_results)["text"]


@tool
def fetch_arxiv(id_or_url: str) -> str:
    """Fetch a specific arXiv paper by its id or URL (e.g. '2301.12345') and download the PDF."""
    return arxiv_fetch(id_or_url)["text"]


@tool
def search_semantic_scholar(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar for papers (strong CS coverage, citations, abstracts)."""
    return s2_search(query, max_results)["text"]


@tool
def resolve_title(title: str) -> str:
    """Resolve a fuzzy/approximate paper title to the single best-matching paper."""
    return s2_resolve_title(title)["text"]


@tool
def search_openalex(query: str, max_results: int = 5) -> str:
    """Search OpenAlex (250M+ works across all fields) with citation counts. Good for breadth."""
    return openalex_search(query, max_results)["text"]


@tool
def resolve_doi(title_or_doi: str) -> str:
    """Resolve a DOI or messy citation/title to canonical metadata via Crossref."""
    return crossref_resolve(title_or_doi)["text"]


@tool
def search_datasets(query: str, max_results: int = 6) -> str:
    """Search HuggingFace Hub and Papers with Code for datasets matching a topic/task."""
    return dataset_search(query, max_results)["text"]


TOOL_SCHEMAS = [
    search_arxiv,
    fetch_arxiv,
    search_semantic_scholar,
    resolve_title,
    search_openalex,
    resolve_doi,
    search_datasets,
]

__all__ = ["TOOL_FUNCS", "TOOL_SCHEMAS"]
