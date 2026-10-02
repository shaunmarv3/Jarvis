"""Tool registry.

Two views of the same tools:
  * `TOOL_SCHEMAS` — LangChain `@tool` objects, bound to the LLM so it can choose
    tools (the docstrings/signatures become the tool schema — written, per
    Anthropic's tool-design guidance, to say *when* to use each tool).
  * `TOOL_FUNCS`   — the raw callables that return *structured* results
    ({'papers'|'datasets'|'web': [...], 'text': ...}). The subagent executes these
    so it can register every result as a citable source.

Logic lives once in the raw functions; the @tool wrappers just describe them.
Any argument that takes a paper or URL also accepts a source id like "S3".
"""

from __future__ import annotations

from langchain_core.tools import tool

from .arxiv_tool import arxiv_fetch, arxiv_search
from .community import hn_search, reddit_search
from .crossref import crossref_resolve
from .datasets import dataset_search
from .github import github_search
from .hf_inspect import dataset_inspect
from .hf_models import hf_model_search
from .openalex import openalex_search
from .reader import paper_read
from .semantic_scholar import s2_resolve_title, s2_search
from .web import web_read, web_search

TOOL_FUNCS = {
    "search_arxiv": arxiv_search,
    "fetch_arxiv": lambda id_or_url: arxiv_fetch(id_or_url, download=False),
    "search_semantic_scholar": s2_search,
    "resolve_title": s2_resolve_title,
    "search_openalex": openalex_search,
    "resolve_doi": crossref_resolve,
    "read_paper": paper_read,
    "search_datasets": dataset_search,
    "inspect_dataset": dataset_inspect,
    "search_web": web_search,
    "read_web": web_read,
    "search_github": github_search,
    "search_hf_models": hf_model_search,
    "search_hn": hn_search,
    "search_reddit": reddit_search,
}


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arXiv preprints. Best for recent ML/AI/CS/physics work (often newer than journals).
    Use short keyword queries (2-6 words)."""
    return ""


@tool
def search_semantic_scholar(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar: strong CS/ML coverage with citation counts and venues.
    Good for finding influential (highly cited) papers on a topic."""
    return ""


@tool
def search_openalex(query: str, max_results: int = 5) -> str:
    """Search OpenAlex (250M+ works, all fields) by title/abstract with citation counts.
    Good for breadth outside CS; results can be noisy for jargon — prefer specific phrases."""
    return ""


@tool
def resolve_title(title: str) -> str:
    """Find the single paper matching an exact or approximate title."""
    return ""


@tool
def resolve_doi(title_or_doi: str) -> str:
    """Resolve a DOI (or a full citation string) to canonical metadata via Crossref."""
    return ""


@tool
def fetch_arxiv(id_or_url: str) -> str:
    """Fetch metadata for one arXiv paper by id or URL (e.g. '2309.15217')."""
    return ""


@tool
def read_paper(paper: str, focus: str) -> str:
    """Read a paper's FULL TEXT and get the passages relevant to `focus`.
    `paper` = a source id like 'S3', an arXiv id, a DOI, or a PDF URL.
    Use this to verify methods, numbers and results instead of relying on the abstract."""
    return ""


@tool
def search_datasets(query: str, max_results: int = 6) -> str:
    """Search the HuggingFace Hub for datasets (matches dataset names; use short keywords)."""
    return ""


@tool
def inspect_dataset(dataset: str) -> str:
    """Inspect a HuggingFace dataset (columns, row count, sample rows, README) without downloading.
    `dataset` is a hub id like 'squad' or 'rajpurkar/squad_v2'."""
    return ""


@tool
def search_web(query: str, max_results: int = 6) -> str:
    """Search the open web for blogs, docs, news, industry practice and tools.
    Prefer primary sources (official docs, original authors) over SEO listicles."""
    return ""


@tool
def read_web(url: str, focus: str = "") -> str:
    """Read a web page's main text (passages relevant to `focus`).
    `url` = a source id like 'S7' or a full URL. Use when a snippet isn't enough."""
    return ""


@tool
def search_github(query: str, max_results: int = 5) -> str:
    """Search GitHub repositories (implementations, libraries, tools) with stars and recency.
    Use 1-4 generic keywords, e.g. 'rag evaluation'."""
    return ""


@tool
def search_hf_models(query: str, max_results: int = 5) -> str:
    """Search HuggingFace models (pretrained checkpoints) by name, most-downloaded first."""
    return ""


@tool
def search_hn(query: str, max_results: int = 5) -> str:
    """Search Hacker News stories — what engineers/practitioners discuss and link to."""
    return ""


@tool
def search_reddit(query: str, max_results: int = 5) -> str:
    """Search Reddit posts — practitioner experience and opinions (may be unavailable)."""
    return ""


TOOL_SCHEMAS = [
    search_arxiv,
    search_semantic_scholar,
    search_openalex,
    resolve_title,
    resolve_doi,
    fetch_arxiv,
    read_paper,
    search_datasets,
    inspect_dataset,
    search_web,
    read_web,
    search_github,
    search_hf_models,
    search_hn,
]
# search_reddit is registered in TOOL_FUNCS but not offered to agents: Reddit now answers
# anonymous API clients with HTTP 403. Add it back here if you configure Reddit access.

__all__ = ["TOOL_FUNCS", "TOOL_SCHEMAS"]
