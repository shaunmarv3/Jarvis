"""Chat-with-paper: index a read paper into its own store, then Q&A it alone."""

from __future__ import annotations

from .config import SUMMARIES_DIR
from .llm import get_llm
from .store import _pid, find_folder, index_paper, search
from .tools.pdf_reader import _ensure_pdf

_QA_PROMPT = """You are answering a question about a research paper using ONLY the excerpts below.
If the answer isn't in the excerpts, say so plainly.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER (cite which excerpt supports each claim where possible):"""


def ensure_indexed(paper: dict) -> tuple[bool, str, str]:
    """Index the paper into its own folder if needed. Returns (ok, note, folder)."""
    folder = find_folder(paper)
    if folder:
        return True, "already indexed", folder

    path = _ensure_pdf(paper)
    if not path:
        return False, "no PDF available to index", ""
    try:
        import pymupdf4llm

        text = pymupdf4llm.to_markdown(path)
    except Exception as exc:
        return False, f"parse failed: {exc}", ""

    folder, n = index_paper(paper, text)
    return bool(folder), f"indexed {n} chunks", folder


def ask_folder(question: str, folder: str, backend: str | None = None, k: int = 5):
    """Answer a question scoped to a single paper's store (by folder)."""
    docs = search(question, folder, k=k)
    if not docs:
        return "No relevant content found in that paper.", []
    context = "\n\n---\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))
    llm = get_llm(backend, temperature=0.2, num_predict=900)
    answer = llm.invoke(_QA_PROMPT.format(context=context, question=question)).content
    return answer, docs


def save_summary(paper: dict, summary: str) -> str:
    """Persist a paper summary as markdown; returns the path."""
    path = SUMMARIES_DIR / f"{_pid(paper)}.md"
    path.write_text(summary, encoding="utf-8")
    return str(path)
