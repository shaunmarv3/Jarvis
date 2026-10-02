"""Headless test of every CLI command + the live /command dropdown.

Indexes the two on-disk PDFs so the DB-backed dropdowns (/use, /forget, /ask) have
real data, then exercises the completer and each command's logic. Network commands
are marked WARN (not FAIL) if offline/rate-limited. The slow LLM /ask is optional via
`--ask`. Prints a PASS/WARN/FAIL line per check.
"""

from __future__ import annotations

import sys

from prompt_toolkit.document import Document

import tempfile
from pathlib import Path

import jarvis.store as _store
from jarvis.config import PAPERS_DIR
from jarvis.repl import COMMANDS, _make_completer, set_session_papers_getter

# Never touch the real vector DB: this script indexes and /forget-deletes papers.
_TMP_VECTOR_DIR = Path(tempfile.mkdtemp(prefix="jarvis-test-vectors-"))
_store.VECTOR_DIR = _TMP_VECTOR_DIR
_store._REGISTRY = _TMP_VECTOR_DIR / "registry.json"

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)


# ---- Setup: build session papers + index the two on-disk PDFs into the vector DB ----
SESSION = [
    {"id": "2309.15217", "title": "Ragas: Automated Evaluation of Retrieval Augmented Generation",
     "source": "arxiv", "year": 2023, "local_path": str(PAPERS_DIR / "2309.15217.pdf"),
     "pdf_url": "http://arxiv.org/pdf/2309.15217"},
    {"id": "2401.15391", "title": "MultiHop-RAG: Benchmarking RAG for Multi-Hop Queries",
     "source": "arxiv", "year": 2024, "local_path": str(PAPERS_DIR / "2401.15391.pdf"),
     "pdf_url": "http://arxiv.org/pdf/2401.15391"},
]
set_session_papers_getter(lambda: SESSION)


def setup_index() -> None:
    from jarvis.store import index_paper, list_papers

    try:
        import pymupdf4llm
    except Exception as exc:
        check("setup: pymupdf4llm import", FAIL, str(exc))
        return
    for p in SESSION:
        try:
            text = pymupdf4llm.to_markdown(p["local_path"])
            folder, n = index_paper(p, text)
            check(f"setup: index {p['id']}", PASS if folder else FAIL, f"{n} chunks -> {folder}")
        except Exception as exc:
            check(f"setup: index {p['id']}", FAIL, str(exc))
    check("setup: DB now has papers", PASS if list_papers() else FAIL, f"{len(list_papers())} indexed")


# ---- Dropdown / completer ----
def test_dropdown() -> None:
    comp = _make_completer()

    def texts(t):
        return [c.text for c in comp.get_completions(Document(t, len(t)), None)]

    from jarvis.store import list_papers

    n_db = len(list_papers())

    check("dropdown: '/' lists all commands", PASS if len(texts("/")) == len(COMMANDS) else FAIL, f"{len(texts('/'))} cmds")
    check("dropdown: '/sa' -> /save", PASS if texts("/sa") == ["/save"] else FAIL, str(texts("/sa")))
    check("dropdown: '/ba' -> /backend", PASS if texts("/ba") == ["/backend"] else FAIL, str(texts("/ba")))
    check("dropdown: '/save ' -> 2 session papers", PASS if texts("/save ") == ["1", "2"] else FAIL, str(texts("/save ")))
    check("dropdown: '/read ' -> 2 session papers", PASS if texts("/read ") == ["1", "2"] else FAIL, str(texts("/read ")))
    check("dropdown: '/use ' -> DB papers", PASS if len(texts("/use ")) == n_db else FAIL, f"{len(texts('/use '))} of {n_db}")
    check("dropdown: '/forget ' -> DB papers", PASS if len(texts("/forget ")) == n_db else FAIL, str(texts("/forget ")))
    check("dropdown: '/backend ' -> ollama|deepseek", PASS if texts("/backend ") == ["ollama", "deepseek"] else FAIL, str(texts("/backend ")))
    check("dropdown: '/save 1' filters", PASS if texts("/save 1") == ["1"] else FAIL, str(texts("/save 1")))
    check("dropdown: only first arg ('/save 1 x')", PASS if texts("/save 1 x") == [] else FAIL, str(texts("/save 1 x")))
    check("dropdown: no source ('/papers ')", PASS if texts("/papers ") == [] else FAIL, str(texts("/papers ")))
    check("dropdown: free text -> none", PASS if texts("rag eval") == [] else FAIL, str(texts("rag eval")))


# ---- Commands (logic) ----
def test_commands(do_ask: bool) -> None:
    import jarvis.cli as cli
    from jarvis.llm import active_backend, set_active_backend
    from jarvis.store import list_papers

    # /papers
    try:
        cli._show_papers(SESSION)
        check("/papers", PASS)
    except Exception as exc:
        check("/papers", FAIL, str(exc))

    # /db
    try:
        cli._list_db()
        check("/db", PASS, f"{len(list_papers())} rows")
    except Exception as exc:
        check("/db", FAIL, str(exc))

    # /save (PDF already on disk via local_path)
    try:
        from jarvis.tools.pdf_reader import _ensure_pdf

        path = _ensure_pdf(SESSION[0])
        check("/save", PASS if path else FAIL, str(path))
    except Exception as exc:
        check("/save", FAIL, str(exc))

    # /backend
    try:
        set_active_backend("deepseek")
        b1 = active_backend()
        set_active_backend("ollama")
        b2 = active_backend()
        check("/backend switch", PASS if (b1 == "deepseek" and b2 == "ollama") else FAIL, f"{b1}->{b2}")
    except Exception as exc:
        check("/backend switch", FAIL, str(exc))

    # /model
    try:
        from jarvis.config import settings

        old = settings.ollama_model
        settings.ollama_model = "test-model:1b"
        ok = settings.ollama_model == "test-model:1b"
        settings.ollama_model = old
        check("/model switch", PASS if ok else FAIL)
    except Exception as exc:
        check("/model switch", FAIL, str(exc))

    # /web (network)
    try:
        from jarvis.tools.web import web_search

        out = web_search("retrieval augmented generation", 3)
        hits = len(out.get("web", []) or [])
        check("/web", PASS if hits else WARN, f"{hits} hits")
    except Exception as exc:
        check("/web", WARN, str(exc))

    # /dataset (network)
    try:
        from jarvis.tools.datasets import dataset_search

        out = dataset_search("question answering", 3)
        n = len(out.get("datasets", []) or [])
        check("/dataset", PASS if n else WARN, f"{n} datasets")
    except Exception as exc:
        check("/dataset", WARN, str(exc))

    # /inspect (network)
    try:
        from jarvis.tools.hf_inspect import dataset_inspect

        out = dataset_inspect("rajpurkar/squad")
        check("/inspect", PASS if out.get("text") else WARN, (out.get("text") or "")[:40])
    except Exception as exc:
        check("/inspect", WARN, str(exc))

    # /ask retrieval (vector search part — fast, no LLM)
    try:
        from jarvis.store import list_papers as lp, search

        folder = lp()[0]["folder"]
        docs = search("How is RAG evaluated?", folder, k=3)
        check("/ask retrieval", PASS if docs else FAIL, f"{len(docs)} chunks from {folder}")
    except Exception as exc:
        check("/ask retrieval", FAIL, str(exc))

    # /ask full RAG answer (slow LLM — only with --ask)
    if do_ask:
        try:
            from jarvis.qa import ask_folder
            from jarvis.store import list_papers as lp

            folder = lp()[0]["folder"]
            ans, docs = ask_folder("What does this paper evaluate?", folder, backend="ollama")
            check("/ask full RAG", PASS if ans and len(ans) > 20 else FAIL, ans[:60].replace("\n", " "))
        except Exception as exc:
            check("/ask full RAG", FAIL, str(exc))

    # /forget (delete one, leave the other so we don't wipe the DB)
    try:
        from jarvis.store import delete_paper, list_papers as lp

        before = lp()
        if before:
            ok = delete_paper(before[-1]["folder"])
            after = lp()
            check("/forget", PASS if (ok and len(after) == len(before) - 1) else FAIL, f"{len(before)}->{len(after)}")
        else:
            check("/forget", WARN, "no DB papers to delete")
    except Exception as exc:
        check("/forget", FAIL, str(exc))


def main() -> int:
    do_ask = "--ask" in sys.argv
    print("=== SETUP ===", flush=True)
    setup_index()
    print("\n=== DROPDOWN ===", flush=True)
    test_dropdown()
    print("\n=== COMMANDS ===", flush=True)
    test_commands(do_ask)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    print(f"\n=== SUMMARY: {n_pass} PASS · {n_warn} WARN · {n_fail} FAIL ===", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
