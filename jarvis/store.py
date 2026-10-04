"""Per-paper vector stores.

Each indexed paper gets its OWN folder under data/vectorstore/, named
``<paper-slug>__<HHMMSSmmm>`` (hour/min/sec/millisecond), holding an independent
Chroma collection. A registry.json tracks folder ↔ paper. Keeping papers in
separate stores means /ask can target ONE paper and never mixes content.
"""

from __future__ import annotations

import gc
import json
import re
import shutil
import time
from datetime import datetime

from .config import VECTOR_DIR
from .embeddings import get_embeddings
from .utils import chunk_text, write_json_atomic

_REGISTRY = VECTOR_DIR / "registry.json"
_META = "jarvis_meta.json"  # each store folder also keeps its own registry entry


def _rebuild_registry() -> list[dict]:
    """Recreate the registry from the store folders on disk (each holds its own entry).
    Folders indexed before entries were stored per folder get a minimal entry."""
    reg = []
    for d in sorted(p for p in VECTOR_DIR.iterdir() if p.is_dir() and re.fullmatch(r".+__\d{9}", p.name)):
        try:
            reg.append(json.loads((d / _META).read_text(encoding="utf-8")))
        except Exception:
            reg.append({"folder": d.name, "paper_id": "", "title": d.name.split("__")[0].replace("-", " "),
                        "created_at": "", "chunks": ""})
    return reg


def _load_registry() -> list[dict]:
    if not _REGISTRY.exists():
        return []
    try:
        reg = json.loads(_REGISTRY.read_text(encoding="utf-8"))
        if isinstance(reg, list):
            return reg
    except Exception:
        pass
    # Corrupt registry: keep the bad file for inspection and rebuild from the folders, so a
    # later save can't silently drop every paper indexed before.
    try:
        _REGISTRY.replace(_REGISTRY.with_name(f"registry.corrupt-{datetime.now():%Y%m%d-%H%M%S}.json"))
    except Exception:
        pass
    reg = _rebuild_registry()
    _save_registry(reg)
    return reg


def _save_registry(reg: list[dict]) -> None:
    write_json_atomic(_REGISTRY, reg)


def _pid(paper: dict) -> str:
    raw = paper.get("id") or paper.get("title") or "paper"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw))[:120]


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (title or "paper").lower()).strip("-")[:50] or "paper"


def _open(folder: str):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="paper",
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_DIR / folder),
    )


def find_folder(paper: dict) -> str | None:
    """Return the existing folder for this paper, if already indexed."""
    pid = _pid(paper)
    for e in _load_registry():
        if e.get("paper_id") == pid and (VECTOR_DIR / e["folder"]).exists():
            return e["folder"]
    return None


def is_indexed(paper: dict) -> bool:
    return find_folder(paper) is not None


def index_paper(paper: dict, text: str) -> tuple[str, int]:
    """Create a dedicated store for this paper. Returns (folder, #chunks).

    If already indexed, returns the existing folder with 0 new chunks.
    """
    existing = find_folder(paper)
    if existing:
        return existing, 0

    chunks = chunk_text(text, size=1500, overlap=150)
    if not chunks:
        return "", 0

    now = datetime.now()
    ts = f"{now:%H%M%S}{now.microsecond // 1000:03d}"  # hr+min+sec+millisec
    folder = f"{_slug(paper.get('title') or paper.get('id'))}__{ts}"

    store = _open(folder)
    store.add_texts(
        texts=chunks,
        metadatas=[{"chunk": i} for i in range(len(chunks))],
        ids=[str(i) for i in range(len(chunks))],
    )
    entry = {
        "folder": folder,
        "paper_id": _pid(paper),
        "title": paper.get("title", ""),
        "created_at": now.isoformat(timespec="seconds"),
        "chunks": len(chunks),
    }
    write_json_atomic(VECTOR_DIR / folder / _META, entry)
    reg = _load_registry()
    reg.append(entry)
    _save_registry(reg)
    return folder, len(chunks)


def list_papers() -> list[dict]:
    """Registry entries whose folder still exists on disk."""
    return [e for e in _load_registry() if (VECTOR_DIR / e["folder"]).exists()]


def search(question: str, folder: str, k: int = 5):
    """Similarity search within a single paper's store."""
    if not folder or not (VECTOR_DIR / folder).exists():
        return []
    try:
        return _open(folder).similarity_search(question, k=k)
    except Exception:
        return []


def delete_paper(folder: str) -> bool:
    """Forget a paper: remove its registry entry and (best-effort) its folder.

    Always logically forgets it (registry entry gone). File removal is retried to
    work around Windows holding the Chroma SQLite handle briefly.
    """
    _save_registry([e for e in _load_registry() if e["folder"] != folder])

    path = VECTOR_DIR / folder
    if not path.exists():
        return True
    for _ in range(4):
        gc.collect()
        try:
            shutil.rmtree(path)
            return True
        except Exception:
            time.sleep(0.3)
    return True  # logically forgotten even if files linger on disk
