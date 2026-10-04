"""Small shared helpers: JSON extraction, chunking, dedup."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def write_json_atomic(path, data: Any, indent: int | None = 2) -> None:
    """Write JSON so a crash can never leave a half-written file: write a temp file in the
    same folder, then swap it in with an atomic rename."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    os.replace(tmp, path)


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort pull of a JSON object out of an LLM response.

    Handles ```json fences, leading prose, and trailing commentary. Returns an
    empty dict if nothing parseable is found (callers should have sane defaults).
    """
    if not text:
        return {}

    # Strip code fences.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        # Grab the first balanced-looking {...} block.
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if start != -1 and end > start else text

    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return {}


def chunk_text(text: str, size: int = 6000, overlap: int = 200) -> list[str]:
    """Split text into character chunks for map-reduce summarization."""
    text = text or ""
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def dedup_papers(papers: list[dict]) -> list[dict]:
    """De-duplicate papers by (lowercased) title or id, preserving order."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in papers:
        key = (p.get("id") or p.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
