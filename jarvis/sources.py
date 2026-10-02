"""Source registry + deterministic citations.

Every result a tool returns (paper, web page, dataset) is registered once and gets a
stable id like ``S12``. Subagents see results tagged with those ids and must cite
them; the lead keeps the tags in its report; finally `finalize_citations` renumbers
the tags into ``[1] [2] …`` and builds the Sources list *from the registry*.

So a citation can only ever point at something that was actually retrieved —
fabricated sources are impossible by construction, and unknown tags are dropped
and counted (a useful eval signal). This replaces the old "LLM rewrites the whole
report with citations" pass, which hallucinated and truncated long reports.
"""

from __future__ import annotations

import re
import threading

from .utils import truncate

# (registry kind, key in a tool's structured result)
RESULT_KINDS = [
    ("paper", "papers"),
    ("web", "web"),
    ("dataset", "datasets"),
    ("code", "code"),
    ("model", "models"),
    ("community", "community"),
]

_ARXIV = re.compile(r"(\d{4}\.\d{4,5})")
_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _norm_url(u: str) -> str:
    u = (u or "").strip().lower().split("#", 1)[0]
    u = re.sub(r"^https?://(www\.)?", "", u)
    return u.rstrip("/")


def source_keys(kind: str, item: dict) -> list[str]:
    """All identity keys for an item (any shared key = same source)."""
    keys: list[str] = []
    if kind == "paper":
        blob = " ".join(str(item.get(k) or "") for k in ("id", "url", "pdf_url", "doi"))
        if (m := _ARXIV.search(blob)) and ("arxiv" in blob.lower() or item.get("source") == "arxiv"
                                           or re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", str(item.get("id") or ""))):
            keys.append("arxiv:" + m.group(1))
        if m := _DOI.search(blob):
            keys.append("doi:" + m.group(1).lower().rstrip("."))
        if t := _norm_title(item.get("title", "")):
            keys.append("title:" + t)
    else:
        if u := _norm_url(item.get("url", "")):
            keys.append("url:" + u)
        elif t := _norm_title(item.get("title", "")):
            keys.append("title:" + t)
    return keys


class SourceRegistry:
    """Thread-safe (parallel subagents share one registry per run)."""

    def __init__(self, existing: dict | None = None) -> None:
        self._lock = threading.Lock()
        self.items: dict[str, dict] = {}
        self._by_key: dict[str, str] = {}
        for sid, entry in (existing or {}).items():
            self.items[sid] = entry
            for k in source_keys(entry["kind"], entry["item"]):
                self._by_key.setdefault(k, sid)

    def _next_id(self) -> str:
        nums = [int(s[1:]) for s in self.items if s[1:].isdigit()]
        return f"S{max(nums, default=0) + 1}"

    def add(self, kind: str, item: dict) -> str:
        keys = source_keys(kind, item)
        with self._lock:
            for k in keys:
                if k in self._by_key:
                    sid = self._by_key[k]
                    entry = self.items[sid]["item"]
                    for f, v in item.items():  # enrich (e.g. S2 adds a pdf_url arXiv lacked)
                        if v and not entry.get(f):
                            entry[f] = v
                    for k2 in keys:
                        self._by_key.setdefault(k2, sid)
                    return sid
            sid = self._next_id()
            self.items[sid] = {"kind": kind, "item": dict(item)}
            for k in keys:
                self._by_key[k] = sid
            return sid

    def get(self, sid: str) -> dict | None:
        return self.items.get((sid or "").strip().upper())

    def to_dict(self) -> dict:
        with self._lock:
            return {sid: {"kind": e["kind"], "item": dict(e["item"])} for sid, e in self.items.items()}


# --------------------------------------------------------------------------- rendering


def render_item(sid: str, kind: str, item: dict) -> str:
    """One line describing a source, as shown to models."""
    if kind == "paper":
        bits = [str(item.get("year") or "n.d.")]
        if item.get("cited_by"):
            bits.append(f"cited {item['cited_by']}×")
        if item.get("venue"):
            bits.append(truncate(item["venue"], 40))
        return f"[{sid}] {item.get('title', '')} ({', '.join(bits)}) — {truncate(item.get('abstract') or '(no abstract)', 320)}"
    if kind == "dataset":
        return (f"[{sid}] dataset {item.get('title', '')} ({item.get('downloads') or 0} downloads) — "
                f"{truncate(item.get('description', ''), 160)} ({item.get('url', '')})")
    if kind == "code":
        return (f"[{sid}] repo {item.get('title', '')} ★{item.get('stars', 0)} (pushed {item.get('updated', '?')}) — "
                f"{truncate(item.get('snippet', ''), 180)} ({item.get('url', '')})")
    if kind == "model":
        return (f"[{sid}] model {item.get('title', '')} ({item.get('downloads') or 0} downloads, "
                f"{item.get('snippet', '')}) ({item.get('url', '')})")
    if kind == "community":
        return f"[{sid}] {item.get('source', 'discussion')}: {item.get('title', '')} — {truncate(item.get('snippet', ''), 200)} ({item.get('url', '')})"
    return f"[{sid}] {item.get('title', '')} — {truncate(item.get('snippet', ''), 220)} ({item.get('url', '')})"


def source_link(item: dict) -> str:
    return item.get("url") or item.get("pdf_url") or ""


# --------------------------------------------------------------------------- citations

_TAG = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]", re.IGNORECASE)
_TRAILING_SOURCES = re.compile(r"\n#{1,3}\s*(sources|references|bibliography)\b.*\Z", re.IGNORECASE | re.DOTALL)


def finalize_citations(report: str, sources: dict) -> tuple[str, list[dict], dict]:
    """Turn [S#] tags into [1], [2]… (order of first use) and append a Sources list.

    Returns (final_text, cited_entries, stats) where stats counts valid/invalid tags.
    """
    report = _TRAILING_SOURCES.sub("", report or "").rstrip()
    order: list[str] = []
    invalid: list[str] = []

    def repl(m: re.Match) -> str:
        nums = []
        for sid in re.split(r"\s*[,;]\s*", m.group(1).upper()):
            if sid not in sources:
                invalid.append(sid)
                continue
            if sid not in order:
                order.append(sid)
            n = order.index(sid) + 1
            if n not in nums:
                nums.append(n)
        return "".join(f"[{n}]" for n in nums)

    text = _TAG.sub(repl, report)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)  # tidy "claim [S99] ." after dropped tags

    cited = []
    lines = []
    for n, sid in enumerate(order, 1):
        entry = sources[sid]
        item = entry["item"]
        year = f" ({item['year']})" if entry["kind"] == "paper" and item.get("year") else ""
        lines.append(f"- [{n}] {item.get('title', '(untitled)')}{year} — {source_link(item)}")
        cited.append({"n": n, "sid": sid, **entry})
    if lines:
        text += "\n\n## Sources\n" + "\n".join(lines)
    stats = {"cited": len(order), "invalid_tags": len(invalid), "retrieved": len(sources)}
    return text, cited, stats
