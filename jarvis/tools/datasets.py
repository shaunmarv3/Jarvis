"""Dataset search on the HuggingFace Hub (free, no key).

Papers with Code was shut down in July 2025 (it now redirects to HF), so the HF Hub
is the single source here.
"""

from __future__ import annotations

import re

from ..utils import truncate


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _hf_search(query: str, limit: int) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    # Newer huggingface_hub dropped `direction` (sort="downloads" is descending already).
    try:
        hits = api.list_datasets(search=query, limit=limit, sort="downloads", full=True)
    except TypeError:
        hits = api.list_datasets(search=query, limit=limit)
    out = []
    for d in hits:
        out.append(
            {
                "source": "huggingface",
                "id": d.id,
                "title": d.id,
                "url": f"https://huggingface.co/datasets/{d.id}",
                "downloads": getattr(d, "downloads", None),
                "description": _clean(getattr(d, "description", "") or "")[:300],
            }
        )
    return out


def dataset_search(query: str, max_results: int = 6) -> dict:
    """Search the HuggingFace Hub for datasets (most-downloaded first)."""
    try:
        datasets = _hf_search(query, max(1, min(int(max_results), 20)))
    except Exception as exc:
        return {"datasets": [], "text": f"Dataset search failed: {exc}"}
    if not datasets:
        return {
            "datasets": [],
            "text": f"No datasets found for '{query}'. HF search matches dataset names — try a shorter keyword (e.g. 'squad', 'qa').",
        }
    lines = [
        f"- {d['title']} ({d.get('downloads') or 0} downloads) — {truncate(d.get('description', ''), 140)} ({d['url']})"
        for d in datasets
    ]
    return {"datasets": datasets, "text": f"Datasets for '{query}':\n" + "\n".join(lines)}
