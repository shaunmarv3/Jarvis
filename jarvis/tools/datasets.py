"""Dataset search across HuggingFace Hub + Papers with Code (both free)."""

from __future__ import annotations

import requests

from ..config import settings
from ..utils import truncate


def _hf_search(query: str, limit: int) -> list[dict]:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        out = []
        # Newer huggingface_hub dropped the `direction` kwarg (sort="downloads" is
        # already descending). Keep a fallback so either version works.
        try:
            hits = api.list_datasets(search=query, limit=limit, sort="downloads")
        except TypeError:
            hits = api.list_datasets(search=query, limit=limit)
        for d in hits:
            out.append(
                {
                    "source": "huggingface",
                    "id": d.id,
                    "title": d.id,
                    "url": f"https://huggingface.co/datasets/{d.id}",
                    "downloads": getattr(d, "downloads", None),
                    "description": (getattr(d, "description", "") or "")[:300],
                }
            )
        return out
    except Exception:
        return []


def _pwc_search(query: str, limit: int) -> list[dict]:
    try:
        resp = requests.get(
            "https://paperswithcode.com/api/v1/datasets/",
            params={"q": query, "items_per_page": limit},
            timeout=settings.request_timeout,
            headers={"User-Agent": "jarvis-research-agent"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", []) or []
        return [
            {
                "source": "paperswithcode",
                "id": r.get("id") or r.get("name"),
                "title": r.get("name") or r.get("full_name"),
                "url": r.get("url") or "https://paperswithcode.com",
                "description": (r.get("description") or "")[:300],
            }
            for r in results
        ]
    except Exception:
        return []


def dataset_search(query: str, max_results: int = 5) -> dict:
    """Search HuggingFace Hub and Papers with Code for datasets."""
    half = max(2, max_results // 2)
    datasets = _hf_search(query, half) + _pwc_search(query, half)
    if not datasets:
        return {"datasets": [], "text": f"No datasets found for '{query}'."}

    lines = [
        f"- [{d['source']}] {d['title']} — {truncate(d.get('description', ''), 140)} ({d['url']})"
        for d in datasets
    ]
    return {
        "datasets": datasets,
        "text": f"Datasets for '{query}':\n" + "\n".join(lines),
    }
