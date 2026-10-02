"""HuggingFace Hub model search (free, no key)."""

from __future__ import annotations


def hf_model_search(query: str, max_results: int = 5) -> dict:
    """Search HF models, most-downloaded first."""
    try:
        from huggingface_hub import HfApi

        hits = HfApi().list_models(search=query, limit=max(1, min(int(max_results), 15)), sort="downloads")
        models = [
            {
                "source": "huggingface",
                "title": m.id,
                "url": f"https://huggingface.co/{m.id}",
                "snippet": f"task: {getattr(m, 'pipeline_tag', None) or 'n/a'}",
                "downloads": getattr(m, "downloads", None) or 0,
                "likes": getattr(m, "likes", None) or 0,
            }
            for m in hits
        ]
    except Exception as exc:
        return {"models": [], "text": f"HF model search failed: {exc}"}
    if not models:
        return {"models": [], "text": f"No HF models for '{query}'. HF search matches model names — use short keywords."}
    lines = [f"- {m['title']} ({m['downloads']} downloads, {m['likes']} likes, {m['snippet']})" for m in models]
    return {"models": models, "text": f"HF models for '{query}':\n" + "\n".join(lines)}
