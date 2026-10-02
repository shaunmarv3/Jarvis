"""GitHub repository search (keyless: ~10 req/min; GITHUB_TOKEN raises the limit)."""

from __future__ import annotations

from ..config import settings
from ..utils import truncate
from . import _http


def github_search(query: str, max_results: int = 5) -> dict:
    """Search GitHub repositories, best match first, with stars and last-push date."""
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    try:
        data = _http.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": max(1, min(int(max_results), 15))},
            headers=headers,
        )
    except Exception as exc:
        return {"code": [], "text": f"GitHub search failed: {exc}"}

    repos = []
    for r in data.get("items", []) or []:
        repos.append(
            {
                "source": "github",
                "title": r.get("full_name", ""),
                "url": r.get("html_url", ""),
                "snippet": (r.get("description") or "").strip(),
                "stars": r.get("stargazers_count", 0),
                "updated": (r.get("pushed_at") or "")[:10],
                "language": r.get("language") or "",
            }
        )
    if not repos:
        return {"code": [], "text": f"No GitHub repos for '{query}'. Try fewer, more generic words."}
    lines = [f"- {r['title']} ★{r['stars']} (pushed {r['updated']}) — {truncate(r['snippet'], 140)}" for r in repos]
    return {"code": repos, "text": f"GitHub repos for '{query}':\n" + "\n".join(lines)}
