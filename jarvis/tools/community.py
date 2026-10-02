"""Practitioner discussion: Hacker News (Algolia API) and Reddit (public JSON). Both keyless."""

from __future__ import annotations

from ..utils import truncate
from . import _http


def hn_search(query: str, max_results: int = 5) -> dict:
    """Search Hacker News stories by relevance."""
    try:
        data = _http.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "tags": "story", "hitsPerPage": max(1, min(int(max_results), 15))},
        )
    except Exception as exc:
        return {"community": [], "text": f"Hacker News search failed: {exc}"}
    posts = []
    for h in data.get("hits", []) or []:
        oid = h.get("objectID")
        posts.append(
            {
                "source": "hackernews",
                "title": h.get("title") or "",
                "url": f"https://news.ycombinator.com/item?id={oid}",
                "link": h.get("url") or "",
                "snippet": f"{h.get('points') or 0} points, {h.get('num_comments') or 0} comments, "
                f"{(h.get('created_at') or '')[:10]}" + (f" — links to {h['url']}" if h.get("url") else ""),
            }
        )
    if not posts:
        return {"community": [], "text": f"No Hacker News stories for '{query}'."}
    return {"community": posts, "text": f"Hacker News for '{query}':\n" + "\n".join(f"- {p['title']} ({p['snippet']})" for p in posts)}


def reddit_search(query: str, max_results: int = 5) -> dict:
    """Search Reddit posts by relevance (public JSON endpoint)."""
    try:
        data = _http.get(
            "https://www.reddit.com/search.json",
            params={"q": query, "limit": max(1, min(int(max_results), 15)), "sort": "relevance", "t": "all"},
            headers={"User-Agent": "jarvis-research-agent/0.3 (research CLI)"},
        )
    except Exception as exc:
        return {"community": [], "text": f"Reddit search failed: {exc} (Reddit often blocks anonymous clients — try search_hn)."}
    posts = []
    for c in (data.get("data") or {}).get("children", []) or []:
        d = c.get("data") or {}
        posts.append(
            {
                "source": "reddit",
                "title": d.get("title") or "",
                "url": "https://www.reddit.com" + (d.get("permalink") or ""),
                "snippet": f"r/{d.get('subreddit')} · {d.get('score') or 0} upvotes, {d.get('num_comments') or 0} comments — "
                + truncate((d.get("selftext") or "").replace("\n", " "), 160),
            }
        )
    if not posts:
        return {"community": [], "text": f"No Reddit posts for '{query}'."}
    return {"community": posts, "text": f"Reddit for '{query}':\n" + "\n".join(f"- {p['title']} ({truncate(p['snippet'], 120)})" for p in posts)}
