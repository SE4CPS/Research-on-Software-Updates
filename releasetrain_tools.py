# releasetrain_tools.py
import requests
from typing import Dict, Any, List, Optional
from langchain.tools import tool

BASE = "https://releasetrain.io/api"

def _get(path: str, params: Optional[Dict[str, Any]] = None):
    url = f"{BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

@tool("rt_os", return_direct=False)
def rt_os(q: str = "os", limit: int = 10) -> str:
    """
    Pull OS/component items via /api/component?q=<query>.
    Returns a concise bulleted list with titles, dates, and links.
    """
    data = _get("component", {"q": q})
    items: List[Dict[str, Any]] = data.get("results") if isinstance(data, dict) else data
    items = items[:limit] if items else []
    if not items:
        return "No OS entries."
    out = ["OS RELEASES:"]
    for i, it in enumerate(items, 1):
        title = it.get("title") or it.get("name") or it.get("versionProductName") or "Untitled"
        url = it.get("url") or it.get("link") or ""
        date = it.get("updatedAt") or it.get("createdAt") or it.get("date") or it.get("published") or ""
        desc = (it.get("versionReleaseNotes") or it.get("summary") or it.get("description") or it.get("content") or "")[:240]
        out.append(f"{i}. {title} ({date})\n  {desc}\n  {url}")
    return "\n".join(out)

@tool("rt_reddit", return_direct=False)
def rt_reddit(query: str = "operating system update", limit: int = 10) -> str:
    """
    Pull Reddit items via /api/reddit?q=<query>.
    Returns a concise bulleted list with subreddit, score, comments, and links.
    """
    data = _get("reddit", {"q": query})
    items: List[Dict[str, Any]] = data.get("results") if isinstance(data, dict) else data
    items = items[:limit] if items else []
    if not items:
        return "No Reddit discussions."
    out = ["REDDIT THREADS:"]
    for i, it in enumerate(items, 1):
        title = it.get("title") or "Untitled"
        sub = it.get("subreddit") or ""
        score = it.get("score") or ""
        comments = it.get("num_comments") or ""
        url = it.get("url") or ""
        excerpt = (it.get("selftext") or it.get("content") or "")[:240]
        out.append(f"{i}. r/{sub} ({score}↑, {comments}💬) {title}\n  {excerpt}\n  {url}")
    return "\n".join(out)