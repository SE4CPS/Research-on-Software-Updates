# hardcoded_last_week_problems.py
import re
import requests
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

BASE = "https://releasetrain.io/api"

def _iso_parse(s: Optional[str]) -> Optional[datetime]:
    """Parse ISO-ish strings (with optional 'Z' and millis) into UTC-aware datetimes."""
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        core = s.split(".")[0]  # drop millis if present
        dt = datetime.fromisoformat(core)
    except Exception:
        try:
            core = s.split(".")[0]
            dt = datetime.strptime(core, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None
    # Make UTC-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def _within_last_week(dt: Optional[datetime], now: datetime) -> bool:
    return bool(dt) and (now - timedelta(days=7) <= dt <= now)

def _get(path: str, params: Optional[Dict[str, Any]] = None):
    url = f"{BASE}/{path.lstrip('/')}"
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    # Endpoint sometimes returns {"results": [...]} or just [...]
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data

def _score_problem(text: str) -> int:
    """Very simple heuristic for 'problem' mentions."""
    text = text.lower()
    hits = 0
    patterns = [
        r"\bbug(s)?\b", r"\bissue(s)?\b", r"\bproblem(s)?\b",
        r"\bcrash(es|ed|ing)?\b", r"\bfreeze(s|d|ing)?\b",
        r"\bhang(s|ing)?\b", r"\bregression(s)?\b",
        r"\bfail(s|ed|ure)?\b", r"\bbroken\b", r"\bnot working\b",
        r"\bhotfix\b", r"\bworkaround\b",
    ]
    for p in patterns:
        if re.search(p, text):
            hits += 1
    return hits

def _summarize_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        title = it.get("title") or it.get("name") or it.get("versionProductName") or "Untitled"
        url = it.get("url") or it.get("link") or ""
        notes = (
            it.get("versionReleaseNotes")
            or it.get("summary")
            or it.get("description")
            or it.get("content")
            or ""
        )
        dt = _iso_parse(
            it.get("updatedAt")
            or it.get("createdAt")
            or it.get("date")
            or it.get("published")
            or it.get("created_utc")
        )
        out.append(
            {
                "title": title,
                "url": url,
                "date": dt.isoformat() if dt else "",
                "problem_score": _score_problem(f"{title}\n{notes}"),
                "excerpt": (notes[:300] + "…") if notes and len(notes) > 300 else notes,
            }
        )
    return out

def fetch_last_week_problem_updates(limit_os: int = 100, limit_reddit: int = 100):
    now = datetime.now(timezone.utc)

    os_items = _get("component", {"q": "os"})
    if not isinstance(os_items, list):
        os_items = []
    os_items = os_items[:limit_os]

    reddit_items = _get("reddit")
    if not isinstance(reddit_items, list):
        reddit_items = []
    reddit_items = reddit_items[:limit_reddit]

    def is_last_week(it: Dict[str, Any]) -> bool:
        dt = _iso_parse(
            it.get("updatedAt")
            or it.get("createdAt")
            or it.get("date")
            or it.get("published")
            or it.get("created_utc")
        )
        return _within_last_week(dt, now)

    os_last_week = [it for it in os_items if is_last_week(it)]
    rd_last_week = [it for it in reddit_items if is_last_week(it)]

    os_summ = _summarize_items(os_last_week)
    rd_summ = _summarize_items(rd_last_week)

    os_problems = [x for x in os_summ if x["problem_score"] > 0]
    rd_problems = [x for x in rd_summ if x["problem_score"] > 0]

    return os_problems, rd_problems, now

def build_prompt(os_hits: List[Dict[str, Any]], rd_hits: List[Dict[str, Any]], now: datetime) -> str:
    header = (
        "You are ReleaseNotesRec. Summarize problematic patch updates from the last 7 days.\n"
        "Rules:\n"
        "- Group by OS vs Reddit.\n"
        "- For each item: Title, one-line issue summary, date (YYYY-MM-DD), and link.\n"
        "- 6–10 bullets total. If nothing found, say so.\n\n"
    )
    lines = [
        header,
        "Question: Which new patch updates released by software companies last week have problems?",
        f"Today (UTC): {now.date().isoformat()}",
        "",
    ]
    if os_hits:
        lines.append("OS / Components (problems):")
        for x in os_hits:
            date = x["date"][:10] if x["date"] else ""
            link = f" ({x['url']})" if x["url"] else ""
            lines.append(f"- {x['title']} — {x['excerpt'] or 'issue detected'} [{date}]{link}")
    if rd_hits:
        lines.append("\nReddit (user-reported problems):")
        for x in rd_hits:
            date = x["date"][:10] if x["date"] else ""
            link = f" ({x['url']})" if x["url"] else ""
            lines.append(f"- {x['title']} — {x['excerpt'] or 'issue reported'} [{date}]{link}")
    if not os_hits and not rd_hits:
        lines.append("No clearly problematic updates detected in the last 7 days.")
    lines.append("\nAnswer:")
    return "\n".join(lines)