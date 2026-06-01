#!/usr/bin/env python3
"""
Create per-post sentiment trajectory plots where x-axis is comment sequence number.

For each endpoint (minScore_0.5, maxScore_0.5), select top N posts by engagement
(num_comments desc, score desc). For each selected post, plot:
- author stream: opening post text at x=0 + OP comments at their comment index
- community stream: non-OP comments at their comment index

Outputs: 2 * N individual PNGs (default N=5).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import matplotlib

os.environ["MPLCONFIGDIR"] = "/tmp/mplcache"
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install nltk first: pip install nltk") from e


ROOT = Path(__file__).resolve().parents[2]

URL_MIN = "https://releasetrain.io/api/reddit/query/filter?minComments=3&minScore=0.5"
URL_MAX = "https://releasetrain.io/api/reddit/query/filter?minComments=3&maxScore=0.5"


def _slug_from_url(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    if q.get("minScore"):
        return f"minScore_{q['minScore'][0]}"
    if q.get("maxScore"):
        return f"maxScore_{q['maxScore'][0]}"
    return "endpoint"


def _clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def comment_sort_key(c: dict[str, Any]) -> float:
    ts = c.get("created_utc_ts")
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    t = _parse_time(c.get("created_utc"))
    return t if t is not None else 0.0


def is_op_comment(c: dict[str, Any], post: dict[str, Any]) -> bool:
    v = c.get("is_submitter")
    if v is True:
        return True
    if v is False:
        return False
    pa = (post.get("author") or "").strip().lower()
    ca = (c.get("author") or "").strip().lower()
    return bool(pa) and pa == ca


def engagement_tuple(post: dict[str, Any]) -> tuple[float, float]:
    return (float(post.get("num_comments") or 0), float(post.get("score") or 0))


def post_text_opening(post: dict[str, Any]) -> str:
    parts = [
        _clean_text(post.get("title") or ""),
        _clean_text(post.get("author_description") or ""),
        _clean_text(post.get("body") or ""),
    ]
    return _clean_text(" ".join(x for x in parts if x))


def ensure_vader() -> None:
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def fetch_posts(url: str, timeout: float = 120.0, retries: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            return payload.get("filters") or {}, payload.get("data") or []
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def build_cumulative_mean_series_for_post(
    post: dict[str, Any], sia: SentimentIntensityAnalyzer
) -> tuple[list[int], list[float], list[float], int, int]:
    """
    Build exactly two trajectories on comment-index axis:
    - author cumulative mean sentiment up to each comment index
    - community cumulative mean sentiment up to each comment index
    """
    opening = post_text_opening(post)
    opening_score = float(sia.polarity_scores(opening)["compound"]) if opening else 0.0

    comments_sorted = sorted(post.get("comments") or [], key=comment_sort_key)
    scored_comments: list[tuple[bool, float]] = []
    for c in comments_sorted:
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        score = float(sia.polarity_scores(body)["compound"])
        scored_comments.append((is_op_comment(c, post), score))

    n = len(scored_comments)
    xs = list(range(0, n + 1))
    author_mean: list[float] = []
    community_mean: list[float] = []

    author_sum = opening_score
    author_cnt = 1
    community_sum = 0.0
    community_cnt = 0

    author_mean.append(author_sum / author_cnt)
    community_mean.append(np.nan)

    for is_author, sc in scored_comments:
        if is_author:
            author_sum += sc
            author_cnt += 1
        else:
            community_sum += sc
            community_cnt += 1
        author_mean.append(author_sum / author_cnt)
        community_mean.append((community_sum / community_cnt) if community_cnt > 0 else np.nan)

    return xs, author_mean, community_mean, author_cnt - 1, community_cnt


def build_raw_series_for_post(
    post: dict[str, Any], sia: SentimentIntensityAnalyzer
) -> tuple[list[int], list[float], list[float], int, int]:
    """
    Build exactly two raw trajectories on comment-index axis:
    - author raw sentiment at each index (NaN when event is community)
    - community raw sentiment at each index (NaN when event is author)
    """
    opening = post_text_opening(post)
    opening_score = float(sia.polarity_scores(opening)["compound"]) if opening else 0.0

    comments_sorted = sorted(post.get("comments") or [], key=comment_sort_key)
    xs = [0]
    author_y = [opening_score]
    community_y = [np.nan]
    n_author_comments = 0
    n_community_comments = 0

    comment_idx = 0
    for c in comments_sorted:
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        comment_idx += 1
        score = float(sia.polarity_scores(body)["compound"])
        xs.append(comment_idx)
        if is_op_comment(c, post):
            author_y.append(score)
            community_y.append(np.nan)
            n_author_comments += 1
        else:
            author_y.append(np.nan)
            community_y.append(score)
            n_community_comments += 1

    return xs, author_y, community_y, n_author_comments, n_community_comments


def plot_post(
    slug: str,
    rank: int,
    post: dict[str, Any],
    sia: SentimentIntensityAnalyzer,
    out_path: Path,
    mode: str,
) -> None:
    if mode == "cumulative":
        xs, author_vals, community_vals, n_author_comments, n_community_comments = build_cumulative_mean_series_for_post(post, sia)
        author_label = f"Author cumulative mean (opening + OP, n={n_author_comments + 1})"
        community_label = f"Community cumulative mean (n={n_community_comments})"
        mode_title = "Cumulative trajectory by comment index"
    else:
        xs, author_vals, community_vals, n_author_comments, n_community_comments = build_raw_series_for_post(post, sia)
        author_label = f"Author raw sentiment (opening + OP comments, n={n_author_comments + 1})"
        community_label = f"Community raw sentiment (n={n_community_comments})"
        mode_title = "Raw sentiment trajectory by comment index"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        xs,
        author_vals,
        color="#c05621",
        marker="o",
        markersize=4,
        linewidth=3.0,
        label=author_label,
    )
    ax.plot(
        xs,
        community_vals,
        color="#2c5282",
        marker="o",
        markersize=4,
        linewidth=3.0,
        linestyle="--",
        label=community_label,
    )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Comment number (chronological within post)")
    ax.set_ylabel("VADER compound")
    ax.set_ylim(-1.05, 1.05)

    title = str(post.get("title") or "")
    title_short = title[:115] + ("..." if len(title) > 115 else "")
    rid = post.get("redditId") or ""
    sub = post.get("subreddit") or ""
    n_comments = int(post.get("num_comments") or 0)
    ax.set_title(
        f"{slug} — top engagement post #{rank} | r/{sub} | id {rid}\n"
        f"{mode_title} | n_comments={n_comments} | {title_short}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-post trajectories by comment index")
    parser.add_argument("--posts-per-endpoint", type=int, default=5, help="How many top engagement posts per endpoint")
    parser.add_argument(
        "--mode",
        choices=["raw", "cumulative"],
        default="raw",
        help="Trajectory style on comment-index axis",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tps_gds_classification" / "outputs" / "vader_endpoint_comment_index_trajectories",
    )
    parser.add_argument("--min-url", type=str, default=URL_MIN)
    parser.add_argument("--max-url", type=str, default=URL_MAX)
    args = parser.parse_args()

    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    for url in (args.min_url, args.max_url):
        slug = _slug_from_url(url)
        _, posts = fetch_posts(url)
        selected = sorted(posts, key=engagement_tuple, reverse=True)[: args.posts_per_endpoint]
        endpoint_out = args.out_dir / slug
        endpoint_out.mkdir(parents=True, exist_ok=True)

        for rank, post in enumerate(selected, start=1):
            rid = str(post.get("redditId") or "unknown")
            safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in rid)
            out_path = endpoint_out / f"post_{rank:02d}_{safe_id}_by_comment_index.png"
            plot_post(slug, rank, post, sia, out_path, args.mode)
            print(f"Wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
