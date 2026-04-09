#!/usr/bin/env python3
"""
Per-endpoint sentiment trajectories: opening post + OP comments vs community comments over time.

Fetches releasetrain filter JSON for minScore and maxScore endpoints, takes top K posts by
engagement (num_comments, then score), runs VADER on each text unit, plots compound vs
hours since post creation. Includes per-post faint lines and mean curves across the K posts.

Writes one combined PNG per endpoint plus one PNG per selected post (under <slug>_posts/).
Per-post traces use distinct colors (solid = author, dashed = community); mean curves stay orange/blue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import matplotlib

os.environ["MPLCONFIGDIR"] = "/tmp/mplcache"
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:  # pragma: no cover
    print("Install nltk: pip install nltk", file=sys.stderr)
    raise e


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


def _parse_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    from datetime import datetime, timezone

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


def _clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


def post_text_opening(post: dict[str, Any]) -> str:
    parts = [
        _clean_text(post.get("title") or ""),
        _clean_text(post.get("author_description") or ""),
        _clean_text(post.get("body") or ""),
    ]
    return _clean_text(" ".join(x for x in parts if x))


def engagement_tuple(post: dict[str, Any]) -> tuple[float, float]:
    return (float(post.get("num_comments") or 0), float(post.get("score") or 0))


def fetch_posts(url: str, timeout: float = 120.0, retries: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            return payload.get("filters") or {}, payload.get("data") or []
        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def ensure_vader() -> None:
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def build_author_series_hours(
    post: dict[str, Any], sia: SentimentIntensityAnalyzer
) -> tuple[np.ndarray, np.ndarray]:
    """Hours since post creation vs compound for OP: opening text then OP comments in time order."""
    t0 = _parse_time(post.get("created_utc"))
    if t0 is None:
        t0 = 0.0
    hours: list[float] = []
    compounds: list[float] = []
    opening = post_text_opening(post)
    if opening:
        hours.append(0.0)
        compounds.append(float(sia.polarity_scores(opening)["compound"]))
    for c in sorted(post.get("comments") or [], key=comment_sort_key):
        if not is_op_comment(c, post):
            continue
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        ts = comment_sort_key(c)
        h = (ts - t0) / 3600.0
        hours.append(h)
        compounds.append(float(sia.polarity_scores(body)["compound"]))
    return np.array(hours, dtype=float), np.array(compounds, dtype=float)


def build_community_series_hours(
    post: dict[str, Any], sia: SentimentIntensityAnalyzer
) -> tuple[np.ndarray, np.ndarray]:
    t0 = _parse_time(post.get("created_utc"))
    if t0 is None:
        t0 = 0.0
    hours: list[float] = []
    compounds: list[float] = []
    for c in sorted(post.get("comments") or [], key=comment_sort_key):
        if is_op_comment(c, post):
            continue
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        ts = comment_sort_key(c)
        h = (ts - t0) / 3600.0
        hours.append(h)
        compounds.append(float(sia.polarity_scores(body)["compound"]))
    return np.array(hours, dtype=float), np.array(compounds, dtype=float)


def interp_on_grid(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linear interpolation with constant extrapolation; x must be sorted."""
    if x.size == 0:
        return np.full_like(grid, np.nan, dtype=float)
    if x.size == 1:
        return np.full_like(grid, float(y[0]), dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    return np.interp(grid, xs, ys, left=float(ys[0]), right=float(ys[-1]))


def post_color_palette(n: int) -> list[tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def rgba_to_hex(c: tuple[float, ...]) -> str:
    return mcolors.to_hex(c[:3])


def mean_on_grid(series_list: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> np.ndarray:
    mats = []
    for x, y in series_list:
        if x.size == 0:
            continue
        mats.append(interp_on_grid(x, y, grid))
    if not mats:
        return np.full_like(grid, np.nan, dtype=float)
    return np.nanmean(np.stack(mats, axis=0), axis=0)


def plot_single_post(
    slug: str,
    post: dict[str, Any],
    rank: int,
    color: tuple[float, float, float, float],
    sia: SentimentIntensityAnalyzer,
    out_path: Path,
    max_hours_cap: float = 168.0,
) -> None:
    ah, ac = build_author_series_hours(post, sia)
    ch, cc = build_community_series_hours(post, sia)
    max_h = 1.0
    if ah.size:
        max_h = max(max_h, float(ah.max()))
    if ch.size:
        max_h = max(max_h, float(ch.max()))
    max_h = min(max_h, max_hours_cap)

    fig, ax = plt.subplots(figsize=(12, 6))
    if ah.size:
        ax.plot(ah, ac, color=color, linestyle="-", linewidth=2.8, alpha=0.95, label="Author (post + OP comments)", zorder=3)
    if ch.size:
        ax.plot(ch, cc, color=color, linestyle="--", linewidth=2.8, alpha=0.95, label="Community comments", zorder=3)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Time (hours since post creation)")
    ax.set_ylabel("VADER compound")
    rid = post.get("redditId") or ""
    sub = post.get("subreddit") or ""
    title_text = post.get("title") or ""
    ax.set_title(
        f"{slug} — post {rank}/{rid}\nr/{sub}\n{title_text[:120]}{'…' if len(title_text) > 120 else ''}",
        fontsize=10,
    )
    ax.set_xlim(0.0, max_h)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint(
    slug: str,
    posts: list[dict[str, Any]],
    sia: SentimentIntensityAnalyzer,
    out_path: Path,
    color_mean_author: str,
    color_mean_community: str,
    per_post_linewidth: float = 2.5,
    mean_linewidth: float = 3.4,
    per_post_dir: Path | None = None,
) -> dict[str, Any]:
    author_series: list[tuple[np.ndarray, np.ndarray]] = []
    community_series: list[tuple[np.ndarray, np.ndarray]] = []
    meta_rows: list[dict[str, Any]] = []

    for rank, post in enumerate(posts, start=1):
        ah, ac = build_author_series_hours(post, sia)
        ch, cc = build_community_series_hours(post, sia)
        author_series.append((ah, ac))
        community_series.append((ch, cc))
        meta_rows.append(
            {
                "rank": rank,
                "redditId": post.get("redditId"),
                "subreddit": post.get("subreddit"),
                "num_comments": post.get("num_comments"),
                "score": post.get("score"),
                "title": post.get("title"),
                "n_author_points": int(ah.size),
                "n_community_points": int(ch.size),
                "post_color_tab10_index": (rank - 1) % 10,
            }
        )

    max_h = 1.0
    for ah, _ in author_series:
        if ah.size:
            max_h = max(max_h, float(ah.max()))
    for ch, _ in community_series:
        if ch.size:
            max_h = max(max_h, float(ch.max()))
    max_h = min(max_h, 168.0)

    grid = np.linspace(0.0, max_h, 120)
    mean_author = mean_on_grid(author_series, grid)
    mean_community = mean_on_grid(community_series, grid)
    post_colors = post_color_palette(len(posts))

    if per_post_dir is not None:
        for idx, post in enumerate(posts):
            rank = idx + 1
            rid = str(post.get("redditId") or "unknown")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in rid)
            single_path = per_post_dir / f"post_{rank:02d}_{safe}_trajectory.png"
            plot_single_post(slug, post, rank, post_colors[idx], sia, single_path, max_hours_cap=168.0)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.22)
    ax = fig.add_subplot(gs[0])
    ax_detail = fig.add_subplot(gs[1])
    ax_detail.axis("off")

    for i, (((ah, ac), (ch, cc)), c) in enumerate(zip(zip(author_series, community_series), post_colors)):
        if ah.size:
            ax.plot(
                ah,
                ac,
                color=c,
                alpha=0.88,
                linewidth=per_post_linewidth,
                linestyle="-",
                zorder=2,
            )
        if ch.size:
            ax.plot(
                ch,
                cc,
                color=c,
                alpha=0.88,
                linewidth=per_post_linewidth,
                linestyle="--",
                zorder=2,
            )

    ax.plot(
        grid,
        mean_author,
        color=color_mean_author,
        linewidth=mean_linewidth,
        label="Mean — author (post + OP comments)",
        zorder=5,
    )
    ax.plot(
        grid,
        mean_community,
        color=color_mean_community,
        linewidth=mean_linewidth,
        label="Mean — community comments",
        zorder=5,
    )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Time (hours since post creation)")
    ax.set_ylabel("VADER compound")
    ax.set_title(
        f"Sentiment trajectory — {slug}\n"
        f"Top {len(posts)} posts by engagement; each color = one post "
        f"(solid = author, dashed = community); bold = mean across posts"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(0.0, max_h)

    lines = [
        f"{'#':<3} {'subreddit':<14} {'id':<12} {'n_cmt':>5}  title",
        "-" * 72,
    ]
    for row in meta_rows:
        title = (row.get("title") or "")[:52]
        if len((row.get("title") or "")) > 52:
            title += "…"
        lines.append(
            f"{row['rank']:<3} r/{str(row.get('subreddit') or '')[:12]:<12} "
            f"{str(row.get('redditId') or ''):<12} {int(row.get('num_comments') or 0):>5}  {title}"
        )
    ax_detail.text(0.0, 1.0, "\n".join(lines), transform=ax_detail.transAxes, va="top", ha="left", fontsize=7.5, family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    interactive_traces: list[dict[str, Any]] = []
    for i, post in enumerate(posts):
        ah, ac = author_series[i]
        ch, cc = community_series[i]
        c_hex = rgba_to_hex(post_colors[i])
        rid = str(post.get("redditId") or "")
        title = str(post.get("title") or "")
        sub = str(post.get("subreddit") or "")
        rank = i + 1
        if ah.size:
            interactive_traces.append(
                {
                    "stream": "author",
                    "rank": rank,
                    "redditId": rid,
                    "title": title,
                    "subreddit": sub,
                    "color": c_hex,
                    "hours": [float(x) for x in ah],
                    "compound": [float(x) for x in ac],
                }
            )
        if ch.size:
            interactive_traces.append(
                {
                    "stream": "community",
                    "rank": rank,
                    "redditId": rid,
                    "title": title,
                    "subreddit": sub,
                    "color": c_hex,
                    "hours": [float(x) for x in ch],
                    "compound": [float(x) for x in cc],
                }
            )

    return {
        "slug": slug,
        "n_posts": len(posts),
        "hours_grid_max": max_h,
        "posts": meta_rows,
        "mean_author": [float(x) for x in mean_author],
        "mean_community": [float(x) for x in mean_community],
        "grid_hours": [float(x) for x in grid],
        "interactive_traces": interactive_traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Author vs community VADER trajectories per endpoint")
    parser.add_argument("--top-posts", type=int, default=10, help="Posts per endpoint (by engagement)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tps_gds_classification" / "outputs" / "vader_endpoint_author_community_trajectories",
    )
    parser.add_argument("--min-url", type=str, default=URL_MIN)
    parser.add_argument("--max-url", type=str, default=URL_MAX)
    parser.add_argument(
        "--no-per-post-pngs",
        action="store_true",
        help="Only write combined endpoint PNGs (skip individual post figures)",
    )
    parser.add_argument(
        "--interactive-json",
        type=Path,
        default=ROOT / "web" / "data" / "vader_thread_trajectories.json",
        help="Write Plotly-ready trajectory JSON for the dashboard",
    )
    parser.add_argument(
        "--no-interactive-json",
        action="store_true",
        help="Skip writing web/data/vader_thread_trajectories.json",
    )
    args = parser.parse_args()

    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    summaries: list[dict[str, Any]] = []
    color_mean_author = "#c05621"
    color_mean_community = "#2c5282"

    for url in (args.min_url, args.max_url):
        slug = _slug_from_url(url)
        filters, all_posts = fetch_posts(url)
        ranked = sorted(all_posts, key=engagement_tuple, reverse=True)[: args.top_posts]
        out_png = args.out_dir / f"{slug}_author_community_trajectory.png"
        per_post_dir: Path | None = None if args.no_per_post_pngs else args.out_dir / f"{slug}_posts"
        block = plot_endpoint(
            slug,
            ranked,
            sia,
            out_png,
            color_mean_author,
            color_mean_community,
            per_post_dir=per_post_dir,
        )
        block["filters"] = filters
        block["source_url"] = url
        block["combined_png"] = str(out_png)
        block["per_post_dir"] = str(per_post_dir) if per_post_dir else None
        if per_post_dir:
            block["per_post_pngs"] = sorted(str(p) for p in per_post_dir.glob("post_*_trajectory.png"))
        summaries.append(block)
        print(f"Wrote {out_png}")
        if per_post_dir:
            n_single = len(list(per_post_dir.glob("post_*_trajectory.png")))
            print(f"Wrote {n_single} per-post PNGs under {per_post_dir}")

    manifest = {
        "selection": f"top {args.top_posts} per endpoint by num_comments desc, score desc",
        "vader": "NLTK VADER per text unit (opening post text; each comment body separately)",
        "author_definition": "title + author_description + body at t=0; then comments with is_submitter true or author name match",
        "community_definition": "comments that are not OP",
        "endpoints": summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "trajectory_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest:", args.out_dir / "trajectory_manifest.json")

    if not args.no_interactive_json:
        interactive_payload: dict[str, Any] = {}
        for block in summaries:
            slug = block["slug"]
            interactive_payload[slug] = {
                "max_hours": block["hours_grid_max"],
                "grid_hours": block["grid_hours"],
                "mean_author": block["mean_author"],
                "mean_community": block["mean_community"],
                "traces": block["interactive_traces"],
            }
        args.interactive_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.interactive_json, "w", encoding="utf-8") as f:
            json.dump(interactive_payload, f, indent=2)
        print("Wrote interactive JSON:", args.interactive_json)


if __name__ == "__main__":
    main()
