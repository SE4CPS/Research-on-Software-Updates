#!/usr/bin/env python3
"""
VADER endpoint comparison (v2).

Changes vs v1:
- Compute VADER separately for:
  - post text: title + author_description + body
  - comments: each comment.body individually, then aggregate (mean or median)
- Final per-post sentiment is a weighted average of:
  post_compound and comments_aggregate_compound
- Adds two sampling strategies per endpoint:
  - engagement_top_n: rank by num_comments then score
  - random_n: uniform random sample with fixed seed (sensitivity check)

Writes to a NEW output folder by default to avoid overwriting v1 outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Literal

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


URL_MIN = "https://releasetrain.io/api/reddit/query/filter?minComments=3&minScore=0.5"
URL_MAX = "https://releasetrain.io/api/reddit/query/filter?minComments=3&maxScore=0.5"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_text(text: str) -> str:
    s = text or ""
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compound_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def fetch_posts(url: str, timeout: float = 60.0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("filters") or {}, payload.get("data") or []


def engagement_tuple(post: dict[str, Any]) -> tuple[float, float]:
    num_comments = float(post.get("num_comments") or 0)
    score = float(post.get("score") or 0)
    return (num_comments, score)


def post_text_no_comments(post: dict[str, Any]) -> str:
    fields = [
        clean_text(post.get("title") or ""),
        clean_text(post.get("author_description") or ""),
        clean_text(post.get("body") or ""),
    ]
    return clean_text(" ".join(x for x in fields if x))


def comment_bodies(post: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for c in post.get("comments") or []:
        body = clean_text(c.get("body") or "")
        if body:
            out.append(body)
    return out


AggMethod = Literal["mean", "median"]


def aggregate_comment_compound(
    sia: SentimentIntensityAnalyzer, bodies: list[str], method: AggMethod
) -> tuple[float | None, int]:
    if not bodies:
        return None, 0
    vals = [sia.polarity_scores(b)["compound"] for b in bodies]
    if method == "median":
        return float(median(vals)), len(vals)
    return float(mean(vals)), len(vals)


def weighted_average(a: float, b: float, wa: float, wb: float) -> float:
    denom = wa + wb
    if denom <= 0:
        return a
    return (wa * a + wb * b) / denom


def ensure_vader() -> None:
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def select_posts(
    posts: list[dict[str, Any]],
    strategy: Literal["engagement", "random"],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if strategy == "engagement":
        return sorted(posts, key=engagement_tuple, reverse=True)[:n]
    rng = random.Random(seed)
    if len(posts) <= n:
        return list(posts)
    return rng.sample(posts, n)


def score_selected_posts(
    posts: list[dict[str, Any]],
    endpoint_name: str,
    strategy_name: str,
    sia: SentimentIntensityAnalyzer,
    comment_agg: AggMethod,
    w_post: float,
    w_comments: float,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, post in enumerate(posts, start=1):
        post_text = post_text_no_comments(post)
        post_scores = sia.polarity_scores(post_text) if post_text else {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

        bodies = comment_bodies(post)
        comments_comp, n_comments_scored = aggregate_comment_compound(sia, bodies, comment_agg)

        final_compound = (
            weighted_average(post_scores["compound"], comments_comp, w_post, w_comments)
            if comments_comp is not None
            else float(post_scores["compound"])
        )

        out.append(
            {
                "endpoint": endpoint_name,
                "strategy": strategy_name,
                "rank_or_index": idx,
                "seed": seed,
                "redditId": post.get("redditId"),
                "title": post.get("title"),
                "subreddit": post.get("subreddit"),
                "num_comments": post.get("num_comments", 0),
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio"),
                "post_compound": float(post_scores["compound"]),
                "comments_compound_agg": comments_comp,
                "comments_agg_method": comment_agg,
                "comments_scored_count": n_comments_scored,
                "final_compound": float(final_compound),
                "label": compound_label(float(final_compound)),
                "weights": {"post": w_post, "comments": w_comments},
                "post_text_length": len(post_text),
                "comments_count_in_payload": len(post.get("comments") or []),
            }
        )
    return out


def label_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in rows:
        counts[r["label"]] += 1
    total = len(rows) or 1
    return {k: 100.0 * v / total for k, v in counts.items()}


def summary(rows: list[dict[str, Any]], filters: dict[str, Any]) -> dict[str, Any]:
    vals = [r["final_compound"] for r in rows]
    return {
        "n_selected_posts": len(rows),
        "compound_mean": mean(vals) if vals else None,
        "compound_std": pstdev(vals) if len(vals) > 1 else 0.0,
        "label_percentages": label_distribution(rows),
        "filters": filters,
    }


def smooth(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    w = max(1, int(window))
    if w <= 1:
        return arr
    if len(arr) < w:
        return np.array([arr.mean()] * len(arr), dtype=float)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(arr, kernel, mode="same")


def plot_raw_smoothed_trend(
    min_rows: list[dict[str, Any]],
    max_rows: list[dict[str, Any]],
    window: int,
    title: str,
    out_path: Path,
) -> None:
    min_y = [r["final_compound"] for r in min_rows]
    max_y = [r["final_compound"] for r in max_rows]
    min_x = np.arange(1, len(min_y) + 1)
    max_x = np.arange(1, len(max_y) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(min_x, min_y, color="#2b6cb0", alpha=0.35, linewidth=1.4, marker="o", markersize=3, label="minScore raw")
    ax.plot(max_x, max_y, color="#c53030", alpha=0.35, linewidth=1.4, marker="o", markersize=3, label="maxScore raw")
    ax.plot(min_x, smooth(min_y, window), color="#2b6cb0", linewidth=2.7, label=f"minScore smoothed (w={window})")
    ax.plot(max_x, smooth(max_y, window), color="#c53030", linewidth=2.7, label=f"maxScore smoothed (w={window})")

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.7)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Post order in this sample (1..30)")
    ax.set_ylabel("VADER compound (final per-post score)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Endpoint-ranked VADER comparative analysis (v2)")
    parser.add_argument("--n", type=int, default=30, help="Number of posts per endpoint per strategy")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window for trend line")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling strategy")
    parser.add_argument("--comment-agg", choices=["mean", "median"], default="mean", help="Aggregate method for per-comment compounds")
    parser.add_argument("--w-post", type=float, default=1.0, help="Weight for title+desc+body compound")
    parser.add_argument("--w-comments", type=float, default=1.0, help="Weight for aggregated comments compound")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root() / "tps_gds_classification" / "outputs" / "vader_endpoint_comparison_v2",
        help="Output directory (new, does not overwrite v1)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    min_filters, min_posts_all = fetch_posts(URL_MIN)
    max_filters, max_posts_all = fetch_posts(URL_MAX)

    strategies: list[tuple[str, Literal["engagement", "random"]]] = [
        ("engagement_top_n", "engagement"),
        ("random_n", "random"),
    ]

    results: dict[str, Any] = {
        "inputs": {
            "endpoints": {"minScore_0.5": URL_MIN, "maxScore_0.5": URL_MAX},
            "n": args.n,
            "random_seed": args.seed,
            "comment_aggregation": args.comment_agg,
            "final_compound_definition": {
                "post_compound": "VADER(title + author_description + body)",
                "comments_compound": f"{args.comment_agg}( VADER(comment.body).compound for each comment )",
                "final_compound": "weighted_average(post_compound, comments_compound)",
                "weights": {"post": args.w_post, "comments": args.w_comments},
            },
        },
        "strategies": {},
    }

    for strat_name, strat_key in strategies:
        min_sel = select_posts(min_posts_all, strat_key, args.n, seed=args.seed)
        max_sel = select_posts(max_posts_all, strat_key, args.n, seed=args.seed)

        # For engagement strategy, keep it in ranked order. For random, keep stable ordering by redditId for plot consistency.
        if strat_key == "engagement":
            min_sel_ordered = min_sel
            max_sel_ordered = max_sel
        else:
            min_sel_ordered = sorted(min_sel, key=lambda p: str(p.get("redditId") or ""))
            max_sel_ordered = sorted(max_sel, key=lambda p: str(p.get("redditId") or ""))

        min_rows = score_selected_posts(
            min_sel_ordered,
            "minScore_0.5",
            strat_name,
            sia,
            comment_agg=args.comment_agg,
            w_post=args.w_post,
            w_comments=args.w_comments,
            seed=args.seed,
        )
        max_rows = score_selected_posts(
            max_sel_ordered,
            "maxScore_0.5",
            strat_name,
            sia,
            comment_agg=args.comment_agg,
            w_post=args.w_post,
            w_comments=args.w_comments,
            seed=args.seed,
        )

        strat_dir = args.out_dir / strat_name
        strat_dir.mkdir(parents=True, exist_ok=True)

        with open(strat_dir / "selected_posts_minScore_0.5.json", "w", encoding="utf-8") as f:
            json.dump(min_rows, f, indent=2)
        with open(strat_dir / "selected_posts_maxScore_0.5.json", "w", encoding="utf-8") as f:
            json.dump(max_rows, f, indent=2)

        s_min = summary(min_rows, min_filters)
        s_max = summary(max_rows, max_filters)
        results["strategies"][strat_name] = {"minScore_0.5": s_min, "maxScore_0.5": s_max}

        plot_raw_smoothed_trend(
            min_rows,
            max_rows,
            window=args.smooth_window,
            title=f"Post-to-post sentiment (v2) — {strat_name} — raw + smoothed",
            out_path=strat_dir / "raw_smoothed_post_trend.png",
        )

    with open(args.out_dir / "comparative_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Wrote v2 analysis to:", args.out_dir)
    print("Trend charts:")
    for s in ("engagement_top_n", "random_n"):
        print(" -", args.out_dir / s / "raw_smoothed_post_trend.png")


if __name__ == "__main__":
    main()

