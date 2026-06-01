#!/usr/bin/env python3
"""
Compare VADER sentiment patterns across two releasetrain Reddit endpoints.

Flow:
1) Fetch endpoint data.
2) Rank posts within each endpoint by engagement (num_comments, then score).
3) Select top N posts per endpoint.
4) Build text from title + author_description + body + comments.
5) Compute VADER scores and labels.
6) Write JSON summaries and comparison plots.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

# Avoid ~/.matplotlib permission issues in restricted environments.
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
    # Rank by comments first, then score.
    num_comments = float(post.get("num_comments") or 0)
    score = float(post.get("score") or 0)
    return (num_comments, score)


def comment_text(comments: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for c in comments or []:
        body = clean_text(c.get("body") or "")
        if body:
            chunks.append(body)
    return " ".join(chunks)


def build_post_text(post: dict[str, Any]) -> str:
    fields = [
        clean_text(post.get("title") or ""),
        clean_text(post.get("author_description") or ""),
        clean_text(post.get("body") or ""),
        comment_text(post.get("comments") or []),
    ]
    return clean_text(" ".join(x for x in fields if x))


def score_posts(
    posts: list[dict[str, Any]],
    endpoint_name: str,
    sia: SentimentIntensityAnalyzer,
    top_n: int,
) -> list[dict[str, Any]]:
    ranked = sorted(posts, key=engagement_tuple, reverse=True)[:top_n]
    out: list[dict[str, Any]] = []
    for rank, post in enumerate(ranked, start=1):
        text = build_post_text(post)
        scores = sia.polarity_scores(text) if text else {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
        out.append(
            {
                "endpoint": endpoint_name,
                "rank": rank,
                "redditId": post.get("redditId"),
                "title": post.get("title"),
                "subreddit": post.get("subreddit"),
                "num_comments": post.get("num_comments", 0),
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio"),
                "compound": scores["compound"],
                "pos": scores["pos"],
                "neg": scores["neg"],
                "neu": scores["neu"],
                "label": compound_label(scores["compound"]),
                "text_length": len(text),
            }
        )
    return out


def label_distribution(scored: list[dict[str, Any]]) -> dict[str, float]:
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for row in scored:
        counts[row["label"]] += 1
    total = len(scored) or 1
    return {k: 100.0 * v / total for k, v in counts.items()}


def endpoint_summary(scored: list[dict[str, Any]], endpoint_name: str, filters: dict[str, Any]) -> dict[str, Any]:
    compounds = [r["compound"] for r in scored]
    return {
        "endpoint": endpoint_name,
        "filters": filters,
        "n_selected_posts": len(scored),
        "engagement_ranking": "num_comments desc, score desc",
        "compound_mean": mean(compounds) if compounds else None,
        "compound_std": pstdev(compounds) if len(compounds) > 1 else 0.0,
        "label_percentages": label_distribution(scored),
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
    out_path: Path,
) -> None:
    min_y = [r["compound"] for r in min_rows]
    max_y = [r["compound"] for r in max_rows]
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
    ax.set_xlabel("Post rank within endpoint (high engagement to lower)")
    ax.set_ylabel("VADER compound")
    ax.set_title("Post-to-post sentiment trend: raw + smoothed")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sentiment_distribution(min_pct: dict[str, float], max_pct: dict[str, float], out_path: Path) -> None:
    labels = ["Positive", "Neutral", "Negative"]
    x = np.arange(len(labels))
    width = 0.35
    min_vals = [min_pct[k] for k in labels]
    max_vals = [max_pct[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, min_vals, width, label="minScore=0.5", color="#2b6cb0")
    ax.bar(x + width / 2, max_vals, width, label="maxScore=0.5", color="#c53030")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage of selected posts")
    ax.set_title("Sentiment label distribution")
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compound_boxplot(min_rows: list[dict[str, Any]], max_rows: list[dict[str, Any]], out_path: Path) -> None:
    min_vals = [r["compound"] for r in min_rows]
    max_vals = [r["compound"] for r in max_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([min_vals, max_vals], tick_labels=["minScore=0.5", "maxScore=0.5"], patch_artist=True)
    ax.set_ylabel("VADER compound")
    ax.set_title("Compound score variability by endpoint")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compound_histogram(min_rows: list[dict[str, Any]], max_rows: list[dict[str, Any]], out_path: Path) -> None:
    min_vals = [r["compound"] for r in min_rows]
    max_vals = [r["compound"] for r in max_rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1, 1, 25)
    ax.hist(min_vals, bins=bins, alpha=0.5, label="minScore=0.5", color="#2b6cb0")
    ax.hist(max_vals, bins=bins, alpha=0.5, label="maxScore=0.5", color="#c53030")
    ax.set_xlabel("VADER compound")
    ax.set_ylabel("Count of selected posts")
    ax.set_title("Compound score distribution")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ensure_vader() -> None:
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Endpoint-ranked VADER comparative analysis")
    parser.add_argument("--top-n", type=int, default=30, help="Top posts per endpoint by engagement")
    parser.add_argument("--smooth-window", type=int, default=5, help="Rolling smoothing window for trend line")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root() / "tps_gds_classification" / "outputs" / "vader_endpoint_comparison",
        help="Output directory",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    min_filters, min_posts = fetch_posts(URL_MIN)
    max_filters, max_posts = fetch_posts(URL_MAX)

    min_scored = score_posts(min_posts, "minScore_0.5", sia, args.top_n)
    max_scored = score_posts(max_posts, "maxScore_0.5", sia, args.top_n)

    min_summary = endpoint_summary(min_scored, "minScore_0.5", min_filters)
    max_summary = endpoint_summary(max_scored, "maxScore_0.5", max_filters)

    comparison = {
        "selection_rule": f"top {args.top_n} posts per endpoint by num_comments desc then score desc",
        "text_fields_for_vader": ["title", "author_description", "body", "comments.body"],
        "thresholds": {"positive_gte": 0.05, "negative_lte": -0.05},
        "endpoints": {"minScore_0.5": min_summary, "maxScore_0.5": max_summary},
    }

    with open(args.out_dir / "selected_posts_minScore_0.5.json", "w", encoding="utf-8") as f:
        json.dump(min_scored, f, indent=2)
    with open(args.out_dir / "selected_posts_maxScore_0.5.json", "w", encoding="utf-8") as f:
        json.dump(max_scored, f, indent=2)
    with open(args.out_dir / "comparative_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    plot_raw_smoothed_trend(min_scored, max_scored, args.smooth_window, args.out_dir / "raw_smoothed_post_trend.png")
    plot_sentiment_distribution(
        min_summary["label_percentages"],
        max_summary["label_percentages"],
        args.out_dir / "sentiment_distribution_bar.png",
    )
    plot_compound_boxplot(min_scored, max_scored, args.out_dir / "compound_boxplot.png")
    plot_compound_histogram(min_scored, max_scored, args.out_dir / "compound_histogram.png")

    print("Wrote analysis to:", args.out_dir)
    print("Main result chart:", args.out_dir / "raw_smoothed_post_trend.png")


if __name__ == "__main__":
    main()
