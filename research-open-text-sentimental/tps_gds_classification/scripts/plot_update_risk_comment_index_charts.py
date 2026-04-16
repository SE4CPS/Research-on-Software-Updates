#!/usr/bin/env python3
"""
Professor-style charts: two figures with at most 5 lines each.

- X-axis: chronological comment index (1 = first comment, 2 = second, ...).
- Y-axis: VADER compound for that comment body.
- Chart A: top engagement posts labeled update-risk (corrected_label == 1).
- Chart B: top engagement posts labeled non-update-risk (corrected_label == 0).

Labels are read from tps_gds_classification/data/updated_labeled_dataset.csv (reddit_id, corrected_label).
Posts are taken from both releasetrain filter endpoints (merged, deduped by redditId).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Any

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
    raise SystemExit("Install nltk: pip install nltk") from e


ROOT = Path(__file__).resolve().parents[2]

URL_MIN = "https://releasetrain.io/api/reddit/query/filter?minComments=3&minScore=0.5"
URL_MAX = "https://releasetrain.io/api/reddit/query/filter?minComments=3&maxScore=0.5"


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
    from datetime import datetime, timezone

    s = str(c.get("created_utc") or "").strip().replace("Z", "+00:00")
    if not s:
        return 0.0
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def engagement_tuple(post: dict[str, Any]) -> tuple[float, float]:
    return (float(post.get("num_comments") or 0), float(post.get("score") or 0))


def fetch_posts(url: str, timeout: float = 120.0, retries: int = 4) -> list[dict[str, Any]]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json().get("data") or []
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    assert last_err is not None
    raise last_err


def load_labels_csv(path: Path) -> dict[str, int]:
    """reddit_id -> 1 (update risk) or 0 (non-update risk)."""
    out: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("reddit_id") or "").strip()
            if not rid:
                continue
            raw = (row.get("corrected_label") or "").strip()
            if raw not in ("0", "1"):
                continue
            out[rid] = int(raw)
    return out


def ensure_vader() -> None:
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def comment_only_compound_series(post: dict[str, Any], sia: SentimentIntensityAnalyzer) -> list[float]:
    """Chronological comments only (no opening post body). Each = VADER compound."""
    vals: list[float] = []
    for c in sorted(post.get("comments") or [], key=comment_sort_key):
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        vals.append(float(sia.polarity_scores(body)["compound"]))
    return vals


def plot_five_line_chart(
    title: str,
    posts: list[dict[str, Any]],
    sia: SentimentIntensityAnalyzer,
    out_path: Path,
    cmap_name: str = "tab10",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap(cmap_name)
    for i, post in enumerate(posts):
        y = comment_only_compound_series(post, sia)
        if not y:
            continue
        x = np.arange(1, len(y) + 1, dtype=float)
        rid = str(post.get("redditId") or "")
        sub = str(post.get("subreddit") or "")
        t = str(post.get("title") or "")
        t_short = (t[:50] + "…") if len(t) > 50 else t
        label = f"#{i + 1} {rid} r/{sub} — {t_short}"
        ax.plot(x, y, color=cmap(i % 10), linewidth=2.4, marker="o", markersize=3.5, label=label)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Comment number (chronological; 1 = first comment)")
    ax.set_ylabel("VADER compound (per comment)")
    ax.set_title(title)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update-risk vs non-update-risk comment-index charts")
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=ROOT / "tps_gds_classification" / "data" / "updated_labeled_dataset.csv",
    )
    parser.add_argument("--lines", type=int, default=5, help="Max lines per chart (default 5)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "tps_gds_classification" / "outputs" / "vader_update_risk_comment_index",
    )
    args = parser.parse_args()

    labels = load_labels_csv(args.labels_csv)
    ensure_vader()
    sia = SentimentIntensityAnalyzer()

    merged: dict[str, dict[str, Any]] = {}
    for url in (URL_MIN, URL_MAX):
        for p in fetch_posts(url):
            rid = str(p.get("redditId") or "")
            if rid and rid not in merged:
                merged[rid] = p

    labeled = [p for p in merged.values() if str(p.get("redditId") or "") in labels]
    risk_posts = [p for p in labeled if labels[str(p["redditId"])] == 1]
    non_posts = [p for p in labeled if labels[str(p["redditId"])] == 0]

    risk_posts = sorted(risk_posts, key=engagement_tuple, reverse=True)[: args.lines]
    non_posts = sorted(non_posts, key=engagement_tuple, reverse=True)[: args.lines]

    plot_five_line_chart(
        f"Update-risk posts (corrected_label=1) — top {len(risk_posts)} by engagement\n"
        "VADER compound vs chronological comment index (comments only)",
        risk_posts,
        sia,
        args.out_dir / "update_risk_top5_by_comment_index.png",
    )
    plot_five_line_chart(
        f"Non–update-risk posts (corrected_label=0) — top {len(non_posts)} by engagement\n"
        "VADER compound vs chronological comment index (comments only)",
        non_posts,
        sia,
        args.out_dir / "non_update_risk_top5_by_comment_index.png",
    )

    meta = {
        "labels_csv": str(args.labels_csv),
        "lines_per_chart": args.lines,
        "update_risk_reddit_ids": [str(p.get("redditId")) for p in risk_posts],
        "non_update_risk_reddit_ids": [str(p.get("redditId")) for p in non_posts],
    }
    import json

    with open(args.out_dir / "selection_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", args.out_dir / "update_risk_top5_by_comment_index.png")
    print("Wrote:", args.out_dir / "non_update_risk_top5_by_comment_index.png")
    print("Wrote:", args.out_dir / "selection_meta.json")


if __name__ == "__main__":
    main()
