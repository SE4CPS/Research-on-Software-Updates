#!/usr/bin/env python3
"""
Fetch Reddit-filter JSON from releasetrain.io and analyze VADER sentiment trajectories
per post (submission text, then comments in chronological order).

Separate outputs per endpoint (minScore vs maxScore filter).
No ML training — NLTK VADER only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
from urllib.parse import urlparse, parse_qs

import matplotlib

# Avoid ~/.matplotlib permission issues in restricted environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:  # pragma: no cover
    print("Install nltk: pip install nltk", file=sys.stderr)
    raise e


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def compound_label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def comment_sort_key(c: dict[str, Any]) -> float:
    ts = c.get("created_utc_ts")
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    t = _parse_time(c.get("created_utc"))
    return t if t is not None else 0.0


def build_thread_sequence(post: dict[str, Any]) -> list[tuple[str, float, str]]:
    """Return ordered (kind, timestamp, text) for VADER. Root first, then comments by time."""
    title = _clean_text(post.get("title") or "")
    desc = _clean_text(post.get("author_description") or "")
    root_text = title if not desc else f"{title}\n{desc}"
    root_ts = _parse_time(post.get("created_utc")) or 0.0
    items: list[tuple[str, float, str]] = [("root", root_ts, root_text)]
    for c in post.get("comments") or []:
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        items.append(("comment", comment_sort_key(c), body))
    root = items[0]
    rest = sorted(items[1:], key=lambda x: x[1])
    return [root] + rest


def score_thread(sia: SentimentIntensityAnalyzer, post: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kind, ts, text in build_thread_sequence(post):
        if not text:
            continue
        pol = sia.polarity_scores(text)
        out.append(
            {
                "kind": kind,
                "ts": ts,
                "compound": pol["compound"],
                "pos": pol["pos"],
                "neg": pol["neg"],
                "neu": pol["neu"],
                "label": compound_label(pol["compound"]),
            }
        )
    return out


def fetch_json(url: str, timeout: float = 60.0) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def aggregate_by_step(threads: list[list[dict[str, Any]]]) -> dict[str, Any]:
    by_step: dict[int, list[float]] = defaultdict(list)
    labels_by_step: dict[int, list[str]] = defaultdict(list)
    deltas: list[float] = []
    opening_compounds: list[float] = []
    final_compounds: list[float] = []
    thread_lengths: list[int] = []
    for seq in threads:
        if not seq:
            continue
        opening_compounds.append(seq[0]["compound"])
        final_compounds.append(seq[-1]["compound"])
        thread_lengths.append(len(seq))
        if len(seq) >= 2:
            deltas.append(seq[-1]["compound"] - seq[0]["compound"])
        for i, row in enumerate(seq):
            by_step[i].append(row["compound"])
            labels_by_step[i].append(row["label"])

    def step_stats(i: int) -> dict[str, Any]:
        vals = by_step.get(i, [])
        if not vals:
            return {"n": 0}
        lbls = labels_by_step.get(i, [])
        counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for lb in lbls:
            counts[lb] = counts.get(lb, 0) + 1
        total = len(lbls)
        pct = {k: 100.0 * v / total for k, v in counts.items()}
        return {
            "n": len(vals),
            "mean_compound": mean(vals),
            "std_compound": pstdev(vals) if len(vals) > 1 else 0.0,
            "label_pct": pct,
        }

    max_len = max(by_step.keys(), default=-1)
    per_step = {str(i): step_stats(i) for i in range(max_len + 1)}
    return {
        "per_step": per_step,
        "max_steps": max_len + 1,
        "opening_mean_compound": mean(opening_compounds) if opening_compounds else None,
        "final_in_thread_mean_compound": mean(final_compounds) if final_compounds else None,
        "median_thread_length": float(np.median(thread_lengths)) if thread_lengths else None,
        "delta_first_to_last": {
            "n": len(deltas),
            "mean": mean(deltas) if deltas else None,
            "std": pstdev(deltas) if len(deltas) > 1 else 0.0,
        },
    }


def plot_mean_compound_by_step(per_step: dict[str, Any], out_path: Path, title: str) -> None:
    indices = sorted(int(k) for k, v in per_step.items() if v.get("n", 0) > 0)
    if not indices:
        return
    xs = indices
    means = [per_step[str(i)]["mean_compound"] for i in indices]
    stds = [per_step[str(i)]["std_compound"] for i in indices]
    ns = [per_step[str(i)]["n"] for i in indices]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, means, marker="o", color="#2c5282")
    ax.fill_between(
        xs,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.2,
        color="#2c5282",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Thread position (0 = title+description, 1+ = comments by time)")
    ax.set_ylabel("Mean VADER compound")
    ax.set_title(title)
    for j, (x, n) in enumerate(zip(xs, ns)):
        ax.annotate(
            f"n={n}",
            (x, means[j]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_mix_stacked(per_step: dict[str, Any], out_path: Path, title: str) -> None:
    indices = sorted(int(k) for k, v in per_step.items() if v.get("n", 0) > 0)
    if not indices:
        return
    pos = [per_step[str(i)]["label_pct"]["Positive"] for i in indices]
    neu = [per_step[str(i)]["label_pct"]["Neutral"] for i in indices]
    neg = [per_step[str(i)]["label_pct"]["Negative"] for i in indices]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.array(indices, dtype=float)
    ax.bar(x, pos, label="Positive", color="#38a169")
    ax.bar(x, neu, bottom=pos, label="Neutral", color="#718096")
    bottom = np.array(pos) + np.array(neu)
    ax.bar(x, neg, bottom=bottom, label="Negative", color="#e53e3e")
    ax.set_xlabel("Thread position")
    ax.set_ylabel("Percent of texts at this position")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_histogram(deltas: list[float], out_path: Path, title: str) -> None:
    if not deltas:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deltas, bins=min(20, max(5, len(deltas) // 3)), color="#805ad5", edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Δ compound (last comment − opening post text)")
    ax.set_ylabel("Number of threads")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def conclusion_text(
    slug: str,
    filters: dict[str, Any],
    total_posts: int,
    threads_scored: int,
    agg: dict[str, Any],
) -> str:
    dinfo = agg["delta_first_to_last"]
    lines = [
        f"Endpoint `{slug}` (filters: {filters})",
        f"Posts in payload: {total_posts}; threads with at least one scored text: {threads_scored}.",
    ]
    om = agg.get("opening_mean_compound")
    fm = agg.get("final_in_thread_mean_compound")
    med_len = agg.get("median_thread_length")
    if om is not None:
        lines.append(f"Mean compound on opening text (title + description) ≈ {om:.3f}.")
    if fm is not None and threads_scored:
        lines.append(
            f"Mean compound on each thread's final text (last comment or opening if no comments) ≈ {fm:.3f}."
        )
    if med_len is not None:
        lines.append(f"Median thread length (scored texts): {med_len:.1f}.")
    if dinfo.get("n"):
        lines.append(
            f"For threads with ≥2 scored texts, mean (final−opening) compound = {dinfo['mean']:.3f} "
            f"(std {dinfo['std']:.3f}, n={dinfo['n']})."
        )
    mean_delta = dinfo.get("mean")
    if mean_delta is not None:
        if mean_delta > 0.02:
            lines.append("Trajectory tends to move slightly more positive from opening to final comment in this sample.")
        elif mean_delta < -0.02:
            lines.append("Trajectory tends to move slightly more negative from opening to final comment in this sample.")
        else:
            lines.append("Average sentiment change from opening to last comment is small (roughly flat trajectory).")
    return " ".join(lines)


def run_endpoint(url: str, out_dir: Path, sia: SentimentIntensityAnalyzer) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug_from_url(url)
    payload = fetch_json(url)
    filters = payload.get("filters") or {}
    posts = payload.get("data") or []
    threads: list[list[dict[str, Any]]] = []
    raw_meta: list[dict[str, Any]] = []
    for p in posts:
        seq = score_thread(sia, p)
        if seq:
            threads.append(seq)
        raw_meta.append(
            {
                "redditId": p.get("redditId"),
                "n_steps": len(seq),
                "compound_path": [round(x["compound"], 4) for x in seq],
            }
        )

    agg = aggregate_by_step(threads)
    summary = {
        "url": url,
        "slug": slug,
        "filters": filters,
        "total_posts": len(posts),
        "threads_scored": len(threads),
        "aggregate": agg,
        "conclusion": conclusion_text(slug, filters, len(posts), len(threads), agg),
    }
    with open(out_dir / "trajectory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_post_trajectories.json", "w", encoding="utf-8") as f:
        json.dump(raw_meta, f, indent=2)

    per_step = agg["per_step"]
    plot_mean_compound_by_step(
        per_step,
        out_dir / "mean_compound_by_thread_position.png",
        f"Mean VADER compound by thread position — {slug}",
    )
    plot_label_mix_stacked(
        per_step,
        out_dir / "sentiment_label_mix_by_position.png",
        f"Label mix by thread position — {slug}",
    )
    deltas = []
    for seq in threads:
        if len(seq) >= 2:
            deltas.append(seq[-1]["compound"] - seq[0]["compound"])
    plot_delta_histogram(
        deltas,
        out_dir / "delta_last_minus_first_hist.png",
        f"Distribution of last−first compound — {slug}",
    )

    print("\n" + "=" * 72)
    print(summary["conclusion"])
    print(f"Wrote outputs under: {out_dir}")


def ensure_vader_lexicon() -> None:
    try:
        SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def main() -> None:
    default_urls = [
        "https://releasetrain.io/api/reddit/query/filter?minComments=3&minScore=0.5",
        "https://releasetrain.io/api/reddit/query/filter?minComments=3&maxScore=0.5",
    ]
    parser = argparse.ArgumentParser(description="VADER sentiment trajectories for releasetrain Reddit filter API")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_project_root() / "tps_gds_classification" / "outputs" / "reddit_trajectory",
        help="Directory for per-endpoint subfolders",
    )
    parser.add_argument("urls", nargs="*", default=default_urls, help="API URLs (default: both filter endpoints)")
    args = parser.parse_args()

    ensure_vader_lexicon()
    sia = SentimentIntensityAnalyzer()

    for url in args.urls:
        slug = _slug_from_url(url)
        run_endpoint(url, args.out_root / slug, sia)


if __name__ == "__main__":
    main()
