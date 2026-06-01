#!/usr/bin/env python3
"""
Aggregate sentiment trajectory figure (TPS vs GDS) for ICMLA Results.

Builds chronological cumulative author/community VADER means per thread,
normalizes progress to 0--100%, aggregates by corrected_label.

Data: C1′ raw posts + labels (same pipeline as data/c1_prime/).

Usage:
  python3 scripts/render_sentiment_trajectory_figure.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from trajectory.build import load_raw_posts_by_id  # noqa: E402

LABELS_CSV = _REPO / "data" / "c1_prime" / "c1_prime_metrics.csv"
RAW_DIR = _REPO / "tps_gds_classification" / "data" / "raw"
OUT_PATH = _REPO / "analysis" / "outputs" / "paper_figures" / "fig_results_sentiment_trajectory.png"
META_PATH = _REPO / "analysis" / "outputs" / "paper_figures" / "fig_results_sentiment_trajectory_meta.json"

GRID_PCT = np.linspace(0, 100, 51)
MIN_SCORED_COMMENTS = 2
MIN_COMMUNITY_COMMENTS = 1
DPI = 300
Z_CI = 1.96


def _clean_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_time(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def comment_sort_key(c: dict[str, Any]) -> float:
    ts = c.get("created_utc_ts")
    if ts is not None:
        try:
            return float(ts)
        except (TypeError, ValueError):
            pass
    return _parse_time(c.get("created_utc"))


def is_op_comment(c: dict[str, Any], post: dict[str, Any]) -> bool:
    v = c.get("is_submitter")
    if v is True:
        return True
    if v is False:
        return False
    pa = (post.get("author") or "").strip().lower()
    ca = (c.get("author") or "").strip().lower()
    return bool(pa) and pa == ca


def post_opening_compound(post: dict[str, Any], sia: SentimentIntensityAnalyzer) -> float:
    parts = [
        _clean_text(post.get("title") or ""),
        _clean_text(post.get("author_description") or post.get("body") or ""),
    ]
    text = _clean_text(" ".join(p for p in parts if p))
    return float(sia.polarity_scores(text)["compound"]) if text else 0.0


def cumulative_chronology_series(
    post: dict[str, Any], sia: SentimentIntensityAnalyzer
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Returns (pct, author_cum, community_cum, abs_gap) for chronological comment stream.
    pct in [0, 100]. Points only where community cumulative is defined.
    """
    opening = post_opening_compound(post, sia)
    comments = sorted(post.get("comments") or [], key=comment_sort_key)
    scored: list[tuple[bool, float]] = []
    for c in comments:
        body = _clean_text(c.get("body") or "")
        if not body:
            continue
        scored.append((is_op_comment(c, post), float(sia.polarity_scores(body)["compound"])))

    if len(scored) < MIN_SCORED_COMMENTS:
        return None

    n = len(scored)
    pct_steps: list[float] = []
    author_cum: list[float] = []
    community_cum: list[float] = []
    gap: list[float] = []

    author_sum, author_cnt = opening, 1
    community_sum, community_cnt = 0.0, 0

    for i, (is_author, sc) in enumerate(scored, start=1):
        if is_author:
            author_sum += sc
            author_cnt += 1
        else:
            community_sum += sc
            community_cnt += 1
        if community_cnt < MIN_COMMUNITY_COMMENTS:
            continue
        a_mean = author_sum / author_cnt
        c_mean = community_sum / community_cnt
        pct_steps.append(100.0 * i / n)
        author_cum.append(a_mean)
        community_cum.append(c_mean)
        gap.append(abs(a_mean - c_mean))

    if len(pct_steps) < 2:
        return None

    return (
        np.array(pct_steps, dtype=float),
        np.array(author_cum, dtype=float),
        np.array(community_cum, dtype=float),
        np.array(gap, dtype=float),
    )


def resample_to_grid(pct: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear interpolation onto GRID_PCT; edge-fill outside range."""
    if len(pct) < 2:
        return np.full_like(GRID_PCT, np.nan, dtype=float)
    order = np.argsort(pct)
    pct = pct[order]
    values = values[order]
    return np.interp(GRID_PCT, pct, values, left=values[0], right=values[-1])


def aggregate_class(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 95% CI (normal approx via SEM across threads) at each grid point."""
    stack = np.vstack(curves)
    mean = np.nanmean(stack, axis=0)
    n = np.sum(~np.isnan(stack), axis=0).astype(float)
    std = np.nanstd(stack, axis=0, ddof=1)
    sem = np.where(n > 1, std / np.sqrt(n), 0.0)
    ci = Z_CI * sem
    return mean, mean - ci, mean + ci


def main() -> None:
    labels = pd.read_csv(LABELS_CSV)
    posts_by_id = load_raw_posts_by_id(RAW_DIR)
    sia = SentimentIntensityAnalyzer()

    tps_curves_gap: list[np.ndarray] = []
    tps_curves_author: list[np.ndarray] = []
    tps_curves_community: list[np.ndarray] = []
    gds_curves_gap: list[np.ndarray] = []
    gds_curves_author: list[np.ndarray] = []
    gds_curves_community: list[np.ndarray] = []

    skipped = 0
    for _, row in labels.iterrows():
        rid = str(row["reddit_id"]).strip()
        post = posts_by_id.get(rid)
        if not post:
            skipped += 1
            continue
        series = cumulative_chronology_series(post, sia)
        if series is None:
            skipped += 1
            continue
        pct, author, community, gap = series
        gap_g = resample_to_grid(pct, gap)
        author_g = resample_to_grid(pct, author)
        community_g = resample_to_grid(pct, community)
        label = int(row["corrected_label"])
        if label == 1:
            tps_curves_gap.append(gap_g)
            tps_curves_author.append(author_g)
            tps_curves_community.append(community_g)
        else:
            gds_curves_gap.append(gap_g)
            gds_curves_author.append(author_g)
            gds_curves_community.append(community_g)

    if not tps_curves_gap or not gds_curves_gap:
        raise SystemExit("Insufficient trajectory curves for TPS or GDS.")

    tps_mean, tps_lo, tps_hi = aggregate_class(tps_curves_gap)
    gds_mean, gds_lo, gds_hi = aggregate_class(gds_curves_gap)
    tps_auth_m, _, _ = aggregate_class(tps_curves_author)
    tps_comm_m, _, _ = aggregate_class(tps_curves_community)
    gds_auth_m, _, _ = aggregate_class(gds_curves_author)
    gds_comm_m, _, _ = aggregate_class(gds_curves_community)

    # Light smoothing on aggregate means only (display); CI from unsmoothed per-thread curves
    def smooth(y: np.ndarray, w: int = 5) -> np.ndarray:
        if len(y) < w:
            return y
        kernel = np.ones(w) / w
        return np.convolve(y, kernel, mode="same")

    tps_plot = smooth(tps_mean)
    gds_plot = smooth(gds_mean)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.4), facecolor="white")

    ax.fill_between(GRID_PCT, tps_lo, tps_hi, color="#2980b9", alpha=0.22, linewidth=0)
    ax.fill_between(GRID_PCT, gds_lo, gds_hi, color="#7f8c8d", alpha=0.18, linewidth=0)
    ax.plot(GRID_PCT, tps_plot, color="#1a5276", linewidth=2.4, label="TPS")
    ax.plot(GRID_PCT, gds_plot, color="#566573", linewidth=2.4, linestyle="-", label="GDS")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Thread progress (% of scored comments, chronological)")
    ax.set_ylabel("Running |author − community| sentiment (VADER)")
    ax.set_title("Aggregate author–community divergence over thread progress (C1′)")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="#bdc3c7")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)

    end_tps = float(tps_mean[-1])
    end_gds = float(gds_mean[-1])
    ax.annotate(
        f"End mean gap: TPS {end_tps:.2f}, GDS {end_gds:.2f}",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        color="#2c3e50",
    )

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_tps_threads": len(tps_curves_gap),
        "n_gds_threads": len(gds_curves_gap),
        "n_skipped": skipped,
        "grid_points": len(GRID_PCT),
        "end_mean_gap_tps": end_tps,
        "end_mean_gap_gds": end_gds,
        "mean_gap_area_tps": float(np.trapezoid(tps_mean, GRID_PCT)),
        "mean_gap_area_gds": float(np.trapezoid(gds_mean, GRID_PCT)),
        "end_author_mean_tps": float(tps_auth_m[-1]),
        "end_community_mean_tps": float(tps_comm_m[-1]),
        "end_author_mean_gds": float(gds_auth_m[-1]),
        "end_community_mean_gds": float(gds_comm_m[-1]),
        "ci_method": "mean ± 1.96·SEM across threads per grid point",
        "sources": [str(LABELS_CSV), str(RAW_DIR)],
    }
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    print(f"Meta: {META_PATH}")
    print(f"TPS threads={len(tps_curves_gap)}, GDS threads={len(gds_curves_gap)}, skipped={skipped}")


if __name__ == "__main__":
    main()
