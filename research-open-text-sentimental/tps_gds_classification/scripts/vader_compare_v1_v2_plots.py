#!/usr/bin/env python3
"""
Create detailed comparison visualizations across:
- v1 (merged text including all comments)
- v2 engagement_top_n (per-comment aggregation + weighted final)
- v2 random_n (per-comment aggregation + weighted final, random sample)

Reads existing JSON outputs and writes NEW PNGs to a dedicated folder
so nothing gets overwritten.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib

os.environ["MPLCONFIGDIR"] = "/tmp/mplcache"
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]

V1_DIR = ROOT / "tps_gds_classification" / "outputs" / "vader_endpoint_comparison"
V2_DIR = ROOT / "tps_gds_classification" / "outputs" / "vader_endpoint_comparison_v2"

OUT_DIR = ROOT / "tps_gds_classification" / "outputs" / "vader_endpoint_comparison_compare_v1_v2"

METHOD_LABELS = {
    "v1": "Baseline Combined-Text Method",
    "v2e": "Per-Comment Aggregation (High-Engagement Sample)",
    "v2r": "Per-Comment Aggregation (Random Sample)",
}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_series_v1(path: Path) -> tuple[list[float], list[str]]:
    rows = load_json(path)
    return [float(r["compound"]) for r in rows], [str(r["label"]) for r in rows]


def get_series_v2(path: Path) -> tuple[list[float], list[str]]:
    rows = load_json(path)
    return [float(r["final_compound"]) for r in rows], [str(r["label"]) for r in rows]


def smooth(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    w = max(1, int(window))
    if w <= 1:
        return arr
    if len(arr) < w:
        return np.array([arr.mean()] * len(arr), dtype=float)
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(arr, kernel, mode="same")


def label_pct(labels: list[str]) -> dict[str, float]:
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for lb in labels:
        if lb in counts:
            counts[lb] += 1
    total = len(labels) or 1
    return {k: 100.0 * v / total for k, v in counts.items()}


def plot_trend_compare(
    endpoint_slug: str,
    v1_vals: list[float],
    v2e_vals: list[float],
    v2r_vals: list[float],
    window: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    def draw(vals: list[float], color: str, name: str) -> None:
        x = np.arange(1, len(vals) + 1)
        ax.plot(x, vals, color=color, alpha=0.25, linewidth=1.2, marker="o", markersize=2.6, label=f"{name} raw")
        ax.plot(x, smooth(vals, window), color=color, linewidth=2.4, label=f"{name} smoothed (w={window})")

    draw(v1_vals, "#2c5282", METHOD_LABELS["v1"])
    draw(v2e_vals, "#c05621", METHOD_LABELS["v2e"])
    draw(v2r_vals, "#2f855a", METHOD_LABELS["v2r"])

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(0.05, color="green", linestyle=":", linewidth=0.7)
    ax.axhline(-0.05, color="red", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Post order (1..30) within each method")
    ax.set_ylabel("VADER compound")
    ax.set_title(f"Comparative Sentiment Trend (Raw and Smoothed) — {endpoint_slug}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_distribution_compare(
    endpoint_slug: str,
    v1_labels: list[str],
    v2e_labels: list[str],
    v2r_labels: list[str],
    out_path: Path,
) -> None:
    methods = [METHOD_LABELS["v1"], METHOD_LABELS["v2e"], METHOD_LABELS["v2r"]]
    pcts = [label_pct(v1_labels), label_pct(v2e_labels), label_pct(v2r_labels)]
    cats = ["Positive", "Neutral", "Negative"]

    x = np.arange(len(methods))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Positive": "#38a169", "Neutral": "#718096", "Negative": "#e53e3e"}
    for i, cat in enumerate(cats):
        vals = [p[cat] for p in pcts]
        ax.bar(x + (i - 1) * width, vals, width, label=cat, color=colors[cat])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel("Percent of posts")
    ax.set_ylim(0, 100)
    ax.set_title(f"Comparative Sentiment Label Distribution — {endpoint_slug}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_boxplot_compare(
    endpoint_slug: str,
    v1_vals: list[float],
    v2e_vals: list[float],
    v2r_vals: list[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(
        [v1_vals, v2e_vals, v2r_vals],
        tick_labels=[METHOD_LABELS["v1"], METHOD_LABELS["v2e"], METHOD_LABELS["v2r"]],
        patch_artist=True,
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.set_ylabel("VADER compound")
    ax.set_title(f"Comparative Compound-Score Variability — {endpoint_slug}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hist_compare(
    endpoint_slug: str,
    v1_vals: list[float],
    v2e_vals: list[float],
    v2r_vals: list[float],
    out_path: Path,
) -> None:
    bins = np.linspace(-1, 1, 25)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(v1_vals, bins=bins, alpha=0.40, label=METHOD_LABELS["v1"], color="#2c5282")
    ax.hist(v2e_vals, bins=bins, alpha=0.40, label=METHOD_LABELS["v2e"], color="#c05621")
    ax.hist(v2r_vals, bins=bins, alpha=0.40, label=METHOD_LABELS["v2r"], color="#2f855a")
    ax.set_xlabel("VADER compound")
    ax.set_ylabel("Count of posts")
    ax.set_title(f"Comparative Compound-Score Distribution — {endpoint_slug}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    endpoints = {
        "minScore_0.5": {
            "v1": V1_DIR / "selected_posts_minScore_0.5.json",
            "v2e": V2_DIR / "engagement_top_n" / "selected_posts_minScore_0.5.json",
            "v2r": V2_DIR / "random_n" / "selected_posts_minScore_0.5.json",
        },
        "maxScore_0.5": {
            "v1": V1_DIR / "selected_posts_maxScore_0.5.json",
            "v2e": V2_DIR / "engagement_top_n" / "selected_posts_maxScore_0.5.json",
            "v2r": V2_DIR / "random_n" / "selected_posts_maxScore_0.5.json",
        },
    }

    window = 5

    for slug, paths in endpoints.items():
        v1_vals, v1_labels = get_series_v1(paths["v1"])
        v2e_vals, v2e_labels = get_series_v2(paths["v2e"])
        v2r_vals, v2r_labels = get_series_v2(paths["v2r"])

        plot_trend_compare(
            slug,
            v1_vals=v1_vals,
            v2e_vals=v2e_vals,
            v2r_vals=v2r_vals,
            window=window,
            out_path=OUT_DIR / f"{slug}_trend_compare.png",
        )
        plot_label_distribution_compare(
            slug,
            v1_labels=v1_labels,
            v2e_labels=v2e_labels,
            v2r_labels=v2r_labels,
            out_path=OUT_DIR / f"{slug}_label_distribution_compare.png",
        )
        plot_boxplot_compare(
            slug,
            v1_vals=v1_vals,
            v2e_vals=v2e_vals,
            v2r_vals=v2r_vals,
            out_path=OUT_DIR / f"{slug}_boxplot_compare.png",
        )
        plot_hist_compare(
            slug,
            v1_vals=v1_vals,
            v2e_vals=v2e_vals,
            v2r_vals=v2r_vals,
            out_path=OUT_DIR / f"{slug}_hist_compare.png",
        )

    print("Wrote comparison plots to:", OUT_DIR)


if __name__ == "__main__":
    main()

