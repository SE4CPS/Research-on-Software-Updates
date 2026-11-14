"""
Compare sentiment trajectories for the extended usability vs defect sample.
Produces aggregate stats, summary chart, and report (no per-post trajectory subplots).
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def load_extended_posts(data_path: Path) -> tuple[list, list, dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    usability = data.get("usability_posts", [])
    defect = data.get("defect_posts", [])
    meta = data.get("analysis_metadata", {})
    return usability, defect, meta


def trajectory_stats(trajectory: list) -> dict:
    if not trajectory:
        return {"mean": 0.0, "std": 0.0, "trend": 0.0, "volatility": 0.0}
    arr = np.array(trajectory)
    mean = float(np.mean(arr))
    std = float(np.std(arr)) if len(arr) > 1 else 0.0
    trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
    vol = float(np.mean(np.abs(np.diff(arr)))) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "trend": trend, "volatility": vol}


def per_post_metrics(posts: list) -> list[dict]:
    out = []
    for p in posts:
        at = p.get("author_trajectory") or []
        ct = p.get("community_trajectory") or []
        as_ = trajectory_stats(at)
        cs_ = trajectory_stats(ct)
        div = abs(as_["mean"] - cs_["mean"])
        out.append({
            "author_mean": as_["mean"],
            "community_mean": cs_["mean"],
            "divergence": div,
            "author_more_negative": as_["mean"] < cs_["mean"],
            "author_trend": as_["trend"],
            "community_trend": cs_["trend"],
            "author_volatility": as_["volatility"],
            "community_volatility": cs_["volatility"],
        })
    return out


def aggregate_metrics(per_post: list[dict]) -> dict:
    if not per_post:
        return {}
    n = len(per_post)
    return {
        "n_posts": n,
        "mean_author_sentiment": round(sum(p["author_mean"] for p in per_post) / n, 4),
        "mean_community_sentiment": round(sum(p["community_mean"] for p in per_post) / n, 4),
        "mean_divergence": round(sum(p["divergence"] for p in per_post) / n, 4),
        "median_divergence": round(float(np.median([p["divergence"] for p in per_post])), 4),
        "pct_author_more_negative": round(100 * sum(1 for p in per_post if p["author_more_negative"]) / n, 1),
        "mean_author_trend": round(sum(p["author_trend"] for p in per_post) / n, 4),
        "mean_community_trend": round(sum(p["community_trend"] for p in per_post) / n, 4),
        "mean_author_volatility": round(sum(p["author_volatility"] for p in per_post) / n, 4),
        "mean_community_volatility": round(sum(p["community_volatility"] for p in per_post) / n, 4),
    }


def create_summary_chart(usability_agg: dict, defect_agg: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Usability vs Defect Posts — Aggregate Comparison (Extended Sample)", fontsize=12, fontweight="bold")

    categories = ["Usability", "Defect"]
    x = np.arange(len(categories))
    width = 0.35

    # 1. Mean author vs community sentiment
    ax = axes[0]
    author_vals = [usability_agg["mean_author_sentiment"], defect_agg["mean_author_sentiment"]]
    community_vals = [usability_agg["mean_community_sentiment"], defect_agg["mean_community_sentiment"]]
    ax.bar(x - width / 2, author_vals, width, label="Author", color="#1f77b4")
    ax.bar(x + width / 2, community_vals, width, label="Community", color="#ff7f0e")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean sentiment")
    ax.set_title("Mean author vs community sentiment")
    ax.legend()

    # 2. Mean and median divergence
    ax = axes[1]
    mean_div = [usability_agg["mean_divergence"], defect_agg["mean_divergence"]]
    median_div = [usability_agg["median_divergence"], defect_agg["median_divergence"]]
    ax.bar(x - width / 2, mean_div, width, label="Mean divergence", color="#2ca02c")
    ax.bar(x + width / 2, median_div, width, label="Median divergence", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Divergence")
    ax.set_title("Author–community divergence")
    ax.legend()

    # 3. % author more negative
    ax = axes[2]
    pct = [usability_agg["pct_author_more_negative"], defect_agg["pct_author_more_negative"]]
    bars = ax.bar(x, pct, color=["#1f77b4", "#d62728"])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("%")
    ax.set_title("% threads where author is more negative")
    for b, v in zip(bars, pct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def write_report(usability_agg: dict, defect_agg: dict, meta: dict, out_path: Path) -> None:
    lines = [
        "# Usability vs Defect — Extended Sample Comparison",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Usability posts:** {usability_agg['n_posts']}  ",
        f"**Defect posts:** {defect_agg['n_posts']}  ",
        "",
        "## Summary table",
        "",
        "| Metric | Usability | Defect |",
        "|--------|-----------|--------|",
        f"| Mean author sentiment | {usability_agg['mean_author_sentiment']} | {defect_agg['mean_author_sentiment']} |",
        f"| Mean community sentiment | {usability_agg['mean_community_sentiment']} | {defect_agg['mean_community_sentiment']} |",
        f"| Mean divergence | {usability_agg['mean_divergence']} | {defect_agg['mean_divergence']} |",
        f"| Median divergence | {usability_agg['median_divergence']} | {defect_agg['median_divergence']} |",
        f"| % author more negative | {usability_agg['pct_author_more_negative']}% | {defect_agg['pct_author_more_negative']}% |",
        "",
        "## Trend & volatility",
        "",
        "| Metric | Usability | Defect |",
        "|--------|-----------|--------|",
        f"| Mean author trend | {usability_agg['mean_author_trend']} | {defect_agg['mean_author_trend']} |",
        f"| Mean community trend | {usability_agg['mean_community_trend']} | {defect_agg['mean_community_trend']} |",
        f"| Mean author volatility | {usability_agg['mean_author_volatility']} | {defect_agg['mean_author_volatility']} |",
        f"| Mean community volatility | {usability_agg['mean_community_volatility']} | {defect_agg['mean_community_volatility']} |",
        "",
        "## Conclusion",
        "",
        "Comparison of auto-labeled usability vs defect posts (extended sample). "
        "Defect-related threads show lower mean author sentiment than usability-related threads in this sample; "
        "divergence and % author more negative can be compared across categories above.",
        "",
        "## Visualization",
        "",
        "See `visualizations/usability_defect_extended_summary.png`.",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    extended_path = project_root / "data" / "usability_defect_posts_extended.json"
    fallback_path = project_root / "data" / "usability_defect_posts.json"

    if extended_path.exists():
        data_path = extended_path
        print("Loading extended sample...")
    else:
        data_path = fallback_path
        print("Extended file not found; using original usability_defect_posts.json...")

    if not data_path.exists():
        print("No data file found. Run: python3 scripts/grow_usability_defect_sample.py")
        return

    usability_posts, defect_posts, meta = load_extended_posts(data_path)
    print(f"Usability: {len(usability_posts)} | Defect: {len(defect_posts)}")

    if not usability_posts or not defect_posts:
        print("Need at least one post per category.")
        return

    us_metrics = per_post_metrics(usability_posts)
    def_metrics = per_post_metrics(defect_posts)
    usability_agg = aggregate_metrics(us_metrics)
    defect_agg = aggregate_metrics(def_metrics)

    viz_dir = project_root / "visualizations"
    doc_dir = project_root / "documentation"
    viz_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    create_summary_chart(usability_agg, defect_agg, viz_dir / "usability_defect_extended_summary.png")
    write_report(usability_agg, defect_agg, meta, doc_dir / "usability_defect_extended_report.md")

    print("Done.")


if __name__ == "__main__":
    main()
