"""
Task 4: Compare sentiment and trajectory metrics by subreddit.
Uses 324 posts from enhanced_automated_sentiment_results.json.
Outputs: report (table by subreddit) and visualizations (bar charts).
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def load_data(data_path: Path) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def trajectory_mean(compound_list: list) -> float:
    if not compound_list:
        return 0.0
    return sum(compound_list) / len(compound_list)


def compute_post_stats(post: dict) -> dict | None:
    author_replies = post.get("author_replies") or []
    community_comments = post.get("community_comments") or []
    if not author_replies or not community_comments:
        return None
    author_compounds = [r["sentiment"]["compound"] for r in author_replies]
    community_compounds = [c["sentiment"]["compound"] for c in community_comments]
    author_mean = trajectory_mean(author_compounds)
    community_mean = trajectory_mean(community_compounds)
    return {
        "subreddit": post.get("subreddit", ""),
        "author_mean": author_mean,
        "community_mean": community_mean,
        "divergence": abs(author_mean - community_mean),
        "author_more_negative": author_mean < community_mean,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "enhanced_automated_sentiment_results.json"
    report_path = project_root / "documentation" / "subreddit_comparison_report.md"
    viz_dir = project_root / "visualizations"

    min_posts_per_subreddit = 5  # only include subreddits with at least this many posts

    print("Loading data...")
    data = load_data(data_path)
    posts = data.get("all_analyzed_posts", [])

    post_stats = []
    for post in posts:
        s = compute_post_stats(post)
        if s is not None:
            post_stats.append(s)

    # Group by subreddit
    by_sub = defaultdict(list)
    for p in post_stats:
        by_sub[p["subreddit"]].append(p)

    # Filter and aggregate
    subreddit_agg = []
    for sub, plist in by_sub.items():
        if len(plist) < min_posts_per_subreddit:
            continue
        n = len(plist)
        subreddit_agg.append({
            "subreddit": sub,
            "n_posts": n,
            "mean_author_sentiment": round(sum(p["author_mean"] for p in plist) / n, 4),
            "mean_community_sentiment": round(sum(p["community_mean"] for p in plist) / n, 4),
            "mean_divergence": round(sum(p["divergence"] for p in plist) / n, 4),
            "pct_author_more_negative": round(100 * sum(1 for p in plist if p["author_more_negative"]) / n, 1),
        })

    subreddit_agg.sort(key=lambda x: x["n_posts"], reverse=True)

    # Report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Subreddit Comparison (Task 4)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Source:** 324 posts from `enhanced_automated_sentiment_results.json`  ",
        f"**Included:** Subreddits with ≥ {min_posts_per_subreddit} posts  ",
        "",
        "## Summary by subreddit",
        "",
        "| Subreddit | N posts | Mean author sentiment | Mean community sentiment | Mean divergence | % author more negative |",
        "|-----------|---------|------------------------|--------------------------|-----------------|------------------------|",
    ]
    for r in subreddit_agg:
        lines.append(f"| {r['subreddit']} | {r['n_posts']} | {r['mean_author_sentiment']} | {r['mean_community_sentiment']} | {r['mean_divergence']} | {r['pct_author_more_negative']}% |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Comparison of sentiment and author–community alignment across subreddits. Differences may reflect community norms, topic mix (e.g. support vs. announcements), or sample size.",
        "",
    ])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    # Visualizations
    viz_dir.mkdir(parents=True, exist_ok=True)
    subs = [r["subreddit"] for r in subreddit_agg]
    n_subs = len(subs)
    if n_subs == 0:
        print("No subreddits with enough posts; skipping plots.")
        return

    x = np.arange(n_subs)
    width = 0.35

    # Chart 1: Mean author vs community sentiment by subreddit
    fig, ax = plt.subplots(figsize=(max(10, n_subs * 0.5), 5))
    author_vals = [r["mean_author_sentiment"] for r in subreddit_agg]
    community_vals = [r["mean_community_sentiment"] for r in subreddit_agg]
    ax.bar(x - width / 2, author_vals, width, label="Author", color="#1f77b4")
    ax.bar(x + width / 2, community_vals, width, label="Community", color="#ff7f0e")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=45, ha="right")
    ax.set_ylabel("Mean sentiment")
    ax.set_title("Task 4: Mean author vs community sentiment by subreddit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "subreddit_comparison_sentiment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {viz_dir / 'subreddit_comparison_sentiment.png'}")

    # Chart 2: Mean divergence and % author more negative by subreddit
    fig, ax1 = plt.subplots(figsize=(max(10, n_subs * 0.5), 5))
    div_vals = [r["mean_divergence"] for r in subreddit_agg]
    pct_vals = [r["pct_author_more_negative"] for r in subreddit_agg]
    ax1.bar(x - width / 2, div_vals, width, label="Mean divergence", color="#2ca02c")
    ax1.set_ylabel("Mean divergence", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, pct_vals, width, label="% author more negative", color="#9467bd", alpha=0.8)
    ax2.set_ylabel("% author more negative", color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax1.set_xticks(x)
    ax1.set_xticklabels(subs, rotation=45, ha="right")
    ax1.set_title("Task 4: Divergence and % author more negative by subreddit")
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(viz_dir / "subreddit_comparison_divergence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {viz_dir / 'subreddit_comparison_divergence.png'}")

    print(f"\nIncluded {len(subreddit_agg)} subreddits (≥ {min_posts_per_subreddit} posts each).")
    print("Done.")


if __name__ == "__main__":
    main()
