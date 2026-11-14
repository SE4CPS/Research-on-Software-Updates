"""
Compute aggregate statistics over all 324 analyzed posts.
Outputs: mean author vs community sentiment, average divergence,
% threads where author is more negative, and optional trend/volatility summaries.
"""

import json
from pathlib import Path


def load_data(data_path: Path) -> dict:
    """Load enhanced automated sentiment results."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def trajectory_mean(compound_list: list) -> float:
    """Mean of compound sentiment values."""
    if not compound_list:
        return 0.0
    return sum(compound_list) / len(compound_list)


def trajectory_trend(compound_list: list) -> float:
    """Linear slope (trend) per comment; 0 if < 2 points."""
    if not compound_list or len(compound_list) < 2:
        return 0.0
    n = len(compound_list)
    x = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(compound_list) / n
    num = sum((x[i] - x_mean) * (compound_list[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def trajectory_volatility(compound_list: list) -> float:
    """Mean absolute change between consecutive values."""
    if not compound_list or len(compound_list) < 2:
        return 0.0
    changes = [abs(compound_list[i] - compound_list[i - 1]) for i in range(1, len(compound_list))]
    return sum(changes) / len(changes)


def compute_post_stats(post: dict) -> dict | None:
    """Compute per-post stats. Returns None if missing required data."""
    author_replies = post.get("author_replies") or []
    community_comments = post.get("community_comments") or []
    if not author_replies or not community_comments:
        return None

    author_compounds = [r["sentiment"]["compound"] for r in author_replies]
    community_compounds = [c["sentiment"]["compound"] for c in community_comments]

    author_mean = trajectory_mean(author_compounds)
    community_mean = trajectory_mean(community_compounds)
    divergence = abs(author_mean - community_mean)
    author_more_negative = author_mean < community_mean

    return {
        "post_id": post.get("post_id"),
        "subreddit": post.get("subreddit"),
        "author_mean": round(author_mean, 4),
        "community_mean": round(community_mean, 4),
        "divergence": round(divergence, 4),
        "author_more_negative": author_more_negative,
        "author_trend": round(trajectory_trend(author_compounds), 4),
        "community_trend": round(trajectory_trend(community_compounds), 4),
        "author_volatility": round(trajectory_volatility(author_compounds), 4),
        "community_volatility": round(trajectory_volatility(community_compounds), 4),
        "n_author_replies": len(author_replies),
        "n_community_comments": len(community_comments),
    }


def compute_aggregates(post_stats: list[dict]) -> dict:
    """Compute aggregate statistics over all posts."""
    n = len(post_stats)
    if n == 0:
        return {}

    author_means = [p["author_mean"] for p in post_stats]
    community_means = [p["community_mean"] for p in post_stats]
    divergences = [p["divergence"] for p in post_stats]
    author_more_negative_count = sum(1 for p in post_stats if p["author_more_negative"])
    author_trends = [p["author_trend"] for p in post_stats]
    community_trends = [p["community_trend"] for p in post_stats]
    author_volatilities = [p["author_volatility"] for p in post_stats]
    community_volatilities = [p["community_volatility"] for p in post_stats]

    return {
        "n_posts": n,
        "overall_mean_author_sentiment": round(sum(author_means) / n, 4),
        "overall_mean_community_sentiment": round(sum(community_means) / n, 4),
        "average_divergence": round(sum(divergences) / n, 4),
        "median_divergence": round(sorted(divergences)[n // 2] if n else 0, 4),
        "pct_author_more_negative": round(100 * author_more_negative_count / n, 1),
        "count_author_more_negative": author_more_negative_count,
        "author_mean_min": round(min(author_means), 4),
        "author_mean_max": round(max(author_means), 4),
        "community_mean_min": round(min(community_means), 4),
        "community_mean_max": round(max(community_means), 4),
        "divergence_min": round(min(divergences), 4),
        "divergence_max": round(max(divergences), 4),
        "average_author_trend": round(sum(author_trends) / n, 4),
        "average_community_trend": round(sum(community_trends) / n, 4),
        "average_author_volatility": round(sum(author_volatilities) / n, 4),
        "average_community_volatility": round(sum(community_volatilities) / n, 4),
        "pct_author_trend_positive": round(100 * sum(1 for t in author_trends if t > 0) / n, 1),
        "pct_community_trend_positive": round(100 * sum(1 for t in community_trends if t > 0) / n, 1),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "enhanced_automated_sentiment_results.json"
    out_json = project_root / "data" / "aggregate_stats.json"
    out_report = project_root / "documentation" / "aggregate_stats_report.md"

    print("Loading data...")
    data = load_data(data_path)
    posts = data.get("all_analyzed_posts", [])
    meta = data.get("analysis_metadata", {})

    print(f"Computing per-post stats for {len(posts)} posts...")
    post_stats = []
    for post in posts:
        s = compute_post_stats(post)
        if s is not None:
            post_stats.append(s)

    print(f"Valid posts with author + community data: {len(post_stats)}")
    aggregates = compute_aggregates(post_stats)

    # Save JSON
    output = {
        "source": "enhanced_automated_sentiment_results.json",
        "source_metadata": meta,
        "n_posts_analyzed": len(post_stats),
        "aggregate_stats": aggregates,
        "per_post_stats_sample": post_stats[:5],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_json}")

    # Write report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Aggregate Statistics — 324 Posts",
        "",
        f"**Source:** `data/enhanced_automated_sentiment_results.json`  ",
        f"**Posts with valid author + community data:** {len(post_stats)}  ",
        "",
        "## Summary Table",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall mean author sentiment | {aggregates['overall_mean_author_sentiment']} |",
        f"| Overall mean community sentiment | {aggregates['overall_mean_community_sentiment']} |",
        f"| Average author–community divergence | {aggregates['average_divergence']} |",
        f"| Median divergence | {aggregates['median_divergence']} |",
        f"| % of threads where author is more negative than community | {aggregates['pct_author_more_negative']}% |",
        f"| Count (author more negative) | {aggregates['count_author_more_negative']} |",
        "",
        "## Ranges",
        "",
        "| Metric | Min | Max |",
        "|--------|-----|-----|",
        f"| Author mean sentiment | {aggregates['author_mean_min']} | {aggregates['author_mean_max']} |",
        f"| Community mean sentiment | {aggregates['community_mean_min']} | {aggregates['community_mean_max']} |",
        f"| Divergence | {aggregates['divergence_min']} | {aggregates['divergence_max']} |",
        "",
        "## Trajectory (trend & volatility)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average author trend (slope per comment) | {aggregates['average_author_trend']} |",
        f"| Average community trend | {aggregates['average_community_trend']} |",
        f"| % of threads with positive author trend | {aggregates['pct_author_trend_positive']}% |",
        f"| % of threads with positive community trend | {aggregates['pct_community_trend_positive']}% |",
        f"| Average author volatility | {aggregates['average_author_volatility']} |",
        f"| Average community volatility | {aggregates['average_community_volatility']} |",
        "",
        "## Interpretation",
        "",
        "- **Author vs community:** Positive overall means indicate that on average both authors and community lean positive; compare means to see who is more positive/negative.",
        "- **Divergence:** Low average/median divergence means author and community sentiment are often aligned; high divergence means they often differ.",
        "- **Author more negative:** This percentage is the share of threads where the OP's mean sentiment is lower than the community's.",
        "- **Trend:** Positive trend = sentiment tends to improve over the thread; negative = sentiment tends to worsen.",
        "- **Volatility:** Higher community volatility is expected (many different commenters); author trajectory is typically smoother.",
        "",
    ]
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_report}")
    print("\nDone.")


if __name__ == "__main__":
    main()
