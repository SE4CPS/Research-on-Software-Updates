"""
Generate visualizations for aggregate statistics (324 posts).
Produces: summary bar/pie charts, distribution histograms, author vs community scatter.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(data_path: Path) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def trajectory_mean(compound_list: list) -> float:
    if not compound_list:
        return 0.0
    return sum(compound_list) / len(compound_list)


def trajectory_trend(compound_list: list) -> float:
    if not compound_list or len(compound_list) < 2:
        return 0.0
    n = len(compound_list)
    x = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(compound_list) / n
    num = sum((x[i] - x_mean) * (compound_list[i] - y_mean) for i in range(n))
    den = sum((x[i] - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def trajectory_volatility(compound_list: list) -> float:
    if not compound_list or len(compound_list) < 2:
        return 0.0
    changes = [abs(compound_list[i] - compound_list[i - 1]) for i in range(1, len(compound_list))]
    return sum(changes) / len(changes)


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
        "author_mean": author_mean,
        "community_mean": community_mean,
        "divergence": abs(author_mean - community_mean),
        "author_more_negative": author_mean < community_mean,
        "author_trend": trajectory_trend(author_compounds),
        "community_trend": trajectory_trend(community_compounds),
        "author_volatility": trajectory_volatility(author_compounds),
        "community_volatility": trajectory_volatility(community_compounds),
    }


def plot_summary(aggregates: dict, out_path: Path) -> None:
    """Summary: author vs community mean, divergence, pie (author more negative), trend/volatility bars."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("Aggregate Statistics — 324 Reddit Posts (Software Updates)", fontsize=12, fontweight="bold")

    # 1. Bar: mean author vs community sentiment
    ax = axes[0, 0]
    labels = ["Author", "Community"]
    values = [aggregates["overall_mean_author_sentiment"], aggregates["overall_mean_community_sentiment"]]
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Mean sentiment (compound)")
    ax.set_title("Overall mean sentiment")
    ax.set_ylim(-0.1, 0.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # 2. Bar: average and median divergence
    ax = axes[0, 1]
    labels = ["Average", "Median"]
    values = [aggregates["average_divergence"], aggregates["median_divergence"]]
    bars = ax.bar(labels, values, color=["#2ca02c", "#9467bd"], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Divergence")
    ax.set_title("Author–community divergence")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # 3. Pie: author more negative vs not
    ax = axes[1, 0]
    pct_neg = aggregates["pct_author_more_negative"]
    sizes = [pct_neg, 100 - pct_neg]
    labels = [f"Author more negative\n({pct_neg:.1f}%)", f"Author ≥ community\n({100 - pct_neg:.1f}%)"]
    colors = ["#d62728", "#bcbd22"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Author vs community sentiment")

    # 4. Bar: average trend and volatility (author vs community)
    ax = axes[1, 1]
    x = np.arange(4)
    vals = [
        aggregates["average_author_trend"],
        aggregates["average_community_trend"],
        aggregates["average_author_volatility"],
        aggregates["average_community_volatility"],
    ]
    labels = ["Author trend", "Community trend", "Author vol.", "Community vol."]
    colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Slope / volatility")
    ax.set_title("Trend (slope) & volatility")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_distributions(post_stats: list[dict], out_path: Path) -> None:
    """Histograms: author mean, community mean, divergence."""
    author_means = [p["author_mean"] for p in post_stats]
    community_means = [p["community_mean"] for p in post_stats]
    divergences = [p["divergence"] for p in post_stats]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Distribution of per-thread statistics (324 posts)", fontsize=12, fontweight="bold")

    for ax, data, title, xlabel in [
        (axes[0], author_means, "Author mean sentiment", "Mean sentiment"),
        (axes[1], community_means, "Community mean sentiment", "Mean sentiment"),
        (axes[2], divergences, "Author–community divergence", "Divergence"),
    ]:
        ax.hist(data, bins=25, color="#1f77b4", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=2, label=f"Mean = {np.mean(data):.3f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scatter(post_stats: list[dict], out_path: Path) -> None:
    """Scatter: author_mean vs community_mean; diagonal line; color by author_more_negative."""
    author_means = [p["author_mean"] for p in post_stats]
    community_means = [p["community_mean"] for p in post_stats]
    author_more_neg = [p["author_more_negative"] for p in post_stats]

    fig, ax = plt.subplots(figsize=(7, 7))
    for i, (am, cm) in enumerate(zip(author_means, community_means)):
        color = "#d62728" if author_more_neg[i] else "#2ca02c"
        ax.scatter(am, cm, c=color, alpha=0.5, s=25, edgecolors="none")
    lims = [-0.7, 1.0]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Author = Community")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Author mean sentiment")
    ax.set_ylabel("Community mean sentiment")
    ax.set_title("Author vs community mean sentiment (324 posts)\nRed = author more negative; Green = author ≥ community")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "enhanced_automated_sentiment_results.json"
    viz_dir = project_root / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(data_path)
    posts = data.get("all_analyzed_posts", [])

    print("Computing per-post stats...")
    post_stats = []
    for post in posts:
        s = compute_post_stats(post)
        if s is not None:
            post_stats.append(s)

    n = len(post_stats)
    aggregates = {
        "n_posts": n,
        "overall_mean_author_sentiment": sum(p["author_mean"] for p in post_stats) / n,
        "overall_mean_community_sentiment": sum(p["community_mean"] for p in post_stats) / n,
        "average_divergence": sum(p["divergence"] for p in post_stats) / n,
        "median_divergence": sorted(p["divergence"] for p in post_stats)[n // 2],
        "pct_author_more_negative": 100 * sum(1 for p in post_stats if p["author_more_negative"]) / n,
        "count_author_more_negative": sum(1 for p in post_stats if p["author_more_negative"]),
        "average_author_trend": sum(p["author_trend"] for p in post_stats) / n,
        "average_community_trend": sum(p["community_trend"] for p in post_stats) / n,
        "average_author_volatility": sum(p["author_volatility"] for p in post_stats) / n,
        "average_community_volatility": sum(p["community_volatility"] for p in post_stats) / n,
    }

    print(f"Generating visualizations ({n} posts)...")
    plot_summary(aggregates, viz_dir / "aggregate_summary.png")
    plot_distributions(post_stats, viz_dir / "aggregate_distributions.png")
    plot_scatter(post_stats, viz_dir / "aggregate_scatter.png")
    print("Done.")


if __name__ == "__main__":
    main()
