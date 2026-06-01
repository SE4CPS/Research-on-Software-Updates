"""
Generate additional visualizations for Task 2 (extended usability vs defect sample).
Produces: distributions by category, scatter (author vs community by category), box plots.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_data(data_path: Path) -> tuple[list, list, dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    usability = data.get("usability_posts", [])
    defect = data.get("defect_posts", [])
    meta = data.get("analysis_metadata", {})
    return usability, defect, meta


def trajectory_mean(trajectory: list) -> float:
    if not trajectory:
        return 0.0
    return float(np.mean(trajectory))


def per_post_metrics(posts: list) -> list[dict]:
    out = []
    for p in posts:
        at = p.get("author_trajectory") or []
        ct = p.get("community_trajectory") or []
        author_mean = trajectory_mean(at)
        community_mean = trajectory_mean(ct)
        div = abs(author_mean - community_mean)
        out.append({
            "author_mean": author_mean,
            "community_mean": community_mean,
            "divergence": div,
            "author_trend": float(np.polyfit(np.arange(len(at)), at, 1)[0]) if len(at) > 1 else 0.0,
            "community_trend": float(np.polyfit(np.arange(len(ct)), ct, 1)[0]) if len(ct) > 1 else 0.0,
            "author_volatility": float(np.mean(np.abs(np.diff(at)))) if len(at) > 1 else 0.0,
            "community_volatility": float(np.mean(np.abs(np.diff(ct)))) if len(ct) > 1 else 0.0,
        })
    return out


def plot_distributions(us_metrics: list, def_metrics: list, out_path: Path) -> None:
    """Histograms: author mean, community mean, divergence — usability vs defect overlaid."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Usability vs Defect — Distribution of per-post metrics (30 + 30 posts)", fontsize=12, fontweight="bold")

    us_author = [p["author_mean"] for p in us_metrics]
    us_community = [p["community_mean"] for p in us_metrics]
    us_div = [p["divergence"] for p in us_metrics]
    def_author = [p["author_mean"] for p in def_metrics]
    def_community = [p["community_mean"] for p in def_metrics]
    def_div = [p["divergence"] for p in def_metrics]

    bins = np.linspace(-0.6, 1.0, 20)
    for ax, us_vals, def_vals, title in [
        (axes[0], us_author, def_author, "Author mean sentiment"),
        (axes[1], us_community, def_community, "Community mean sentiment"),
        (axes[2], us_div, def_div, "Divergence"),
    ]:
        ax.hist(us_vals, bins=bins, alpha=0.6, label="Usability", color="#1f77b4", edgecolor="white")
        ax.hist(def_vals, bins=bins, alpha=0.6, label="Defect", color="#d62728", edgecolor="white")
        ax.axvline(np.mean(us_vals), color="#1f77b4", linestyle="--", linewidth=2, label=f"Usability mean = {np.mean(us_vals):.3f}")
        ax.axvline(np.mean(def_vals), color="#d62728", linestyle="--", linewidth=2, label=f"Defect mean = {np.mean(def_vals):.3f}")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scatter(usability_posts: list, defect_posts: list, us_metrics: list, def_metrics: list, out_path: Path) -> None:
    """Scatter: author mean vs community mean; points colored by category (usability vs defect)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_title("Usability vs Defect — Author mean vs community mean (30 + 30 posts)", fontsize=12, fontweight="bold")

    us_author = [p["author_mean"] for p in us_metrics]
    us_community = [p["community_mean"] for p in us_metrics]
    def_author = [p["author_mean"] for p in def_metrics]
    def_community = [p["community_mean"] for p in def_metrics]

    ax.scatter(us_author, us_community, c="#1f77b4", alpha=0.6, s=40, label="Usability", edgecolors="white")
    ax.scatter(def_author, def_community, c="#d62728", alpha=0.6, s=40, label="Defect", edgecolors="white")
    lims = [-0.6, 1.0]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Author = Community")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Author mean sentiment")
    ax.set_ylabel("Community mean sentiment")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_boxplots(us_metrics: list, def_metrics: list, out_path: Path) -> None:
    """Box plots: author mean, community mean, divergence by category (usability vs defect)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Usability vs Defect — Box plots by category (30 + 30 posts)", fontsize=12, fontweight="bold")

    us_author = [p["author_mean"] for p in us_metrics]
    us_community = [p["community_mean"] for p in us_metrics]
    us_div = [p["divergence"] for p in us_metrics]
    def_author = [p["author_mean"] for p in def_metrics]
    def_community = [p["community_mean"] for p in def_metrics]
    def_div = [p["divergence"] for p in def_metrics]

    data = [
        ([us_author, def_author], "Author mean sentiment", ["Usability", "Defect"]),
        ([us_community, def_community], "Community mean sentiment", ["Usability", "Defect"]),
        ([us_div, def_div], "Divergence", ["Usability", "Defect"]),
    ]
    colors = ["#1f77b4", "#d62728"]
    for ax, (vals_list, title, labels) in zip(axes, data):
        bp = ax.boxplot(vals_list, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_trend_volatility(us_metrics: list, def_metrics: list, out_path: Path) -> None:
    """Bar chart: mean trend and mean volatility by category (author vs community, usability vs defect)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Usability vs Defect — Mean trend and volatility by category", fontsize=12, fontweight="bold")

    categories = ["Usability", "Defect"]
    x = np.arange(len(categories))
    width = 0.35

    # Trend: author vs community by category
    ax = axes[0]
    us_author_trend = np.mean([p["author_trend"] for p in us_metrics])
    us_comm_trend = np.mean([p["community_trend"] for p in us_metrics])
    def_author_trend = np.mean([p["author_trend"] for p in def_metrics])
    def_comm_trend = np.mean([p["community_trend"] for p in def_metrics])
    author_vals = [us_author_trend, def_author_trend]
    community_vals = [us_comm_trend, def_comm_trend]
    ax.bar(x - width / 2, author_vals, width, label="Author trend", color="#1f77b4")
    ax.bar(x + width / 2, community_vals, width, label="Community trend", color="#ff7f0e")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean slope per comment")
    ax.set_title("Mean trend (slope)")
    ax.legend()

    # Volatility: author vs community by category
    ax = axes[1]
    us_author_vol = np.mean([p["author_volatility"] for p in us_metrics])
    us_comm_vol = np.mean([p["community_volatility"] for p in us_metrics])
    def_author_vol = np.mean([p["author_volatility"] for p in def_metrics])
    def_comm_vol = np.mean([p["community_volatility"] for p in def_metrics])
    author_vals = [us_author_vol, def_author_vol]
    community_vals = [us_comm_vol, def_comm_vol]
    ax.bar(x - width / 2, author_vals, width, label="Author volatility", color="#1f77b4")
    ax.bar(x + width / 2, community_vals, width, label="Community volatility", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Mean volatility")
    ax.set_title("Mean volatility")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "usability_defect_posts_extended.json"
    viz_dir = project_root / "visualizations"

    if not data_path.exists():
        print("Extended data not found. Run: python3 scripts/grow_usability_defect_sample.py")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)
    print("Loading extended usability vs defect sample...")
    usability_posts, defect_posts, _ = load_data(data_path)
    us_metrics = per_post_metrics(usability_posts)
    def_metrics = per_post_metrics(defect_posts)
    print(f"Usability: {len(us_metrics)} | Defect: {len(def_metrics)}")
    print("Generating Task 2 visualizations...")

    plot_distributions(us_metrics, def_metrics, viz_dir / "usability_defect_distributions.png")
    plot_scatter(usability_posts, defect_posts, us_metrics, def_metrics, viz_dir / "usability_defect_scatter.png")
    plot_boxplots(us_metrics, def_metrics, viz_dir / "usability_defect_boxplots.png")
    plot_trend_volatility(us_metrics, def_metrics, viz_dir / "usability_defect_trend_volatility.png")

    print("Done.")


if __name__ == "__main__":
    main()
