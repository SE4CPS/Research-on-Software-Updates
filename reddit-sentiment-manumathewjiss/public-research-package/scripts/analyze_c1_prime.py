#!/usr/bin/env python3
"""
Statistical analysis: TPS vs GDS on C1′ trajectory/divergence metrics.

Usage (from research-open-text-sentimental/):
  python3 scripts/analyze_c1_prime.py
  python3 scripts/analyze_c1_prime.py --tier-a-only
  python3 scripts/analyze_c1_prime.py --plot

Outputs:
  analysis/outputs/c1_prime_divergence/
    divergence_report.md
    divergence_report_tier_a.md   (if --tier-a-only or always both)
    summary.json
    comparisons_full.csv
    figure_divergence_violin.png    (with --plot)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.c1_prime_stats import (  # noqa: E402
    author_more_negative_rate,
    comparisons_to_dataframe,
    filter_tier_a,
    format_markdown_report,
    run_all_comparisons,
)

DEFAULT_METRICS = _REPO_ROOT / "data" / "c1_prime" / "c1_prime_metrics.csv"
DEFAULT_OUT = _REPO_ROOT / "analysis" / "outputs" / "c1_prime_divergence"


def _plot_violin(df: pd.DataFrame, out_path: Path) -> None:
    """Violin plot for key metrics by class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["sentiment_divergence", "author_mean_sentiment", "community_mean_sentiment"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    if len(metrics) == 1:
        axes = [axes]
    labels_map = {0: "GDS", 1: "TPS"}
    for ax, col in zip(axes, metrics):
        data = [
            df.loc[df["corrected_label"] == 0, col].dropna().to_numpy(),
            df.loc[df["corrected_label"] == 1, col].dropna().to_numpy(),
        ]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["GDS", "TPS"])
        ax.set_title(col.replace("_", " "))
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.suptitle("C1′ metrics by discourse class (VADER)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_cohort(
    df: pd.DataFrame,
    cohort_name: str,
    out_dir: Path,
    n_boot: int,
    random_state: int,
    write_prefix: str,
) -> dict:
    comparisons = run_all_comparisons(df, n_boot=n_boot, random_state=random_state)
    author_neg = author_more_negative_rate(df)
    comp_df = comparisons_to_dataframe(comparisons)

    comp_df.to_csv(out_dir / f"comparisons_{write_prefix}.csv", index=False)

    report = format_markdown_report(cohort_name, df, comparisons, author_neg)
    (out_dir / f"divergence_report_{write_prefix}.md").write_text(report, encoding="utf-8")

    return {
        "cohort": cohort_name,
        "n_posts": int(len(df)),
        "n_tps": int((df["corrected_label"] == 1).sum()),
        "n_gds": int((df["corrected_label"] == 0).sum()),
        "author_more_negative": author_neg,
        "comparisons": [c.to_dict() for c in comparisons],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="C1′ TPS vs GDS divergence statistical analysis.")
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--tier-a-only", action="store_true", help="Only analyze Tier-A subreddits.")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="Save violin figure for full cohort.")
    args = p.parse_args()

    if not args.metrics_csv.exists():
        print(f"Missing {args.metrics_csv}. Run: python3 scripts/build_c1_prime.py", file=sys.stderr)
        return 1

    df = pd.read_csv(args.metrics_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_csv": str(args.metrics_csv),
        "cohorts": {},
    }

    if args.tier_a_only:
        tier_df = filter_tier_a(df)
        summary["cohorts"]["tier_a"] = analyze_cohort(
            tier_df,
            "Tier-A software/dev subreddits",
            args.out_dir,
            args.bootstrap_samples,
            args.random_state,
            "tier_a",
        )
    else:
        summary["cohorts"]["full"] = analyze_cohort(
            df,
            "Full C1′ (542 labeled)",
            args.out_dir,
            args.bootstrap_samples,
            args.random_state,
            "full",
        )
        tier_df = filter_tier_a(df)
        summary["cohorts"]["tier_a"] = analyze_cohort(
            tier_df,
            "Tier-A software/dev subreddits",
            args.out_dir,
            args.bootstrap_samples,
            args.random_state,
            "tier_a",
        )
        if args.plot:
            _plot_violin(df, args.out_dir / "figure_divergence_violin_full.png")
            _plot_violin(tier_df, args.out_dir / "figure_divergence_violin_tier_a.png")

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console summary for divergence (key paper metric)
    full = summary["cohorts"].get("full") or summary["cohorts"].get("tier_a")
    for c in full["comparisons"]:
        if c["metric"] == "sentiment_divergence":
            print("\n=== sentiment_divergence (|author − community|) ===")
            print(f"  TPS mean: {c['mean_tps']}  GDS mean: {c['mean_gds']}  Δ: {c['mean_diff_tps_minus_gds']}")
            print(f"  Welch p: {c['welch_p']}  Mann–Whitney p: {c['mannwhitney_p']}")
            print(f"  Bootstrap CI (Δ): [{c['bootstrap_mean_diff_ci_lower']}, {c['bootstrap_mean_diff_ci_upper']}]")
    an = full["author_more_negative"]
    print("\n=== % author more negative than community ===")
    print(f"  TPS: {an['TPS']['rate']}  GDS: {an['GDS']['rate']}  Δ: {an.get('rate_diff_tps_minus_gds')}")
    print(f"  p (Mann–Whitney): {an.get('mannwhitney_p')}")
    print(f"\nReports: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
