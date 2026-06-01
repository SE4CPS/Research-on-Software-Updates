#!/usr/bin/env python3
"""
Render ICMLA Results-section figures from verified evaluation/analysis outputs.

Usage:
  python3 scripts/render_icmla_results_figures.py

Writes PNGs to analysis/outputs/paper_figures/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "analysis" / "outputs" / "paper_figures"

CV_FULL = _REPO / "evaluation/outputs/paper_cv_full/summary.json"
ROBERTA = _REPO / "evaluation/outputs/paper_roberta_5fold/summary.json"
CV_TIER = _REPO / "evaluation/outputs/paper_cv_tier_a/summary.json"
ABL_FULL = _REPO / "evaluation/outputs/ablations/ablations_v1/ablation_table.csv"
C1_METRICS = _REPO / "data/c1_prime/c1_prime_metrics.csv"
DIV_SUMMARY = _REPO / "analysis/outputs/c1_prime_divergence/summary.json"
NB_CM = _REPO / "evaluation/outputs/paper_cv_full/naive_bayes/confusion_matrix_oof.csv"

DPI = 300
IEEE_COLORS = {
    "nb": "#2c3e50",
    "lr": "#3498db",
    "roberta": "#9b59b6",
    "vader": "#e74c3c",
    "tps": "#2980b9",
    "gds": "#95a5a6",
    "d1": "#27ae60",
    "d3": "#f39c12",
    "d4": "#8e44ad",
}


def _style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "sans-serif",
        }
    )


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def fig_classification_benchmark() -> None:
    full = _load_json(CV_FULL)
    roberta = _load_json(ROBERTA)

    models = [
        ("Naive Bayes", "naive_bayes", IEEE_COLORS["nb"]),
        ("Logistic Reg.", "logistic_regression", IEEE_COLORS["lr"]),
        ("RoBERTa", "roberta", IEEE_COLORS["roberta"]),
        ("VADER rules", "vader_rules", IEEE_COLORS["vader"]),
    ]

    macro, macro_std, tps_f1, tps_std, colors = [], [], [], [], []
    for _label, key, color in models:
        if key == "roberta":
            agg = roberta["models"]["roberta"]["aggregate"]
        else:
            agg = full["models"][key]["aggregate"]
        macro.append(agg["f1_macro_mean"])
        macro_std.append(agg["f1_macro_std"])
        tps_f1.append(agg["f1_tps_mean"])
        tps_std.append(agg["f1_tps_std"])
        colors.append(color)

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(3.5, 2.6), facecolor="white")

    b1 = ax.bar(
        x - w / 2,
        macro,
        w,
        yerr=macro_std,
        capsize=3,
        label="Macro-F1",
        color=colors,
        edgecolor="#2c3e50",
        linewidth=0.5,
        error_kw={"elinewidth": 0.8, "ecolor": "#34495e"},
    )
    b2 = ax.bar(
        x + w / 2,
        tps_f1,
        w,
        yerr=tps_std,
        capsize=3,
        label="TPS-F1",
        color=colors,
        edgecolor="#2c3e50",
        linewidth=0.5,
        alpha=0.55,
        hatch="//",
        error_kw={"elinewidth": 0.8, "ecolor": "#34495e"},
    )

    ax.set_ylabel("Score")
    ax.set_xticks(x, [m[0] for m in models], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("TPS/GDS classification (5-fold CV, n=542)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.axhline(0.5, color="#bdc3c7", linewidth=0.6, linestyle="--", alpha=0.6)

    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    fig.tight_layout()
    _save(fig, "fig_results_classification_benchmark.png")


def fig_divergence_violin() -> None:
    df = pd.read_csv(C1_METRICS)
    df = df[["corrected_label", "sentiment_divergence"]].dropna()
    df["class"] = df["corrected_label"].map({1: "TPS", 0: "GDS"})

    div = _load_json(DIV_SUMMARY)
    row = next(
        c
        for c in div["cohorts"]["full"]["comparisons"]
        if c["metric"] == "sentiment_divergence"
    )
    mean_tps = row["mean_tps"]
    mean_gds = row["mean_gds"]
    p_val = row["mannwhitney_p"]

    fig, ax = plt.subplots(figsize=(3.2, 2.6), facecolor="white")

    data_tps = df.loc[df["class"] == "TPS", "sentiment_divergence"].values
    data_gds = df.loc[df["class"] == "GDS", "sentiment_divergence"].values
    positions = [1, 2]
    parts = ax.violinplot(
        [data_tps, data_gds],
        positions=positions,
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(IEEE_COLORS["tps"] if i == 0 else IEEE_COLORS["gds"])
        body.set_alpha(0.35)
        body.set_edgecolor("#2c3e50")
        body.set_linewidth(0.6)

    bp = ax.boxplot(
        [data_tps, data_gds],
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#2c3e50", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    bp["boxes"][0].set_facecolor(IEEE_COLORS["tps"])
    bp["boxes"][1].set_facecolor(IEEE_COLORS["gds"])
    for box in bp["boxes"]:
        box.set_alpha(0.85)

    ax.scatter([1], [mean_tps], marker="D", s=28, color="#c0392b", zorder=5, label="Mean")
    ax.scatter([2], [mean_gds], marker="D", s=28, color="#c0392b", zorder=5)

    ax.set_xticks(positions, [f"TPS\n(n={len(data_tps)})", f"GDS\n(n={len(data_gds)})"])
    ax.set_ylabel("|Author − community| (VADER)")
    ax.set_title("Sentiment divergence by discourse class (C1′)")
    ax.text(
        0.5,
        0.97,
        f"Δ mean = {mean_tps - mean_gds:.3f}, MW p = {p_val:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
    )
    ax.set_ylim(0, max(df["sentiment_divergence"].max() * 1.05, 0.5))
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, "fig_results_divergence_violin.png")


def fig_ablation_features() -> None:
    abl = pd.read_csv(ABL_FULL)
    keep = ["D1_title_only", "D2_title_body", "D3_full_thread", "D4_full_plus_trajectory"]
    abl = abl[abl["ablation_id"].isin(keep)].copy()
    labels = {
        "D1_title_only": "D1 Title",
        "D2_title_body": "D2 Title+body",
        "D3_full_thread": "D3 Full thread",
        "D4_full_plus_trajectory": "D4 + trajectory",
    }
    abl["label"] = abl["ablation_id"].map(labels)
    abl = abl.iloc[::-1]

    y = np.arange(len(abl))
    h = 0.32
    fig, ax = plt.subplots(figsize=(3.6, 2.4), facecolor="white")

    ax.barh(
        y - h / 2,
        abl["macro_f1_mean"],
        height=h,
        xerr=abl["macro_f1_std"],
        capsize=2,
        label="Macro-F1",
        color=IEEE_COLORS["d1"],
        edgecolor="#2c3e50",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7},
    )
    ax.barh(
        y + h / 2,
        abl["tps_f1_mean"],
        height=h,
        xerr=abl["tps_f1_std"],
        capsize=2,
        label="TPS-F1",
        color=IEEE_COLORS["d4"],
        edgecolor="#2c3e50",
        linewidth=0.5,
        alpha=0.85,
        error_kw={"elinewidth": 0.7},
    )

    ax.set_yticks(y, abl["label"])
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.0)
    ax.set_title("Feature ablations (logistic regression, 5-fold CV)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.axvline(0.632, color=IEEE_COLORS["d3"], linewidth=0.8, linestyle=":", alpha=0.7)

    fig.tight_layout()
    _save(fig, "fig_results_ablation_features.png")


def fig_tier_a_robustness() -> None:
    full = _load_json(CV_FULL)
    tier = _load_json(CV_TIER)
    div = _load_json(DIV_SUMMARY)

    metrics = [
        ("NB macro-F1", "naive_bayes", "f1_macro_mean"),
        ("NB TPS-F1", "naive_bayes", "f1_tps_mean"),
        ("Divergence Δ", None, None),
    ]

    full_vals, tier_vals = [], []
    for title, model, key in metrics:
        if title == "Divergence Δ":
            f_row = next(
                c
                for c in div["cohorts"]["full"]["comparisons"]
                if c["metric"] == "sentiment_divergence"
            )
            t_row = next(
                c
                for c in div["cohorts"]["tier_a"]["comparisons"]
                if c["metric"] == "sentiment_divergence"
            )
            full_vals.append(f_row["mean_diff_tps_minus_gds"])
            tier_vals.append(t_row["mean_diff_tps_minus_gds"])
        else:
            full_vals.append(full["models"][model]["aggregate"][key])
            tier_vals.append(tier["models"][model]["aggregate"][key])

    x = np.arange(len(metrics))
    w = 0.32
    fig, ax = plt.subplots(figsize=(3.5, 2.4), facecolor="white")
    ax.bar(x - w / 2, full_vals, w, label="Full C1 (n=542)", color="#7f8c8d", edgecolor="#2c3e50", linewidth=0.5)
    ax.bar(x + w / 2, tier_vals, w, label="Tier-A (n=304)", color=IEEE_COLORS["tps"], edgecolor="#2c3e50", linewidth=0.5)

    ax.set_xticks(x, [m[0] for m in metrics])
    ax.set_ylabel("Score / mean Δ")
    ax.set_ylim(0, max(max(full_vals), max(tier_vals)) * 1.15)
    ax.set_title("Tier-A software-subreddit robustness")
    ax.legend(loc="upper left", framealpha=0.95)

    for i, (fv, tv) in enumerate(zip(full_vals, tier_vals)):
        ax.text(i - w / 2, fv + 0.02, f"{fv:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, tv + 0.02, f"{tv:.3f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    _save(fig, "fig_results_tier_a_robustness.png")


def fig_nb_confusion_oof() -> None:
    cm = pd.read_csv(NB_CM, index_col=0)
    mat = cm.values.astype(int)
    labels = [["GDS", "TPS"], ["GDS", "TPS"]]

    fig, ax = plt.subplots(figsize=(2.8, 2.4), facecolor="white")
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=mat.max())

    ax.set_xticks([0, 1], ["Pred GDS", "Pred TPS"])
    ax.set_yticks([0, 1], ["True GDS", "True TPS"])
    ax.set_title("Naive Bayes OOF confusion (n=542)")

    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] > mat.max() * 0.55 else "#2c3e50"
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, "fig_appendix_nb_confusion_oof.png")


def main() -> None:
    _style()
    fig_classification_benchmark()
    fig_divergence_violin()
    fig_ablation_features()
    fig_tier_a_robustness()
    fig_nb_confusion_oof()
    print(f"\nAll figures saved under: {OUT}")


if __name__ == "__main__":
    main()
