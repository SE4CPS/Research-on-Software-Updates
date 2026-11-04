"""Publication-candidate trajectory figures (match plot_data tables exactly)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_all_figures(
    out: Path,
    *,
    fixed_plot: list[dict],
    elapsed_plot: list[dict],
    gap_plot: list[dict],
    four_plot: list[dict],
) -> dict[str, Any]:
    _style()
    fig_dir = out / "figures"
    info: dict[str, Any] = {}

    info["fig_C"] = _fig_fixed(fig_dir / "fig_C_fixed_eligibility_author_path.png", fixed_plot)
    info["fig_B"] = _fig_elapsed(fig_dir / "fig_B_elapsed_time_author_community.png", elapsed_plot)
    info["fig_E"] = _fig_gap(fig_dir / "fig_E_exploratory_author_community_gap.png", gap_plot)
    info["fig_F"] = _fig_four(fig_dir / "fig_F_four_model_two_event.png", four_plot)
    info["fig_eligibility"] = _fig_eligibility_bars(
        fig_dir / "fig_eligibility_counts.png", fixed_plot, elapsed_plot
    )
    return info


def _fig_fixed(path: Path, rows: list[dict]) -> dict:
    order = ["post", "first", "second"]
    xlabels = ["Event 0\n(post)", "First author\nreply", "Second author\nreply"]
    cohorts = ["ai", "frozen"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {"TPS": "#1f4e79", "GDS": "#b8572a"}

    for ax, cohort in zip(axes, cohorts):
        for lab in ("TPS", "GDS"):
            sub = [r for r in rows if r["cohort"] == cohort and r["label"] == lab]
            sub = sorted(sub, key=lambda r: order.index(r["point"]))
            xs = list(range(len(order)))
            means = [r["mean"] for r in sub]
            lo = [r["ci95_low"] for r in sub]
            hi = [r["ci95_high"] for r in sub]
            n = sub[0]["n"] if sub else 0
            yerr = np.array(
                [
                    [m - l if m is not None and l is not None else 0 for m, l in zip(means, lo)],
                    [h - m if m is not None and h is not None else 0 for m, h in zip(means, hi)],
                ]
            )
            ax.errorbar(
                xs,
                means,
                yerr=yerr,
                fmt="-o",
                color=colors[lab],
                label=f"{lab} (n={n})",
                capsize=3,
                linewidth=1.5,
                markersize=6,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
        ax.set_title("AI-assisted purpose labels" if cohort == "ai" else "Frozen-score comparable")
        ax.set_xlabel("Fixed-eligibility author path (same threads at all points)")
        ax.axhline(0, color="#888", linewidth=0.6, linestyle=":")
        ax.legend(frameon=False, loc="best")
    axes[0].set_ylabel("Mean VADER compound")
    fig.suptitle(
        "Fixed-eligibility original-author sentiment path (VADER)\n"
        "95% whole-thread bootstrap CIs; Event 0 = title+body",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "n_rows": len(rows)}


def _fig_elapsed(path: Path, rows: list[dict]) -> dict:
    cohorts = ["ai", "frozen"]
    labels = ["TPS", "GDS"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=True)
    role_style = {
        "post": {"color": "#333333", "marker": "D", "ls": "None", "label": "Event 0 (post)"},
        "author": {"color": "#1f4e79", "marker": "o", "ls": "-", "label": "Author replies"},
        "community": {"color": "#2a7a4b", "marker": "s", "ls": "--", "label": "Community replies"},
    }
    bin_order = ["Event0", "0-6h", "6-24h", "1-3d", ">3d"]

    for i, cohort in enumerate(cohorts):
        for j, lab in enumerate(labels):
            ax = axes[i][j]
            for role, style in role_style.items():
                sub = [
                    r
                    for r in rows
                    if r["cohort"] == cohort and r["label"] == lab and r["role"] == role
                ]
                # order
                sub = sorted(
                    sub,
                    key=lambda r: bin_order.index(r["bin"]) if r["bin"] in bin_order else 99,
                )
                if not sub:
                    continue
                xs = [r["bin_mid_hours"] for r in sub]
                means = [r["mean"] for r in sub]
                # plot sparse as open markers without connecting strongly
                for r, x, m in zip(sub, xs, means):
                    if m is None:
                        continue
                    yerr = None
                    if r["ci95_low"] is not None and r["ci95_high"] is not None:
                        yerr = np.array([[m - r["ci95_low"]], [r["ci95_high"] - m]])
                    face = "none" if r.get("sparse") else style["color"]
                    ax.errorbar(
                        [x],
                        [m],
                        yerr=yerr,
                        fmt=style["marker"],
                        color=style["color"],
                        markerfacecolor=face,
                        capsize=2,
                        markersize=6,
                    )
                # connect non-sparse role points (excluding Event0 post for role lines)
                if role != "post":
                    solid = [r for r in sub if r["mean"] is not None and not r.get("sparse")]
                    if len(solid) >= 2:
                        ax.plot(
                            [r["bin_mid_hours"] for r in solid],
                            [r["mean"] for r in solid],
                            linestyle=style["ls"],
                            color=style["color"],
                            linewidth=1.2,
                            alpha=0.85,
                        )
                # annotate n only for sparse cells to avoid overlap; full n in plot_data CSV
                for r in sub:
                    if r["mean"] is None or not r.get("sparse"):
                        continue
                    ax.annotate(
                        f"n={r['n_threads']}*",
                        (r["bin_mid_hours"], r["mean"]),
                        textcoords="offset points",
                        xytext=(0, -12),
                        ha="center",
                        fontsize=7,
                        color="#8a3b12",
                    )
            # compact per-panel n strip (Event0 / role×bin) under title
            e0 = next(
                (
                    r
                    for r in rows
                    if r["cohort"] == cohort
                    and r["label"] == lab
                    and r["role"] == "post"
                    and r["bin"] == "Event0"
                ),
                None,
            )
            n_strip = f"Event0 n={e0['n_threads']}" if e0 else ""
            ax.set_title(f"{'AI' if cohort=='ai' else 'Frozen'} · {lab}\n{n_strip}", fontsize=10)
            ax.axhline(0, color="#888", linewidth=0.6, linestyle=":")
            ax.set_xlabel("Elapsed hours since post (bin midpoints)")
            if j == 0:
                ax.set_ylabel("Mean VADER compound\n(thread-equal weight)")
            # legend once
            if i == 0 and j == 1:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=s["color"],
                        marker=s["marker"],
                        linestyle=s["ls"] if s["ls"] != "None" else "None",
                        label=s["label"],
                    )
                    for s in role_style.values()
                ]
                ax.legend(handles=handles, frameon=False, loc="best")

    fig.suptitle(
        "Elapsed-time author vs community sentiment (VADER)\n"
        "Predeclared bins; hollow markers / n* = sparse (n_threads < 20); full per-bin n in plot_data CSV; no imputation",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "n_rows": len(rows)}


def _fig_gap(path: Path, rows: list[dict]) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = {"TPS": "#1f4e79", "GDS": "#b8572a"}
    for ax, cohort in zip(axes, ("ai", "frozen")):
        for lab in ("TPS", "GDS"):
            sub = [r for r in rows if r["cohort"] == cohort and r["label"] == lab]
            sub = sorted(sub, key=lambda r: r["bin_mid_hours"])
            plotted = False
            for r in sub:
                if r["mean_author_minus_community"] is None:
                    continue
                m = r["mean_author_minus_community"]
                yerr = None
                if r["ci95_low"] is not None:
                    yerr = np.array([[m - r["ci95_low"]], [r["ci95_high"] - m]])
                ax.errorbar(
                    [r["bin_mid_hours"]],
                    [m],
                    yerr=yerr,
                    fmt="o",
                    color=colors[lab],
                    markerfacecolor=("none" if r.get("sparse") else colors[lab]),
                    capsize=2,
                    label=(lab if not plotted else None),
                )
                plotted = True
                if r.get("sparse") or (r["n_threads_both_roles"] or 0) < 40:
                    ax.annotate(
                        f"n={r['n_threads_both_roles']}",
                        (r["bin_mid_hours"], m),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=7,
                        color="#555",
                    )
            # connect non-sparse
            solid = [r for r in sub if r["mean_author_minus_community"] is not None and not r.get("sparse")]
            if len(solid) >= 2:
                ax.plot(
                    [r["bin_mid_hours"] for r in solid],
                    [r["mean_author_minus_community"] for r in solid],
                    color=colors[lab],
                    linewidth=1.2,
                )
        ax.axhline(0, color="#888", linewidth=0.6, linestyle=":")
        ax.set_title("AI-assisted" if cohort == "ai" else "Frozen-score")
        ax.set_xlabel("Elapsed hours (bin midpoints)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Author − community VADER mean\n(same thread & bin)")
    fig.suptitle(
        "Exploratory within-bin author−community gap (VADER)\n"
        "Requires both roles in-bin; hollow = sparse; not a causal claim",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "n_rows": len(rows), "exploratory": True}


def _fig_four(path: Path, rows: list[dict]) -> dict:
    models = [
        ("vader", "VADER compound"),
        ("textblob", "TextBlob polarity"),
        ("cardiff", "Cardiff sign"),
        ("senti4sd", "Senti4SD sign"),
    ]
    # AI primary panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = {"TPS": "#1f4e79", "GDS": "#b8572a"}
    for ax, (mid, title) in zip(axes.ravel(), models):
        for lab in ("TPS", "GDS"):
            sub = [
                r
                for r in rows
                if r["cohort"] == "ai" and r["label"] == lab and r["model"] == mid
            ]
            sub = sorted(sub, key=lambda r: 0 if r["point"] == "post" else 1)
            if not sub:
                continue
            xs = [0, 1]
            means = [r["mean"] for r in sub]
            lo = [r["ci95_low"] for r in sub]
            hi = [r["ci95_high"] for r in sub]
            n = sub[0]["n"]
            yerr = np.array(
                [
                    [m - l for m, l in zip(means, lo)],
                    [h - m for m, h in zip(means, hi)],
                ]
            )
            ax.errorbar(
                xs,
                means,
                yerr=yerr,
                fmt="-o",
                color=colors[lab],
                label=f"{lab} (n={n})",
                capsize=3,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Event 0\n(post)", "First author\nreply"])
        ax.set_title(title)
        ax.axhline(0, color="#888", linewidth=0.6, linestyle=":")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Four-model two-event sensitivity (AI matched pairs only)\n"
        "Native scales; 6,996 pairs total across cohorts — AI panels shown; no invented later scores",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path), "n_rows": len(rows), "constraint": "two-event only"}


def _fig_eligibility_bars(path: Path, fixed_plot: list[dict], elapsed_plot: list[dict]) -> dict:
    """Mandatory count/eligibility companion figure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # fixed ge2 n from fixed_plot
    for ax, cohort, title in zip(
        axes,
        ("ai", "frozen"),
        ("AI-assisted", "Frozen-score"),
    ):
        vals = []
        labs = []
        for lab in ("TPS", "GDS"):
            sub = [r for r in fixed_plot if r["cohort"] == cohort and r["label"] == lab and r["point"] == "post"]
            n = sub[0]["n"] if sub else 0
            vals.append(n)
            labs.append(lab)
        ax.bar(labs, vals, color=["#1f4e79", "#b8572a"])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title}: fixed ≥2 author path n")
        ax.set_ylabel("Threads (fixed across Event0/first/second)")
    fig.suptitle("Eligibility counts for connected author-path figure", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(path)}
