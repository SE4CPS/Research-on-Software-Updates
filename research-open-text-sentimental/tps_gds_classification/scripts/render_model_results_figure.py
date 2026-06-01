#!/usr/bin/env python3
"""
Single PNG figure comparing VADER, Naive Bayes, and BERT on test metrics.
Reads outputs/*/metrics.json and writes outputs/model_results_figure.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUT = _PROJECT_ROOT / "outputs"


def load_metrics(subdir: str) -> dict:
    with (_OUT / subdir / "metrics.json").open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    v = load_metrics("vader_baseline")
    nb = load_metrics("naive_bayes")
    b = load_metrics("bert")

    names = ["VADER\n(rule baseline)", "Naive Bayes\n(TF-IDF + NB)", "BERT\n(fine-tuned)"]
    acc = [v["test_accuracy"], nb["test_accuracy"], b["test_accuracy"]]
    f1_tps = [v["test_f1_tps"], nb["test_f1_tps"], b["test_f1_tps"]]
    f1_macro = [v["test_f1_macro"], nb["test_f1_macro"], b["test_f1_macro"]]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")

    x = np.arange(len(names))
    w = 0.24
    c_acc = "#2ecc71"
    c_tps = "#3498db"
    c_mac = "#9b59b6"

    b1 = ax.bar(x - w, acc, w, label="Test accuracy", color=c_acc, edgecolor="#2c3e50", linewidth=0.6)
    b2 = ax.bar(x, f1_tps, w, label="Test F1 (TPS)", color=c_tps, edgecolor="#2c3e50", linewidth=0.6)
    b3 = ax.bar(x + w, f1_macro, w, label="Test F1 (macro)", color=c_mac, edgecolor="#2c3e50", linewidth=0.6)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#2c3e50",
            )

    autolabel(b1)
    autolabel(b2)
    autolabel(b3)

    ax.set_ylabel("Score (0–1)", fontsize=11)
    ax.set_xticks(x, names, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.set_title(
        "TPS vs GDS — three models on the same held-out test set (n = 40)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.axhline(1.0, color="#bdc3c7", linewidth=0.8, linestyle="--", alpha=0.7)

    fig.text(
        0.5,
        0.02,
        "Verified labels · GDS undersampled to 175 · stratified split random_state=42",
        ha="center",
        fontsize=8,
        color="#7f8c8d",
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    out_path = _OUT / "model_results_figure.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
