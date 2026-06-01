#!/usr/bin/env python3
"""Build model_comparison.png from outputs/*/metrics.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUT = _PROJECT_ROOT / "outputs"


def load_metrics(subdir: str) -> dict:
    p = _OUT / subdir / "metrics.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    v = load_metrics("vader_baseline")
    nb = load_metrics("naive_bayes")
    b = load_metrics("bert")

    bert_val_f1 = b.get("val_eval", {}).get("eval_f1_tps")
    if bert_val_f1 is None:
        bert_val_f1 = float("nan")

    rows = [
        [
            "VADER (rule, technical_first)",
            f"{v['val_f1_tps']:.3f}",
            f"{v['test_accuracy']:.3f}",
            f"{v['test_f1_tps']:.3f}",
            f"{v['test_f1_macro']:.3f}",
            f"{v['test_precision_tps']:.3f}",
            f"{v['test_recall_tps']:.3f}",
        ],
        [
            "Naive Bayes (TF-IDF + MultinomialNB)",
            f"{nb['val_f1_tps']:.3f}",
            f"{nb['test_accuracy']:.3f}",
            f"{nb['test_f1_tps']:.3f}",
            f"{nb['test_f1_macro']:.3f}",
            f"{nb['test_precision_tps']:.3f}",
            f"{nb['test_recall_tps']:.3f}",
        ],
        [
            "BERT (bert-base-uncased, fine-tuned)",
            f"{bert_val_f1:.3f}",
            f"{b['test_accuracy']:.3f}",
            f"{b['test_f1_tps']:.3f}",
            f"{b['test_f1_macro']:.3f}",
            f"{b['test_precision_tps']:.3f}",
            f"{b['test_recall_tps']:.3f}",
        ],
    ]

    col_labels = [
        "Model",
        "Val F1\n(TPS)",
        "Test\naccuracy",
        "Test F1\n(TPS)",
        "Test F1\n(macro)",
        "Test prec.\n(TPS)",
        "Test rec.\n(TPS)",
    ]

    fig_w, fig_h = 12.0, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(
        "TPS vs GDS — model comparison (same 40-example test set, random_state=42)",
        fontsize=11,
        pad=12,
        fontweight="bold",
    )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 2.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff")

    fig.tight_layout()
    out_path = _OUT / "model_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
