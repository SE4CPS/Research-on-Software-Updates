"""
Task 3: Compute agreement between manual (human) sentiment labels and VADER.
Reads data/vader_validation_template.csv (human_sentiment column must be filled).
Outputs: documentation/vader_validation_report.md and confusion matrix figures.
"""

import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


def load_labeled_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            human = (row.get("human_sentiment") or "").strip()
            if human:
                row["human_sentiment"] = human.capitalize()
                if row["human_sentiment"] not in ("Positive", "Negative", "Neutral"):
                    row["human_sentiment"] = "Neutral"
                rows.append(row)
    return rows


def accuracy_and_confusion(rows: list[dict], vader_col: str = "vader_title_label") -> tuple[float, dict]:
    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    for r in rows:
        human = r.get("human_sentiment", "")
        vader = r.get(vader_col, "")
        if human and vader:
            confusion[vader][human] += 1
            if human == vader:
                correct += 1
    acc = correct / len(rows) if rows else 0.0
    return acc, dict(confusion)


def write_report(rows: list[dict], acc_title: float, acc_author: float,
                 conf_title: dict, conf_author: dict, out_path: Path) -> None:
    n = len(rows)
    labels = ["Positive", "Neutral", "Negative"]
    lines = [
        "# VADER Validation Report (Task 3)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Labeled posts:** {n} (manual labels in `data/vader_validation_template.csv`)  ",
        "",
        "## Agreement with VADER",
        "",
        "| VADER reference | Accuracy |",
        "|-----------------|----------|",
        f"| Title sentiment (VADER label) | {acc_title:.1%} ({int(acc_title * n)}/{n} correct) |",
        f"| Author mean bucket (VADER)   | {acc_author:.1%} ({int(acc_author * n)}/{n} correct) |",
        "",
        "## Confusion matrix: Human vs VADER title label",
        "",
        "|  | " + " | ".join(f"Human {l}" for l in labels) + " |",
        "|--|" + "|".join(["---"] * len(labels)) + "|",
    ]
    for v in labels:
        row_vals = [str(conf_title.get(v, {}).get(h, 0)) for h in labels]
        lines.append(f"| **VADER {v}** | " + " | ".join(row_vals) + " |")
    lines.extend([
        "",
        "## Confusion matrix: Human vs VADER author-mean bucket",
        "",
        "|  | " + " | ".join(f"Human {l}" for l in labels) + " |",
        "|--|" + "|".join(["---"] * len(labels)) + "|",
    ])
    for v in labels:
        row_vals = [str(conf_author.get(v, {}).get(h, 0)) for h in labels]
        lines.append(f"| **VADER author {v}** | " + " | ".join(row_vals) + " |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Accuracy:** Proportion of posts where human label matches VADER.",
        "- **Title label:** VADER sentiment of the post title only.",
        "- **Author mean bucket:** VADER compound averaged over author replies, then bucketed (Positive ≥ 0.05, Negative ≤ -0.05, else Neutral).",
        "",
    ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def plot_confusion_matrix(conf: dict, title: str, out_path: Path) -> None:
    labels = ["Positive", "Neutral", "Negative"]
    matrix = np.zeros((3, 3))
    for i, v in enumerate(labels):
        for j, h in enumerate(labels):
            matrix[i, j] = conf.get(v, {}).get(h, 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Reference (human)")
    ax.set_ylabel("VADER")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    csv_path = project_root / "data" / "vader_validation_template.csv"
    report_path = project_root / "documentation" / "vader_validation_report.md"
    viz_dir = project_root / "visualizations"

    if not csv_path.exists():
        print("Template not found. Run first: python3 scripts/create_vader_validation_sample.py")
        return

    rows = load_labeled_csv(csv_path)
    if not rows:
        print("No rows with 'human_sentiment' filled. Add Positive/Negative/Neutral for each post.")
        return

    print(f"Loaded {len(rows)} labeled posts")
    acc_title, conf_title = accuracy_and_confusion(rows, "vader_title_label")
    acc_author, conf_author = accuracy_and_confusion(rows, "vader_author_bucket")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(rows, acc_title, acc_author, conf_title, conf_author, report_path)

    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_confusion_matrix(conf_title, "VADER title label vs reference", viz_dir / "vader_validation_confusion_title.png")
        plot_confusion_matrix(conf_author, "VADER author bucket vs reference", viz_dir / "vader_validation_confusion_author.png")
    except Exception as e:
        print(f"  Skipping plots: {e}")

    print(f"\nVADER title accuracy: {acc_title:.1%}")
    print(f"VADER author bucket accuracy: {acc_author:.1%}")
    print("Done.")


if __name__ == "__main__":
    main()
