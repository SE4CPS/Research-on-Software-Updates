#!/usr/bin/env python3
"""
Compute human validation agreement after filling human_eval_template_80.csv.

Usage:
  python3 scripts/analyze_human_eval.py
  python3 scripts/analyze_human_eval.py --input data/human_eval/human_eval_filled.csv

Outputs: analysis/outputs/human_validation/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_INPUT = _REPO_ROOT / "data" / "human_eval" / "human_eval_template_80.csv"
DEFAULT_OUT = _REPO_ROOT / "analysis" / "outputs" / "human_validation"


def _normalize_label(s: str) -> int | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().upper()
    if not t:
        return None
    if t in ("TPS", "1", "TECHNICAL", "TECHNICAL PROBLEM-SOLVING"):
        return 1
    if t in ("GDS", "0", "DISCONTENT", "GENERAL DISCONTENT"):
        return 0
    if t in ("UNCLEAR", "UNKNOWN", "NA", "N/A", "-"):
        return None
    return None


def _normalize_divergence(s: str) -> str | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().lower().replace(" ", "_").replace("-", "_")
    if not t:
        return None
    mapping = {
        "aligned": "aligned",
        "similar": "aligned",
        "author_more_negative": "author_more_negative",
        "op_more_negative": "author_more_negative",
        "community_more_negative": "community_more_negative",
        "crowd_more_negative": "community_more_negative",
        "unclear": "unclear",
    }
    for k, v in mapping.items():
        if k in t:
            return v
    return t if t else None


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    labels = sorted(set(y1) | set(y2))
    cm = np.zeros((len(labels), len(labels)))
    li = {l: i for i, l in enumerate(labels)}
    for a, b in zip(y1, y2):
        cm[li[a], li[b]] += 1
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if pe == 1:
        return 1.0
    return float((po - pe) / (1 - pe))


def interpret_kappa(k: float) -> str:
    if np.isnan(k):
        return "not computed"
    if k < 0:
        return "poor (less than chance)"
    if k < 0.21:
        return "slight agreement"
    if k < 0.41:
        return "fair agreement"
    if k < 0.61:
        return "moderate agreement"
    if k < 0.81:
        return "substantial agreement"
    return "almost perfect agreement"


def vader_divergence_bucket(author_more_negative: bool, divergence: float, thresh: float = 0.15) -> str:
    if divergence < thresh:
        return "aligned"
    return "author_more_negative" if author_more_negative else "community_more_negative"


def main() -> int:
    p = argparse.ArgumentParser(description="Human validation agreement analysis.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    if not args.input.exists():
        print(f"Missing {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df["human_label_int"] = df["human_label_tps_gds"].map(_normalize_label)
    df["human_div"] = df["human_divergence_rating"].map(_normalize_divergence)

    n_label = df["human_label_int"].notna().sum()
    n_div = df["human_div"].notna().sum()

    report_lines = [
        "# Human Validation Report",
        "",
        f"**Source:** `{args.input.name}`",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Rows:** {len(df)} | **Label ratings filled:** {n_label} | **Divergence ratings filled:** {n_div}",
        "",
    ]

    summary: dict = {"n_rows": len(df), "n_label_filled": int(n_label), "n_div_filled": int(n_div)}

    if n_label == 0:
        report_lines.extend(
            [
                "## Status: awaiting human labels",
                "",
                "Fill columns `human_label_tps_gds` and `human_divergence_rating`, then re-run this script.",
                "",
                "See `documentation/HUMAN_VALIDATION_GUIDE.md` for labeling instructions.",
                "",
            ]
        )
    else:
        sub = df[df["human_label_int"].notna()].copy()
        gold = sub["gold_corrected_label"].astype(int).to_numpy()
        human = sub["human_label_int"].astype(int).to_numpy()
        acc = float((gold == human).mean())
        kappa = cohens_kappa(gold, human)
        summary["label_accuracy_vs_gold"] = acc
        summary["cohens_kappa"] = kappa
        summary["kappa_interpretation"] = interpret_kappa(kappa)

        # confusion
        tp = int(((gold == 1) & (human == 1)).sum())
        tn = int(((gold == 0) & (human == 0)).sum())
        fp = int(((gold == 0) & (human == 1)).sum())
        fn = int(((gold == 1) & (human == 0)).sum())

        report_lines.extend(
            [
                "## TPS/GDS agreement with gold (`corrected_label`)",
                "",
                f"- **Accuracy:** {acc:.1%} ({int((gold==human).sum())}/{len(gold)})",
                f"- **Cohen's κ:** {kappa:.3f} ({interpret_kappa(kappa)})",
                "",
                "| | Human TPS | Human GDS |",
                "|--|-----------|-----------|",
                f"| **Gold TPS** | {tp} | {fn} |",
                f"| **Gold GDS** | {fp} | {tn} |",
                "",
            ]
        )

        disagree = sub[sub["human_label_int"] != sub["gold_corrected_label"]][
            ["reddit_id", "subreddit", "title_raw", "gold_label_name", "human_label_tps_gds", "human_notes"]
        ].head(15)
        if len(disagree):
            report_lines.append("### Example label disagreements (up to 15)\n")
            for _, r in disagree.iterrows():
                report_lines.append(
                    f"- **{r['reddit_id']}** (r/{r['subreddit']}): \"{str(r['title_raw'])[:80]}…\" "
                    f"gold={r['gold_label_name']} human={r['human_label_tps_gds']}"
                )
            report_lines.append("")

    if n_div > 0 and "vader_author_more_negative" in df.columns:
        sub_d = df[df["human_div"].notna()].copy()
        sub_d["vader_bucket"] = sub_d.apply(
            lambda r: vader_divergence_bucket(
                bool(r.get("vader_author_more_negative")),
                float(r.get("vader_divergence") or 0),
            ),
            axis=1,
        )
        agree_div = (sub_d["human_div"] == sub_d["vader_bucket"]).mean()
        agree_aligned = (
            (sub_d["human_div"] == "aligned") & (sub_d["vader_bucket"] == "aligned")
        ).sum()
        summary["divergence_agreement_rate"] = float(agree_div)
        report_lines.extend(
            [
                "## Divergence: human vs VADER proxy bucket",
                "",
                f"- **Exact bucket agreement:** {agree_div:.1%} (n={len(sub_d)})",
                "",
                "VADER bucket = `aligned` if divergence < 0.15 else author/community more negative.",
                "",
            ]
        )
        sub_d[["reddit_id", "human_div", "vader_bucket", "vader_divergence"]].to_csv(
            args.out_dir / "divergence_comparison.csv", index=False
        )

    (args.out_dir / "human_validation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n".join(report_lines[:20]))
    print(f"\nWrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
