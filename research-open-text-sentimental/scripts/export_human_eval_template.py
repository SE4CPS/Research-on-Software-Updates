#!/usr/bin/env python3
"""
Export stratified human evaluation template (~80 posts).

Usage:
  python3 scripts/export_human_eval_template.py
  python3 scripts/export_human_eval_template.py --n 80

Output: data/human_eval/human_eval_template_80.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.config import EvaluationConfig  # noqa: E402
from evaluation.data_loader import load_c1_extended_frame  # noqa: E402


def stratified_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    """Sample n rows preserving label ratio (TPS/GDS)."""
    n = min(n, len(df))
    tps = df[df["label"] == 1]
    gds = df[df["label"] == 0]
    n_tps = max(1, round(n * len(tps) / len(df)))
    n_tps = min(n_tps, len(tps))
    n_gds = n - n_tps
    n_gds = min(n_gds, len(gds))
  # adjust if rounding short
    parts = []
    if n_tps > 0:
        parts.append(tps.sample(n=n_tps, random_state=random_state))
    if n_gds > 0:
        parts.append(gds.sample(n=n_gds, random_state=random_state + 1))
    out = pd.concat(parts, ignore_index=True)
    if len(out) < n:
        got = set(out["reddit_id"].astype(str))
        remaining = df[~df["reddit_id"].astype(str).isin(got)]
        need = n - len(out)
        if need > 0 and len(remaining) > 0:
            out = pd.concat(
                [
                    out,
                    remaining.sample(
                        n=min(need, len(remaining)),
                        random_state=random_state + 2,
                    ),
                ],
                ignore_index=True,
            )
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Export human eval CSV template.")
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "data" / "human_eval" / "human_eval_template_80.csv")
    args = p.parse_args()

    config = EvaluationConfig()
    df = load_c1_extended_frame(
        config.labels_path,
        config.data_json_path,
        c1_prime_csv=config.c1_prime_csv,
    )

    sample = stratified_sample(df, args.n, args.random_state)

    out = sample[
        [
            "reddit_id",
            "url",
            "subreddit",
            "title_raw",
            "label",
        ]
    ].copy()
    out.rename(columns={"label": "gold_corrected_label"}, inplace=True)
    out["gold_label_name"] = out["gold_corrected_label"].map({1: "TPS", 0: "GDS"})

  # Human fields
    out["human_label_tps_gds"] = ""  # TPS | GDS | Unclear
    out["human_divergence_rating"] = ""  # aligned | author_more_negative | community_more_negative | unclear
    out["human_notes"] = ""

    if "sentiment_divergence" in sample.columns:
        out["vader_divergence"] = sample["sentiment_divergence"].tolist()
    if "author_more_negative" in sample.columns:
        out["vader_author_more_negative"] = sample["author_more_negative"].tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows → {args.out}")
    print(f"  TPS: {(out['gold_corrected_label']==1).sum()}  GDS: {(out['gold_corrected_label']==0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
