#!/usr/bin/env python3
"""
Build a spreadsheet-friendly CSV for manual label review.

Each row has a link (url), current API label, and empty columns for you to fill:
  corrected_label : 0 = GDS, 1 = TPS (or leave blank if correct)
  notes             : optional short comment

Run from repo root:
  python tps_gds_classification/scripts/export_manual_review_template.py
  python tps_gds_classification/scripts/export_manual_review_template.py --sample 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JSON = _PROJECT_ROOT / "data" / "tps_gds_dataset.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=_DEFAULT_JSON)
    p.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "data" / "manual_label_review.csv",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, randomly sample this many rows (stratified by label when possible).",
    )
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    with args.data.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records") or []

    if args.sample is not None and args.sample < len(records):
        import pandas as pd

        df = pd.DataFrame(records)
        # Stratified sample: ~half TPS / half GDS when possible
        tps = df[df["label"] == 1]
        gds = df[df["label"] == 0]
        half = max(1, args.sample // 2)
        n_tps = min(len(tps), half)
        n_gds = min(len(gds), args.sample - n_tps)
        if n_tps + n_gds < args.sample:
            n_gds = min(len(gds), args.sample - n_tps)
        parts = []
        if n_tps:
            parts.append(tps.sample(n=n_tps, random_state=args.random_state))
        if n_gds:
            parts.append(gds.sample(n=n_gds, random_state=args.random_state + 1))
        df = pd.concat(parts, ignore_index=True)
        records = df.to_dict("records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reddit_id",
        "url",
        "subreddit",
        "label_current",
        "label_name_hint",
        "title",
        "corrected_label",
        "notes",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in records:
            lab = r.get("label")
            w.writerow(
                {
                    "reddit_id": r.get("reddit_id", ""),
                    "url": r.get("url", ""),
                    "subreddit": r.get("subreddit", ""),
                    "label_current": lab,
                    "label_name_hint": "TPS" if lab == 1 else "GDS",
                    "title": (r.get("title_raw") or "").replace("\n", " ")[:500],
                    "corrected_label": "",
                    "notes": "",
                }
            )

    print(f"Wrote {len(records)} rows to {args.out}")
    print("Fill in: corrected_label (0 or 1) and optional notes. Leave corrected_label empty if the current label is OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
