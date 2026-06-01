#!/usr/bin/env python3
"""
Build C1′ trajectory/divergence metrics for all 542 TPS/GDS labeled posts.

Usage (from research-open-text-sentimental/):
  python scripts/build_c1_prime.py
  python scripts/build_c1_prime.py --no-trajectory-arrays  # smaller JSON

Outputs:
  data/c1_prime/c1_prime_metrics.csv
  data/c1_prime/c1_prime_dataset.json
  data/c1_prime/build_metadata.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trajectory.build import (  # noqa: E402
    DEFAULT_LABELS,
    DEFAULT_OUT_DIR,
    DEFAULT_RAW_DIR,
    build_c1_prime,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Build C1′ trajectory metrics (542 labeled posts).")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--min-author", type=int, default=1, help="Min author replies for trajectory_eligible.")
    p.add_argument("--min-community", type=int, default=1, help="Min community comments for trajectory_eligible.")
    p.add_argument(
        "--no-trajectory-arrays",
        action="store_true",
        help="Omit per-comment arrays from JSON (metrics CSV still complete).",
    )
    args = p.parse_args()

    build_c1_prime(
        labels_path=args.labels,
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        include_trajectory_arrays=not args.no_trajectory_arrays,
        min_author=args.min_author,
        min_community=args.min_community,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
