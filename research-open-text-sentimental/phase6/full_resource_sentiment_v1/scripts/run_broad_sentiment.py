"""Run frozen Workstream G scoring. No live API. Does not mark G validated."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from full_resource_sentiment_v1.broad_sentiment import run_broad_sentiment  # noqa: E402
from full_resource_sentiment_v1.paths import (  # noqa: E402
    BROAD_CAPTURE_DIR,
    BROAD_COMPACT_INDEX,
    BROAD_SENTIMENT_RUN_DIR,
    LOCAL_COMPACT_RECORDS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Broad Reddit-URL sentiment scoring (Workstream G)")
    parser.add_argument("--capture-dir", type=Path, default=BROAD_CAPTURE_DIR)
    parser.add_argument("--compact-index", type=Path, default=BROAD_COMPACT_INDEX)
    parser.add_argument("--local-records", type=Path, default=LOCAL_COMPACT_RECORDS)
    parser.add_argument("--run-dir", type=Path, default=BROAD_SENTIMENT_RUN_DIR)
    parser.add_argument("--skip-denom-gate", action="store_true", help="For tests only; do not use on the frozen run")
    args = parser.parse_args()
    manifest = run_broad_sentiment(
        capture_dir=args.capture_dir,
        compact_path=args.compact_index,
        local_path=args.local_records,
        run_dir=args.run_dir,
        validate_frozen_denominators=not args.skip_denom_gate,
    )
    print(
        json.dumps(
            {
                "status": manifest.get("status"),
                "n_post_scores": manifest.get("n_post_scores"),
                "n_comment_scores": manifest.get("n_comment_scores"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "run_dir": str(args.run_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
