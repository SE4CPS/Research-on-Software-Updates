"""Apply the handle privacy correction to prepared packets. Does not redraw the sample."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from full_resource_sentiment_v1.relevance_audit import correct_prepared_packet_handles  # noqa: E402


def main() -> int:
    correct_prepared_packet_handles()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
