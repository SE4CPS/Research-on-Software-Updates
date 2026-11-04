"""Prepare the 240-post blinded relevance-audit sample. Does not label or modify G/I outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from full_resource_sentiment_v1.relevance_audit import prepare_relevance_audit_sample  # noqa: E402


def main() -> int:
    prepare_relevance_audit_sample()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
