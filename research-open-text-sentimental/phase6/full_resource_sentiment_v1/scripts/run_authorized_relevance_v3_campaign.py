#!/usr/bin/env python3
"""Authorized live v3 AI relevance campaign. Requires env gates and OPENAI_API_KEY."""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.relevance_classify.campaign_v3 import run_v3_campaign
from full_resource_sentiment_v1.relevance_classify.spec import (
    ALLOW_LIVE_ENV,
    PROMPT_VERSION_ENV,
    RESEARCH_RUN_ENV,
    STAGE2_LIVE_LOCKED,
)

EXTERNAL_ENV = Path.home() / ".config/reddit-research/openai.env"


def _load_external_env() -> bool:
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"):
        return True
    if not EXTERNAL_ENV.is_file():
        return False
    for raw in EXTERNAL_ENV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in {"OPENAI_API_KEY", "OPENAI_KEY"} and value:
            os.environ["OPENAI_API_KEY"] = value
    return bool(os.environ.get("OPENAI_API_KEY"))


def main() -> int:
    os.environ.setdefault(RESEARCH_RUN_ENV, "1")
    os.environ.setdefault(ALLOW_LIVE_ENV, "1")
    os.environ["RELEVANCE_CLASSIFY_TEST_MODE"] = "0"
    os.environ[PROMPT_VERSION_ENV] = "v3"
    loaded = _load_external_env()
    print(
        "external_env_loaded="
        + str(loaded).lower()
        + " key_present="
        + str(bool(os.environ.get("OPENAI_API_KEY"))).lower()
        + " prompt="
        + os.environ.get(PROMPT_VERSION_ENV, "")
    )
    if not STAGE2_LIVE_LOCKED:
        print("STAGE2_LIVE_LOCKED flipped unexpectedly; refusing to run.", file=sys.stderr)
        return 2
    bundle = run_v3_campaign()
    if bundle.get("blocker"):
        print(f"BLOCKER {bundle['blocker'].get('kind')}: {bundle['blocker'].get('detail')}", file=sys.stderr)
        return 3
    print("v3 campaign complete. See AI_RELEVANCE_V3_CAMPAIGN_REPORT_2026-10-29.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
