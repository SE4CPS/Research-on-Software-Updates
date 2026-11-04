"""Shared IO and hashing helpers for master research 2026-11-03."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from full_resource_sentiment_v1.paths import WORKSPACE

OUT_ROOT = WORKSPACE / "results/master_research_2026-11-03"
RESTRICTED_ROOT = WORKSPACE / "restricted/master_research_2026-11-03"
DATASET_JSONL = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl"
DATASET_CSV = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.csv"
POST_SCORES = WORKSPACE / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"
THREAD_METRICS = WORKSPACE / "results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl"
COMMENT_SCORES = WORKSPACE / "results/broad_sentiment_2026-10-22-main/comment_scores.jsonl"
STAGE1_DIR = WORKSPACE / "results/tps_gds_sentiment_v2_2026-11-03/stage1_corrected_vader"
CAPTURE_DIR = WORKSPACE / "restricted/broad_release_train/2026-10-22-main"
PILOT_IDS = WORKSPACE / "planning/dataset_expansion_2026-11-03/restricted/PILOT_SELECTED_IDS.csv"
PILOT_STRATA = WORKSPACE / "planning/dataset_expansion_2026-11-03/PILOT_STRATA.csv"
V3_PREDICTIONS = WORKSPACE / "restricted/ai_relevance_exploratory_full_2026-10-29/full/predictions.jsonl"
EXPECTED_JSONL = "99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd"
EXPECTED_CSV = "39a500f2332c17bc041000db7a66e7bc360b14cd6ecde70ef2a7720288867124"
FROZEN_DEF = "frozen_broad_release_train_package_scores"


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESTRICTED_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(RESTRICTED_ROOT, 0o700)
    except OSError:
        pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
