"""Paths for Codex-required evidence corrections."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from full_resource_sentiment_v1.paths import WORKSPACE

OUT_ROOT = WORKSPACE / "results/pre_manuscript_evidence_corrections_2026-11-03"
PRIOR_RECON = WORKSPACE / "results/pre_manuscript_evidence_reconciliation_2026-11-03"
AUDIT_OUT = WORKSPACE / "results/audit_corrections_2026-11-03"
LENGTH_OUT = WORKSPACE / "results/sentiment_length_sensitivity_2026-11-03"
SENTI4SD_OUT = WORKSPACE / "results/senti4sd_comparison_2026-11-03"
CORR_OUT = WORKSPACE / "results/correction_rq_cardiff_2026-11-03"
DATASET_JSONL = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl"
PROTECTED_IDS = WORKSPACE / "restricted/relevance_audit_sample_2026-10-28/excluded_protected_ids.json"
EXPECTED_JSONL = "99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd"
FROZEN_DEF = "frozen_broad_release_train_package_scores"
HIST_DEF = "historical_542_harmonized_package_comparison"

# Codex-stated flow arithmetic (verified against sealed artifacts)
BROAD_CAPTURE = 37596
SCREENED_EXCLUDED = 30998
CAPTURE_LINKED_ORIGINAL = 5905
PROTECTED_IDS_N = 693
ORIGINAL_SELECTED = 6425
HISTORICAL_IN_ORIGINAL = 520
AI_NEW_CANDIDATES = 12053
ANALYTICAL_UNION = 18478


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(OUT_ROOT, 0o755)
    except OSError:
        pass


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
