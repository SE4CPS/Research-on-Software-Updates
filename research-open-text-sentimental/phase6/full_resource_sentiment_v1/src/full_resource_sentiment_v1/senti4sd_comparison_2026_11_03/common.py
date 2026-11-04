"""Paths and IO for Senti4SD comparison tree."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from full_resource_sentiment_v1.paths import WORKSPACE

OUT_ROOT = WORKSPACE / "results/senti4sd_comparison_2026-11-03"
RESTRICTED_ROOT = WORKSPACE / "restricted/senti4sd_comparison_2026-11-03"
LENGTH_OUT = WORKSPACE / "results/sentiment_length_sensitivity_2026-11-03"
LENGTH_RESTRICTED = WORKSPACE / "restricted/sentiment_length_sensitivity_2026-11-03"
SEALED_AUDIT = WORKSPACE / "results/audit_corrections_2026-11-03"

VENDOR = WORKSPACE / "vendor/senti4sd_2026-11-03"
PYSENTI = VENDOR / "pySenti4SD"
VENV_SENTI = VENDOR / ".venv_senti4sd"
JAVA_BIN = Path("/opt/homebrew/opt/openjdk/bin/java")

DATASET_JSONL = WORKSPACE / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl"
EXPECTED_JSONL = "99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd"


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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
