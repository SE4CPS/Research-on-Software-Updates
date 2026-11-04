"""Tests for pre-manuscript evidence reconciliation."""

from __future__ import annotations

from full_resource_sentiment_v1.pre_manuscript_evidence_2026_11_03.common import (
    AUDIT_OUT,
    DATASET_JSONL,
    EXPECTED_JSONL,
    LENGTH_OUT,
    SENTI4SD_OUT,
    sha256_file,
)


def test_dataset_hash():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_sealed_trees_present():
    assert (AUDIT_OUT / "OUTPUT_HASHES_FINAL.json").exists()
    assert (LENGTH_OUT / "OUTPUT_HASHES_FINAL.json").exists()
    assert (SENTI4SD_OUT / "OUTPUT_HASHES_FINAL.json").exists()


def test_ledger_builder_imports():
    from full_resource_sentiment_v1.pre_manuscript_evidence_2026_11_03.build import build_evidence_ledger

    rows = build_evidence_ledger()
    assert any(r["evidence_id"] == "RQ1-AI-VADER" for r in rows)
    assert any(str(r["evidence_id"]).startswith("SUPERSEDED") for r in rows)
    assert any("Senti4SD" in str(r["instrument"]) and "REPLY-ELIGIBLE" in str(r["adjustment_or_sensitivity"]) for r in rows)
