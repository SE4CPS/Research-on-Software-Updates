"""Tests for Codex-required evidence corrections."""

from __future__ import annotations

from full_resource_sentiment_v1.pre_manuscript_evidence_corrections_2026_11_03.common import (
    ANALYTICAL_UNION,
    BROAD_CAPTURE,
    CAPTURE_LINKED_ORIGINAL,
    DATASET_JSONL,
    EXPECTED_JSONL,
    HISTORICAL_IN_ORIGINAL,
    ORIGINAL_SELECTED,
    PRIOR_RECON,
    PROTECTED_IDS_N,
    SCREENED_EXCLUDED,
    sha256_file,
)
from full_resource_sentiment_v1.pre_manuscript_evidence_corrections_2026_11_03.build import (
    verify_frozen_provenance,
)


def test_dataset_hash():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_flow_arithmetic():
    assert BROAD_CAPTURE == SCREENED_EXCLUDED + CAPTURE_LINKED_ORIGINAL + PROTECTED_IDS_N
    assert ORIGINAL_SELECTED == CAPTURE_LINKED_ORIGINAL + HISTORICAL_IN_ORIGINAL
    assert ANALYTICAL_UNION == ORIGINAL_SELECTED + 12053


def test_provenance_matches_codex():
    report = verify_frozen_provenance()
    assert report["match_expected"] is True


def test_prior_reconciliation_still_sealed():
    assert (PRIOR_RECON / "OUTPUT_HASHES_FINAL.json").exists()
