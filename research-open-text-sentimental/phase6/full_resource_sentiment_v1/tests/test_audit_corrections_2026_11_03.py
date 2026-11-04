"""Tests for audit corrections (Cliff's δ sign, seals preserved)."""

from __future__ import annotations

from full_resource_sentiment_v1.audit_corrections_2026_11_03.common import (
    DATASET_JSONL,
    EXPECTED_JSONL,
    SEALED_CORR,
    protected_ids_only,
    sha256_file,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)


def test_frozen_hash():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_protected_count():
    assert len(protected_ids_only()) == 693


def test_sealed_correction_tree_preserved():
    assert (SEALED_CORR / "OUTPUT_HASHES_FINAL.json").exists()


def test_cliffs_delta_sign_matches_gds_minus_tps():
    """GDS scores systematically higher than TPS → Cliff's δ must be positive."""
    gds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tps = [0.0, 0.1, 0.2, 0.3, 0.4]
    out = independent_stratified_mean_diff(gds, tps, seed=1, n_boot=200)
    assert out["mean_diff"] > 0
    assert out["cliffs_delta"] > 0
    # Symmetric reverse
    out_rev = independent_stratified_mean_diff(tps, gds, seed=1, n_boot=200)
    assert out_rev["cliffs_delta"] < 0
    assert abs(out["cliffs_delta"] + out_rev["cliffs_delta"]) < 1e-9
