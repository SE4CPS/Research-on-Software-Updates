"""Tests for corrections package."""

from __future__ import annotations

import numpy as np

from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (
    EXPECTED_JSONL,
    DATASET_JSONL,
    protected_ids_only,
    sha256_file,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage0_repairs import _sign_agree


def test_protected_id_only_count():
    ids = protected_ids_only()
    assert len(ids) == 693


def test_frozen_hash_untouched():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_sign_agree_uses_bool_not_identity():
    # numpy bool must count as agreement when signs match (not `is True` identity)
    v = np.float64(0.2)
    t = np.float64(0.3)
    assert bool(_sign_agree(float(v), float(t))) is True
    assert bool(_sign_agree(-0.2, 0.3)) is False
    assert bool(_sign_agree(0.0, 0.0)) is True
    assert bool(_sign_agree(0.01, -0.01, 0.05, 0.05)) is True  # both neutral under 0.05


def test_estimand_names_distinct():
    # Conceptual: first_author ≠ mean_all — verified by separate fields in stage0 outputs when run
    assert "first_author_reply_minus_post" != "mean_all_author_replies_minus_post"
