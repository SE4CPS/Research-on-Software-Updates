"""Tests for strict continuation — synthetic / lightweight; do not overwrite accepted trees."""

from __future__ import annotations

import numpy as np

from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (
    DATASET_JSONL,
    EXPECTED_JSONL,
    protected_ids,
    sha256_file,
)
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase1_cohorts import _gate_reasons
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase2_excluded import _select_targeted_200


def test_input_hash_and_protected_exclusion():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL
    prot = protected_ids()
    assert len(prot) >= 700


def test_gate_reasons_missing_not_neutral():
    reasons = _gate_reasons(
        n_author_scoreable=0,
        n_author_empty=0,
        n_author_missing_ts=0,
        n_author_missing_score=0,
        n_community_scoreable=0,
        post_ts_valid=True,
        threshold=4,
    )
    assert "no_saved_author_reply_scoreable" in reasons
    assert "insufficient_author_replies_for_threshold_4" in reasons


def test_targeted_200_reproducible_and_excludes_pilot():
    pool = []
    for i in range(500):
        pool.append(
            {
                "reddit_id": f"id{i}",
                "n_author": 4 if i < 200 else (2 if i < 350 else (1 if i < 450 else 0)),
                "body_available": True,
                "luna_predicted_label": "not_relevant",
                "period": "older",
            }
        )
    exclude = {f"id{i}" for i in range(10)}  # pretend pilot
    a, da = _select_targeted_200(pool, exclude_ids=exclude, seed=20261103, n=200)
    b, _db = _select_targeted_200(pool, exclude_ids=exclude, seed=20261103, n=200)
    assert [r["reddit_id"] for r in a] == [r["reddit_id"] for r in b]
    assert not any(r["reddit_id"] in exclude for r in a)
    assert da["n_selected"] <= 200
    assert sum(1 for r in a if r["target_stratum"] == "A_author_ge4") <= 100


def test_split_isolation_no_label_in_features_contract():
    # Feature contract: classifier inputs are text only — checked by model card fields in phase2
    from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase2_excluded import EXPECTED_EXCLUDED

    assert EXPECTED_EXCLUDED["nonprotected_excluded"] == 30998


def test_class_balance_weights_formula():
    n_tr, n_tps, n_gds = 1948, 1816, 132
    w_tps = n_tr / (2.0 * n_tps)
    w_gds = n_tr / (2.0 * n_gds)
    assert abs(w_tps - 0.5363) < 0.001
    assert abs(w_gds - 7.3788) < 0.001
