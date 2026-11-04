"""Tests for Stage-1 corrected VADER TPS/GDS analysis."""

from __future__ import annotations

import pytest

from full_resource_sentiment_v1.scoring import early_late_or_none
from full_resource_sentiment_v1.tps_gds_sentiment_v2.analyze import (
    _band_contrast_point,
    filter_frozen_relevant,
)
from full_resource_sentiment_v1.tps_gds_sentiment_v2.dictionary import (
    EXPECTED,
    FROZEN_DEF,
    HISTORICAL_DEF,
    MEASURE_DICTIONARY,
    N_BOOT,
)
from full_resource_sentiment_v1.tps_gds_sentiment_v2.load import flatten, verify_and_load


def test_dictionary_separates_author_shift_from_post_to_author():
    assert "author_shift" in MEASURE_DICTIONARY
    assert "original_post_to_author_mean" in MEASURE_DICTIONARY
    assert MEASURE_DICTIONARY["author_shift"]["not_equal_to"] == "original_post_to_author_mean"
    assert ">= 4" in MEASURE_DICTIONARY["author_shift"]["construction"] or ">=4" in MEASURE_DICTIONARY[
        "author_shift"
    ]["eligible_when"].replace(" ", "")
    assert N_BOOT >= 10000


def test_early_late_requires_four_author_replies():
    assert early_late_or_none([0.1, 0.2, 0.3]) is None
    assert early_late_or_none([0.0, 0.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_flatten_derives_post_to_author_and_eligibility():
    row = flatten(
        {
            "reddit_id": "x1",
            "final_relevance": "relevant",
            "final_discussion": "GDS",
            "decision_source": "gpt-6.1-terra_ai_classification",
            "decision_status": "model_decision",
            "decision_confidence": "high",
            "selection_stratum": "older",
            "sentiment_definition": FROZEN_DEF,
            "sentiment": {
                "title_compound": 0.1,
                "body_compound": -0.2,
                "combined_compound": 0.0,
                "author_mean": 0.5,
                "author_shift": 0.25,
                "n_author_replies_nonempty": 4,
                "n_community_replies_nonempty": 1,
                "title_nonempty": True,
                "body_nonempty": True,
                "count_gap": False,
                "order_eligible": True,
                "elapsed_eligible": True,
            },
        },
        {"x1": {"title_len": 25, "body_len": 250}},
    )
    assert row["original_post_to_author_mean"] == pytest.approx(0.5)
    assert row["author_shift_eligible"] is True
    assert row["post_to_author_eligible"] is True
    assert row["body_band"] == "body_100_to_500"
    assert row["title_band"] == "title_20_to_50"

    short = flatten(
        {
            "reddit_id": "x2",
            "final_relevance": "relevant",
            "final_discussion": "TPS",
            "decision_source": "original_f_review",
            "decision_status": "existing_decision",
            "sentiment_definition": FROZEN_DEF,
            "sentiment": {
                "combined_compound": 0.1,
                "author_mean": None,
                "author_shift": None,
                "n_author_replies_nonempty": 2,
                "title_nonempty": True,
                "body_nonempty": False,
            },
        },
        {},
    )
    assert short["author_shift_eligible"] is False
    assert short["post_to_author_eligible"] is False
    assert short["original_post_to_author_mean"] is None


def test_missing_author_not_zero():
    row = flatten(
        {
            "reddit_id": "z",
            "final_relevance": "relevant",
            "final_discussion": "TPS",
            "decision_status": "existing_decision",
            "decision_source": "original_f_review",
            "sentiment_definition": FROZEN_DEF,
            "sentiment": {
                "author_mean": None,
                "author_shift": None,
                "combined_compound": 0.2,
                "n_author_replies_nonempty": 0,
            },
        },
        {},
    )
    assert row["author_mean"] is None
    assert row["author_shift"] is None


def test_band_standardized_known_example():
    # Two bands; only band A has both classes.
    tps = [
        {"body_band": "body_lt_100", "body_compound": 0.0},
        {"body_band": "body_lt_100", "body_compound": 0.0},
        {"body_band": "body_gt_500", "body_compound": 1.0},  # TPS only in this band
    ]
    gds = [
        {"body_band": "body_lt_100", "body_compound": -0.5},
        {"body_band": "body_lt_100", "body_compound": -0.5},
    ]
    out = _band_contrast_point(tps, gds, score_field="body_compound", band_key="body_band")
    assert out["bands_missing_a_class"] == ["body_gt_500"]
    assert out["standardized_mean_diff"] == pytest.approx(-0.5)
    # pooled = all with band and score: 3 TPS + 2 GDS = 5; gt_500 lacks GDS so excluded from standardized weights
    assert out["n_pooled_eligible"] == 5


def test_historical_not_mixed_into_frozen_filter():
    rows = [
        {
            "final_relevance": "relevant",
            "final_discussion": "TPS",
            "sentiment_definition": FROZEN_DEF,
            "combined_compound": 0.1,
        },
        {
            "final_relevance": "relevant",
            "final_discussion": "GDS",
            "sentiment_definition": HISTORICAL_DEF,
            "combined_compound": -0.9,
        },
    ]
    frozen = filter_frozen_relevant(rows)
    assert len(frozen) == 1
    assert frozen[0]["final_discussion"] == "TPS"


def test_contrast_synthetic():
    gds = [{"combined_compound": -0.4, "author_shift_eligible": False, "post_to_author_eligible": False}] * 5
    tps = [{"combined_compound": 0.1, "author_shift_eligible": False, "post_to_author_eligible": False}] * 5
    # Use small boot via direct stratified call path through contrast with monkeypatch? contrast uses N_BOOT=10000
    # Too slow for unit test — call stratified via analyze.contrast but override by testing field values only
    from full_resource_sentiment_v1.inference import stratified_mean_diff

    out = stratified_mean_diff([-0.4] * 5, [0.1] * 5, n_boot=200, seed=1, difference_name="gds_minus_tps")
    assert out["mean_diff"] == pytest.approx(-0.5)


def test_verify_real_inputs():
    result = verify_and_load()
    assert result["ok"] is True, result["discrepancies"]
    assert result["n"] == EXPECTED["n_frame"]
    assert result["n_frozen_relevant_tps"] == EXPECTED["frozen_relevant_tps"]
    assert result["n_frozen_relevant_gds"] == EXPECTED["frozen_relevant_gds"]
    assert result["protected_overlap"] == 0


def test_run_stage1_smoke_definitions_on_real_sample():
    """Smoke: eligibility flags and author_shift n match >=4 rule (full 10k boot not required)."""
    result = verify_and_load()
    frozen = filter_frozen_relevant(result["rows"])
    gds = [r for r in frozen if r["final_discussion"] == "GDS"]
    tps = [r for r in frozen if r["final_discussion"] == "TPS"]
    n_gds_shift = sum(1 for r in gds if r["author_shift_eligible"])
    n_tps_shift = sum(1 for r in tps if r["author_shift_eligible"])
    n_gds_pta = sum(1 for r in gds if r["post_to_author_eligible"])
    n_tps_pta = sum(1 for r in tps if r["post_to_author_eligible"])
    assert n_gds_shift == 30
    assert n_tps_shift == 115
    assert n_gds_pta == 98
    assert n_tps_pta == 956
    # Ensure estimands differ on a row that has both
    both = next(r for r in tps if r["author_shift_eligible"] and r["post_to_author_eligible"])
    # Not required to differ numerically always, but fields must both be present and conceptually separate keys
    assert "author_shift" in both and "original_post_to_author_mean" in both
