"""Tests for TPS/GDS sentiment analysis eligibility, joins, and contrasts."""

from __future__ import annotations

import pytest

from full_resource_sentiment_v1.tps_gds_sentiment_analysis.analyze import (
    contrast_gds_minus_tps,
    filter_rows,
    reply_count_bucket,
    run_primary_analysis,
)
from full_resource_sentiment_v1.tps_gds_sentiment_analysis.dictionary import (
    EXPECTED_COUNTS,
    FROZEN_DEF,
    HISTORICAL_DEF,
    MEASURE_DICTIONARY,
)
from full_resource_sentiment_v1.tps_gds_sentiment_analysis.load import flatten_row, verify_and_load


def test_measure_dictionary_covers_required_fields():
    required = {
        "title_compound",
        "body_compound",
        "combined_compound",
        "author_mean",
        "community_mean",
        "first_reply_compound",
        "last_minus_first",
        "order_slope_normalized",
        "elapsed_hours",
        "body_minus_title",
    }
    assert required <= set(MEASURE_DICTIONARY)


def test_synthetic_missingness_not_zero():
    row = flatten_row(
        {
            "reddit_id": "abc",
            "final_relevance": "relevant",
            "final_discussion": "TPS",
            "decision_source": "gpt-6.1-terra_ai_classification",
            "decision_status": "model_decision",
            "decision_confidence": "high",
            "selection_stratum": "older",
            "sentiment_definition": FROZEN_DEF,
            "sentiment": {
                "title_compound": 0.1,
                "body_compound": None,
                "combined_compound": 0.1,
                "author_mean": None,
                "community_mean": None,
                "title_nonempty": True,
                "body_nonempty": False,
                "n_author_replies_nonempty": 0,
                "n_community_replies_nonempty": 0,
                "count_gap": False,
            },
        },
        {},
    )
    assert row["body_compound"] is None
    assert row["author_mean"] is None
    assert row["body_minus_title"] is None


def test_synthetic_contrast_known_values():
    tps = [{"body_compound": 0.0, "order_eligible": True} for _ in range(10)]
    gds = [{"body_compound": -0.5, "order_eligible": True} for _ in range(10)]
    out = contrast_gds_minus_tps(tps, gds, fields=("body_compound",))
    assert out["body_compound"]["mean_diff"] == pytest.approx(-0.5)
    assert out["body_compound"]["n_a"] == 10
    assert out["body_compound"]["n_b"] == 10


def test_reply_buckets():
    assert reply_count_bucket({"reply_role_total": 0}) == "0"
    assert reply_count_bucket({"reply_role_total": 1}) == "1"
    assert reply_count_bucket({"reply_role_total": 3}) == "2_3"
    assert reply_count_bucket({"reply_role_total": 8}) == "4_10"
    assert reply_count_bucket({"reply_role_total": 12}) == "11_plus"


def test_filter_keeps_definitions_separate():
    rows = [
        {
            "final_relevance": "relevant",
            "final_discussion": "TPS",
            "sentiment_definition": FROZEN_DEF,
            "combined_compound": 0.2,
        },
        {
            "final_relevance": "relevant",
            "final_discussion": "GDS",
            "sentiment_definition": HISTORICAL_DEF,
            "combined_compound": -0.2,
        },
    ]
    frozen = filter_rows(rows, sentiment_definition=FROZEN_DEF)
    hist = filter_rows(rows, sentiment_definition=HISTORICAL_DEF)
    assert len(frozen) == 1 and frozen[0]["final_discussion"] == "TPS"
    assert len(hist) == 1 and hist[0]["final_discussion"] == "GDS"


def test_verify_real_frame_counts_and_hashes():
    result = verify_and_load()
    assert result["ok"] is True, result["discrepancies"]
    assert result["n"] == EXPECTED_COUNTS["n_frame"]
    assert result["protected_overlap"] == 0
    assert result["jsonl_sha256"] == EXPECTED_COUNTS["full_dataset_jsonl_sha256"]
    assert result["csv_sha256"] == EXPECTED_COUNTS["full_dataset_csv_sha256"]
    assert result["sentiment_definition_counts"][FROZEN_DEF] == 5905
    assert result["sentiment_definition_counts"][HISTORICAL_DEF] == 520
    # Master table retains every row.
    assert len(result["rows"]) == 6425
    # Length join available for frozen package posts.
    assert result["n_with_length_join"] >= 5905


def test_primary_analysis_does_not_merge_other_into_gds():
    result = verify_and_load()
    primary = run_primary_analysis(result["rows"])
    assert primary["n_by_discussion"]["TPS"] == 3028  # frozen relevant TPS only
    assert primary["n_by_discussion"]["GDS"] == 231
    assert primary["n_by_discussion"]["Other"] == 1514
    assert "Other" in primary["groups"]
    assert primary["gds_minus_tps"]["combined_compound"]["n_a"] <= 231
    assert primary["gds_minus_tps"]["combined_compound"]["n_b"] <= 3028
