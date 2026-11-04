"""Regression tests for exact comment-ID join (no positional fallback)."""

from __future__ import annotations

from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (
    DATASET_JSONL,
    EXPECTED_JSONL,
    protected_ids_only,
    sha256_file,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.exact_join import join_score_row


def test_frozen_hash_untouched():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_protected_id_only_count():
    assert len(protected_ids_only()) == 693


def test_exact_join_two_same_role_reordered():
    """Two author comments; score original_index points to second; order-join would grab first."""
    comments = [
        {"id": "c_first", "body": "thanks fixed", "is_submitter": True, "created_utc_ts": 100},
        {"id": "c_second", "body": "still broken", "is_submitter": True, "created_utc_ts": 200},
    ]
    score = {
        "reddit_id": "abc123",
        "capture_page": 1,
        "original_index": 1,
        "role": "author",
        "is_author": True,
        "author_role_basis": "is_submitter",
        "compound": -0.5,
        "comment_key": "abc123:1",
        "chronological_index": 0,
    }
    joined = join_score_row(score, comments)
    assert joined["join_status"] == "ok"
    assert joined["comment_id"] == "c_second"
    assert joined["_body_for_scoring"] == "still broken"
    assert joined["adapter_role"] == "author"
    # Must not silently use position 0
    assert joined["comment_id"] != "c_first"


def test_is_submitter_disagrees_with_adapter_role():
    comments = [{"id": "c1", "body": "hello", "is_submitter": False}]
    score = {
        "reddit_id": "xyz",
        "capture_page": 2,
        "original_index": 0,
        "role": "author",  # adapter says author
        "is_author": True,
        "author_role_basis": "name_match",
        "compound": 0.1,
        "comment_key": "xyz:0",
    }
    joined = join_score_row(score, comments)
    assert joined["join_status"] == "ok"
    assert joined["role_flag_disagree"] is True
    assert joined["adapter_role"] == "author"  # retained
    assert joined["capture_is_submitter"] is False


def test_blank_comment_and_missing_index():
    comments = [{"id": "c0", "body": "   ", "is_submitter": True}]
    blank_score = {
        "reddit_id": "r1",
        "capture_page": 3,
        "original_index": 0,
        "role": "author",
        "compound": 0.0,
        "comment_key": "r1:0",
    }
    blank = join_score_row(blank_score, comments)
    assert blank["join_status"] == "blank_body"

    missing = join_score_row(
        {**blank_score, "original_index": 9},
        comments,
    )
    assert missing["join_status"] == "index_out_of_range"

    no_cap = join_score_row(blank_score, None)
    assert no_cap["join_status"] == "missing_capture"


def test_no_positional_fallback_on_bad_index():
    comments = [
        {"id": "a", "body": "one", "is_submitter": True},
        {"id": "b", "body": "two", "is_submitter": True},
    ]
    score = {
        "reddit_id": "r2",
        "capture_page": 4,
        "original_index": 99,
        "role": "author",
        "compound": 0.2,
        "comment_key": "r2:99",
    }
    joined = join_score_row(score, comments)
    assert joined["join_status"] == "index_out_of_range"
    assert joined["comment_id"] is None
    assert joined["_body_for_scoring"] == ""
