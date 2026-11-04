"""Tests for removal correction and dataset integrity."""

from __future__ import annotations

from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.common import (
    CONTINUATION_OUT,
    DATASET_JSONL,
    EXPECTED_JSONL,
    protected_ids_only,
    sha256_file,
)
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.removal_rule import (
    deterministic_unassessable,
    field_is_whole_removal_notice,
)


def test_frozen_hash():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_protected_count():
    assert len(protected_ids_only()) == 693


def test_continuation_seal_preserved():
    assert (CONTINUATION_OUT / "OUTPUT_HASHES_FINAL.json").exists()


def test_exact_reddit_removal_notice_unassessable():
    title = "[ Removed by Reddit ]"
    body = "[ Removed by Reddit on account of violating the [content policy](/help/contentpolicy). ]"
    assert field_is_whole_removal_notice(title)
    assert field_is_whole_removal_notice(body)
    assert deterministic_unassessable(title, body) is not None


def test_mixed_real_text_plus_notice_stays_assessable():
    title = "My app broke after update"
    body = "Something was removed from the UI but I need a workaround"
    assert deterministic_unassessable(title, body) is None
    assert not field_is_whole_removal_notice(body)


def test_ordinary_prose_about_removals_not_unassessable():
    title = "Feature removed in latest release"
    body = "They removed dark mode and I want it back after the update"
    assert deterministic_unassessable(title, body) is None


def test_classic_bracket_removed_whole_field():
    assert deterministic_unassessable("[removed]", "[deleted]") is not None
    assert deterministic_unassessable("[removed]", "still have a real body about the update") is None
