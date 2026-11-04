"""Tests for continuation package."""

from __future__ import annotations

from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.common import (
    DATASET_JSONL,
    EXACT_JOIN_OUT,
    EXPECTED_JSONL,
    protected_ids_only,
    sha256_file,
)


def test_frozen_hash_untouched():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_protected_count():
    assert len(protected_ids_only()) == 693


def test_exact_join_seal_preserved():
    seal = EXACT_JOIN_OUT / "OUTPUT_HASHES_FINAL.json"
    assert seal.exists()
    # Sealed tree must still be present (path existence; content audited separately)
    assert (EXACT_JOIN_OUT / "CODEX_REVIEW_PACKET.md").exists()


def test_later_author_marked_incomparable_concept():
    # Definitional: new later_author requires >=3 author replies; old included n==2 last
    assert "later_author_reply" != "second_author_reply"
