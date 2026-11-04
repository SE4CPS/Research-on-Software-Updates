"""Tests for sentiment-length sensitivity package."""

from __future__ import annotations

from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03.analysis import _prefix_pair
from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03.common import (
    DATASET_JSONL,
    EXPECTED_JSONL,
    SEALED_AUDIT,
    protected_ids_only,
    sha256_file,
)


def test_frozen_hash():
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL


def test_protected_count():
    assert len(protected_ids_only()) == 693


def test_sealed_audit_preserved():
    assert (SEALED_AUDIT / "OUTPUT_HASHES_FINAL.json").exists()


def test_equal_length_prefix_rule():
    post = "ABCDEFGHIJ"  # 10
    reply = "XYZ"  # 3
    p, r, n = _prefix_pair(post, reply)
    assert n == 3
    assert p == "ABC"
    assert r == "XYZ"
    assert len(p) == len(r) == n


def test_prefix_when_reply_longer():
    post = "AB"
    reply = "WXYZ"
    p, r, n = _prefix_pair(post, reply)
    assert n == 2
    assert p == "AB"
    assert r == "WX"
