"""Workstream I contract tests. Do not modify G outputs."""

from __future__ import annotations

import json

from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.inference import bootstrap_mean_median_ci
from full_resource_sentiment_v1.paths import BROAD_SENTIMENT_RUN_DIR, LOCAL_COMPACT_RECORDS
from full_resource_sentiment_v1.robustness_i_common import (
    EXPECTED_N,
    I_SEED,
    assert_g_guard,
    assign_cut_bin,
    first_author_comment,
    first_k_word_text,
    hash_g_outputs,
    load_jsonl_by_id,
    original_post_text,
    pooled_character_bins,
    ratio_bin_ia,
    reading_aid,
    recompute_g_points,
    timed_comments,
    verify_f_membership,
    word_count,
)
from full_resource_sentiment_v1.robustness_i_length import bin_support_ok
from full_resource_sentiment_v1.scoring import score_observed_text

GENUINE_ZERO_TEXT = "The table is made of wood."


def test_word_prefix_uses_split_tokens_and_package():
    text = "Alpha, beta gamma delta"
    assert word_count(text) == 4
    assert first_k_word_text(text, 3) == "Alpha, beta gamma"
    prefix = score_observed_text(first_k_word_text(text, 3))
    direct = score_observed_text("Alpha, beta gamma")
    assert prefix is not None and direct is not None
    assert prefix["compound"] == direct["compound"]
    full = score_observed_text(text)
    assert full is not None
    assert original_post_text("Hello", "world") == "Hello world"
    assert original_post_text("Hello", "") == "Hello"


def test_missing_text_is_null_genuine_zero_stays_zero():
    assert score_observed_text("") is None
    assert score_observed_text("   ") is None
    scored = score_observed_text(GENUINE_ZERO_TEXT)
    assert scored is not None
    assert scored["compound"] == 0.0


def test_first_author_reply_uses_saved_chronological_order():
    comments = [
        {"is_author": True, "chronological_index": 2, "original_index": 0, "compound": 0.1, "timestamp_valid": True, "timestamp_epoch_seconds": 30},
        {"is_author": False, "chronological_index": 0, "original_index": 1, "compound": -0.2, "timestamp_valid": True, "timestamp_epoch_seconds": 10},
        {"is_author": True, "chronological_index": 1, "original_index": 2, "compound": 0.3, "timestamp_valid": True, "timestamp_epoch_seconds": 20},
    ]
    first = first_author_comment(comments)
    assert first["original_index"] == 2
    timed = timed_comments(comments)
    assert [row["original_index"] for row in timed] == [1, 2, 0]


def test_ratio_and_length_bin_support_rules():
    assert ratio_bin_ia(0.5) == "le_1"
    assert ratio_bin_ia(1.0) == "le_1"
    assert ratio_bin_ia(2.0) == "(1,2]"
    assert ratio_bin_ia(5.1) == "gt_5"
    cuts = pooled_character_bins([10, 10, 10, 50, 50, 50, 90, 90, 90, 90], n_bins=5)
    assert cuts == sorted(cuts)
    assert len(cuts) <= 6
    assert assign_cut_bin(cuts[0], cuts) == 0
    assert assign_cut_bin(cuts[-1], cuts) == len(cuts) - 2
    assert bin_support_ok(30, 10) is True
    assert bin_support_ok(29, 10) is False
    assert bin_support_ok(30, 9) is False
    aid = reading_aid(
        reference=-0.29,
        diagnostic=-0.10,
        strata={"count_gap": {"n": 500, "mean": -0.2}, "no_gap": {"n": 500, "mean": -0.3}},
        estimand_note="x",
    )
    assert aid["materially_attenuated"] is True
    assert aid["directionally_stable"] is True


def test_bootstrap_seed_is_workstream_i_seed():
    values = [0.1, -0.2, 0.0, 0.4]
    first = bootstrap_mean_median_ci(values, n_boot=20, seed=I_SEED)
    second = bootstrap_mean_median_ci(values, n_boot=20, seed=I_SEED)
    other = bootstrap_mean_median_ci(values, n_boot=20, seed=20261022)
    assert first["mean_ci95"] == second["mean_ci95"]
    assert first["seed"] == 20261024
    assert other["mean_ci95"] != first["mean_ci95"]


def test_g_hashes_points_f_labels_and_count_gap_retention():
    hashes = hash_g_outputs(BROAD_SENTIMENT_RUN_DIR)
    assert hashes["post_scores.jsonl"] == sha256_file(BROAD_SENTIMENT_RUN_DIR / "post_scores.jsonl")
    posts = load_jsonl_by_id(BROAD_SENTIMENT_RUN_DIR / "post_scores.jsonl")
    threads = load_jsonl_by_id(BROAD_SENTIMENT_RUN_DIR / "thread_metrics.jsonl")
    points = recompute_g_points(posts, threads)
    assert_g_guard(points)
    f_info = verify_f_membership(threads, LOCAL_COMPACT_RECORDS)
    assert f_info["n"] == 1864
    assert f_info["protected_in_f"] == 0
    n_gap = sum(1 for row in posts.values() if row.get("count_gap"))
    assert n_gap == EXPECTED_N["count_gap"]
    a_gap = sum(
        1
        for row in posts.values()
        if row.get("count_gap") and row.get("title_compound") is not None and row.get("body_compound") is not None
    )
    assert a_gap > 0
    hash_g_outputs(BROAD_SENTIMENT_RUN_DIR)


def test_timestamp_coverage_can_be_computed_from_saved_comments():
    n_valid = 0
    n_all = 0
    with (BROAD_SENTIMENT_RUN_DIR / "comment_scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            n_all += 1
            n_valid += bool(row.get("timestamp_valid"))
    assert n_all == EXPECTED_N["comments"]
    coverage = n_valid / n_all
    assert 0.0 <= coverage <= 1.0
