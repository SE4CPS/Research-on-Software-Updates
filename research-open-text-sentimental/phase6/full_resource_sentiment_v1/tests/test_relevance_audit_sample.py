"""Relevance-audit sampling contract tests. Do not read protected outcomes."""

from __future__ import annotations

from full_resource_sentiment_v1.paths import FRESH_500, RESERVED_TEST
from full_resource_sentiment_v1.io_utils import iter_jsonl
from full_resource_sentiment_v1.relevance_audit import (
    HANDLE_CORRECTION_CASES,
    RESTRICTED_DIR,
    SAMPLE_SEED,
    _sequence_hash,
    assert_packets_have_zero_handles,
    draw_stratified_sample,
    load_primary_frame,
    packet_is_blinded,
    redact_pii,
    unredacted_handles,
    verify_g_hashes,
    verify_i_archive,
)


def test_sample_is_deterministic_unique_and_stratified():
    october = [f"oct{i:04d}" for i in range(200)]
    older = [f"old{i:04d}" for i in range(180)]
    first = draw_stratified_sample(october, older, seed=SAMPLE_SEED, n_per_stratum=120)
    second = draw_stratified_sample(october, older, seed=SAMPLE_SEED, n_per_stratum=120)
    assert first["october_ids"] == second["october_ids"]
    assert first["older_ids"] == second["older_ids"]
    assert first["all_ids"] == second["all_ids"]
    assert first["reviewer_a_order"] == second["reviewer_a_order"]
    assert first["reviewer_b_order"] == second["reviewer_b_order"]
    assert first["reviewer_a_order"] != first["reviewer_b_order"]
    assert _sequence_hash(first["reviewer_a_order"]) != _sequence_hash(first["reviewer_b_order"])
    assert len(set(first["all_ids"])) == 240
    assert not set(first["october_ids"]) & set(first["older_ids"])
    assert set(first["october_ids"]).issubset(set(october))
    assert set(first["older_ids"]).issubset(set(older))
    other = draw_stratified_sample(october, older, seed=20261027, n_per_stratum=120)
    assert other["all_ids"] != first["all_ids"]


def test_missing_text_is_retained_not_dropped():
    packet = {
        "case_code": "C-000001",
        "title": "",
        "body": "   ",
        "relevance": "",
        "evidence_phrase": "",
        "uncertainty_note": "",
    }
    assert packet_is_blinded(packet) == []
    readable = bool(packet["title"].strip() or packet["body"].strip())
    assert readable is False


def test_blinding_rejects_ids_scores_and_preassigned_labels():
    clean = {
        "case_code": "C-000002",
        "title": "App crashed after 2.1",
        "body": "The installer failed.",
        "relevance": "",
        "evidence_phrase": "",
        "uncertainty_note": "",
    }
    assert packet_is_blinded(clean) == []
    dirty = dict(clean)
    dirty["reddit_id"] = "abc123"
    assert "reddit_id" in packet_is_blinded(dirty)
    labeled = dict(clean)
    labeled["relevance"] = "relevant"
    assert "preassigned_relevance" in packet_is_blinded(labeled)


def test_pii_redaction_and_handle_flag():
    text = "Email me at a.b@example.com see https://reddit.com/r/x and u/someone 415-555-1212"
    redacted, counts = redact_pii(text)
    assert "example.com" not in redacted
    assert "https://" not in redacted
    assert "415-555-1212" not in redacted
    assert "u/someone" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_URL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_HANDLE]" in redacted
    assert counts["email"] == 1
    assert counts["url"] == 1
    assert counts["phone"] == 1
    assert counts["handle"] == 1
    assert counts["handle_flagged_not_redacted"] == 0
    assert unredacted_handles(redacted) == []


def test_prepared_packets_have_zero_unredacted_handles():
    for name in ("reviewer_A_packet.jsonl", "reviewer_B_packet.jsonl"):
        rows = list(iter_jsonl(RESTRICTED_DIR / name))
        assert_packets_have_zero_handles(rows, source=name)
        by_code = {row["case_code"]: row for row in rows}
        for code in HANDLE_CORRECTION_CASES:
            assert "[REDACTED_HANDLE]" in by_code[code]["body"]
        leftover = unredacted_handles((RESTRICTED_DIR / name.replace(".jsonl", ".csv")).read_text(encoding="utf-8"))
        assert leftover == []


def test_protected_exclusion_cohort_counts_and_no_outcome_files():
    frame = load_primary_frame()
    assert frame["counts"]["n_primary"] == 37596
    assert frame["counts"]["n_eligible_october"] == 20337
    assert frame["counts"]["n_eligible_older"] == 16566
    assert frame["counts"]["n_protected_union"] == 693
    sample = draw_stratified_sample(frame["october_ids"], frame["older_ids"])
    selected = set(sample["all_ids"])
    assert selected.isdisjoint(set(frame["protected_union_ids"]))
    for rid in sample["october_ids"]:
        assert frame["eligible"][rid]["stratum"] == "october_2026"
    for rid in sample["older_ids"]:
        assert frame["eligible"][rid]["stratum"] == "older"
    assert not any("s2_target" in row for row in frame["eligible"].values())
    # Membership-only: this test never opens reserved/fresh outcome files.
    assert RESERVED_TEST.name == "reserved_test.jsonl"
    assert FRESH_500.name == "all_500_reviewed.jsonl"


def test_g_and_i_archive_hashes_unchanged():
    verify_g_hashes()
    verify_i_archive()
    verify_g_hashes()
    verify_i_archive()
