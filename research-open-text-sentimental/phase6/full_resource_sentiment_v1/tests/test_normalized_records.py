from __future__ import annotations

from full_resource_sentiment_v1.comments import public_comment
from full_resource_sentiment_v1.inventory import load_all_sources
from full_resource_sentiment_v1.records import (
    PROTECTED_OUTCOME_FIELDS,
    audit_normalized_records,
    build_normalized_record,
    choose_analysis_snapshot,
    iter_normalized_records,
    record_matches_schema,
    union_reddit_ids,
)
from full_resource_sentiment_v1.schema import LABEL_FIELD_NAMES, RECORD_REQUIRED_KEYS, SNAPSHOT_REQUIRED_KEYS

MULTI_SOURCE_ID = "1nt4jih"
TITLE_CONFLICT_ID = "1vsvieb"
MARCH_ID = "1o0nzld"
FRESH_ID = "1n7ryh0"
RESERVED_ID = "1ka0kq8"


def test_materialized_record_matches_documented_schema(loaded_sources):
    record = build_normalized_record(loaded_sources, MULTI_SOURCE_ID)
    assert record_matches_schema(record) == []
    for key in RECORD_REQUIRED_KEYS:
        assert key in record
    for key in LABEL_FIELD_NAMES:
        assert key in record["labels"]
    assert record["snapshots"]
    for snap in record["snapshots"]:
        for key in SNAPSHOT_REQUIRED_KEYS:
            assert key in snap
        for comment in snap["comments"]:
            assert "author" not in comment
            assert "text" not in comment


def test_one_id_in_multiple_sources_keeps_separate_snapshots(loaded_sources):
    record = build_normalized_record(loaded_sources, MULTI_SOURCE_ID)
    memberships = set(record["source_memberships"])
    assert "historical_labels" in memberships
    assert "historical_raw_gds" in memberships
    assert "phase2_master" in memberships
    assert "engagement_cohort_324" in memberships
    source_keys = [snap["source_key"] for snap in record["snapshots"]]
    assert "historical_raw_gds" in source_keys
    assert "phase2_master" in source_keys
    assert "engagement_cohort_324" in source_keys
    historical = choose_analysis_snapshot(record, "historical_reproduction")
    master_conv = choose_analysis_snapshot(record, "master_source_conversations")
    cohort = choose_analysis_snapshot(record, "engagement_selected_cohort")
    assert historical is not None and historical["source_key"] == "historical_raw_gds"
    assert cohort is not None and cohort["source_key"] == "engagement_cohort_324"
    assert master_conv is None or master_conv["source_key"] == "october5_api_pages"
    assert historical["capture_time"]
    assert historical["source_file"].endswith("gds_response.json")


def test_incompatible_text_snapshot_is_flagged(loaded_sources):
    record = build_normalized_record(loaded_sources, TITLE_CONFLICT_ID)
    assert record["snapshot_conflicts"]["title_hash_conflict"] is True
    pairs = record["snapshot_conflicts"]["conflicting_snapshot_pairs"]
    assert any(pair["title_hash_conflict"] for pair in pairs)
    oct_snap = next(snap for snap in record["snapshots"] if snap["source_key"] == "october5_api_pages")
    master_snap = next(snap for snap in record["snapshots"] if snap["source_key"] == "phase2_master")
    assert oct_snap["title_sha256"] != master_snap["title_sha256"]
    assert oct_snap["body_sha256"] == master_snap["body_sha256"]


def test_october_repeated_ids_are_not_silently_overwritten(loaded_sources):
    reddit_id = "repeatid1"
    first = loaded_sources["oct5"]["by_id"]["1wyg8ze"]
    second = dict(first)
    second["title"] = (first.get("title") or "") + " SECOND PAGE COPY"
    oct5 = dict(loaded_sources["oct5"])
    oct5["ids"] = list(loaded_sources["oct5"]["ids"]) + [reddit_id]
    oct5["inventory"] = dict(loaded_sources["oct5"]["inventory"])
    oct5["inventory"]["ids"] = oct5["ids"]
    oct5["by_id"] = dict(loaded_sources["oct5"]["by_id"])
    oct5["by_id"][reddit_id] = first
    oct5["candidate_route_by_id"] = dict(loaded_sources["oct5"]["candidate_route_by_id"])
    oct5["candidate_route_by_id"][reddit_id] = "tps"
    oct5["metadata_predicted_by_id"] = dict(loaded_sources["oct5"]["metadata_predicted_by_id"])
    oct5["metadata_predicted_by_id"][reddit_id] = {"isUpdateRelated": True}
    oct5["metadata_labeled_by_id"] = dict(loaded_sources["oct5"]["metadata_labeled_by_id"])
    oct5["metadata_labeled_by_id"][reddit_id] = {"isUpdateRelated": None}
    oct5["occurrences_by_id"] = dict(loaded_sources["oct5"].get("occurrences_by_id") or {})
    oct5["occurrences_by_id"][reddit_id] = [
        {
            "page_name": "tps_response.json",
                "source_file": loaded_sources["oct5"]["pages"][0]["file"],
                "route": "tps",
                "capture_time": "first",
                "post": first,
            },
            {
                "page_name": "tps_page_2.json",
                "source_file": loaded_sources["oct5"]["pages"][1]["file"],
            "route": "tps",
            "capture_time": "second",
            "post": second,
        },
    ]
    loaded = dict(loaded_sources)
    loaded["oct5"] = oct5
    record = build_normalized_record(loaded, reddit_id)
    oct_snaps = [snap for snap in record["snapshots"] if snap["source_key"] == "october5_api_pages"]
    assert len(oct_snaps) == 2
    assert record["duplicate_id_occurrences"]["october5_api_pages"] == 2
    assert oct_snaps[0]["title_sha256"] != oct_snaps[1]["title_sha256"]
    assert record["snapshot_conflicts"]["title_hash_conflict"] is True
    chosen = choose_analysis_snapshot(record, "master_source_conversations")
    assert chosen is not None
    assert chosen["occurrence_index"] == 0
    assert chosen["title_sha256"] == oct_snaps[0]["title_sha256"]


def test_role_and_timestamp_retention_on_march_snapshot(loaded_sources, march_post):
    record = build_normalized_record(loaded_sources, MARCH_ID)
    snap = next(s for s in record["snapshots"] if s["source_key"] == "historical_raw_tps")
    assert snap["comments_status"] == "present_nonempty"
    assert snap["chronological_original_indices"] == [2, 1, 0]
    comments = snap["comments"]
    assert [c["original_index"] for c in comments] == [0, 1, 2]
    by_original = {c["original_index"]: c for c in comments}
    assert by_original[1]["is_author"] is True
    assert by_original[1]["author_role_basis"] == "is_submitter"
    assert by_original[0]["is_author"] is False
    assert by_original[2]["is_author"] is False
    assert all(c["timestamp_valid"] for c in comments)
    assert by_original[0]["timestamp_raw"] == march_post["comments"][0]["created_utc"]
    assert by_original[1]["body_nonempty"] is True
    availability = record["comment_availability_by_snapshot"]["historical_raw_tps#0"]
    assert availability["n_author_replies_nonempty"] == 1
    assert availability["n_community_replies_nonempty"] == 2


def test_protected_outcomes_are_not_read_from_fresh_or_reserved(loaded_sources):
    assert loaded_sources["reserved"].get("ids_only") is True
    assert loaded_sources["fresh"].get("ids_only") is True
    assert "by_id" not in loaded_sources["reserved"]
    assert "by_id" not in loaded_sources["fresh"]
    fresh = build_normalized_record(loaded_sources, FRESH_ID)
    reserved = build_normalized_record(loaded_sources, RESERVED_ID)
    assert fresh["labels"]["split_fresh"] is True
    assert fresh["labels"]["split_reserved"] is False
    assert reserved["labels"]["split_reserved"] is True
    assert reserved["labels"]["split_fresh"] is False
    assert "development_v2" not in fresh["source_memberships"]
    assert "development_v2" not in reserved["source_memberships"]
    assert fresh["labels"]["s2_target"] is None
    assert fresh["labels"]["s2_in_scope"] is None
    assert reserved["labels"]["s2_target"] is None
    assert reserved["labels"]["s2_in_scope"] is None
    assert fresh["protected_outcomes_copied"] is False
    loaded = dict(loaded_sources)
    loaded["fresh"] = dict(loaded_sources["fresh"])
    loaded["reserved"] = dict(loaded_sources["reserved"])
    loaded["fresh"]["by_id"] = {FRESH_ID: {field: "LEAK" for field in PROTECTED_OUTCOME_FIELDS}}
    loaded["reserved"]["by_id"] = {RESERVED_ID: {"s2_target": 1, "s2_in_scope": True}}
    still_fresh = build_normalized_record(loaded, FRESH_ID)
    still_reserved = build_normalized_record(loaded, RESERVED_ID)
    assert still_fresh["labels"]["s2_target"] is None
    assert still_reserved["labels"]["s2_target"] is None


def test_historical_overlap_keeps_both_march_snapshots(loaded_sources):
    overlap_id = loaded_sources["historical_raw"]["ids_where_gds_overwrites_tps"][0]
    record = build_normalized_record(loaded_sources, overlap_id)
    keys = [snap["source_key"] for snap in record["snapshots"]]
    assert "historical_raw_tps" in keys
    assert "historical_raw_gds" in keys
    chosen = choose_analysis_snapshot(record, "historical_reproduction")
    assert chosen is not None
    assert chosen["source_key"] == "historical_raw_gds"


def test_union_iterator_count_and_audit_protected_flags(loaded_sources):
    ids = union_reddit_ids(loaded_sources)
    assert len(ids) >= 3362 + 500 + (324 - 18)
    sample = list(iter_normalized_records(loaded_sources, reddit_ids=[MULTI_SOURCE_ID, FRESH_ID, MARCH_ID]))
    assert [row["reddit_id"] for row in sample] == sorted([MULTI_SOURCE_ID, FRESH_ID, MARCH_ID])
    audit = audit_normalized_records(sample)
    assert audit["n_records"] == 3
    assert audit["protected"]["n_split_fresh"] == 1
    assert audit["protected"]["n_fresh_with_s2_target"] == 0


def test_public_comment_helper_strips_text_and_username():
    stripped = public_comment(
        {
            "original_index": 0,
            "chronological_index": 1,
            "body_nonempty": True,
            "timestamp_valid": True,
            "is_author": False,
            "author_role_basis": "not_author",
            "text": "secret body",
            "author": "secret_user",
        }
    )
    assert "text" not in stripped
    assert "author" not in stripped
    assert stripped["original_index"] == 0
