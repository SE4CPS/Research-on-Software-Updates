"""Documented normalized schema. API predictions never fill human-label fields."""

from __future__ import annotations

from typing import Any

NORMALIZED_SCHEMA: dict[str, Any] = {
    "schema_id": "full_resource_sentiment_v1.normalized_record",
    "version": "1",
    "join_key": "nonempty exact redditId/reddit_id/post_id; never title",
    "public_export_policy": (
        "Public result files store IDs, hashes, flags, and metrics. "
        "They do not store usernames or full title/body/comment text. "
        "Optional local-only dumps may live under local_normalized/."
    ),
    "fields": {
        "reddit_id": "str; canonical nonempty join key",
        "source_memberships": "list of source keys that contain this ID",
        "duplicate_id_occurrences": "per-source occurrence counts; duplicates are flagged, not silently dropped",
        "snapshots": {
            "description": "One object per source snapshot; incompatible texts are kept side by side",
            "per_snapshot": {
                "source_key": "str",
                "source_file": "absolute path of the read-only input",
                "capture_time": "fetched_at_utc, pagination header date, or null",
                "analysis_priority": (
                    "historical reproduction uses March raw; master-source conversations "
                    "use October 5 saved pages; live responses are a new snapshot"
                ),
                "title_present": "bool",
                "title_sha256": "hash of original title string, including empty",
                "body_field": "author_description | post_body | selftext | description | absent",
                "body_present": "bool",
                "body_sha256": "hash of original body string",
                "comments_status": "absent | present_empty | present_nonempty | malformed",
                "n_stored_comments": "int",
                "n_nonempty_comments": "int",
                "num_comments_field": "raw num_comments or null; completeness signal only",
            },
        },
        "labels": {
            "historical_corrected_label": "0 GDS / 1 TPS from updated_labeled_dataset_unique.csv; null otherwise",
            "phase2_about_software": "master/development human decision; never an API route",
            "phase2_update_related": "master/development human decision",
            "phase2_discussion_type": "master/development human decision",
            "phase2_review_status": "includes manu_assisted_uncertain for five master IDs",
            "s2_in_scope": "development field; not read from protected tests for scoring",
            "s2_target": "development field; protected outcomes are not copied into scoring",
            "api_candidate_route": "tps | gds | null; candidate membership only",
            "api_metadata_predicted": "copied from metadata.predicted; not a human label",
            "api_metadata_labeled": "copied from metadata.labeled; may be null",
            "split_reserved": "ID is in reserved_test.jsonl; overlap audit only",
            "split_fresh": "ID is in all_500_reviewed.jsonl; overlap audit only",
        },
        "comments": {
            "original_index": "position in the stored array",
            "chronological_index": "valid timestamp order; original_index tie-break",
            "body_nonempty": "bool",
            "timestamp_valid": "bool; invalid timestamps are flagged, not guessed as elapsed time",
            "is_author": "bool",
            "author_role_basis": "is_submitter | name_match | not_author | blank_names_not_matched",
            "username_policy": "usernames are not exported in public results",
        },
        "missingness": [
            "comments key absent",
            "comments present empty array",
            "comment with empty body",
            "no observed author reply",
            "no observed community reply",
            "insufficient role-specific observations for a shift (n < 4 nonempty role comments)",
            "helper returned 0.0 because a role has no comments — not observed neutrality",
        ],
        "snapshot_conflicts": "title/body SHA-256 mismatches across snapshots for the same reddit_id",
        "analysis_snapshot_choice": (
            "historical_reproduction uses March GDS then TPS; "
            "master_source_conversations uses October 5 saved pages; "
            "engagement_selected_cohort uses the 324 reply-selected source and is not representative"
        ),
    },
}

RECORD_REQUIRED_KEYS = (
    "reddit_id",
    "source_memberships",
    "duplicate_id_occurrences",
    "snapshots",
    "snapshot_conflicts",
    "labels",
    "comment_availability_by_snapshot",
    "protected_outcomes_copied",
    "analysis_snapshot_choice",
)

LABEL_FIELD_NAMES = tuple(NORMALIZED_SCHEMA["fields"]["labels"].keys())

SNAPSHOT_REQUIRED_KEYS = tuple(NORMALIZED_SCHEMA["fields"]["snapshots"]["per_snapshot"].keys())

ANALYSIS_SNAPSHOT_RULES = {
    "historical_reproduction": ["historical_raw_gds", "historical_raw_tps"],
    "master_source_conversations": ["october5_api_pages"],
    "engagement_selected_cohort": ["engagement_cohort_324"],
}


def schema_document() -> dict[str, Any]:
    return NORMALIZED_SCHEMA
