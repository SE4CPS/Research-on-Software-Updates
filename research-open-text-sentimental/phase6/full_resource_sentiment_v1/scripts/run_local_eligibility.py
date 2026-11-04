"""Audit conversation coverage from existing compact records; never score protected outcomes.

One exact Reddit ID is counted at most once in each named view. Different source
snapshots are kept separate; this program does not merge their comment arrays.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "normalized_records_compact.jsonl"
OUTPUT = ROOT / "results" / "local_conversation_eligibility.json"

VIEWS = {
    "historical_542": "historical_labels",
    "engagement_cohort_324": "engagement_cohort_324",
    "phase2_master_3362": "phase2_master",
    "development_3151": "development_v2",
    "reddit_development_3121": "reddit_development",
    "october5_candidate_pages": "october5_api_pages",
    "local_union_excluding_fresh_only": None,
}
RAW_PRIORITY = ("october5_api_pages", "historical_raw_gds", "historical_raw_tps", "engagement_cohort_324")
TEXT_PRIORITY = ("reddit_development", "development_v2", "phase2_master", "october5_api_pages", "historical_raw_gds", "historical_raw_tps", "engagement_cohort_324")


def parsed_time(comment):
    if not comment.get("timestamp_valid"):
        return None
    try:
        return datetime.fromisoformat(comment["timestamp_parsed_utc"].replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None


def choose_snapshot(record, key):
    snapshots = record["snapshots"]
    if key == "historical_542":
        order = ("historical_raw_gds", "historical_raw_tps")
    elif key == "engagement_cohort_324":
        order = ("engagement_cohort_324",)
    elif key == "october5_candidate_pages":
        order = ("october5_api_pages",)
    else:
        order = RAW_PRIORITY
    for source in order:
        candidates = [s for s in snapshots if s["source_key"] == source]
        if candidates:
            return min(candidates, key=lambda s: s.get("occurrence_index", 0))
    return None


def choose_text(record, key, conversation_snapshot):
    if key in {"historical_542", "engagement_cohort_324", "october5_candidate_pages"}:
        return conversation_snapshot
    view_priority = {
        "phase2_master_3362": ("phase2_master",),
        "development_3151": ("development_v2",),
        "reddit_development_3121": ("reddit_development",),
    }
    for source in view_priority.get(key, TEXT_PRIORITY):
        candidates = [s for s in record["snapshots"] if s["source_key"] == source]
        if candidates:
            return min(candidates, key=lambda s: s.get("occurrence_index", 0))
    return None


def summarize(records):
    views = {key: Counter() for key in VIEWS}
    sources = {key: Counter() for key in VIEWS}
    seen = set()
    protected = Counter()
    for record in records:
        rid = record["reddit_id"]
        if rid in seen:
            raise ValueError(f"duplicate normalized ID: {rid}")
        seen.add(rid)
        if record.get("protected_outcomes_copied"):
            raise ValueError(f"protected outcome in normalized record: {rid}")
        membership = set(record["source_memberships"])
        protected["reserved_memberships"] += "reserved_test" in membership
        protected["fresh_memberships"] += "fresh_500" in membership
        for view, source_membership in VIEWS.items():
            if source_membership is not None and source_membership not in membership:
                continue
            if view == "local_union_excluding_fresh_only" and membership <= {"reserved_test", "fresh_500"}:
                continue
            c = views[view]
            c["ids"] += 1
            conv = choose_snapshot(record, view)
            txt = choose_text(record, view, conv)
            if conv:
                sources[view][conv["source_key"]] += 1
            else:
                c["no_conversation_snapshot"] += 1
            if txt:
                c["title_nonempty"] += bool(txt["title_present"])
                c["body_nonempty"] += bool(txt["body_present"])
                c["title_and_body_nonempty"] += bool(txt["title_present"] and txt["body_present"])
                c["title_or_body_nonempty"] += bool(txt["title_present"] or txt["body_present"])
            if not conv:
                continue
            status = conv["comments_status"]
            c[f"comments_{status}"] += 1
            n = conv["n_nonempty_comments"]
            a = conv["n_author_replies_nonempty"]
            b = conv["n_community_replies_nonempty"]
            c["stored_comment_turns"] += conv["n_stored_comments"]
            c["nonempty_comment_turns"] += n
            c["valid_timestamp_turns"] += conv["n_valid_timestamps_nonempty"]
            c["invalid_or_missing_timestamp_turns"] += conv["n_invalid_or_missing_timestamps"]
            c["nonempty_comments_ge1"] += n >= 1
            c["author_replies_ge1"] += a >= 1
            c["community_replies_ge1"] += b >= 1
            c["both_roles_ge1"] += a >= 1 and b >= 1
            c["author_replies_ge4"] += a >= 4
            c["community_replies_ge4"] += b >= 4
            c["nonempty_comments_ge2"] += n >= 2
            comments = [x for x in conv["comments"] if x.get("body_nonempty")]
            valid = [parsed_time(x) for x in comments]
            valid = [x for x in valid if x is not None]
            c["valid_timestamp_comments_ge2"] += len(valid) >= 2
            c["all_nonempty_comments_timestamp_valid"] += n >= 1 and len(valid) == n
            c["order_eligible_comment_time"] += len(valid) >= 2
            c["elapsed_between_comments_eligible"] += len(set(valid)) >= 2
            c["role_ambiguous_nonempty_turns"] += sum(
                x.get("author_role_basis") == "blank_names_not_matched" for x in comments
            )
            c["role_conflict_nonempty_turns"] += sum(
                bool(x.get("role_conflict_is_submitter_vs_name") or x.get("cohort_role_conflict"))
                for x in comments
            )
            reported = conv.get("num_comments_field")
            try:
                reported = int(reported)
            except (ValueError, TypeError):
                reported = None
            c["num_comments_reported"] += reported is not None
            c["reported_more_than_stored"] += reported is not None and reported > conv["n_stored_comments"]
    return {
        "input": str(INPUT),
        "total_distinct_joined_ids": len(seen),
        "definitions": {
            "unit": "exact Reddit ID, once per named view; views overlap and must not be summed",
            "text": "nonempty title/body from documented source priority; conversation snapshot may differ but snapshots never merge",
            "order": "at least two nonempty stored comments with valid timestamps, sorted by timestamp; invalid times are excluded",
            "elapsed": "at least two distinct valid comment timestamps; this is comment-to-comment time only because normalized snapshots do not retain post-creation time",
            "role": "author if is_submitter or nonblank exact author match; blank/unknown names can inflate non-author count and are separately flagged",
            "absence": "no stored author reply is missing observation, not a measured neutral sentiment",
            "protected": "reserved/fresh membership counted only; outcomes never read or scored; local union includes reserved master IDs but excludes fresh-only IDs",
        },
        "protected_memberships": dict(protected),
        "views": {key: {"counts": dict(sorted(views[key].items())), "conversation_source_counts": dict(sorted(sources[key].items()))} for key in VIEWS},
    }


def main():
    with INPUT.open(encoding="utf-8") as fh:
        result = summarize(json.loads(line) for line in fh if line.strip())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ids": result["total_distinct_joined_ids"], "views": result["views"]}, indent=2))


if __name__ == "__main__":
    main()
