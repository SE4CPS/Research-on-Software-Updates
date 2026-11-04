"""Tests for eligibility rules, not sentiment outcomes."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_local_eligibility.py"
spec = importlib.util.spec_from_file_location("run_local_eligibility", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def record(reddit_id="x", times=("2026-10-01T00:00:00+00:00", "2026-10-01T01:00:00+00:00")):
    comments = [
        {"body_nonempty": True, "timestamp_valid": True, "timestamp_parsed_utc": t,
         "is_author": i == 0, "author_role_basis": "is_submitter" if i == 0 else "not_author"}
        for i, t in enumerate(times)
    ]
    snap = {"source_key": "october5_api_pages", "occurrence_index": 0,
            "title_present": True, "body_present": True, "comments_status": "present_nonempty",
            "n_stored_comments": len(comments), "n_nonempty_comments": len(comments),
            "n_author_replies_nonempty": 1, "n_community_replies_nonempty": max(0, len(comments)-1),
            "n_valid_timestamps_nonempty": len(comments), "n_invalid_or_missing_timestamps": 0,
            "num_comments_field": len(comments), "comments": comments}
    return {"reddit_id": reddit_id, "source_memberships": ["october5_api_pages", "phase2_master"],
            "snapshots": [snap], "protected_outcomes_copied": False}


def test_distinct_times_required_for_elapsed():
    r = record(times=("2026-10-01T00:00:00+00:00",) * 2)
    c = audit.summarize([r])["views"]["phase2_master_3362"]["counts"]
    assert c["order_eligible_comment_time"] == 1
    assert c.get("elapsed_between_comments_eligible", 0) == 0


def test_exact_id_once_and_protected_outcome_rejected():
    r = record()
    with pytest.raises(ValueError, match="duplicate"):
        audit.summarize([r, r])
    r["protected_outcomes_copied"] = True
    with pytest.raises(ValueError, match="protected"):
        audit.summarize([r])


def test_missing_author_is_not_neutral_or_author_present():
    r = record()
    r["snapshots"][0]["n_author_replies_nonempty"] = 0
    c = audit.summarize([r])["views"]["phase2_master_3362"]["counts"]
    assert c.get("author_replies_ge1", 0) == 0
    assert "author_mean" not in c
