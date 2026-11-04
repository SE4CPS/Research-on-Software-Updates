from __future__ import annotations

import json
from pathlib import Path

from full_resource_sentiment_v1.comments import comments_status, normalize_comments
from full_resource_sentiment_v1.loaders import _id_inventory
from full_resource_sentiment_v1.roles import infer_author_role

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "synthetic_posts.json").read_text(encoding="utf-8"))


def test_nonempty_and_empty_post_body_status():
    nonempty = comments_status(FIXTURES["nonempty_body"])
    absent = comments_status(FIXTURES["comments_key_absent"])
    empty = comments_status(FIXTURES["comments_empty_array"])
    assert nonempty == "present_nonempty"
    assert absent == "absent"
    assert empty == "present_empty"
    assert FIXTURES["empty_body_title_only"]["author_description"] == ""
    assert FIXTURES["nonempty_body"]["author_description"].strip()


def test_empty_comment_body_is_flagged_not_scored_as_reply_text():
    norm = normalize_comments(FIXTURES["empty_comment_body"])
    assert norm["n_stored"] == 2
    assert norm["n_empty_body"] == 1
    assert norm["n_nonempty"] == 1
    assert norm["comments_original_order"][0]["body_nonempty"] is False


def test_author_by_is_submitter_even_when_names_differ():
    post = FIXTURES["author_by_is_submitter"]
    comment = post["comments"][0]
    role = infer_author_role(post["author"], comment["author"], comment["is_submitter"])
    assert role["is_author"] is True
    assert role["author_role_basis"] == "is_submitter"
    assert role["role_conflict_is_submitter_vs_name"] is True


def test_author_by_nonempty_name_match():
    post = FIXTURES["author_by_name_match"]
    comment = post["comments"][0]
    role = infer_author_role(post["author"], comment["author"], comment["is_submitter"])
    assert role["is_author"] is True
    assert role["author_role_basis"] == "name_match"


def test_blank_author_names_do_not_match():
    post = FIXTURES["blank_author_names"]
    comment = post["comments"][0]
    role = infer_author_role(post["author"], comment["author"], comment["is_submitter"])
    assert role["is_author"] is False
    assert role["author_role_basis"] == "blank_names_not_matched"
    assert role["blank_post_author"] is True
    assert role["blank_comment_author"] is True


def test_conflicting_role_fields_preserve_is_submitter_and_flag():
    post = FIXTURES["conflicting_role_fields"]
    comment = post["comments"][0]
    role = infer_author_role(post["author"], comment["author"], comment["is_submitter"])
    assert role["is_author"] is True
    assert role["role_conflict_is_submitter_vs_name"] is True


def test_out_of_order_timestamps_sort_by_parsed_time():
    norm = normalize_comments(FIXTURES["out_of_order_timestamps"])
    assert norm["chronological_original_indices"] == [1, 2, 0]
    originals = [c["original_index"] for c in norm["comments_original_order"]]
    assert originals == [0, 1, 2]


def test_equal_timestamps_use_original_index_tie_break():
    norm = normalize_comments(FIXTURES["equal_timestamps"])
    assert norm["chronological_original_indices"] == [0, 1]
    assert norm["comments_original_order"][0]["timestamp_raw"] == norm["comments_original_order"][1]["timestamp_raw"]


def test_missing_timestamp_sorts_after_valid_and_is_flagged():
    norm = normalize_comments(FIXTURES["missing_timestamp"])
    assert norm["n_invalid_or_missing_timestamps"] == 1
    assert norm["chronological_original_indices"] == [0, 1]
    missing = norm["comments_original_order"][1]
    assert missing["timestamp_valid"] is False
    assert missing["timestamp_reason"] == "missing"


def test_duplicate_reddit_ids_are_flagged():
    inv = _id_inventory(FIXTURES["duplicate_ids"])
    assert inv["n_rows"] == 3
    assert inv["n_unique_ids"] == 2
    assert inv["duplicate_ids"] == ["abc123"]
    assert inv["id_counts"]["abc123"] == 2
