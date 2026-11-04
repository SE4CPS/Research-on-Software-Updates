from __future__ import annotations

import json
from pathlib import Path

from releasetrain_sentiment import score_post, trajectory_summary

from full_resource_sentiment_v1.adapter import extract_post_body, package_eligibility, to_package_post
from full_resource_sentiment_v1.comments import comments_status

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "synthetic_posts.json").read_text(encoding="utf-8"))


def test_author_description_maps_to_description_not_selftext():
    post = FIXTURES["nonempty_body"]
    body, field, present = extract_post_body(post)
    assert present is True
    assert field == "author_description"
    assert body == "Nothing compiles since 3.2.0"
    mapped = to_package_post(post)
    assert mapped["description"] == "Nothing compiles since 3.2.0"
    assert mapped["adapter_meta"]["used_fetch_to_scoreable_post"] is False
    assert mapped["adapter_meta"]["post_kept_separate_from_author_replies"] is True


def test_blank_body_is_preserved_as_empty_description():
    mapped = to_package_post(FIXTURES["empty_body_title_only"])
    assert mapped["title"] == "Update broke my build"
    assert mapped["description"] == ""
    assert mapped["adapter_meta"]["body_field"] == "author_description"


def test_absent_comments_key_does_not_become_silent_empty_without_flag():
    raw = FIXTURES["comments_key_absent"]
    assert comments_status(raw) == "absent"
    mapped = to_package_post(raw)
    assert mapped["comments"] == []
    assert mapped["adapter_meta"]["comments_status"] == "absent"


def test_empty_comments_array_status():
    raw = FIXTURES["comments_empty_array"]
    assert comments_status(raw) == "present_empty"
    mapped = to_package_post(raw)
    assert mapped["comments"] == []
    assert mapped["adapter_meta"]["comments_status"] == "present_empty"


def test_empty_comment_body_is_passed_to_package_native_but_flagged():
    mapped = to_package_post(FIXTURES["empty_comment_body"], include_empty_comment_bodies=True)
    assert len(mapped["comments"]) == 2
    assert mapped["comments"][0]["text"].strip() == ""
    assert mapped["adapter_meta"]["comment_body_nonempty"][0] is False
    skipped = to_package_post(FIXTURES["empty_comment_body"], include_empty_comment_bodies=False)
    assert len(skipped["comments"]) == 1
    assert skipped["adapter_meta"]["n_empty_comment_bodies_skipped"] == 1


def test_package_comments_are_chronological():
    mapped = to_package_post(FIXTURES["out_of_order_timestamps"])
    assert [c["text"] for c in mapped["comments"]] == ["early", "mid", "late"]
    assert mapped["adapter_meta"]["chronological_original_indices"] == [1, 2, 0]


def test_fetch_to_scoreable_post_is_not_the_adapter():
    raw = {
        "title": "T",
        "author": "",
        "author_description": "the live body",
        "description": "wrong field",
        "comments": [{"author": "", "body": "blank names", "is_submitter": False, "created_utc": "2026-01-01T00:00:00"}],
    }
    mapped = to_package_post(raw)
    assert mapped["description"] == "the live body"
    assert mapped["comments"][0]["is_author"] is False
    try:
        from releasetrain_sentiment.fetch import to_scoreable_post
    except Exception:
        return
    fetch_mapped = to_scoreable_post(raw)
    assert fetch_mapped["description"] != mapped["description"]
    assert fetch_mapped["comments"][0]["is_author"] is True


def test_score_post_keeps_original_post_separate():
    mapped = to_package_post(FIXTURES["nonempty_body"])
    scored = score_post(mapped)
    assert scored[0]["source"] == "post"
    assert scored[0]["index"] == -1
    assert scored[0]["is_author"] is True
    assert len(scored) == 1 + len(mapped["comments"])
