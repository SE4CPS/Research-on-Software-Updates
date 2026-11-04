from __future__ import annotations

from full_resource_sentiment_v1.adapter import extract_post_body, to_package_post
from full_resource_sentiment_v1.comments import normalize_comments
from full_resource_sentiment_v1.roles import infer_author_role


def test_march_post_matches_raw_snapshot_field_by_field(march_post):
    assert march_post["redditId"] == "1o0nzld"
    assert march_post["title"] == "After update doesn't work FaceDetailer"
    body, field, present = extract_post_body(march_post)
    assert present is True
    assert field == "author_description"
    assert "selftext" not in march_post
    assert "description" not in march_post
    assert isinstance(march_post["comments"], list)
    assert len(march_post["comments"]) == 3
    assert [c["created_utc"] for c in march_post["comments"]] == [
        "2025-10-07T22:24:24",
        "2025-10-07T21:48:34",
        "2025-10-07T21:40:41",
    ]
    assert [c["is_submitter"] for c in march_post["comments"]] == [False, True, False]
    assert [bool(c["body"].strip()) for c in march_post["comments"]] == [True, True, True]
    roles = [
        infer_author_role(march_post["author"], c["author"], c["is_submitter"])["is_author"]
        for c in march_post["comments"]
    ]
    assert roles == [False, True, False]
    mapped = to_package_post(march_post)
    assert mapped["title"] == march_post["title"]
    assert mapped["description"] == march_post["author_description"]
    assert mapped["adapter_meta"]["chronological_original_indices"] == [2, 1, 0]
    assert [c["text"] for c in mapped["comments"]] == [
        march_post["comments"][2]["body"],
        march_post["comments"][1]["body"],
        march_post["comments"][0]["body"],
    ]
    assert [c["is_author"] for c in mapped["comments"]] == [False, True, False]


def test_october_post_maps_nonempty_author_description(oct5_post):
    assert oct5_post["redditId"] == "1wyg8ze"
    body, field, present = extract_post_body(oct5_post)
    assert present is True
    assert field == "author_description"
    assert body.strip()
    assert "selftext" not in oct5_post
    assert body == oct5_post["author_description"]
    assert isinstance(oct5_post["comments"], list)
    assert len(oct5_post["comments"]) == 2
    mapped = to_package_post(oct5_post)
    assert mapped["description"] == oct5_post["author_description"]
    assert mapped["title"] == oct5_post["title"]
    assert len(mapped["comments"]) == 2
    assert mapped["comments"][0]["text"] == oct5_post["comments"][0]["body"]
    assert mapped["comments"][1]["text"] == oct5_post["comments"][1]["body"]
    assert mapped["comments"][0]["is_author"] is False
    assert mapped["comments"][1]["is_author"] is True
    norm = normalize_comments(oct5_post)
    assert norm["comments_status"] == "present_nonempty"
    assert norm["n_author_replies_nonempty"] == 1
    assert norm["n_community_replies_nonempty"] == 1
