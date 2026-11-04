"""Workstream G scoring contract tests. Do not require the 37,596-post run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from releasetrain_sentiment import score_post

from full_resource_sentiment_v1.adapter import to_package_post
from full_resource_sentiment_v1.broad_sentiment import (
    load_local_development_labels,
    public_thread_row,
    run_broad_sentiment,
    select_handcheck_ids,
    verify_capture_pages,
)
from full_resource_sentiment_v1.inference import bootstrap_mean_median_ci, cliffs_delta
from full_resource_sentiment_v1.privacy import scan_obj, scan_paths
from full_resource_sentiment_v1.scoring import score_raw_post

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "synthetic_posts.json").read_text(encoding="utf-8"))
GENUINE_ZERO = 0.0


def test_title_body_combined_and_missing_body_vs_zero():
    body_post = FIXTURES["nonempty_body"]
    scored = score_raw_post(body_post)
    title_only = score_post({"title": body_post["title"], "description": "", "comments": []})[0]
    body_only = score_post({"title": "", "description": body_post["author_description"], "comments": []})[0]
    combined = score_post(
        {"title": body_post["title"], "description": body_post["author_description"], "comments": []}
    )[0]
    assert scored["title"]["compound"] == title_only["compound"]
    assert scored["body"]["compound"] == body_only["compound"]
    assert scored["combined"]["compound"] == combined["compound"]
    empty = score_raw_post(FIXTURES["empty_body_title_only"])
    assert empty["body"] is None
    assert empty["body_minus_title"] is None
    zero = score_raw_post(FIXTURES["genuine_neutral_zero"])
    assert zero["title"]["compound"] == GENUINE_ZERO
    assert zero["body"]["compound"] == GENUINE_ZERO
    assert zero["comment_rows"][0]["compound"] == GENUINE_ZERO


def test_no_author_reply_is_null_not_package_zero():
    none = score_raw_post(FIXTURES["no_observed_reply"])
    zero = score_raw_post(FIXTURES["genuine_neutral_zero"])
    assert none["author_mean"] is None
    assert none["n_author_replies_nonempty"] == 0
    assert zero["community_mean"] == GENUINE_ZERO
    assert zero["community_mean"] is not None


def test_roles_blank_names_and_is_submitter():
    submitter = score_raw_post(FIXTURES["author_by_is_submitter"])
    named = score_raw_post(FIXTURES["author_by_name_match"])
    blank = score_raw_post(FIXTURES["blank_author_names"])
    assert submitter["comment_rows"][0]["role"] == "author"
    assert named["comment_rows"][0]["role"] == "author"
    assert blank["comment_rows"][0]["role"] == "unresolved"
    assert blank["comment_rows"][0]["is_author"] is False
    assert blank["n_community_replies_nonempty"] == 1
    assert blank["n_unresolved_replies_nonempty"] == 1
    empty = score_raw_post(FIXTURES["empty_comment_body"])
    assert empty["n_empty_comment_bodies"] == 1
    assert empty["n_nonempty_comments"] == 1
    assert empty["n_comments_skipped_empty"] == 1
    absent = score_raw_post(FIXTURES["comments_key_absent"])
    assert absent["comments_status"] == "absent"
    assert absent["comment_rows"] == []


def test_timestamp_tie_malformed_and_order():
    tied = score_raw_post(FIXTURES["equal_timestamps"])
    assert [row["original_index"] for row in tied["comment_rows"]] == [0, 1]
    assert tied["order_eligible"] is True
    assert tied["elapsed_eligible"] is False
    missing = score_raw_post(FIXTURES["missing_timestamp"])
    assert missing["n_nonempty_comments"] == 2
    assert missing["n_timed_nonempty"] == 1
    assert missing["order_eligible"] is False
    ordered = score_raw_post(FIXTURES["out_of_order_timestamps"])
    assert [row["original_index"] for row in ordered["comment_rows"]] == [1, 2, 0]


def test_outputs_drop_package_text_and_usernames():
    scored = score_raw_post(FIXTURES["handcalc_thread"])
    blob = json.dumps(scored)
    assert "Known issue" not in blob
    assert "dev1" not in blob
    assert "user2" not in blob
    for row in scored["comment_rows"]:
        assert "text" not in row
        assert "author" not in row
    thread = {"reddit_id": "x", "author_mean": 0.1, "timed_compounds": [0.1, 0.2], "comment_rows": scored["comment_rows"]}
    public = public_thread_row(thread)
    assert "timed_compounds" not in public
    assert "comment_rows" not in public
    assert scan_obj({"reddit_id": "abc", "title_compound": 0.1, "author_mean": None}) == []
    assert scan_obj({"title": "secret", "url": "https://reddit.com/r/x"}) != []


def test_bootstrap_uses_thread_unit_and_fixed_seed():
    values = [0.1, -0.2, 0.0, 0.4, -0.5]
    first = bootstrap_mean_median_ci(values, n_boot=50, seed=20261022)
    second = bootstrap_mean_median_ci(values, n_boot=50, seed=20261022)
    third = bootstrap_mean_median_ci(values, n_boot=50, seed=20261023)
    assert first["mean_ci95"] == second["mean_ci95"]
    assert first["seed"] == 20261022
    assert first["bootstrap_unit"] == "thread"
    assert first["n_bootstrap"] == 50
    assert third["mean_ci95"] != first["mean_ci95"]
    rng = np.random.default_rng(20261022)
    idx = rng.integers(0, 5, size=(50, 5))
    means = np.asarray(values, dtype=float)[idx].mean(axis=1)
    assert abs(first["mean_ci95"][0] - float(np.percentile(means, 2.5))) < 1e-12
    assert cliffs_delta([1.0, 2.0, 3.0], [0.0, 1.0, 2.0]) > 0


def test_handchecked_march_and_october_match_package(march_post, oct5_post):
    march = score_raw_post(march_post)
    oct5 = score_raw_post(oct5_post)
    mapped_m = to_package_post(march_post, include_empty_comment_bodies=False)
    mapped_o = to_package_post(oct5_post, include_empty_comment_bodies=False)
    scored_m = score_post(mapped_m)
    scored_o = score_post(mapped_o)
    assert abs(march["combined"]["compound"] - scored_m[0]["compound"]) < 1e-12
    assert abs(oct5["combined"]["compound"] - scored_o[0]["compound"]) < 1e-12
    assert [row["compound"] for row in march["comment_rows"]] == [item["compound"] for item in scored_m[1:]]
    assert [row["compound"] for row in oct5["comment_rows"]] == [item["compound"] for item in scored_o[1:]]
    assert march["n_author_replies_nonempty"] == 1
    assert march["n_community_replies_nonempty"] == 2
    assert oct5["n_author_replies_nonempty"] == 1
    assert oct5["n_community_replies_nonempty"] == 1
    assert "text" not in json.dumps(march["comment_rows"][0])


def _write_page(path: Path, rows: list[dict]) -> dict:
    payload = {"data": rows, "totalCount": len(rows)}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes(raw)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows), "file": path.name}


def test_mini_capture_dedup_gap_protected_resume_and_privacy(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    both = {
        "redditId": "aaa111",
        "title": "Update broke my build",
        "author": "op_user",
        "author_description": "Nothing compiles since 3.2.0",
        "created_utc": "2026-10-01T00:00:00",
        "source": "reddit",
        "url": "https://www.reddit.com/r/x/comments/aaa111/",
        "num_comments": 9,
        "comments": [
            {"author": "op_user", "body": "Known issue, fix coming soon", "is_submitter": True, "created_utc": "2026-10-01T01:00:00"},
            {"author": "other", "body": "Same here, very annoying", "is_submitter": False, "created_utc": "2026-10-01T02:00:00"},
            {"author": "other", "body": "   ", "is_submitter": False, "created_utc": "2026-10-01T03:00:00"},
        ],
    }
    title_only = {
        "redditId": "bbb222",
        "title": "Only a title about an update",
        "author": "op_user",
        "author_description": "",
        "created_utc": "2025-01-01T00:00:00",
        "source": None,
        "url": "https://reddit.com/r/x/comments/bbb222/",
        "num_comments": 0,
        "comments": [],
    }
    reserved = {
        "redditId": "ccc333",
        "title": "The table is made of wood.",
        "author": "op_user",
        "author_description": "The table is made of wood.",
        "created_utc": "2026-10-02T00:00:00",
        "source": "reddit",
        "url": "https://www.reddit.com/r/x/comments/ccc333/",
        "num_comments": 1,
        "comments": [
            {"author": "other", "body": "The table is made of wood.", "is_submitter": False, "created_utc": "2026-10-02T01:00:00"}
        ],
    }
    other_scope = {
        "redditId": "ddd444",
        "title": "I love this update, everything works perfectly!",
        "author": "op_user",
        "author_description": "I love this update, everything works perfectly!",
        "created_utc": "2026-09-01T00:00:00",
        "source": "reddit",
        "url": "https://www.reddit.com/r/x/comments/ddd444/",
        "num_comments": 0,
        "comments": [],
    }
    non_reddit = {
        "redditId": "so999",
        "title": "stackoverflow",
        "author": "x",
        "author_description": "body",
        "created_utc": "2026-10-01T00:00:00",
        "source": "stackoverflow",
        "url": "https://stackoverflow.com/q/1",
        "num_comments": 0,
        "comments": [],
    }
    page1 = _write_page(capture / "page_00001.json", [both, title_only, non_reddit])
    page2 = _write_page(capture / "page_00002.json", [reserved, other_scope, both])
    manifest = {
        "capture_id": "mini",
        "complete": True,
        "started_utc": "2026-10-22T00:00:00+00:00",
        "finished_utc": "2026-10-22T00:01:00+00:00",
        "pages": [
            {**page1, "page": 1, "request_cursor": None, "next_cursor": "c1", "file": "page_00001.json"},
            {**page2, "page": 2, "request_cursor": "c1", "next_cursor": None, "file": "page_00002.json"},
        ],
    }
    (capture / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    verified = verify_capture_pages(capture)
    assert verified["n_pages"] == 2

    compact_rows = [
        {"reddit_id": "aaa111", "reddit_url": True, "source": "reddit", "created_month": "2026-10",
         "body_nonempty": True, "author_replies": 1, "community_replies": 1, "nonempty_comments": 2,
         "reported_more_than_stored": True, "local_memberships": ["development_v2"]},
        {"reddit_id": "bbb222", "reddit_url": True, "source": None, "created_month": "2025-01",
         "body_nonempty": False, "author_replies": 0, "community_replies": 0, "nonempty_comments": 0,
         "reported_more_than_stored": False, "local_memberships": []},
        {"reddit_id": "ccc333", "reddit_url": True, "source": "reddit", "created_month": "2026-10",
         "body_nonempty": True, "author_replies": 0, "community_replies": 1, "nonempty_comments": 1,
         "reported_more_than_stored": False, "local_memberships": ["reserved_test"]},
        {"reddit_id": "ddd444", "reddit_url": True, "source": "reddit", "created_month": "2026-09",
         "body_nonempty": True, "author_replies": 0, "community_replies": 0, "nonempty_comments": 0,
         "reported_more_than_stored": False, "local_memberships": ["development_v2"]},
        {"reddit_id": "so999", "reddit_url": False, "source": "stackoverflow", "created_month": "2026-10",
         "body_nonempty": True, "author_replies": 0, "community_replies": 0, "nonempty_comments": 0,
         "reported_more_than_stored": False, "local_memberships": []},
    ]
    compact_path = tmp_path / "compact.jsonl"
    compact_path.write_text("".join(json.dumps(row) + "\n" for row in compact_rows), encoding="utf-8")
    local_path = tmp_path / "local.jsonl"
    local_rows = [
        {"reddit_id": "aaa111", "source_memberships": ["development_v2"], "protected_outcomes_copied": False,
         "labels": {"s2_in_scope": True, "s2_target": 1, "split_reserved": False, "split_fresh": False}},
        {"reddit_id": "ccc333", "source_memberships": ["reserved_test"], "protected_outcomes_copied": False,
         "labels": {"s2_in_scope": True, "s2_target": 1, "split_reserved": True, "split_fresh": False}},
        {"reddit_id": "ddd444", "source_memberships": ["development_v2"], "protected_outcomes_copied": False,
         "labels": {"s2_in_scope": True, "s2_target": 0, "split_reserved": False, "split_fresh": False}},
    ]
    local_path.write_text("".join(json.dumps(row) + "\n" for row in local_rows), encoding="utf-8")
    labels = load_local_development_labels(local_path)
    assert labels["aaa111"]["f_eligible"] is True
    assert labels["ccc333"]["f_eligible"] is False
    assert labels["ccc333"]["s2_target"] is None

    run_dir = tmp_path / "run"
    manifest = run_broad_sentiment(
        capture_dir=capture,
        compact_path=compact_path,
        local_path=local_path,
        run_dir=run_dir,
        validate_frozen_denominators=False,
    )
    posts = [json.loads(line) for line in (run_dir / "post_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    comments = [json.loads(line) for line in (run_dir / "comment_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    threads = [json.loads(line) for line in (run_dir / "thread_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [row["reddit_id"] for row in posts]
    assert ids == sorted(set(ids))
    assert set(ids) == {"aaa111", "bbb222", "ccc333", "ddd444"}
    assert "so999" not in ids
    by_id = {row["reddit_id"]: row for row in posts}
    assert by_id["aaa111"]["count_gap"] is True
    assert by_id["bbb222"]["body_compound"] is None
    thread_by = {row["reddit_id"]: row for row in threads}
    assert thread_by["bbb222"]["author_mean"] is None
    assert thread_by["ccc333"]["community_mean"] == GENUINE_ZERO
    assert thread_by["ccc333"]["f_eligible"] is False
    assert thread_by["aaa111"]["f_eligible"] is True
    assert thread_by["aaa111"]["s2_target"] == 1
    assert thread_by["ddd444"]["s2_target"] == 0
    assert by_id["aaa111"]["n_empty_comment_bodies"] == 1
    assert all("text" not in row and "author" not in row for row in comments)
    assert all("https://" not in json.dumps(row) for row in posts)
    hits = scan_paths([run_dir / "post_scores.jsonl", run_dir / "thread_metrics.jsonl", run_dir / "analysis_A.json"])
    assert hits == []
    first_hash = (run_dir / "hashes.json").read_text(encoding="utf-8")
    again = run_broad_sentiment(
        capture_dir=capture,
        compact_path=compact_path,
        local_path=local_path,
        run_dir=run_dir,
        validate_frozen_denominators=False,
    )
    assert again["status"] == "awaiting_codex_review"
    assert (run_dir / "hashes.json").read_text(encoding="utf-8") == first_hash
    sample = select_handcheck_ids(threads, n=3, seed=20261022)
    assert len(set(sample)) == len(sample)
    f_ids = {row["reddit_id"] for row in threads if row["f_eligible"]}
    assert f_ids == {"aaa111", "ddd444"}
    assert manifest["counts"]["duplicate_reddit_url_skipped"] == 1
    assert manifest["counts"]["non_reddit_url_skipped"] == 1
