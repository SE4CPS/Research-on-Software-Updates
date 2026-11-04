"""Independently validate a frozen ReleaseTrain raw-page capture.

Writes aggregate counts and per-ID flags/hashes only; raw text remains restricted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from urllib.parse import urlparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from full_resource_sentiment_v1.roles import infer_author_role  # noqa: E402
from full_resource_sentiment_v1.timestamps import parse_created_utc  # noqa: E402


def audit(capture_id):
    directory = ROOT / "restricted" / "broad_release_train" / capture_id
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise ValueError("capture is not complete")
    local = {}
    with (ROOT / "results" / "normalized_records_compact.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            x = json.loads(line)
            local[x["reddit_id"]] = x["source_memberships"]
    seen = {}
    duplicate_occurrences = 0
    title_conflicts = 0
    body_conflicts = 0
    counters = Counter()
    source_counts = Counter()
    month_counts = Counter()
    page_totals = []
    prior_next = None
    for index, page in enumerate(manifest["pages"], 1):
        if page["page"] != index or page["request_cursor"] != prior_next:
            raise ValueError(f"broken page/cursor sequence at {index}")
        raw = (directory / page["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != page["sha256"] or len(raw) != page["bytes"]:
            raise ValueError(f"page integrity failure {index}")
        payload = json.loads(raw)
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != page["rows"]:
            raise ValueError(f"page schema/count failure {index}")
        page_totals.append(page.get("reported_total"))
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict) or not row.get("redditId"):
                raise ValueError(f"missing ID on page {index}")
            rid = str(row["redditId"])
            title = str(row.get("title") or "")
            body = str(row.get("author_description") or "")
            title_hash = hashlib.sha256(title.encode()).hexdigest()
            body_hash = hashlib.sha256(body.encode()).hexdigest()
            if rid in seen:
                duplicate_occurrences += 1
                title_conflicts += title_hash != seen[rid]["title_hash"]
                body_conflicts += body_hash != seen[rid]["body_hash"]
                continue
            source = row.get("source")
            source_counts[str(source)] += 1
            host = urlparse(str(row.get("url") or "")).hostname or ""
            reddit_url = host == "reddit.com" or host.endswith(".reddit.com")
            counters["reddit_url"] += reddit_url
            counters["reddit_source_and_url"] += str(source).lower() == "reddit" and reddit_url
            counters["reddit_source_nonreddit_url"] += str(source).lower() == "reddit" and not reddit_url
            created = parse_created_utc(row.get("created_utc"))
            counters["post_timestamp_valid"] += created["valid"]
            month_counts[str(created.get("parsed_utc") or "missing")[:7]] += 1
            counters["title_nonempty"] += bool(title.strip())
            counters["body_nonempty"] += bool(body.strip())
            counters["title_and_body_nonempty"] += bool(title.strip() and body.strip())
            counters["post_author_present"] += bool(str(row.get("author") or "").strip())
            counters["api_post_sentiment_present"] += row.get("sentiment") is not None
            comments = row.get("comments")
            counters["comments_absent"] += "comments" not in row or comments is None
            counters["comments_empty"] += isinstance(comments, list) and not comments
            counters["comments_malformed"] += comments is not None and not isinstance(comments, list)
            comments = comments if isinstance(comments, list) else []
            counters["comments_with_any_stored"] += bool(comments)
            counters["stored_comment_turns"] += len(comments)
            n_author = n_community = n_nonempty = n_valid = n_unknown = 0
            times = set()
            comment_epochs = []
            for comment in comments:
                if not isinstance(comment, dict):
                    counters["malformed_comment_turns"] += 1
                    continue
                if not str(comment.get("body") or "").strip():
                    counters["empty_comment_turns"] += 1
                    continue
                n_nonempty += 1
                counters["comment_author_present_turns"] += bool(str(comment.get("author") or "").strip())
                counters["is_submitter_flag_present_turns"] += comment.get("is_submitter") is not None
                counters["api_comment_sentiment_present_turns"] += comment.get("sentiment_score") is not None
                role = infer_author_role(row.get("author"), comment.get("author"), comment.get("is_submitter"))
                n_author += bool(role["is_author"])
                n_community += not bool(role["is_author"])
                n_unknown += role["author_role_basis"] == "blank_names_not_matched"
                counters["role_conflict_turns"] += bool(role["role_conflict_is_submitter_vs_name"])
                parsed = parse_created_utc(comment.get("created_utc"))
                n_valid += parsed["valid"]
                if parsed["valid"]:
                    times.add(parsed["parsed_utc"])
                    comment_epochs.append(parsed["epoch_seconds"])
                    counters["comment_before_post_turns"] += bool(
                        created["valid"] and parsed["epoch_seconds"] < created["epoch_seconds"]
                    )
            counters["nonempty_comment_turns"] += n_nonempty
            counters["valid_timestamp_nonempty_turns"] += n_valid
            counters["unknown_role_nonempty_turns"] += n_unknown
            counters["author_reply_turns"] += n_author
            counters["community_reply_turns"] += n_community
            counters["posts_with_nonempty_comments"] += n_nonempty >= 1
            counters["posts_with_author_replies"] += n_author >= 1
            counters["posts_with_community_replies"] += n_community >= 1
            counters["posts_with_both_roles"] += n_author >= 1 and n_community >= 1
            counters["order_eligible"] += n_valid >= 2
            counters["elapsed_comment_eligible"] += len(times) >= 2
            counters["post_to_first_reply_eligible"] += bool(created["valid"] and comment_epochs)
            counters["author_ge4"] += n_author >= 4
            counters["community_ge4"] += n_community >= 4
            counters["all_nonempty_timestamps_valid"] += n_nonempty > 0 and n_valid == n_nonempty
            num_comments = row.get("num_comments")
            try:
                num_comments = int(num_comments)
            except (ValueError, TypeError):
                num_comments = None
            counters["num_comments_present"] += num_comments is not None
            counters["num_comments_exceeds_stored"] += num_comments is not None and num_comments > len(comments)
            seen[rid] = {
                "reddit_id": rid, "capture_page": index, "capture_row_index": row_index,
                "source": source, "reddit_url": reddit_url,
                "created_month": str(created.get("parsed_utc") or "missing")[:7],
                "title_hash": title_hash, "body_hash": body_hash,
                "title_nonempty": bool(title.strip()), "body_nonempty": bool(body.strip()),
                "post_timestamp_valid": created["valid"], "stored_comments": len(comments),
                "nonempty_comments": n_nonempty, "author_replies": n_author,
                "community_replies": n_community, "valid_reply_times": n_valid,
                "distinct_reply_times": len(times), "unknown_role_replies": n_unknown,
                "post_to_first_reply_eligible": bool(created["valid"] and comment_epochs),
                "num_comments_reported": num_comments,
                "reported_more_than_stored": num_comments is not None and num_comments > len(comments),
                "local_memberships": local.get(rid, []),
            }
        prior_next = page.get("next_cursor")
    counters["unique_ids"] = len(seen)
    counters["rows"] = len(seen) + duplicate_occurrences
    counters["duplicate_id_occurrences"] = duplicate_occurrences
    counters["duplicate_title_conflicts"] = title_conflicts
    counters["duplicate_body_conflicts"] = body_conflicts
    counters["overlap_local_4168"] = sum(bool(v["local_memberships"]) for v in seen.values())
    counters["overlap_master"] = sum("phase2_master" in v["local_memberships"] for v in seen.values())
    counters["overlap_fresh_membership"] = sum("fresh_500" in v["local_memberships"] for v in seen.values())
    counters["overlap_reserved_membership"] = sum("reserved_test" in v["local_memberships"] for v in seen.values())
    reddit_only = [v for v in seen.values() if str(v["source"]).lower() == "reddit"]
    reddit_counts = {
        "unique_ids": len(reddit_only),
        "title_nonempty": sum(v["title_nonempty"] for v in reddit_only),
        "body_nonempty": sum(v["body_nonempty"] for v in reddit_only),
        "post_timestamp_valid": sum(v["post_timestamp_valid"] for v in reddit_only),
        "comments_with_any_stored": sum(v["stored_comments"] > 0 for v in reddit_only),
        "posts_with_nonempty_comments": sum(v["nonempty_comments"] > 0 for v in reddit_only),
        "posts_with_author_replies": sum(v["author_replies"] > 0 for v in reddit_only),
        "posts_with_community_replies": sum(v["community_replies"] > 0 for v in reddit_only),
        "posts_with_both_roles": sum(v["author_replies"] > 0 and v["community_replies"] > 0 for v in reddit_only),
        "order_eligible": sum(v["valid_reply_times"] >= 2 for v in reddit_only),
        "elapsed_comment_eligible": sum(v["distinct_reply_times"] >= 2 for v in reddit_only),
        "overlap_local_4168": sum(bool(v["local_memberships"]) for v in reddit_only),
        "overlap_master": sum("phase2_master" in v["local_memberships"] for v in reddit_only),
    }
    reddit_url_rows = [v for v in seen.values() if v["reddit_url"]]
    reddit_url_counts = {
        "unique_ids": len(reddit_url_rows),
        "source_missing_or_nonreddit": sum(str(v["source"]).lower() != "reddit" for v in reddit_url_rows),
        "title_nonempty": sum(v["title_nonempty"] for v in reddit_url_rows),
        "body_nonempty": sum(v["body_nonempty"] for v in reddit_url_rows),
        "post_timestamp_valid": sum(v["post_timestamp_valid"] for v in reddit_url_rows),
        "posts_with_nonempty_comments": sum(v["nonempty_comments"] > 0 for v in reddit_url_rows),
        "posts_with_author_replies": sum(v["author_replies"] > 0 for v in reddit_url_rows),
        "posts_with_community_replies": sum(v["community_replies"] > 0 for v in reddit_url_rows),
        "posts_with_both_roles": sum(v["author_replies"] > 0 and v["community_replies"] > 0 for v in reddit_url_rows),
        "order_eligible": sum(v["valid_reply_times"] >= 2 for v in reddit_url_rows),
        "elapsed_comment_eligible": sum(v["distinct_reply_times"] >= 2 for v in reddit_url_rows),
        "post_to_first_reply_eligible": sum(v["post_to_first_reply_eligible"] for v in reddit_url_rows),
        "overlap_local_4168": sum(bool(v["local_memberships"]) for v in reddit_url_rows),
        "overlap_master": sum("phase2_master" in v["local_memberships"] for v in reddit_url_rows),
    }
    result = {
        "capture_id": capture_id, "raw_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "started_utc": manifest["started_utc"], "finished_utc": manifest.get("finished_utc"),
        "pages": len(manifest["pages"]), "first_reported_total": page_totals[0],
        "last_reported_total": page_totals[-1], "min_reported_total": min(page_totals),
        "max_reported_total": max(page_totals), "raw_bytes": sum(p["bytes"] for p in manifest["pages"]),
        "counts": dict(sorted(counters.items())), "source_counts": dict(sorted(source_counts.items())),
        "reddit_only_counts": reddit_counts,
        "reddit_url_counts": reddit_url_counts,
        "month_counts": dict(sorted(month_counts.items())),
        "note": "Source and API predictions are candidate metadata. No human outcomes or sentiment scores are inferred. Canonical occurrence is first captured per exact ID; conflicting duplicates are counted.",
    }
    output = ROOT / "results" / f"broad_capture_audit_{capture_id}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = ROOT / "results" / f"broad_capture_compact_{capture_id}.jsonl"
    with compact.open("w", encoding="utf-8") as fh:
        for rid in sorted(seen):
            fh.write(json.dumps(seen[rid], sort_keys=True) + "\n")
    print(json.dumps({"audit": str(output), "compact": str(compact), "counts": result["counts"], "source_counts": result["source_counts"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_id")
    audit(parser.parse_args().capture_id)
