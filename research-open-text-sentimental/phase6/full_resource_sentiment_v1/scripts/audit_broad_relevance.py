"""Read-only label/relevance/denominator audit for the frozen broad capture.

Only development human labels are read. Reserved/fresh outcomes are never opened.
The broad listing itself is not labeled by this script.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = "2026-10-22-main"
BROAD = ROOT / "results" / f"broad_capture_compact_{CAPTURE}.jsonl"
LOCAL = ROOT / "results" / "normalized_records_compact.jsonl"
OUTPUT = ROOT / "results" / "broad_relevance_denominators.json"

FEATURES = (
    "title_nonempty", "body_nonempty", "nonempty_comments_ge1", "author_replies_ge1",
    "community_replies_ge1", "both_roles_ge1", "order_eligible", "elapsed_eligible",
    "post_to_first_reply_eligible", "reported_more_than_stored",
)


def increment(counter, row):
    counter["n"] += 1
    counter["title_nonempty"] += bool(row["title_nonempty"])
    counter["body_nonempty"] += bool(row["body_nonempty"])
    counter["nonempty_comments_ge1"] += row["nonempty_comments"] >= 1
    counter["author_replies_ge1"] += row["author_replies"] >= 1
    counter["community_replies_ge1"] += row["community_replies"] >= 1
    counter["both_roles_ge1"] += row["author_replies"] >= 1 and row["community_replies"] >= 1
    counter["order_eligible"] += row["valid_reply_times"] >= 2
    counter["elapsed_eligible"] += row["distinct_reply_times"] >= 2
    counter["post_to_first_reply_eligible"] += row["post_to_first_reply_eligible"]
    counter["reported_more_than_stored"] += row["reported_more_than_stored"]
    counter["created_2026_10"] += row["created_month"] == "2026-10"
    counter["created_2026_08_or_later"] += row["created_month"] >= "2026-08"


def main():
    local = {}
    with LOCAL.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            local[record["reddit_id"]] = record
    groups = {name: Counter() for name in (
        "broad_reddit_url", "broad_count_gap", "broad_no_count_gap",
        "dev_overlap", "dev_in_scope", "dev_in_scope_tps", "dev_in_scope_other",
        "dev_out_of_scope", "dev_in_scope_count_gap", "dev_in_scope_no_count_gap",
        "broad_unreviewed_dev",
    )}
    label_fields = {key: Counter() for key in (
        "phase2_about_software", "phase2_update_related", "phase2_discussion_type",
        "phase2_review_status", "s2_in_scope",
    )}
    membership = Counter()
    with BROAD.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not row["reddit_url"]:
                continue
            increment(groups["broad_reddit_url"], row)
            increment(groups["broad_count_gap" if row["reported_more_than_stored"] else "broad_no_count_gap"], row)
            local_row = local.get(row["reddit_id"])
            local_membership = set(local_row["source_memberships"]) if local_row else set()
            membership["overlap_any_local"] += bool(local_membership)
            membership["overlap_master"] += "phase2_master" in local_membership
            membership["overlap_development"] += "development_v2" in local_membership
            membership["overlap_reserved_membership"] += "reserved_test" in local_membership
            membership["overlap_fresh_membership"] += "fresh_500" in local_membership
            if "development_v2" not in local_membership:
                increment(groups["broad_unreviewed_dev"], row)
                continue
            if "reserved_test" in local_membership or "fresh_500" in local_membership:
                raise ValueError("protected membership appeared in development analysis")
            increment(groups["dev_overlap"], row)
            labels = local_row["labels"]
            for field in label_fields:
                label_fields[field][str(labels.get(field))] += 1
            if labels.get("s2_in_scope") is True:
                increment(groups["dev_in_scope"], row)
                increment(groups["dev_in_scope_count_gap" if row["reported_more_than_stored"] else "dev_in_scope_no_count_gap"], row)
                target = labels.get("s2_target")
                if target == 1:
                    increment(groups["dev_in_scope_tps"], row)
                elif target == 0:
                    increment(groups["dev_in_scope_other"], row)
                else:
                    raise ValueError("in-scope development row missing target")
            else:
                increment(groups["dev_out_of_scope"], row)
    result = {
        "capture": CAPTURE,
        "population_note": "Broad Reddit-URL candidates are primary; development overlap is nonrandom and only it carries accessible human-reviewed labels. API candidate/source flags are not labels.",
        "protected_note": "Fresh and reserved membership overlap only; no protected outcome read or exported.",
        "count_gap_note": "Reported num_comments > stored array length flags possible incomplete coverage; not an exclusion and not proof recoverable comments exist.",
        "groups": {name: dict(sorted(counter.items())) for name, counter in groups.items()},
        "label_fields_on_dev_overlap": {key: dict(sorted(counter.items())) for key, counter in label_fields.items()},
        "membership": dict(sorted(membership.items())),
    }
    if groups["broad_reddit_url"]["n"] != groups["broad_count_gap"]["n"] + groups["broad_no_count_gap"]["n"]:
        raise ValueError("count-gap partition did not sum")
    if groups["dev_in_scope"]["n"] != groups["dev_in_scope_tps"]["n"] + groups["dev_in_scope_other"]["n"]:
        raise ValueError("in-scope labels did not sum")
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"groups": result["groups"], "membership": result["membership"]}, indent=2))


if __name__ == "__main__":
    main()
