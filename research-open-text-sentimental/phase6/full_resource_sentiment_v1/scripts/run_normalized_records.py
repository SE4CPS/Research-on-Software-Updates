#!/usr/bin/env python3
"""Materialize joined per-ID normalized records.

Does not fetch live API data, score protected tests, train models, or rerun
the historical 542 comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(REPO_ROOT))

from full_resource_sentiment_v1.inventory import load_all_sources  # noqa: E402
from full_resource_sentiment_v1.io_utils import write_json, write_jsonl  # noqa: E402
from full_resource_sentiment_v1.paths import RESULTS_DIR  # noqa: E402
from full_resource_sentiment_v1.records import (  # noqa: E402
    audit_normalized_records,
    iter_normalized_records,
    record_matches_schema,
    union_reddit_ids,
)
from full_resource_sentiment_v1.schema import schema_document  # noqa: E402


def main() -> int:
    loaded = load_all_sources()
    if loaded["reserved"].get("by_id") or loaded["fresh"].get("by_id"):
        print("STOP: protected files were loaded beyond ID membership")
        return 2
    ids = union_reddit_ids(loaded)
    records = list(iter_normalized_records(loaded, reddit_ids=ids, include_comment_text=False))
    problems = []
    for record in records[:25]:
        problems.extend(record_matches_schema(record))
    if problems:
        print("STOP: materialized records missing schema fields")
        print(problems[:20])
        return 2
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / "normalized_schema.json", schema_document())
    write_jsonl(RESULTS_DIR / "normalized_records_compact.jsonl", records)
    audit = audit_normalized_records(records)
    write_json(RESULTS_DIR / "normalized_records_audit.json", audit)
    print(f"normalized records: {audit['n_records']}")
    print(f"membership combinations: {audit['n_membership_combinations']}")
    print(
        "title/body snapshot conflicts:",
        audit["snapshot_conflicts"]["n_records_title_hash_conflict"],
        audit["snapshot_conflicts"]["n_records_body_hash_conflict"],
    )
    print(
        "protected reserved/fresh:",
        audit["protected"]["n_split_reserved"],
        audit["protected"]["n_split_fresh"],
        "s2_target copied from protected:",
        audit["protected"]["n_reserved_with_s2_target"] + audit["protected"]["n_fresh_with_s2_target"],
    )
    print(f"wrote {RESULTS_DIR / 'normalized_records_compact.jsonl'}")
    print(f"wrote {RESULTS_DIR / 'normalized_records_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
