"""Resolve full frozen original title/body without the model 8,000-char cap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.hashing import sha256_text
from full_resource_sentiment_v1.relevance_classify.preprocess import redact_one_way
from full_resource_sentiment_v1.validation_200.constants import phase6_root, repo_root


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def build_full_text_index(needed_ids: set[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load uncapped title/body from frozen snapshots. Prefer fuller bodies."""
    root = repo_root()
    phase6 = phase6_root()
    docs: dict[str, dict[str, Any]] = {}

    def consider(rid: str, title: Any, body: Any, snapshot: str) -> None:
        if needed_ids is not None and rid not in needed_ids:
            return
        title_s = _clean(title)
        body_s = _clean(body)
        prev = docs.get(rid)
        if prev is None:
            docs[rid] = {
                "title": title_s,
                "body": body_s,
                "snapshot": snapshot,
                "title_chars": len(title_s),
                "body_chars": len(body_s),
            }
            return
        # Prefer longer body, then longer title, keep first snapshot if equal.
        if len(body_s) > len(prev["body"]) or (
            len(body_s) == len(prev["body"]) and len(title_s) > len(prev["title"])
        ):
            docs[rid] = {
                "title": title_s,
                "body": body_s,
                "snapshot": snapshot,
                "title_chars": len(title_s),
                "body_chars": len(body_s),
                "replaced_previous_snapshot": prev["snapshot"],
            }

    for line in (root / "phase3/revised_development_v2/development_v2.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        consider(str(row["reddit_id"]), row.get("title"), row.get("post_body"), "development_v2")

    for hist_name, label in (
        ("tps_gds_classification/data/raw/tps_response.json", "historical_tps_raw"),
        ("tps_gds_classification/data/raw/gds_response.json", "historical_gds_raw"),
    ):
        payload = json.loads((root / hist_name).read_text(encoding="utf-8"))
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if not rid:
                continue
            consider(rid, row.get("title"), row.get("author_description"), label)

    capture_dir = phase6 / "restricted/broad_release_train/2026-10-22-main"
    for path in sorted(capture_dir.glob("page_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        snap = f"broad_capture:{path.name}"
        for row in payload.get("data") or []:
            rid = str(row.get("redditId") or "")
            if not rid:
                continue
            consider(rid, row.get("title"), row.get("author_description"), snap)

    return docs


def redact_packet_text(title: str, body: str) -> dict[str, Any]:
    original_title = _clean(title)
    original_body = _clean(body)
    red_title, title_counts = redact_one_way(original_title)
    red_body, body_counts = redact_one_way(original_body)
    return {
        "original_title": original_title,
        "original_body": original_body,
        "display_title": red_title,
        "display_body": red_body,
        "original_title_sha256": sha256_text(original_title),
        "original_body_sha256": sha256_text(original_body),
        "display_title_sha256": sha256_text(red_title),
        "display_body_sha256": sha256_text(red_body),
        "redaction_counts_title": title_counts,
        "redaction_counts_body": body_counts,
        "body_truncated_for_model_cap": False,
        "body_chars_original": len(original_body),
        "title_chars_original": len(original_title),
    }
