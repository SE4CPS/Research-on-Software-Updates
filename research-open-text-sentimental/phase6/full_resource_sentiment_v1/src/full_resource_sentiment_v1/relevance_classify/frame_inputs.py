"""Build redacted classifier inputs for the frozen 37,596-post candidate frame."""

from __future__ import annotations

from typing import Any

from full_resource_sentiment_v1.hashing import sha256_text
from full_resource_sentiment_v1.io_utils import read_json
from full_resource_sentiment_v1.paths import BROAD_CAPTURE_DIR, WORKSPACE
from full_resource_sentiment_v1.relevance_audit import collect_selected_text, load_primary_frame
from full_resource_sentiment_v1.relevance_classify.preprocess import redact_one_way, reject_protected_outcomes
from full_resource_sentiment_v1.relevance_classify.spec import FULL_CLASSIFIER_RUN_ID

EXPECTED_FRAME_N = 37596
CAPTURE_MANIFEST_SHA256 = "3239a66564a693c72699dc9c25546f66ed839b05f3c68a26cf43a3db75853b8c"


class FrameInputError(ValueError):
    pass


def load_frame_classifier_inputs() -> dict[str, Any]:
    manifest = read_json(BROAD_CAPTURE_DIR / "manifest.json")
    from full_resource_sentiment_v1.hashing import sha256_file

    digest = sha256_file(BROAD_CAPTURE_DIR / "manifest.json")
    if digest != CAPTURE_MANIFEST_SHA256:
        raise FrameInputError(f"capture manifest hash changed: {digest}")
    frame = load_primary_frame()
    posts = frame["posts"]
    if len(posts) != EXPECTED_FRAME_N:
        raise FrameInputError(f"primary frame n={len(posts)} != {EXPECTED_FRAME_N}")
    texts = collect_selected_text(set(posts))
    if set(texts) != set(posts):
        raise FrameInputError("capture text IDs differ from the primary frame")

    records: list[dict[str, Any]] = []
    october = 0
    older = 0
    protected = 0
    for rid, meta in posts.items():
        text = texts[rid]
        reject_protected_outcomes(text)
        title = "" if text.get("title") is None else str(text.get("title") or "")
        body = "" if text.get("body") is None else str(text.get("body") or "")
        title_redacted, _ = redact_one_way(title)
        body_redacted, _ = redact_one_way(body)
        stratum = meta["stratum"]
        if stratum == "october_2026":
            october += 1
        else:
            older += 1
        is_protected = bool(meta.get("reserved_membership") or meta.get("fresh_membership"))
        if is_protected:
            protected += 1
        records.append(
            {
                "synthetic_id": rid,
                "reddit_id": rid,
                "title": title_redacted,
                "body": body_redacted,
                "stratum": stratum,
                "created_month": meta.get("created_month"),
                "count_gap": bool(meta.get("count_gap")),
                "protected_membership": is_protected,
                "capture_id": "2026-10-22-main",
                "classifier_run_id": FULL_CLASSIFIER_RUN_ID,
                "title_sha256_original": sha256_text(title),
                "body_sha256_original": sha256_text(body),
            }
        )
    records.sort(key=lambda row: row["reddit_id"])
    return {
        "records": records,
        "n": len(records),
        "n_october": october,
        "n_older": older,
        "n_protected": protected,
        "capture_id": manifest.get("capture_id"),
        "capture_manifest_sha256": digest,
        "frame_keys": sorted(frame.keys()) if isinstance(frame, dict) else None,
    }
