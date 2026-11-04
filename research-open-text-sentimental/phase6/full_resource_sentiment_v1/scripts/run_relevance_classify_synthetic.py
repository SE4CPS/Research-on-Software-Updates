#!/usr/bin/env python3
"""Stage 2 synthetic-only classifier run. No live API, no frozen Reddit text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_text, git_revision, utcnow
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.privacy import scan_paths
from full_resource_sentiment_v1.relevance_classify.pipeline import (
    CheckpointStore,
    HumanCorrectionStore,
    assert_no_raw_text,
    classify_records,
    public_row,
)
from full_resource_sentiment_v1.relevance_classify.planner import plan_batches, simulate_full_frame_budget
from full_resource_sentiment_v1.relevance_classify.providers import MockProvider
from full_resource_sentiment_v1.relevance_classify.spec import (
    CLASSIFIER_RUN_ID,
    PROMPT_VERSION,
    SCHEMA_VERSION,
    STAGE2_LIVE_LOCKED,
    prompt_sha256,
    schema_sha256,
)
from full_resource_sentiment_v1.relevance_classify.synthetic_suite import classification_records, mock_scripts

PUBLIC_DIR = WORKSPACE / "results" / "ai_relevance_synthetic_fixes_2026-10-28"
RESTRICTED_DIR = WORKSPACE / "restricted" / "ai_relevance_synthetic_fixes_2026-10-28"


def main() -> int:
    if not STAGE2_LIVE_LOCKED:
        raise RuntimeError("Stage 2 runner refuses to start unless live inference is locked")
    records = classification_records()
    provider = MockProvider(mock_scripts())
    RESTRICTED_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointStore(RESTRICTED_DIR / "checkpoint.jsonl")
    rows = classify_records(records, provider, checkpoint=checkpoint)
    corrections = HumanCorrectionStore(RESTRICTED_DIR / "human_corrections.jsonl")
    if not corrections.rows:
        news = next(row for row in rows if row["synthetic_id"] == "synth-news-update")
        corrections.add(
            synthetic_id=news["synthetic_id"],
            classifier_run_id=news["classifier_run_id"],
            corrected_label="not_relevant",
            reason="Synthetic demonstration that human corrections are a separate layer.",
        )

    public_rows = [public_row(row) for row in rows]
    for item in public_rows:
        assert_no_raw_text(item)

    counts: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in rows:
        label = row.get("predicted_label") or "none"
        counts[str(label)] = counts.get(str(label), 0) + 1
        statuses[str(row["response_status"])] = statuses.get(str(row["response_status"]), 0) + 1

    plan = plan_batches(records, queued_token_limit=2_000_000)
    budget = simulate_full_frame_budget(
        n=37596,
        mean_submitted_chars=1146,
        queued_token_limit=2_000_000,
        max_output_tokens=256,
        proposed_ceiling_usd=25.0,
    )

    summary = {
        "status": "awaiting_codex_synthetic_implementation_validation",
        "classifier_run_id": CLASSIFIER_RUN_ID,
        "created_utc": utcnow(),
        "git_revision": git_revision(),
        "live_locked": STAGE2_LIVE_LOCKED,
        "n_synthetic_records": len(records),
        "n_provider_calls": len(provider.calls),
        "label_counts": counts,
        "status_counts": statuses,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha256(),
        "public_rows": public_rows,
        "upload": False,
        "spend": False,
        "used_frozen_reddit_text": False,
        "protected_outcomes_read": False,
    }
    assert_no_raw_text(summary)
    atomic_write_json(PUBLIC_DIR / "public_summary.json", summary)
    atomic_write_json(PUBLIC_DIR / "batch_plan.json", plan)
    atomic_write_json(PUBLIC_DIR / "full_frame_budget_simulation.json", budget)

    sample = []
    for row in rows:
        sample.append(
            {
                "synthetic_id": row["synthetic_id"],
                "response_status": row["response_status"],
                "predicted_label": row["predicted_label"],
                "evidence_phrase": row["evidence_phrase"],
                "decision_reason": row["decision_reason"],
                "ambiguity_reasons": row["ambiguity_reasons"],
                "review_priority": row["review_priority"],
                "unassessable_basis": row["unassessable_basis"],
                "response_id": row["response_id"],
                "retry_count": row["retry_count"],
                "model_input_sha256": row["model_input_sha256"],
                "redaction_types": row["redaction_types"],
            }
        )
    atomic_write_json(RESTRICTED_DIR / "sample_mock_outputs.json", sample)

    hashes = {
        "public_summary.json": sha256_file(PUBLIC_DIR / "public_summary.json"),
        "batch_plan.json": sha256_file(PUBLIC_DIR / "batch_plan.json"),
        "full_frame_budget_simulation.json": sha256_file(PUBLIC_DIR / "full_frame_budget_simulation.json"),
        "checkpoint.jsonl": sha256_file(RESTRICTED_DIR / "checkpoint.jsonl"),
        "sample_mock_outputs.json": sha256_file(RESTRICTED_DIR / "sample_mock_outputs.json"),
        "human_corrections.jsonl": sha256_file(RESTRICTED_DIR / "human_corrections.jsonl"),
        "prompt_sha256": prompt_sha256(),
        "schema_sha256": schema_sha256(),
    }
    atomic_write_json(PUBLIC_DIR / "hashes.json", hashes)

    public_hits = scan_paths(
        [
            PUBLIC_DIR / "public_summary.json",
            PUBLIC_DIR / "batch_plan.json",
            PUBLIC_DIR / "full_frame_budget_simulation.json",
            PUBLIC_DIR / "hashes.json",
        ]
    )
    if public_hits:
        raise RuntimeError(f"public synthetic outputs failed privacy scan: {public_hits[:12]}")

    atomic_write_text(
        PUBLIC_DIR / "STATUS.txt",
        "Awaiting Codex synthetic implementation validation\n",
    )
    print(json.dumps({"status": summary["status"], "n": len(rows), "hashes": hashes}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
