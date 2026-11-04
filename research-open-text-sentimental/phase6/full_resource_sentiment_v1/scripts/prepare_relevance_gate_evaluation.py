#!/usr/bin/env python3
"""Prepare the 240-case evaluation gate. Synthetic scoring only. No live API."""

from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_text, git_revision, utcnow
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.privacy import scan_paths
from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate, to_classifier_inputs, wilson_95
from full_resource_sentiment_v1.relevance_classify.gate_prepare import (
    GATE_REAL_RUN_LOCKED,
    describe_provider_payload,
    gate_budget_simulation,
    presentation_protocol,
    verify_frozen_fingerprints,
)
from full_resource_sentiment_v1.relevance_classify.spec import STAGE2_LIVE_LOCKED, prompt_sha256, schema_sha256

PUBLIC_DIR = WORKSPACE / "results" / "ai_relevance_gate_prep_2026-10-29"


def _synthetic_demo() -> dict:
    labels = []
    predictions = []
    packets = []
    for index in range(1, 9):
        code = f"C-{index:06d}"
        stratum = "october_2026" if index <= 4 else "older"
        if index <= 3:
            human = "relevant"
            pred = "relevant"
            status = "ok"
        elif index == 4:
            human = "relevant"
            pred = None
            status = "invalid_json"
        elif index == 8:
            human = "unassessable"
            pred = "unassessable"
            status = "skipped_deterministic"
        else:
            human = "not_relevant"
            pred = "not_relevant"
            status = "ok"
        labels.append({"case_code": code, "stratum": stratum, "final_label": human})
        predictions.append(
            {"case_code": code, "predicted_label": pred, "response_status": status, "review_priority": "low"}
        )
        packets.append({"case_code": code, "title": f"synthetic title {index}", "body": "synthetic body", "relevance": ""})
    inputs = to_classifier_inputs(packets)
    scored = evaluate_gate(labels, predictions)
    return {"classifier_inputs_n": len(inputs), "evaluation": scored}


def main() -> int:
    if not STAGE2_LIVE_LOCKED or not GATE_REAL_RUN_LOCKED:
        raise RuntimeError("prepare script refuses to run unless real scoring stays locked")
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    frozen = verify_frozen_fingerprints()
    demo = _synthetic_demo()
    payload = describe_provider_payload()
    budget = gate_budget_simulation(n=240)
    protocol = presentation_protocol()
    summary = {
        "status": "awaiting_codex_gate_preparation_validation",
        "created_utc": utcnow(),
        "git_revision": git_revision(),
        "live_locked": STAGE2_LIVE_LOCKED,
        "real_240_run_locked": GATE_REAL_RUN_LOCKED,
        "used_real_reddit_text": False,
        "used_protected_outcomes": False,
        "scored_locked_audit": False,
        "prompt_sha256": prompt_sha256(),
        "schema_sha256": schema_sha256(),
        "frozen_fingerprints": frozen,
        "presentation_protocol": protocol,
        "provider_payload": payload,
        "budget_240_length_only": budget,
        "wilson_32_of_40": wilson_95(32, 40),
        "synthetic_demo": {
            "classifier_inputs_n": demo["classifier_inputs_n"],
            "complete_join": demo["evaluation"]["join"]["complete"],
            "metrics": {
                "true_positives": demo["evaluation"]["metrics"]["true_positives"],
                "false_positives": demo["evaluation"]["metrics"]["false_positives"],
                "false_negatives": demo["evaluation"]["metrics"]["false_negatives"],
                "precision": demo["evaluation"]["metrics"]["precision"],
                "recall": demo["evaluation"]["metrics"]["recall"],
                "model_errors": demo["evaluation"]["metrics"]["model_errors"],
            },
            "gate_passed": demo["evaluation"]["gate"]["passed"],
        },
        "future_commands_if_authorized": {
            "note": "Not to be run until Codex/PI authorize a real 240-case scoring job.",
            "pytest": ".venv/bin/python -m pytest -q tests/test_relevance_gate_evaluation.py tests/test_relevance_classify_synthetic.py",
            "prepare_only": ".venv/bin/python scripts/prepare_relevance_gate_evaluation.py",
        },
    }
    atomic_write_json(PUBLIC_DIR / "public_summary.json", summary)
    atomic_write_json(PUBLIC_DIR / "hashes.json", {"public_summary.json": sha256_file(PUBLIC_DIR / "public_summary.json")})
    hits = scan_paths([PUBLIC_DIR / "public_summary.json", PUBLIC_DIR / "hashes.json"])
    if hits:
        raise RuntimeError(f"public gate-prep outputs failed privacy scan: {hits[:12]}")
    atomic_write_text(PUBLIC_DIR / "STATUS.txt", "Awaiting Codex gate preparation validation\n")
    print(json.dumps({"status": summary["status"], "budget": budget["estimated_cost_usd_no_cache"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
