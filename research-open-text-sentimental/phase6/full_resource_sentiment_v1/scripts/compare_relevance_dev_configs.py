#!/usr/bin/env python3
"""Compare offline classifier configs on the locked 240 as development data.

Does not call a model, does not use second-review labels, and does not
overwrite the locked audit.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate
from full_resource_sentiment_v1.relevance_classify.scope_primary import apply_second_pass, SCOPE_VERSION

LOCKED = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_labels.csv"
LOCKED_SHA = "07a52b4682844b88cd175da55bc342661310e19eb1a720df6a1ecfc44c9df45e"
PRED = WORKSPACE / "restricted" / "ai_relevance_campaign_2026-10-29" / "gate" / "predictions.jsonl"
PACKET = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "reviewer_A_packet.jsonl"
OUT = WORKSPACE / "results" / "relevance_dev_config_compare_2026-10-29" / "comparison.json"


def public_metrics(evaluation: dict) -> dict:
    m = evaluation["metrics"]
    g = evaluation["gate"]
    return {
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "precision_wilson_95": m["precision_wilson_95"],
        "recall_wilson_95": m["recall_wilson_95"],
        "true_positives": m["true_positives"],
        "false_positives": m["false_positives"],
        "false_negatives": m["false_negatives"],
        "true_negatives": m["true_negatives"],
        "model_errors": m["model_errors"],
        "gate_passed": g["passed"],
        "precision_ok": g["precision_ok"],
        "recall_ok": g["recall_ok"],
        "development_only": True,
        "independent_validation": False,
    }


def main() -> int:
    if sha256_file(LOCKED) != LOCKED_SHA:
        raise SystemExit("locked labels hash changed")
    with LOCKED.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    predictions = [json.loads(line) for line in PRED.read_text(encoding="utf-8").splitlines() if line.strip()]
    packets = {}
    for line in PACKET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            packets[rec["case_code"]] = rec

    baseline = evaluate_gate(labels, predictions, require_complete=True)
    second = []
    demote_counts: dict[str, int] = {}
    for row in predictions:
        rec = packets[row["case_code"]]
        out = apply_second_pass(row, rec.get("title") or "", rec.get("body") or "")
        if out.get("second_pass_applied"):
            reason = str(out.get("second_pass_reason") or "other")
            demote_counts[reason] = demote_counts.get(reason, 0) + 1
        second.append(out)
    second_eval = evaluate_gate(labels, second, require_complete=True)

    payload = {
        "locked_v1_sha256": LOCKED_SHA,
        "used_second_review_labels": False,
        "scope_version": SCOPE_VERSION,
        "paid_model_calls": 0,
        "configs": {
            "A_saved_luna_v1": public_metrics(baseline),
            "B_luna_v1_plus_exclusion_second_pass": public_metrics(second_eval),
        },
        "second_pass_demote_counts": demote_counts,
        "note": (
            "Config B is a deterministic exclusion layer on saved v1 outputs. "
            "It is development-set analysis, not independent accuracy, and not a "
            "new model. Remaining false positives include posts the written scope "
            "would still count as updates even when v1 humans labeled not_relevant."
        ),
    }
    hits = scan_obj(payload)
    if hits:
        raise SystemExit(f"public comparison leaked markers: {hits[:8]}")
    atomic_write_json(OUT, payload)
    a = payload["configs"]["A_saved_luna_v1"]
    b = payload["configs"]["B_luna_v1_plus_exclusion_second_pass"]
    print(f"A P={a['precision']:.3f} R={a['recall']:.3f} gate={a['gate_passed']}")
    print(f"B P={b['precision']:.3f} R={b['recall']:.3f} gate={b['gate_passed']} demotes={demote_counts}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
