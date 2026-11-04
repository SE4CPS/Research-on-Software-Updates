#!/usr/bin/env python3
"""Archive the 2026-10-29 second human review and re-score saved Luna predictions.

Does not overwrite the locked 2026-10-28 audit, does not call a model, and does
not read protected evaluation outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_jsonl, atomic_write_text, utcnow
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.io_utils import read_jsonl
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate
from full_resource_sentiment_v1.relevance_classify.gate_prepare import verify_frozen_fingerprints

SOURCE_A = Path("/Users/manumathewjiss/Documents/Codex/2026-10-28/ple/outputs/reviewer_A_packet_completed.csv")
SOURCE_B = Path("/Users/manumathewjiss/Documents/Codex/2026-10-28/ple/outputs/reviewer_B_packet_completed.csv")
LOCKED_LABELS = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_labels.csv"
LOCKED_LABELS_SHA256 = "07a52b4682844b88cd175da55bc342661310e19eb1a720df6a1ecfc44c9df45e"
PACKET_A = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "reviewer_A_packet.jsonl"
PACKET_B = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "reviewer_B_packet.jsonl"
PREDICTIONS = WORKSPACE / "restricted" / "ai_relevance_campaign_2026-10-29" / "gate" / "predictions.jsonl"
RESTRICTED_DIR = WORKSPACE / "restricted" / "relevance_audit_second_review_2026-10-29"
PUBLIC_DIR = WORKSPACE / "results" / "relevance_second_review_2026-10-29"

ALLOWED_LABELS = {"relevant", "not_relevant", "unclear", "unassessable"}
PACKET_FIELDS = ("case_code", "title", "body", "relevance", "evidence_phrase", "uncertainty_note")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def norm_text(value: str | None) -> str:
    return "" if value is None else str(value).replace("\r\n", "\n").replace("\r", "\n")


def clip(value: str | None, n: int = 180) -> str:
    return " ".join((value or "").split())[:n]


def evidence_exact(ev: str, title: str, body: str) -> bool:
    ev = (ev or "").strip()
    if not ev:
        return False
    return ev in (title or "") or ev in (body or "")


def fn_theme(human_ev: str, unc: str) -> str:
    blob = f"{human_ev} {unc}".lower()
    if any(w in blob for w in ("linux", "dual boot", "windows 10", "operating-system", "os change", "veda")):
        return "os_migration_or_dual_boot"
    if any(w in blob for w in ("launched", "launch", "beta", "pilot", "store", "marketplace", "submitted", "release", "packaged", "installer", "try", "closed test")):
        return "first_launch_or_distribution"
    if "version" in blob or "kernel" in blob:
        return "named_versions_without_update_verb"
    if "update" in blob or "git pull" in blob:
        return "update_language"
    return "other_shipping_or_tooling"


def fp_theme(reason: str, ev: str) -> str:
    blob = f"{reason} {ev}".lower()
    if "sql" in blob or "update actually worked" in blob or "matched zero row" in blob:
        return "database_update_verb"
    if "price" in blob:
        return "pricing_not_software_change"
    if "router" in blob or "unifi" in blob or "edgerouter" in blob or "hardware revision" in blob or "board" in blob:
        return "hardware_or_network_gear"
    if "open router" in blob or "openrouter" in blob:
        return "vendor_or_api_switch"
    if "certificate" in blob or "ssl" in blob:
        return "certificate_or_ops_rotation"
    if "vulnerability" in blob or "cve" in blob or "1.26.0" in blob:
        return "advisory_names_a_version"
    if "latest updates installed" in blob:
        return "system_status_not_the_update"
    if "duckdns" in blob and "ip" in blob:
        return "ip_refresh_not_software_version"
    if "model got updated" in blob or "grok" in blob:
        return "unconfirmed_model_change"
    return "other_update_adjacent"


def main() -> int:
    if sha256_file(LOCKED_LABELS) != LOCKED_LABELS_SHA256:
        raise SystemExit("locked final_labels.csv hash changed; refusing to proceed")
    frozen = verify_frozen_fingerprints()
    rows_a = read_csv(SOURCE_A)
    rows_b = read_csv(SOURCE_B)
    if list(rows_a[0].keys()) != list(PACKET_FIELDS) or list(rows_b[0].keys()) != list(PACKET_FIELDS):
        raise SystemExit("unexpected CSV columns")
    if len(rows_a) != 240 or len(rows_b) != 240:
        raise SystemExit("expected 240 rows in each CSV")
    map_a = {row["case_code"]: row for row in rows_a}
    map_b = {row["case_code"]: row for row in rows_b}
    if len(map_a) != 240 or len(map_b) != 240:
        raise SystemExit("duplicate case_code in a completed packet")
    if set(map_a) != set(map_b):
        raise SystemExit("reviewer A/B case_code sets differ")

    packet_a = {row["case_code"]: row for row in read_jsonl(PACKET_A)}
    packet_b = {row["case_code"]: row for row in read_jsonl(PACKET_B)}
    if set(packet_a) != set(map_a) or set(packet_b) != set(map_b):
        raise SystemExit("completed CSVs do not match original packet case codes")

    text_mismatch = 0
    label_disagree = []
    for code in sorted(map_a):
        a_row, b_row = map_a[code], map_b[code]
        if a_row["relevance"].strip() != b_row["relevance"].strip():
            label_disagree.append(code)
        if a_row["relevance"].strip() not in ALLOWED_LABELS:
            raise SystemExit(f"invalid A label {code}")
        if b_row["relevance"].strip() not in ALLOWED_LABELS:
            raise SystemExit(f"invalid B label {code}")
        if norm_text(a_row["title"]) != norm_text(packet_a[code].get("title") or ""):
            text_mismatch += 1
        if norm_text(a_row["body"]) != norm_text(packet_a[code].get("body") or ""):
            text_mismatch += 1
        if norm_text(b_row["title"]) != norm_text(packet_b[code].get("title") or ""):
            text_mismatch += 1
        if norm_text(b_row["body"]) != norm_text(packet_b[code].get("body") or ""):
            text_mismatch += 1
        if norm_text(a_row["title"]) != norm_text(b_row["title"]) or norm_text(a_row["body"]) != norm_text(b_row["body"]):
            text_mismatch += 1
    if label_disagree:
        raise SystemExit(f"label disagreement: {label_disagree[:12]}")
    if text_mismatch:
        raise SystemExit(f"title/body mismatch vs original packets: {text_mismatch}")

    with LOCKED_LABELS.open(newline="", encoding="utf-8") as handle:
        locked_rows = list(csv.DictReader(handle))
    locked = {row["case_code"]: row for row in locked_rows}
    if set(locked) != set(map_a):
        raise SystemExit("second-review codes do not match locked audit")

    counts = Counter(row["relevance"].strip() for row in rows_a)
    if counts != {"relevant": 73, "not_relevant": 166, "unassessable": 1}:
        raise SystemExit(f"unexpected second-review counts {dict(counts)}")

    evidence_stats = {"relevant_exact": 0, "relevant_paraphrase": 0, "relevant_empty": 0}
    for row in rows_a:
        if row["relevance"].strip() != "relevant":
            continue
        ev = (row.get("evidence_phrase") or "").strip()
        if not ev:
            evidence_stats["relevant_empty"] += 1
        elif evidence_exact(ev, row["title"], row["body"]):
            evidence_stats["relevant_exact"] += 1
        else:
            evidence_stats["relevant_paraphrase"] += 1

    change_rows = []
    for code in sorted(map_a):
        old = locked[code]["final_label"]
        new = map_a[code]["relevance"].strip()
        if old == new:
            continue
        change_rows.append(
            {
                "case_code": code,
                "stratum": locked[code]["stratum"],
                "original_final_label": old,
                "second_review_label": new,
                "reviewer_A_evidence": map_a[code].get("evidence_phrase") or "",
                "reviewer_A_uncertainty": map_a[code].get("uncertainty_note") or "",
                "adjudicated": False,
                "agreement": "A_and_B_identical_labels",
            }
        )

    new_labels = [
        {
            "case_code": code,
            "stratum": locked[code]["stratum"],
            "reviewer_A_label": map_a[code]["relevance"].strip(),
            "reviewer_B_label": map_b[code]["relevance"].strip(),
            "final_label": map_a[code]["relevance"].strip(),
            "adjudicated": "False",
            "agreement": "unanimous",
        }
        for code in sorted(map_a)
    ]
    predictions = read_jsonl(PREDICTIONS)
    evaluation = evaluate_gate(
        [
            {"case_code": row["case_code"], "stratum": row["stratum"], "final_label": row["final_label"]}
            for row in new_labels
        ],
        predictions,
        require_complete=True,
    )
    evaluation["label_set"] = "second_review_2026-10-29"
    evaluation["retrospective_development_evaluation"] = True
    evaluation["independent_validation"] = False
    evaluation["used_real_reddit_text"] = False
    evaluation["used_protected_outcomes"] = False
    metrics = evaluation["metrics"]

    pred_by = {row["case_code"]: row for row in predictions}
    fp_rows = []
    for code in metrics["false_positive_case_codes"]:
        pred = pred_by[code]
        fp_rows.append(
            {
                "case_code": code,
                "stratum": locked[code]["stratum"],
                "human_v2": "not_relevant",
                "ai_label": pred.get("predicted_label"),
                "ai_status": pred.get("response_status"),
                "ai_evidence": pred.get("evidence_phrase"),
                "ai_reason": pred.get("decision_reason"),
                "human_evidence": map_a[code].get("evidence_phrase"),
                "theme": fp_theme(str(pred.get("decision_reason") or ""), str(pred.get("evidence_phrase") or "")),
            }
        )
    fn_rows = []
    for code in metrics["false_negative_case_codes"]:
        pred = pred_by[code]
        human_ev = map_a[code].get("evidence_phrase") or ""
        unc = map_a[code].get("uncertainty_note") or ""
        fn_rows.append(
            {
                "case_code": code,
                "stratum": locked[code]["stratum"],
                "human_v1": locked[code]["final_label"],
                "human_v2": "relevant",
                "ai_label": pred.get("predicted_label"),
                "ai_status": pred.get("response_status"),
                "ai_reason": pred.get("decision_reason"),
                "human_evidence": human_ev,
                "human_uncertainty": unc,
                "theme": fn_theme(human_ev, unc),
            }
        )

    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    dest_a = RESTRICTED_DIR / "reviewer_A_packet_completed.csv"
    dest_b = RESTRICTED_DIR / "reviewer_B_packet_completed.csv"
    shutil.copy2(SOURCE_A, dest_a)
    shutil.copy2(SOURCE_B, dest_b)

    with (RESTRICTED_DIR / "final_labels_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_code",
                "stratum",
                "reviewer_A_label",
                "reviewer_B_label",
                "final_label",
                "adjudicated",
                "agreement",
            ],
        )
        writer.writeheader()
        writer.writerows(new_labels)

    atomic_write_jsonl(RESTRICTED_DIR / "label_changes_vs_locked_v1.jsonl", change_rows)
    atomic_write_jsonl(RESTRICTED_DIR / "disagreement_false_positives.jsonl", fp_rows)
    atomic_write_jsonl(RESTRICTED_DIR / "disagreement_false_negatives.jsonl", fn_rows)
    atomic_write_json(RESTRICTED_DIR / "evaluation_vs_saved_luna.json", evaluation)

    provenance = {
        "created_utc": utcnow(),
        "label_set_id": "relevance_audit_second_review_2026-10-29",
        "status": "human_second_review_unanimous_not_independent_validation",
        "source_a": str(SOURCE_A),
        "source_b": str(SOURCE_B),
        "source_a_sha256": sha256_file(SOURCE_A),
        "source_b_sha256": sha256_file(SOURCE_B),
        "archived_a_sha256": sha256_file(dest_a),
        "archived_b_sha256": sha256_file(dest_b),
        "locked_v1_path": str(LOCKED_LABELS),
        "locked_v1_sha256": LOCKED_LABELS_SHA256,
        "locked_v1_unmodified": sha256_file(LOCKED_LABELS) == LOCKED_LABELS_SHA256,
        "original_packets_text_match": True,
        "reviewer_label_agreement": "240/240",
        "adjudication_needed": False,
        "n": 240,
        "counts": dict(counts),
        "n_changed_from_locked_v1": len(change_rows),
        "change_types": dict(Counter(f"{row['original_final_label']}->{row['second_review_label']}" for row in change_rows)),
        "rubric": "Same four labels as LABEL_GUIDE. Second review is a more careful re-application, not a separately published instrument. Operational addendum documents observed inclusion of OS migrations and first launches/releases.",
        "protected_outcomes_read": False,
        "overwrote_locked_v1": False,
        "frozen_input_hashes": frozen,
    }
    atomic_write_json(RESTRICTED_DIR / "provenance.json", provenance)

    public = {
        "label_set_id": "relevance_audit_second_review_2026-10-29",
        "n": 240,
        "counts": dict(counts),
        "reviewer_agreement": "240/240",
        "n_changed_from_locked_v1": len(change_rows),
        "change_types": provenance["change_types"],
        "locked_v1_unmodified": True,
        "retrospective_metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "precision_wilson_95": metrics["precision_wilson_95"],
            "recall_wilson_95": metrics["recall_wilson_95"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "true_negatives": metrics["true_negatives"],
            "model_unclear": metrics["model_unclear"],
            "model_unassessable": metrics["model_unassessable"],
            "model_errors": metrics["model_errors"],
            "confusion_human_by_prediction": metrics["confusion_human_by_prediction"],
            "by_stratum": metrics["by_stratum"],
            "false_positive_case_codes": metrics["false_positive_case_codes"],
            "false_negative_case_codes": metrics["false_negative_case_codes"],
        },
        "gate_retrospective": {
            "precision_min": 0.75,
            "recall_min": 0.80,
            "precision_ok": bool(evaluation["gate"]["precision_ok"]),
            "recall_ok": bool(evaluation["gate"]["recall_ok"]),
            "passed": bool(evaluation["gate"]["passed"]),
            "not_independent_validation": True,
        },
        "fn_themes": dict(Counter(row["theme"] for row in fn_rows)),
        "fp_themes": dict(Counter(row["theme"] for row in fp_rows)),
        "evidence_stats_reviewer_A_relevant": evidence_stats,
        "ai_predicted_not_human_confirmed": True,
        "paid_model_calls": 0,
    }
    hits = scan_obj(
        {
            "counts": public["counts"],
            "metrics": {k: public["retrospective_metrics"][k] for k in ("precision", "recall", "f1", "true_positives", "false_positives")},
            "gate": public["gate_retrospective"],
        }
    )
    if hits:
        raise SystemExit(f"public summary leaked markers: {hits[:8]}")
    atomic_write_json(PUBLIC_DIR / "public_summary.json", public)
    atomic_write_json(PUBLIC_DIR / "provenance_public.json", {k: provenance[k] for k in provenance if k not in {"source_a", "source_b"}})
    print(f"wrote {RESTRICTED_DIR}")
    print(f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} gate={evaluation['gate']['passed']}")
    print(f"locked_v1_unmodified={provenance['locked_v1_unmodified']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
