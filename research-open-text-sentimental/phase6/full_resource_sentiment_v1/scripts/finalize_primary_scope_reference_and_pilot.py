#!/usr/bin/env python3
"""Finalize the primary-scope 240-case reference, compare saved classifiers, and
draw a blinded unseen validation pilot.

Does not overwrite historical audits, does not classify 37,596 posts, does not
retune models, and does not read protected evaluation outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_jsonl, atomic_write_text, utcnow
from full_resource_sentiment_v1.hashing import sha256_file, sha256_text
from full_resource_sentiment_v1.io_utils import read_json, read_jsonl
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_audit import (
    PACKET_FIELDS,
    assert_packets_have_zero_handles,
    collect_selected_text,
    load_primary_frame,
    packet_csv_text,
    packet_is_blinded,
    redact_pii,
)
from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate
from full_resource_sentiment_v1.relevance_classify.gate_prepare import verify_frozen_fingerprints
from full_resource_sentiment_v1.relevance_classify.preprocess import reject_protected_outcomes
from full_resource_sentiment_v1.relevance_classify.scope_primary import SCOPE_VERSION, apply_second_pass
from full_resource_sentiment_v1.relevance_classify.spec import (
    PROMPT_VERSION_V3,
    PROPOSED_SPEND_CEILING_USD,
    prompt_sha256,
    prompt_v3_sha256,
    schema_sha256,
)

SOURCE_A = Path("/Users/manumathewjiss/Documents/Codex/2026-10-28/ple/outputs/reviewer_A_packet_primary_scope_review_2026-10-29.csv")
SOURCE_B = Path("/Users/manumathewjiss/Documents/Codex/2026-10-28/ple/outputs/reviewer_B_packet_primary_scope_review_2026-10-29.csv")
EXPECTED_A_SHA = "da5d58577e7b0a5355cc30f854027921ec523d109f9008ab27e19b7e329f5381"
EXPECTED_B_SHA = "d99f767e170c70a008a90bbee7e27ef80817202b25bee2de640320fb1f562817"
LOCKED_LABELS = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_labels.csv"
LOCKED_SHA = "07a52b4682844b88cd175da55bc342661310e19eb1a720df6a1ecfc44c9df45e"
PACKET_A = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "reviewer_A_packet.jsonl"
PACKET_B = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "reviewer_B_packet.jsonl"
PRIVATE_KEY = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "private_id_key.jsonl"
EXCLUDED_PROTECTED = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "excluded_protected_ids.json"
V1_PRED = WORKSPACE / "restricted" / "ai_relevance_campaign_2026-10-29" / "gate" / "predictions.jsonl"
V3_PRED = WORKSPACE / "restricted" / "ai_relevance_v3_2026-10-29" / "gate" / "predictions.jsonl"
TRAINING_EXAMPLES = WORKSPACE / "reviewer_training" / "03_REAL_REDDIT_EXAMPLES.md"
TRAINING_EXERCISES = WORKSPACE / "reviewer_training" / "06_REVIEWER_TRAINING_EXERCISES.md"

REF_DIR = WORKSPACE / "restricted" / "relevance_audit_primary_scope_adjudicated_2026-10-29"
REF_PUBLIC = WORKSPACE / "results" / "relevance_audit_primary_scope_adjudicated_2026-10-29"
PILOT_DIR = WORKSPACE / "restricted" / "independent_validation_pilot_2026-10-29"
PILOT_PUBLIC = WORKSPACE / "results" / "independent_validation_pilot_2026-10-29"

ADDED_RELEVANT = (
    "C-121241",
    "C-208451",
    "C-233878",
    "C-491004",
    "C-513597",
    "C-831673",
    "C-832942",
)
REMOVED_RELEVANT = (
    "C-287309",
    "C-399641",
    "C-487265",
    "C-585657",
    "C-619037",
    "C-896971",
    "C-924103",
)
ADJUDICATION_14 = {
    "C-121241": ("relevant", "Beta app already existed; an update added functions. Not inclusion merely because of a first launch."),
    "C-208451": ("relevant", "Failed Ubuntu upgrade is central to the author's explanation for preferring Debian."),
    "C-233878": ("relevant", "Home Assistant update caused loss of previously working remote access."),
    "C-491004": ("relevant", "Breaking changes in existing software projects are discussed."),
    "C-513597": ("relevant", "Change to an existing Apple TV app, with a new Home Assistant integration. The existing-app change is the relevant part."),
    "C-831673": ("relevant", "ComfyUI update caused previously working functionality to fail."),
    "C-832942": ("relevant", "Updating Corsair Hub firmware made it work with Linux tools; firmware is software on the device."),
    "C-287309": ("not_relevant", "Safari crashes despite updates; no update is identified as causing the crashes."),
    "C-585657": ("not_relevant", "The author only speculates about an unspecified update."),
    "C-399641": ("not_relevant", "Ubuntu-to-Debian migration without a particular update."),
    "C-924103": ("not_relevant", "Request for notifications of future Samsung updates; no concrete change is discussed."),
    "C-487265": ("not_relevant", "Feature announcement without an update to existing software in the saved post."),
    "C-896971": ("not_relevant", "First release of model files, not an update to existing software."),
    "C-619037": (
        "not_relevant",
        "Saved title announces a DeepSeek model update to revision 1017, but there is no saved body explaining a change to an app, OS, plugin, or firmware. AI model-weight/revision releases as a class remain an unresolved professor policy boundary; this case is not_relevant under the current scope.",
    ),
}
ALLOWED = {"relevant", "not_relevant", "unclear", "unassessable"}
PILOT_SEED = 202610291
PILOT_N_PER_PERIOD = 120


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def evidence_kind(ev: str, title: str, body: str, label: str) -> str:
    text = (ev or "").strip()
    if not text:
        return "blank"
    if label == "unassessable":
        return "unassessable_note"
    if text in (title or "") or text in (body or ""):
        return "exact_quote"
    return "paraphrase_or_explanation"


def training_reddit_ids() -> set[str]:
    found: set[str] = set()
    heading = re.compile(r"^## \d+\. `([0-9A-Za-z]+)`", re.M)
    for path in (TRAINING_EXAMPLES, TRAINING_EXERCISES):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found.update(heading.findall(text))
    return found


def public_metrics(evaluation: dict) -> dict:
    m = evaluation["metrics"]
    g = evaluation["gate"]
    return {
        "true_positives": m["true_positives"],
        "false_positives": m["false_positives"],
        "false_negatives": m["false_negatives"],
        "true_negatives": m["true_negatives"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "precision_wilson_95": m["precision_wilson_95"],
        "recall_wilson_95": m["recall_wilson_95"],
        "model_errors": m["model_errors"],
        "model_unclear": m["model_unclear"],
        "model_unassessable": m["model_unassessable"],
        "false_positive_case_codes": m["false_positive_case_codes"],
        "false_negative_case_codes": m["false_negative_case_codes"],
        "gate_precision_ok": g["precision_ok"],
        "gate_recall_ok": g["recall_ok"],
        "gate_passed_if_applied_to_development": g["passed"],
        "development_only": True,
        "independent_validation": False,
    }


def error_codes(predictions: list[dict]) -> list[str]:
    out = []
    for row in predictions:
        status = row.get("response_status")
        if status not in {"ok", "skipped_deterministic"}:
            out.append(str(row.get("case_code") or row.get("synthetic_id")))
    return sorted(x for x in out if x)


def finalize_reference() -> dict:
    if sha256_file(LOCKED_LABELS) != LOCKED_SHA:
        raise SystemExit("locked final_labels.csv hash changed; refusing")
    frozen = verify_frozen_fingerprints()
    if sha256_file(SOURCE_A) != EXPECTED_A_SHA or sha256_file(SOURCE_B) != EXPECTED_B_SHA:
        raise SystemExit("primary-scope CSV hashes do not match the intake record")
    rows_a = read_csv(SOURCE_A)
    rows_b = read_csv(SOURCE_B)
    if list(rows_a[0].keys()) != list(PACKET_FIELDS) or list(rows_b[0].keys()) != list(PACKET_FIELDS):
        raise SystemExit("unexpected CSV columns")
    if len(rows_a) != 240 or len(rows_b) != 240:
        raise SystemExit("expected 240 rows")
    map_a = {row["case_code"]: row for row in rows_a}
    map_b = {row["case_code"]: row for row in rows_b}
    if set(map_a) != set(map_b) or len(map_a) != 240:
        raise SystemExit("A/B case codes diverge")
    packet_a = {row["case_code"]: row for row in read_jsonl(PACKET_A)}
    packet_b = {row["case_code"]: row for row in read_jsonl(PACKET_B)}
    locked = read_csv(LOCKED_LABELS)
    locked_by = {row["case_code"]: row for row in locked}
    if set(locked_by) != set(map_a):
        raise SystemExit("CSV codes do not match locked audit")

    text_mismatch = 0
    label_disagree = []
    evidence_rows = []
    for code in sorted(map_a):
        a = map_a[code]
        b = map_b[code]
        pa = packet_a[code]
        pb = packet_b[code]
        reject_protected_outcomes(a)
        reject_protected_outcomes(b)
        if a["title"] != pa["title"] or a["body"] != pa["body"] or b["title"] != pb["title"] or b["body"] != pb["body"]:
            text_mismatch += 1
        if a["relevance"] not in ALLOWED or b["relevance"] not in ALLOWED:
            raise SystemExit(f"invalid label {code}")
        if a["relevance"] != b["relevance"]:
            label_disagree.append(code)
        evidence_rows.append(
            {
                "case_code": code,
                "final_label": a["relevance"],
                "reviewer_A_evidence_kind": evidence_kind(a["evidence_phrase"], a["title"], a["body"], a["relevance"]),
                "reviewer_B_evidence_kind": evidence_kind(b["evidence_phrase"], b["title"], b["body"], b["relevance"]),
                "reviewer_A_evidence_phrase": a["evidence_phrase"],
                "reviewer_B_evidence_phrase": b["evidence_phrase"],
                "reviewer_A_uncertainty_note": a.get("uncertainty_note") or "",
                "reviewer_B_uncertainty_note": b.get("uncertainty_note") or "",
            }
        )
    if text_mismatch:
        raise SystemExit(f"title/body mismatch vs original packets: {text_mismatch}")
    if label_disagree:
        raise SystemExit(f"A/B labels disagree: {label_disagree}")

    counts = Counter(row["relevance"] for row in rows_a)
    if counts != Counter({"relevant": 40, "not_relevant": 199, "unassessable": 1}):
        raise SystemExit(f"unexpected consensus counts {dict(counts)}")

    adjudication_rows = []
    consensus_vs_adjudication = []
    for code, (want, reason) in ADJUDICATION_14.items():
        observed = map_a[code]["relevance"]
        if observed != want:
            consensus_vs_adjudication.append({"case_code": code, "consensus": observed, "adjudicated": want})
        locked_label = locked_by[code]["final_label"]
        adjudication_rows.append(
            {
                "case_code": code,
                "locked_v1_label": locked_label,
                "reviewer_A_label": map_a[code]["relevance"],
                "reviewer_B_label": map_b[code]["relevance"],
                "adjudicated_label": want,
                "reason": reason,
                "professor_policy_unresolved": code == "C-619037",
            }
        )
        if set(ADDED_RELEVANT) | set(REMOVED_RELEVANT) != set(ADJUDICATION_14):
            raise SystemExit("14-case tables drifted")
        if code in ADDED_RELEVANT and (locked_label != "not_relevant" or want != "relevant"):
            raise SystemExit(f"{code} is not a not_relevant→relevant change")
        if code in REMOVED_RELEVANT and (locked_label != "relevant" or want != "not_relevant"):
            raise SystemExit(f"{code} is not a relevant→not_relevant change")

    if consensus_vs_adjudication:
        raise SystemExit(f"adjudication differs from A/B consensus: {consensus_vs_adjudication}")

    label_rows = []
    changes = []
    for locked_row in locked:
        code = locked_row["case_code"]
        final = map_a[code]["relevance"]
        changed = final != locked_row["final_label"]
        label_rows.append(
            {
                "case_code": code,
                "stratum": locked_row["stratum"],
                "reviewer_A_label": map_a[code]["relevance"],
                "reviewer_B_label": map_b[code]["relevance"],
                "final_label": final,
                "adjudicated": str(code in ADJUDICATION_14),
            }
        )
        if changed:
            changes.append(
                {
                    "case_code": code,
                    "stratum": locked_row["stratum"],
                    "locked_v1": locked_row["final_label"],
                    "primary_scope_adjudicated": final,
                }
            )
    if {row["case_code"] for row in changes} != set(ADJUDICATION_14):
        raise SystemExit("change set vs locked v1 is not exactly the 14 adjudicated cases")

    a_exact = sum(1 for row in evidence_rows if row["reviewer_A_evidence_kind"] == "exact_quote")
    b_exact = sum(1 for row in evidence_rows if row["reviewer_B_evidence_kind"] == "exact_quote")
    a_para = sum(1 for row in evidence_rows if row["reviewer_A_evidence_kind"] == "paraphrase_or_explanation")
    b_para = sum(1 for row in evidence_rows if row["reviewer_B_evidence_kind"] == "paraphrase_or_explanation")

    REF_DIR.mkdir(parents=True, exist_ok=True)
    REF_PUBLIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_A, REF_DIR / "reviewer_A_packet_primary_scope_review_2026-10-29.csv")
    shutil.copy2(SOURCE_B, REF_DIR / "reviewer_B_packet_primary_scope_review_2026-10-29.csv")
    write_csv(
        REF_DIR / "final_labels.csv",
        label_rows,
        ["case_code", "stratum", "reviewer_A_label", "reviewer_B_label", "final_label", "adjudicated"],
    )
    write_csv(
        REF_DIR / "evidence_documentation.csv",
        evidence_rows,
        [
            "case_code",
            "final_label",
            "reviewer_A_evidence_kind",
            "reviewer_B_evidence_kind",
            "reviewer_A_evidence_phrase",
            "reviewer_B_evidence_phrase",
            "reviewer_A_uncertainty_note",
            "reviewer_B_uncertainty_note",
        ],
    )
    write_csv(
        REF_DIR / "adjudication_14.csv",
        adjudication_rows,
        [
            "case_code",
            "locked_v1_label",
            "reviewer_A_label",
            "reviewer_B_label",
            "adjudicated_label",
            "reason",
            "professor_policy_unresolved",
        ],
    )
    atomic_write_jsonl(REF_DIR / "label_changes_vs_locked_v1.jsonl", changes)
    provenance = {
        "created_utc": utcnow(),
        "version": "primary_scope_adjudicated_v3ref_2026-10-29",
        "n": 240,
        "label_counts": dict(counts),
        "reviewer_process": {
            "reviewer_A_and_B_reviewed_separately": True,
            "some_discussion_occurred": True,
            "complete_independence_claimed": False,
            "discussion_note": (
                "Reviewers reviewed separately, with some discussion. File agreement is 240/240, "
                "but discussion may have influenced decisions. Do not describe these as fully independent ratings."
            ),
        },
        "evidence_quotes_verified_against_saved_packets": True,
        "reviewer_A_exact_quotes": a_exact,
        "reviewer_B_exact_quotes": b_exact,
        "reviewer_A_paraphrases_kept_as_explanations": a_para,
        "reviewer_B_paraphrases_kept_as_explanations": b_para,
        "source_A_sha256": EXPECTED_A_SHA,
        "source_B_sha256": EXPECTED_B_SHA,
        "locked_v1_sha256_unchanged": LOCKED_SHA,
        "second_review_archive_untouched": True,
        "adjudicated_label_equals_latest_reviewer_consensus": True,
        "differences_vs_latest_consensus": [],
        "n_changed_vs_locked_v1": len(changes),
        "professor_policy_unresolved": [
            {
                "topic": "whether AI model-weight or model-revision releases count as software updates",
                "this_case": "C-619037 labeled not_relevant under current scope",
                "status": "unresolved_pending_Professor_Berhe",
            }
        ],
        "frozen_fingerprints_verified": True,
        "used_protected_outcomes": False,
    }
    atomic_write_json(REF_DIR / "provenance.json", provenance)
    public = {
        "version": provenance["version"],
        "n": 240,
        "label_counts": dict(counts),
        "n_changed_vs_locked_v1": 14,
        "added_relevant": list(ADDED_RELEVANT),
        "removed_relevant": list(REMOVED_RELEVANT),
        "adjudicated_equals_latest_consensus": True,
        "complete_independence_claimed": False,
        "professor_policy_unresolved_n": 1,
        "used_protected_outcomes": False,
        "historical_archives_unchanged": True,
        "path": str(REF_DIR / "final_labels.csv"),
        "final_labels_sha256": sha256_file(REF_DIR / "final_labels.csv"),
    }
    hits = scan_obj(public)
    if hits:
        raise SystemExit(f"public reference leaked markers: {hits[:8]}")
    atomic_write_json(REF_PUBLIC / "public_summary.json", public)
    return {
        "labels": label_rows,
        "provenance": provenance,
        "public": public,
        "frozen": frozen,
        "evidence": evidence_rows,
        "changes": changes,
    }


def compare_classifiers(label_rows: list[dict]) -> dict:
    v1_rows = read_jsonl(V1_PRED)
    v3_rows = read_jsonl(V3_PRED)
    packets = {row["case_code"]: row for row in read_jsonl(PACKET_A)}
    v1_plus = []
    for row in v1_rows:
        rec = packets[row["case_code"]]
        v1_plus.append(apply_second_pass(row, rec.get("title") or "", rec.get("body") or ""))
    evals = {
        "luna_v1": evaluate_gate(label_rows, v1_rows, require_complete=True),
        "luna_v1_plus_exclusions": evaluate_gate(label_rows, v1_plus, require_complete=True),
        "frozen_v3_plus_second_pass": evaluate_gate(label_rows, v3_rows, require_complete=True),
    }
    payload = {
        "reference": "primary_scope_adjudicated_v3ref_2026-10-29",
        "development_only": True,
        "independent_validation": False,
        "used_second_review_labels": False,
        "used_protected_outcomes": False,
        "not_retuned": True,
        "v1_prompt_sha256": prompt_sha256(),
        "v3_prompt_sha256": prompt_v3_sha256(),
        "schema_sha256": schema_sha256(),
        "scope_version": SCOPE_VERSION,
        "methods": {},
    }
    for name, evaluation in evals.items():
        block = public_metrics(evaluation)
        preds = {"luna_v1": v1_rows, "luna_v1_plus_exclusions": v1_plus, "frozen_v3_plus_second_pass": v3_rows}[name]
        block["error_case_codes"] = error_codes(preds)
        payload["methods"][name] = block
    hits = scan_obj({k: v for k, v in payload.items() if k != "methods"} | {"gate_flags": {n: m["gate_passed_if_applied_to_development"] for n, m in payload["methods"].items()}})
    if hits:
        raise SystemExit(f"comparison leaked markers: {hits[:8]}")
    REF_PUBLIC.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REF_PUBLIC / "saved_classifier_comparison.json", payload)
    atomic_write_json(REF_DIR / "saved_classifier_comparison.json", payload)
    return payload


def draw_pilot() -> dict:
    frame = load_primary_frame()
    mapping = read_jsonl(PRIVATE_KEY)
    development_ids = {str(row["reddit_id"]) for row in mapping}
    protected = set(read_json(EXCLUDED_PROTECTED).get("union_ids") or [])
    trained = training_reddit_ids()
    trained_in_frame = trained & set(frame["posts"])
    pools = {"october_2026": [], "older": []}
    excluded_n = {"development_240": 0, "protected": 0, "reviewer_training": 0}
    for rid, meta in frame["posts"].items():
        if meta.get("reserved_membership") or meta.get("fresh_membership"):
            excluded_n["protected"] += 1
            continue
        if rid in protected:
            excluded_n["protected"] += 1
            continue
        if rid in development_ids:
            excluded_n["development_240"] += 1
            continue
        if rid in trained:
            excluded_n["reviewer_training"] += 1
            continue
        pools[meta["stratum"]].append(rid)
    for name in pools:
        pools[name].sort()
    rng = random.Random(PILOT_SEED)
    drawn: dict[str, list[str]] = {}
    inclusion: dict[str, float] = {}
    for name, ids in pools.items():
        take = min(PILOT_N_PER_PERIOD, len(ids))
        sampled = rng.sample(ids, take)
        sampled.sort()
        drawn[name] = sampled
        for rid in sampled:
            inclusion[rid] = take / len(ids)
    ordered = drawn["october_2026"] + drawn["older"]
    by_code = {rid: f"P-{index:06d}" for index, rid in enumerate(ordered, start=1)}
    texts = collect_selected_text(set(ordered))
    packets = []
    private_rows = []
    for rid in ordered:
        text = texts[rid]
        title_redacted, title_pii = redact_pii(text["title"])
        body_redacted, body_pii = redact_pii(text["body"])
        packet = {
            "case_code": by_code[rid],
            "title": title_redacted,
            "body": body_redacted,
            "relevance": "",
            "evidence_phrase": "",
            "uncertainty_note": "",
        }
        hits = packet_is_blinded(packet)
        if hits:
            raise SystemExit(f"packet blinding failure {by_code[rid]}: {hits}")
        packets.append(packet)
        meta = frame["posts"][rid]
        if meta.get("reserved_membership") or meta.get("fresh_membership"):
            raise SystemExit("protected ID leaked into pilot")
        private_rows.append(
            {
                "case_code": by_code[rid],
                "reddit_id": rid,
                "stratum": meta["stratum"],
                "created_month": meta.get("created_month"),
                "inclusion_probability": inclusion[rid],
                "validation_seed": PILOT_SEED,
                "title_sha256": sha256_text(text["title"]),
                "body_sha256": sha256_text(text["body"]),
                "packet_title_sha256": sha256_text(title_redacted),
                "packet_body_sha256": sha256_text(body_redacted),
                "pii_title": title_pii,
                "pii_body": body_pii,
                "protected_membership": False,
                "ai_predicted_not_human_confirmed": True,
                "human_label": "",
            }
        )
    assert_packets_have_zero_handles(packets, source="pilot")
    order_a = list(range(len(packets)))
    order_b = list(range(len(packets)))
    random.Random(PILOT_SEED + 1).shuffle(order_a)
    random.Random(PILOT_SEED + 2).shuffle(order_b)
    packet_a = [packets[i] for i in order_a]
    packet_b = [packets[i] for i in order_b]
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (PILOT_DIR / "validation").mkdir(parents=True, exist_ok=True)
    PILOT_PUBLIC.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(PILOT_DIR / "validation" / "reviewer_A_packet.jsonl", packet_a)
    atomic_write_jsonl(PILOT_DIR / "validation" / "reviewer_B_packet.jsonl", packet_b)
    atomic_write_jsonl(PILOT_DIR / "validation" / "private_id_key.jsonl", private_rows)
    (PILOT_DIR / "validation" / "reviewer_A_packet.csv").write_text(packet_csv_text(packet_a), encoding="utf-8")
    (PILOT_DIR / "validation" / "reviewer_B_packet.csv").write_text(packet_csv_text(packet_b), encoding="utf-8")
    n_oct_eligible = len(pools["october_2026"])
    n_old_eligible = len(pools["older"])
    design = {
        "design_id": "period_stratified_unseen_pilot_240_2026-10-29",
        "seed": PILOT_SEED,
        "n_drawn": len(packets),
        "n_per_period_target": PILOT_N_PER_PERIOD,
        "pools_available": {"october_2026": n_oct_eligible, "older": n_old_eligible},
        "drawn_counts": {name: len(ids) for name, ids in drawn.items()},
        "excluded_development_n": excluded_n["development_240"],
        "excluded_protected_n": len(protected),
        "excluded_reviewer_training_in_frame_n": len(trained_in_frame),
        "reviewer_training_ids_parsed_n": len(trained),
        "eligible_frame_after_exclusions": n_oct_eligible + n_old_eligible,
        "inclusion_probability_october": PILOT_N_PER_PERIOD / n_oct_eligible,
        "inclusion_probability_older": PILOT_N_PER_PERIOD / n_old_eligible,
        "ai_predictions_hidden_from_reviewers": True,
        "human_labels_invented": False,
        "full_frame_not_classified": True,
        "four_stratum_ai_design_not_used": True,
        "four_stratum_reason": (
            "The four-cell AI-selected design requires full-frame predictions. "
            "Full 37,596 classification is not authorized. This pilot samples by period only."
        ),
        "estimator": (
            "After dual review and adjudication, compute Horvitz-Thompson weighted precision "
            "and recall with weight=1/inclusion_probability. Period weights are "
            f"{n_oct_eligible}/{PILOT_N_PER_PERIOD} for October and "
            f"{n_old_eligible}/{PILOT_N_PER_PERIOD} for older. "
            "Use validation_weighted.score_weighted_validation. "
            "Do not report unweighted sample accuracy as full-frame accuracy."
        ),
        "prespecified_decision_rule": {
            "pass_if": "weighted_precision >= 0.75 AND weighted_recall >= 0.80 on assessable labels",
            "human_positive": "relevant",
            "human_negative": "not_relevant",
            "exclude_from_binary": ["unclear", "unassessable"],
            "selected": "frozen v3 predicted_label==relevant and response_status==ok after second pass",
            "do_not_use_development_240": True,
            "do_not_lower_thresholds": True,
            "do_not_retune_v3": True,
            "wilson_or_bootstrap": "report intervals; the pass rule uses the weighted point estimates, matching the development gate",
        },
        "v3_scoring_of_pilot": "optional after packets; predictions stay in the private key",
        "weighted_estimator_module": "full_resource_sentiment_v1.relevance_classify.validation_weighted",
    }
    hits = scan_obj(design)
    if hits:
        raise SystemExit(f"pilot design leaked markers: {hits[:8]}")
    atomic_write_json(PILOT_PUBLIC / "validation_design.json", design)
    atomic_write_json(PILOT_DIR / "validation_design.json", design)
    instructions = (
        "# Independent validation pilot — reviewer instructions\n\n"
        "Status: unseen-case labeling. Do not consult live Reddit, model outputs, "
        "the original 240 labels, or the other reviewer's sheet during your first pass.\n\n"
        "Review separately first. If you later discuss a hard case, record that the "
        "discussion happened. Do not claim complete independence for discussed cases.\n\n"
        "Use the frozen primary scope: relevant means an identifiable update, upgrade, "
        "patch, rollback, or version change to **existing** software/OS/app/plugin/firmware, "
        "or a consequence attributed to that change. Exclude first-ever launches without a "
        "prior version, SQL UPDATE, news/status updates, hardware-only, vendor switches, "
        "certificate/IP refresh, price changes, and advisory-only version mentions. "
        "Flag OS migrations without a named update; do not silently expand relevance. "
        "Whether AI model-weight/revision releases count in general is **not yet confirmed "
        "by Professor Berhe**; if a post is only a model-revision announcement with no app/"
        "OS/plugin/firmware change, prefer `not_relevant` or `unclear` and write a note.\n\n"
        "Copy an exact title/body quote into evidence_phrase when the label is relevant. "
        "If you can only paraphrase, put the paraphrase in uncertainty_note, not as a fake quote.\n"
    )
    atomic_write_text(PILOT_DIR / "validation" / "REVIEWER_INSTRUCTIONS.md", instructions)
    return {
        "design": design,
        "n": len(packets),
        "private_rows": private_rows,
        "packet_a": packet_a,
        "packet_b": packet_b,
        "records_for_model": [
            {
                "synthetic_id": row["reddit_id"],
                "case_code": row["case_code"],
                "reddit_id": row["reddit_id"],
                "title": texts[row["reddit_id"]]["title"],
                "body": texts[row["reddit_id"]]["body"],
                "stratum": row["stratum"],
                "protected_membership": False,
                "classifier_run_id": "ai-relevance-validation-pilot-240-v3-2026-10-29",
            }
            for row in private_rows
        ],
    }


def write_reports(reference: dict, comparison: dict, pilot: dict) -> None:
    methods = comparison["methods"]
    lines = [
        "# Primary-scope adjudicated 240-case reference — 2026-10-29",
        "",
        f"**Path:** `{REF_DIR / 'final_labels.csv'}`",
        f"**SHA-256:** `{sha256_file(REF_DIR / 'final_labels.csv')}`",
        "",
        "This is a new dated reference. Locked v1 (2026-10-28) and the second-review archive remain unchanged.",
        "",
        "## Counts",
        "",
        "- 40 relevant, 199 not_relevant, 1 unassessable (`C-639299`).",
        "- Reviewer A/B file agreement 240/240 on labels.",
        "- Reviewers reviewed separately, **with some discussion**. Complete independence is not claimed.",
        "- Adjudicated labels **equal** the latest 240-case reviewer consensus. Differences vs consensus: none.",
        "- 14 labels differ from locked v1: 7 added to relevant, 7 removed from relevant.",
        "",
        "## Fourteen adjudicated changes vs locked v1",
        "",
        "Added to relevant: " + ", ".join(ADDED_RELEVANT) + ".",
        "",
        "Removed from relevant: " + ", ".join(REMOVED_RELEVANT) + ".",
        "",
        "`C-619037` is `not_relevant` under the current scope. Whether AI model-weight/revision releases ever count remains an **unresolved Professor Berhe policy decision**.",
        "",
        "## Evidence",
        "",
        f"- Reviewer A exact title/body quotes: {reference['provenance']['reviewer_A_exact_quotes']}/240; paraphrases kept as explanations: {reference['provenance']['reviewer_A_paraphrases_kept_as_explanations']}.",
        f"- Reviewer B exact quotes: {reference['provenance']['reviewer_B_exact_quotes']}/240; paraphrases kept as explanations: {reference['provenance']['reviewer_B_paraphrases_kept_as_explanations']}.",
        "- Title/body matched the original saved packets for all 240 rows in both CSVs.",
        "",
        "## Classifier comparison (development only)",
        "",
        "| Method | TP | FP | FN | TN | Precision | Recall | F1 | P≥0.75 | R≥0.80 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    names = [
        ("luna_v1", "Luna v1"),
        ("luna_v1_plus_exclusions", "Luna v1 + exclusions"),
        ("frozen_v3_plus_second_pass", "Frozen v3 + second pass"),
    ]
    for key, label in names:
        m = methods[key]
        lines.append(
            f"| {label} | {m['true_positives']} | {m['false_positives']} | {m['false_negatives']} | {m['true_negatives']} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['gate_precision_ok']} | {m['gate_recall_ok']} |"
        )
    lines.extend(
        [
            "",
            "These numbers reuse the same 240 development cases. They are **not** independent validation.",
            "Models were not retuned.",
            "",
            "## Independent validation pilot",
            "",
            f"- Blinded packets: `{PILOT_DIR / 'validation'}`",
            f"- n={pilot['n']}, seed={PILOT_SEED}, 120 October + 120 older previously unseen posts.",
            "- Excluded original 240, 693 protected IDs, and reviewer-training example IDs found in the frame.",
            "- AI predictions are hidden from reviewers.",
            "- Full 37,596 classification was **not** run. The four-cell AI-selected design is deferred until that is authorized.",
            "",
            "## Go/no-go",
            "",
            "**No-go for full-frame classification.** Fresh human labels on the unseen pilot do not exist yet. The familiar 240 cannot be used as independent validation. Thresholds were not lowered.",
            "",
        ]
    )
    atomic_write_text(WORKSPACE / "PRIMARY_SCOPE_ADJUDICATED_REFERENCE_2026-10-29.md", "\n".join(lines) + "\n")
    brief = [
        "# Decision sheet — Professor Berhe and Codex — 2026-10-29",
        "",
        "## Already completed (no approval needed to archive)",
        "",
        f"1. New adjudicated reference: `{REF_DIR / 'final_labels.csv'}` (40/199/1). Historical audits untouched.",
        "2. Saved-classifier development comparison written. No retuning.",
        "3. Blinded unseen 240-case period-stratified validation packets prepared.",
        "",
        "## Outstanding human/scientific decisions",
        "",
        "1. **Confirm or revise** the unresolved policy: do AI model-weight/revision releases count as software updates? Current this-case decision: `C-619037` = not_relevant.",
        "2. **Label the unseen validation packets** (Reviewer A and B, separately first; record any discussion).",
        "3. After those labels exist, compute weighted P/R against frozen v3 predictions on the *same unseen sample*. Pass only if weighted P≥0.75 and R≥0.80.",
        "4. **Do not authorize 37,596 classification** until that unseen-sample rule passes, or until you explicitly override the protocol in writing.",
        "",
        "No professor approval is invented here.",
        "",
    ]
    atomic_write_text(WORKSPACE / "DECISION_SHEET_BERHE_CODEX_2026-10-29.md", "\n".join(brief))


def main() -> int:
    reference = finalize_reference()
    comparison = compare_classifiers(reference["labels"])
    pilot = draw_pilot()
    write_reports(reference, comparison, pilot)
    print(f"reference={REF_DIR / 'final_labels.csv'}")
    print(f"reference_sha256={sha256_file(REF_DIR / 'final_labels.csv')}")
    print(f"pilot_packets={PILOT_DIR / 'validation'}")
    print("adjudicated_equals_consensus=true")
    print("full_frame_classified=false")
    print("independent_validation_human_labels=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
