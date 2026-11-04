"""Load the locked 240-case audit for authorized scoring. Labels never enter model input."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.hashing import sha256_file, sha256_text
from full_resource_sentiment_v1.io_utils import read_json, read_jsonl
from full_resource_sentiment_v1.paths import WORKSPACE
from full_resource_sentiment_v1.relevance_classify.evaluate import EXPECTED_HUMAN_COUNTS, EXPECTED_N
from full_resource_sentiment_v1.relevance_classify.preprocess import redact_one_way, reject_protected_outcomes
from full_resource_sentiment_v1.relevance_classify.spec import GATE_CLASSIFIER_RUN_ID

LOCKED_LABELS = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_labels.csv"
LOCKED_STATS = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_statistics.json"
SAMPLE_DIR = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28"
PACKET_A = SAMPLE_DIR / "reviewer_A_packet.jsonl"
PACKET_B = SAMPLE_DIR / "reviewer_B_packet.jsonl"
PRIVATE_KEY = SAMPLE_DIR / "private_id_key.jsonl"
EXCLUDED_PROTECTED = SAMPLE_DIR / "excluded_protected_ids.json"
SELECTED_IDS = SAMPLE_DIR / "selected_ids.json"

LOCKED_LABELS_SHA256 = "07a52b4682844b88cd175da55bc342661310e19eb1a720df6a1ecfc44c9df45e"
LOCKED_STATS_SHA256 = "e61173250bc3270b312a01058b6f8cdf4ecaf18aac91429252fdf23d31dcd1c7"


class AuditInputError(ValueError):
    """Raised when the locked 240-case inputs fail an integrity check."""


def _counts(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "")
        out[key] = out.get(key, 0) + 1
    return out


def load_locked_labels() -> list[dict[str, str]]:
    digest = sha256_file(LOCKED_LABELS)
    if digest != LOCKED_LABELS_SHA256:
        raise AuditInputError(f"locked final_labels.csv hash changed: {digest}")
    with LOCKED_LABELS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_N:
        raise AuditInputError(f"expected {EXPECTED_N} labels, found {len(rows)}")
    codes = [row["case_code"] for row in rows]
    if len(set(codes)) != EXPECTED_N:
        raise AuditInputError("duplicate case_code in locked labels")
    counts = _counts(rows, "final_label")
    for label, expected in EXPECTED_HUMAN_COUNTS.items():
        if counts.get(label, 0) != expected:
            raise AuditInputError(f"locked {label} count {counts.get(label, 0)} != {expected}")
    return rows


def load_private_mapping() -> list[dict[str, Any]]:
    rows = read_jsonl(PRIVATE_KEY)
    if len(rows) != EXPECTED_N:
        raise AuditInputError(f"private key n={len(rows)}")
    codes = [str(row["case_code"]) for row in rows]
    rids = [str(row["reddit_id"]) for row in rows]
    if len(set(codes)) != EXPECTED_N or len(set(rids)) != EXPECTED_N:
        raise AuditInputError("private key is not 1:1 case_code/reddit_id")
    if any(row.get("protected_membership") for row in rows):
        raise AuditInputError("protected membership leaked into the 240-case key")
    return rows


def load_classifier_inputs() -> dict[str, Any]:
    """Title/body by case_code. Human labels are returned separately and never mixed in."""
    labels = load_locked_labels()
    mapping = load_private_mapping()
    packets = read_jsonl(PACKET_A)
    packet_b = read_jsonl(PACKET_B)
    if len(packets) != EXPECTED_N or len(packet_b) != EXPECTED_N:
        raise AuditInputError("reviewer packets are not 240 rows")
    map_by_code = {str(row["case_code"]): row for row in mapping}
    packet_by_code = {str(row["case_code"]): row for row in packets}
    label_codes = {row["case_code"] for row in labels}
    if set(packet_by_code) != label_codes or set(map_by_code) != label_codes:
        raise AuditInputError("case_code sets diverge across labels/packets/key")
    excluded = read_json(EXCLUDED_PROTECTED)
    protected_ids = set(excluded.get("union_ids") or [])
    selected = read_json(SELECTED_IDS)

    records: list[dict[str, Any]] = []
    for label in labels:
        code = label["case_code"]
        packet = packet_by_code[code]
        reject_protected_outcomes(packet)
        if packet.get("relevance"):
            # Packets may contain empty label slots from the human form. Nonempty
            # values must never be copied onto the model record.
            pass
        key = map_by_code[code]
        rid = str(key["reddit_id"])
        if rid in protected_ids:
            raise AuditInputError(f"{code} maps to a protected reddit_id")
        title = "" if packet.get("title") is None else str(packet.get("title") or "")
        body = "" if packet.get("body") is None else str(packet.get("body") or "")
        title_redacted, _ = redact_one_way(title)
        body_redacted, _ = redact_one_way(body)
        if sha256_text(title) != key.get("packet_title_sha256"):
            raise AuditInputError(f"{code} packet title hash mismatch")
        if sha256_text(body) != key.get("packet_body_sha256"):
            raise AuditInputError(f"{code} packet body hash mismatch")
        records.append(
            {
                "synthetic_id": code,
                "case_code": code,
                "reddit_id": rid,
                "title": title_redacted,
                "body": body_redacted,
                "stratum": label["stratum"],
                "capture_id": "2026-10-22-main",
                "classifier_run_id": GATE_CLASSIFIER_RUN_ID,
                "protected_membership": False,
            }
        )
    records.sort(key=lambda row: row["case_code"])
    return {
        "records": records,
        "labels": labels,
        "mapping": mapping,
        "n": len(records),
        "label_counts": _counts(labels, "final_label"),
        "stratum_counts": _counts(labels, "stratum"),
        "protected_union_n": len(protected_ids),
        "selected_n": len(selected.get("all_ids_sorted") or []),
        "packet_a_sha256": sha256_file(PACKET_A),
        "packet_b_sha256": sha256_file(PACKET_B),
        "labels_sha256": LOCKED_LABELS_SHA256,
        "stats_sha256": sha256_file(LOCKED_STATS),
        "stats_expected_sha256": LOCKED_STATS_SHA256,
    }
