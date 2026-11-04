"""Post-run scientific quality checks. Never invent labels or hide failures."""

from __future__ import annotations

from collections import Counter
from typing import Any

from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.evaluate import _is_selected
from full_resource_sentiment_v1.relevance_classify.preprocess import evidence_in_title_or_body
from full_resource_sentiment_v1.relevance_classify.spec import LABELS, POSITIVE_LABEL, active_prompt_sha256, schema_sha256

TERMINAL_OK = frozenset({"ok", "skipped_deterministic"})


def quality_report(
    *,
    expected_ids: list[str],
    rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]] | None = None,
    id_field: str = "synthetic_id",
) -> dict[str, Any]:
    expected = [str(x) for x in expected_ids]
    expected_set = set(expected)
    found: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        key = str(row.get(id_field) or "")
        if not key:
            continue
        if key in found:
            duplicates.append(key)
        found[key] = row
    missing = sorted(expected_set - set(found))
    extra = sorted(set(found) - expected_set)

    labels = Counter()
    statuses = Counter()
    selected = 0
    evidence_fail = 0
    residual_pii = 0
    errors = []
    unclear = []
    unassessable = []
    for key in expected:
        row = found.get(key)
        if row is None:
            continue
        status = str(row.get("response_status") or "")
        label = row.get("predicted_label")
        statuses[status] += 1
        if status in TERMINAL_OK and label in LABELS:
            labels[str(label)] += 1
        else:
            labels["error"] += 1
            errors.append(key)
        if _is_selected(row):
            selected += 1
        if label == "unclear":
            unclear.append(key)
        if label == "unassessable":
            unassessable.append(key)
        if row.get("residual_pii_suspected"):
            residual_pii += 1
        if records_by_id and label == POSITIVE_LABEL and status == "ok":
            rec = records_by_id.get(key) or {}
            if not evidence_in_title_or_body(
                row.get("evidence_phrase"),
                str(rec.get("title") or ""),
                str(rec.get("body") or ""),
            ):
                evidence_fail += 1

    by_stratum: dict[str, dict[str, int]] = {}
    for key in expected:
        row = found.get(key) or {}
        rec = (records_by_id or {}).get(key) or row
        stratum = str(rec.get("stratum") or row.get("stratum") or "unspecified")
        block = by_stratum.setdefault(
            stratum,
            {"n": 0, "selected": 0, "relevant": 0, "not_relevant": 0, "unclear": 0, "unassessable": 0, "error": 0},
        )
        block["n"] += 1
        if _is_selected(row):
            block["selected"] += 1
        status = str(row.get("response_status") or "")
        label = row.get("predicted_label")
        if status in TERMINAL_OK and label in LABELS:
            block[str(label)] = block.get(str(label), 0) + 1
        else:
            block["error"] += 1

    public_hits = scan_obj({"labels": dict(labels), "statuses": dict(statuses), "by_stratum": by_stratum})
    return {
        "n_expected": len(expected),
        "n_rows": len(found),
        "complete_coverage": not missing and not extra and not duplicates and len(found) == len(expected),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_duplicates": len(duplicates),
        "missing_ids_head": missing[:25],
        "extra_ids_head": extra[:25],
        "duplicate_ids_head": duplicates[:25],
        "label_counts": dict(labels),
        "status_counts": dict(statuses),
        "n_selected": selected,
        "n_errors": len(errors),
        "n_unclear": len(unclear),
        "n_unassessable": len(unassessable),
        "n_evidence_fail": evidence_fail,
        "n_residual_pii_suspected": residual_pii,
        "by_stratum": by_stratum,
        "prompt_sha256": active_prompt_sha256(),
        "schema_sha256": schema_sha256(),
        "public_scan_hits": public_hits,
        "selected_definition": "predicted_label==relevant and response_status==ok",
        "ai_predicted_not_human_confirmed": True,
    }


def fingerprint_files(paths: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, path in paths.items():
        if path is None:
            continue
        out[name] = sha256_file(path)
    return out
