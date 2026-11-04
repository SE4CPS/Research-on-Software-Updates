"""Classify synthetic posts with checkpointing and a separate human-correction layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.broad_sentiment import atomic_write_jsonl, utcnow
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.io_utils import read_jsonl
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.preprocess import (
    ProtectedOutcomeSuppliedError,
    evidence_in_title_or_body,
    preprocess_record,
    record_id,
)
from full_resource_sentiment_v1.relevance_classify.providers import (
    ClassifierProvider,
    LiveInferenceForbidden,
    TransientProviderError,
)
from full_resource_sentiment_v1.relevance_classify.spec import (
    CLASSIFIER_RUN_ID,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    LABELS,
    LUNA_BATCH_INPUT_USD_PER_MILLION,
    LUNA_BATCH_OUTPUT_USD_PER_MILLION,
    LUNA_SYNC_INPUT_USD_PER_MILLION,
    LUNA_SYNC_OUTPUT_USD_PER_MILLION,
    MAX_DECISION_REASON_WORDS,
    MODEL_ALIAS,
    MODEL_OUTPUT_SCHEMA,
    NULL_EVIDENCE_LABELS,
    POSITIVE_LABEL,
    PROVIDER_NAME,
    PUBLIC_ROW_KEYS,
    REASONING_EFFORT,
    SCHEMA_VERSION,
    TEMPERATURE,
    active_prompt_sha256,
    active_prompt_version,
    active_system_prompt,
    estimate_tokens_from_chars,
    schema_sha256,
)

ACCEPTED_STATUSES = frozenset({"ok", "skipped_deterministic"})
TRANSIENT_RETRIES = 3

REQUIRED_MODEL_FIELDS = tuple(MODEL_OUTPUT_SCHEMA["required"])


class DuplicateIdError(ValueError):
    pass


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            for row in read_jsonl(path):
                self.rows[str(row["synthetic_id"])] = row

    def accepted(self, post_id: str) -> bool:
        row = self.rows.get(post_id)
        return bool(row and row.get("response_status") in ACCEPTED_STATUSES)

    def write_if_absent(self, row: dict[str, Any]) -> bool:
        post_id = str(row["synthetic_id"])
        if self.accepted(post_id):
            return False
        self.rows[post_id] = row
        self.flush()
        return True

    def flush(self) -> None:
        ordered = [self.rows[key] for key in sorted(self.rows)]
        atomic_write_jsonl(self.path, ordered)

    def sha256(self) -> str | None:
        if not self.path.exists():
            return None
        return sha256_file(self.path)


class HumanCorrectionStore:
    """Separate versioned layer. Never overwrites classifier rows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, Any]] = []
        if path.exists():
            self.rows = read_jsonl(path)

    def add(
        self,
        *,
        synthetic_id: str,
        classifier_run_id: str,
        corrected_label: str,
        reason: str,
    ) -> dict[str, Any]:
        if corrected_label not in LABELS:
            raise ValueError(f"invalid corrected_label {corrected_label}")
        row = {
            "synthetic_id": synthetic_id,
            "classifier_run_id": classifier_run_id,
            "corrected_label": corrected_label,
            "reason": reason,
            "created_utc": utcnow(),
            "layer": "human_correction_v1",
        }
        self.rows.append(row)
        atomic_write_jsonl(self.path, self.rows)
        return row


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in PUBLIC_ROW_KEYS if key in row}


def assert_no_raw_text(obj: Any) -> None:
    hits = scan_obj(obj)
    if hits:
        raise ValueError(f"raw text/PII markers in public object: {hits[:8]}")


def review_priority_for(status: str, label: str | None) -> tuple[str, str]:
    if status not in ACCEPTED_STATUSES:
        return "error", "invalid_or_failed_inference"
    if label == "relevant":
        return "high", "predicted_positive_triage_not_confidence"
    if label == "unclear":
        return "medium", "human_guide_unclear"
    if label == "unassessable":
        return "medium", "unassessable_needs_coverage_report"
    return "low", "predicted_negative"


def _empty_model_fields() -> dict[str, Any]:
    return {
        "predicted_label": None,
        "decision_reason": None,
        "evidence_phrase": None,
        "ambiguity_reasons": [],
    }


def _cost_est(
    input_tokens: int,
    output_tokens: int,
    *,
    input_rate: float = LUNA_BATCH_INPUT_USD_PER_MILLION,
    output_rate: float = LUNA_BATCH_OUTPUT_USD_PER_MILLION,
) -> float:
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def _attach_usage(row: dict[str, Any], response: dict[str, Any]) -> None:
    usage = response.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    if input_tokens is None and output_tokens is None:
        return
    in_tok = int(input_tokens or 0)
    out_tok = int(output_tokens or 0)
    row["input_tokens_actual"] = in_tok
    row["output_tokens_actual"] = out_tok
    pricing = str(response.get("pricing") or "batch")
    if pricing == "sync":
        row["cost_usd_actual"] = round(
            _cost_est(
                in_tok,
                out_tok,
                input_rate=LUNA_SYNC_INPUT_USD_PER_MILLION,
                output_rate=LUNA_SYNC_OUTPUT_USD_PER_MILLION,
            ),
            8,
        )
    else:
        row["cost_usd_actual"] = round(_cost_est(in_tok, out_tok), 8)
    if response.get("model"):
        row["resolved_model"] = response.get("model")
    row["pricing_schedule"] = pricing


def _base_row(prepared: dict[str, Any], *, status: str, retry_count: int, provider_name: str) -> dict[str, Any]:
    label = None
    priority, priority_reason = review_priority_for(status, label)
    prompt_digest = active_prompt_sha256()
    schema_digest = schema_sha256()
    output_tokens = 0 if status != "ok" else DEFAULT_MAX_OUTPUT_TOKENS
    return {
        "synthetic_id": prepared["synthetic_id"],
        "case_code": prepared.get("case_code"),
        "reddit_id": prepared.get("reddit_id"),
        "stratum": prepared.get("stratum"),
        "created_month": prepared.get("created_month"),
        "count_gap": prepared.get("count_gap"),
        "capture_id": prepared.get("capture_id") or "synthetic-not-a-capture",
        "classifier_run_id": prepared.get("classifier_run_id") or CLASSIFIER_RUN_ID,
        **_empty_model_fields(),
        "review_priority": priority,
        "review_priority_reason": priority_reason,
        "unassessable_basis": prepared.get("unassessable_basis"),
        "removal_notice_heuristic": bool(prepared.get("removal_notice_heuristic")),
        "title_sha256": prepared["title_sha256"],
        "body_sha256": prepared["body_sha256"],
        "model_input_sha256": prepared["model_input_sha256"],
        "redacted": prepared["redacted"],
        "redaction_types": prepared["redaction_types"],
        "residual_pii_suspected": prepared["residual_pii_suspected"],
        "residual_pii_types": prepared["residual_pii_types"],
        "truncated": prepared["truncated"],
        "title_chars": prepared["title_chars"],
        "body_chars": prepared["body_chars"],
        "submitted_chars": prepared["submitted_chars"],
        "comments_ignored": prepared["comments_ignored"],
        "comments_present": prepared["comments_present"],
        "protected_membership": prepared["protected_membership"],
        "model_id": MODEL_ALIAS,
        "model_snapshot": None,
        "resolved_model": None,
        "provider": provider_name,
        "prompt_version": active_prompt_version(),
        "prompt_sha256": prompt_digest,
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_digest,
        "temperature": TEMPERATURE,
        "reasoning_effort": REASONING_EFFORT,
        "response_id": None,
        "response_status": status,
        "retry_count": retry_count,
        "input_tokens_est": prepared["input_tokens_est"] + estimate_tokens_from_chars(len(active_system_prompt())),
        "output_tokens_est": output_tokens,
        "cost_usd_est": round(
            _cost_est(
                prepared["input_tokens_est"] + estimate_tokens_from_chars(len(active_system_prompt())),
                output_tokens,
            ),
            8,
        ),
        "created_utc": utcnow(),
    }


def _validate_model_payload(parsed: Any, prepared: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(parsed, dict):
        return "invalid_json", None
    allowed = set(REQUIRED_MODEL_FIELDS)
    keys = set(parsed.keys())
    if keys != allowed:
        return "schema_miss", None
    label = parsed.get("predicted_label")
    if label not in LABELS:
        return "schema_miss", None
    reason = parsed.get("decision_reason")
    if not isinstance(reason, str) or not reason.strip():
        return "schema_miss", None
    if len(reason.split()) > MAX_DECISION_REASON_WORDS:
        return "schema_miss", None
    reasons = parsed.get("ambiguity_reasons")
    if not isinstance(reasons, list) or any(
        item
        not in {
            "multiple_update_meanings",
            "implied_link_only",
            "redaction_affects_key_phrase",
            "short_text",
            "other",
        }
        for item in reasons
    ):
        return "schema_miss", None
    evidence = parsed.get("evidence_phrase")
    if label == POSITIVE_LABEL:
        if not evidence_in_title_or_body(evidence, prepared["title_submitted"], prepared["body_submitted"]):
            return "evidence_fail", None
    elif label in NULL_EVIDENCE_LABELS:
        if evidence is not None:
            return "evidence_fail", None
    else:
        return "schema_miss", None
    return "ok", parsed


def classify_prepared(
    prepared: dict[str, Any],
    provider: ClassifierProvider,
    *,
    max_retries: int = TRANSIENT_RETRIES,
) -> dict[str, Any]:
    if prepared["unassessable_basis"]:
        row = _base_row(prepared, status="skipped_deterministic", retry_count=0, provider_name=provider.name)
        row["predicted_label"] = "unassessable"
        row["decision_reason"] = "No recoverable original title/body after whole-field empty/removal-notice rule."
        row["evidence_phrase"] = None
        row["review_priority"], row["review_priority_reason"] = review_priority_for(
            "skipped_deterministic", "unassessable"
        )
        row["output_tokens_est"] = 0
        row["cost_usd_est"] = 0.0
        return row

    if prepared["truncated"]:
        row = _base_row(prepared, status="truncated_blocked", retry_count=0, provider_name=provider.name)
        row["review_priority"], row["review_priority_reason"] = review_priority_for("truncated_blocked", None)
        return row

    last_error = "error"
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            response = provider.complete(
                {
                    "synthetic_id": prepared["synthetic_id"],
                    "submitted_text": prepared["submitted_text"],
                    "model_input_sha256": prepared["model_input_sha256"],
                }
            )
        except LiveInferenceForbidden:
            raise
        except TransientProviderError:
            retries = attempt + 1
            last_error = "exhausted" if attempt >= max_retries else "timeout"
            if attempt >= max_retries:
                row = _base_row(prepared, status="exhausted", retry_count=retries, provider_name=provider.name)
                return row
            continue

        return ingest_model_response(
            prepared,
            response,
            provider_name=provider.name,
            retry_count=retries,
        )

    row = _base_row(prepared, status=last_error, retry_count=retries, provider_name=provider.name)
    return row


def ingest_model_response(
    prepared: dict[str, Any],
    response: dict[str, Any],
    *,
    provider_name: str,
    retry_count: int = 0,
) -> dict[str, Any]:
    if not response.get("ok"):
        status = str(response.get("kind") or "error")
        row = _base_row(prepared, status=status, retry_count=retry_count, provider_name=provider_name)
        row["response_id"] = response.get("response_id")
        _attach_usage(row, response)
        return row

    status, parsed = _validate_model_payload(response.get("parsed"), prepared)
    if status != "ok" or parsed is None:
        row = _base_row(prepared, status=status, retry_count=retry_count, provider_name=provider_name)
        row["response_id"] = response.get("response_id")
        _attach_usage(row, response)
        return row

    row = _base_row(prepared, status="ok", retry_count=retry_count, provider_name=provider_name)
    row["predicted_label"] = parsed["predicted_label"]
    row["decision_reason"] = parsed["decision_reason"]
    row["evidence_phrase"] = parsed["evidence_phrase"]
    row["ambiguity_reasons"] = list(parsed["ambiguity_reasons"])
    row["response_id"] = response.get("response_id")
    row["resolved_model"] = response.get("model") or (MODEL_ALIAS + "-mock")
    row["review_priority"], row["review_priority_reason"] = review_priority_for("ok", parsed["predicted_label"])
    _attach_usage(row, response)
    if prepared["residual_pii_suspected"] and row["review_priority"] != "error":
        row["review_priority_reason"] = row["review_priority_reason"] + "|residual_pii_flagged"
    return row


def assert_unique_ids(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        post_id = record_id(record)
        if post_id in seen:
            raise DuplicateIdError(f"duplicate synthetic_id {post_id}")
        seen.add(post_id)
        ordered.append(post_id)
    return ordered


def classify_records(
    records: list[dict[str, Any]],
    provider: ClassifierProvider,
    *,
    checkpoint: CheckpointStore,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    max_retries: int = TRANSIENT_RETRIES,
) -> list[dict[str, Any]]:
    assert_unique_ids(records)
    results: list[dict[str, Any]] = []
    for record in records:
        post_id = record_id(record)
        if checkpoint.accepted(post_id):
            results.append(checkpoint.rows[post_id])
            continue
        try:
            prepared = preprocess_record(record, max_input_tokens=max_input_tokens)
        except ProtectedOutcomeSuppliedError:
            raise
        row = classify_prepared(prepared, provider, max_retries=max_retries)
        checkpoint.write_if_absent(row)
        results.append(checkpoint.rows[post_id])
    return results
