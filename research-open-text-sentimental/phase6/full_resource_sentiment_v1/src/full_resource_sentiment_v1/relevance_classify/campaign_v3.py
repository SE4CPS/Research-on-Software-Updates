"""Authorized v3 live campaign. Original 240 labels only. Does not use second-review labels."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_jsonl, atomic_write_text, utcnow
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.io_utils import read_json
from full_resource_sentiment_v1.paths import RESULTS_DIR, WORKSPACE
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify import campaign as camp
from full_resource_sentiment_v1.relevance_classify.audit_inputs import LOCKED_LABELS_SHA256
from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate
from full_resource_sentiment_v1.relevance_classify.live_client import MissingAPIKeyError, OpenAIRequestError, api_key_present
from full_resource_sentiment_v1.relevance_classify.preprocess import reject_protected_outcomes
from full_resource_sentiment_v1.relevance_classify.providers import LiveInferenceForbidden, live_inference_allowed
from full_resource_sentiment_v1.relevance_classify.quality import quality_report
from full_resource_sentiment_v1.relevance_classify.scope_primary import SCOPE_VERSION, apply_second_pass
from full_resource_sentiment_v1.relevance_classify.spec import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    FULL_CLASSIFIER_RUN_ID_V3,
    GATE_CLASSIFIER_RUN_ID_V3,
    MODEL_ALIAS,
    PROMPT_VERSION_ENV,
    PROMPT_VERSION_V3,
    PROPOSED_SPEND_CEILING_USD,
    REASONING_EFFORT,
    SCHEMA_VERSION,
    TEMPERATURE,
    prompt_sha256,
    prompt_v3_sha256,
    schema_sha256,
)
from full_resource_sentiment_v1.relevance_classify.spend import SpendCeilingError, SpendLedger

PUBLIC_DIR = RESULTS_DIR / "ai_relevance_v3_2026-10-29"
RESTRICTED_DIR = WORKSPACE / "restricted" / "ai_relevance_v3_2026-10-29"
PRIOR_SPEND = WORKSPACE / "restricted" / "ai_relevance_campaign_2026-10-29" / "spend_ledger.json"
SCOPE_PATH = Path(__file__).with_name("scope_primary.py")


def _bind_paths() -> None:
    camp.PUBLIC_DIR = PUBLIC_DIR
    camp.RESTRICTED_DIR = RESTRICTED_DIR


def freeze_v3() -> dict[str, Any]:
    payload = {
        "frozen_utc": utcnow(),
        "prompt_version": PROMPT_VERSION_V3,
        "prompt_sha256": prompt_v3_sha256(),
        "v1_prompt_sha256_unchanged": prompt_sha256(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha256(),
        "model": MODEL_ALIAS,
        "temperature": TEMPERATURE,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "scope_version": SCOPE_VERSION,
        "scope_primary_sha256": sha256_file(SCOPE_PATH),
        "scope_primary_source": "src/full_resource_sentiment_v1/relevance_classify/scope_primary.py",
        "gate_run_id": GATE_CLASSIFIER_RUN_ID_V3,
        "full_run_id": FULL_CLASSIFIER_RUN_ID_V3,
        "label_set": "locked_v1_final_labels_2026-10-28",
        "locked_labels_sha256": LOCKED_LABELS_SHA256,
        "used_second_review_labels": False,
        "used_protected_outcomes": False,
    }
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PUBLIC_DIR / "V3_FREEZE.json", payload)
    lines = [
        "# Frozen relevance_zero_shot_v3 — 2026-10-29",
        "",
        "This freeze is the authorized v3 development configuration. It does not replace",
        "the locked v1 prompt hashes. Second-review labels were not used.",
        "",
        f"- Prompt version: `{PROMPT_VERSION_V3}`",
        f"- Prompt SHA-256: `{payload['prompt_sha256']}`",
        f"- v1 prompt SHA-256 (unchanged): `{payload['v1_prompt_sha256_unchanged']}`",
        f"- Schema version: `{SCHEMA_VERSION}` SHA-256 `{payload['schema_sha256']}`",
        f"- Model: `{MODEL_ALIAS}`; temperature {TEMPERATURE}; reasoning.effort={REASONING_EFFORT}; max_output_tokens={DEFAULT_MAX_OUTPUT_TOKENS}",
        f"- Second-pass: `{SCOPE_VERSION}` file SHA-256 `{payload['scope_primary_sha256']}`",
        f"- Gate run id: `{GATE_CLASSIFIER_RUN_ID_V3}`",
        f"- Full run id: `{FULL_CLASSIFIER_RUN_ID_V3}`",
        f"- Human labels: locked v1 `{LOCKED_LABELS_SHA256}`",
        "",
        "Do not retune this prompt against the original 240 if the development gate fails.",
        "",
    ]
    atomic_write_text(WORKSPACE / "RELEVANCE_ZERO_SHOT_V3_FREEZE_2026-10-29.md", "\n".join(lines))
    return payload


def apply_pass(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(rec["synthetic_id"]): rec for rec in records}
    out = []
    for row in rows:
        rec = by_id.get(str(row.get("synthetic_id") or row.get("case_code") or ""))
        title = "" if rec is None else str(rec.get("title") or "")
        body = "" if rec is None else str(rec.get("body") or "")
        out.append(apply_second_pass(row, title, body))
    return out


def evidence_fail_codes(rows: list[dict[str, Any]]) -> list[str]:
    codes = []
    for row in rows:
        if row.get("response_status") == "evidence_fail":
            codes.append(str(row.get("case_code") or row.get("synthetic_id") or ""))
    return [code for code in codes if code]


def assert_no_protected_outcomes(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        reject_protected_outcomes(row)


def disagreement_notes(
    labels: list[dict[str, Any]],
    passed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_code = {str(row.get("case_code") or row.get("synthetic_id")): row for row in passed_rows}
    fps: list[dict[str, Any]] = []
    fns: list[dict[str, Any]] = []
    for lab in labels:
        code = str(lab["case_code"])
        pred = by_code[code]
        flags = pred.get("scope_flags") or {}
        selected = pred.get("predicted_label") == "relevant" and pred.get("response_status") == "ok"
        item = {
            "case_code": code,
            "human_label": lab["final_label"],
            "predicted_label": pred.get("predicted_label"),
            "response_status": pred.get("response_status"),
            "second_pass_applied": pred.get("second_pass_applied"),
            "second_pass_reason": pred.get("second_pass_reason"),
            "strong_existing_change": flags.get("strong_existing_change"),
            "os_migration_boundary": flags.get("os_migration_boundary"),
            "first_release_boundary": flags.get("first_release_boundary"),
        }
        if lab["final_label"] == "relevant" and not selected:
            fns.append(item)
        elif lab["final_label"] == "not_relevant" and selected:
            fps.append(item)
    return {
        "false_positives": fps,
        "false_negatives": fns,
        "n_fp_with_strong_existing_change": sum(1 for row in fps if row.get("strong_existing_change")),
        "n_fn_without_strong_existing_change": sum(1 for row in fns if not row.get("strong_existing_change")),
        "interpretation": (
            "If a false positive has strong_existing_change=true, the written primary "
            "scope treats an identifiable existing-software change as relevant while the "
            "locked human label is not_relevant. That is a definition disagreement, not "
            "a reason to retune on these 240 or to overwrite locked labels."
        ),
    }


def write_v3_reports(bundle: dict[str, Any]) -> None:
    freeze = bundle.get("freeze") or {}
    gate = bundle.get("gate") or {}
    evaluation = gate.get("evaluation") or {}
    metrics = evaluation.get("metrics") or {}
    gate_obj = evaluation.get("gate") or {}
    spend = bundle.get("spend") or {}
    blocker = bundle.get("blocker")
    quality = (bundle.get("full") or {}).get("quality") or {}
    disagreement = bundle.get("disagreement") or {}
    n_selected = quality.get("n_selected")
    lines = [
        "# AI relevance v3 campaign — 2026-10-29",
        "",
        "Status: v3 development evaluation against the original locked 240 human labels. "
        "AI labels are predictions, not human-confirmed ground truth. "
        "Second-review labels and protected evaluation outcomes were not used.",
        "",
        "## Freeze",
        "",
        f"- Prompt: `{PROMPT_VERSION_V3}` SHA-256 `{freeze.get('prompt_sha256')}`",
        f"- v1 prompt SHA-256 unchanged: `{freeze.get('v1_prompt_sha256_unchanged')}`",
        f"- Schema: `{freeze.get('schema_version')}` SHA-256 `{freeze.get('schema_sha256')}`",
        f"- Model: `{MODEL_ALIAS}`; temperature {TEMPERATURE}; reasoning.effort={REASONING_EFFORT}",
        f"- Second-pass: `{SCOPE_VERSION}` SHA-256 `{freeze.get('scope_primary_sha256')}`",
        "",
        "## Locked 240-case development evaluation",
        "",
        "- Human labels: 40 relevant, 199 not_relevant, 1 unassessable.",
        f"- Confusion (TP/FP/FN/TN): {metrics.get('true_positives')}/{metrics.get('false_positives')}/{metrics.get('false_negatives')}/{metrics.get('true_negatives')}",
        f"- Precision: {metrics.get('precision')}",
        f"- Recall: {metrics.get('recall')}",
        f"- F1: {metrics.get('f1')}",
        f"- Wilson 95% precision: {metrics.get('precision_wilson_95')}",
        f"- Wilson 95% recall: {metrics.get('recall_wilson_95')}",
        f"- Human-by-prediction: {metrics.get('confusion_human_by_prediction')}",
        f"- Evidence-validation errors: {metrics.get('evidence_validation_errors')} codes {metrics.get('evidence_fail_case_codes')}",
        f"- Unclear / unassessable / errors: {metrics.get('model_unclear')} / {metrics.get('model_unassessable')} / {metrics.get('model_errors')}",
        f"- Cost-control gate passed: {gate_obj.get('passed')} (precision≥0.75 and recall≥0.80)",
        f"- False-positive case codes: {metrics.get('false_positive_case_codes')}",
        f"- False-negative case codes: {metrics.get('false_negative_case_codes')}",
        f"- Spend snapshot: {spend}",
        "",
        "## Written-scope vs original human labels",
        "",
        f"- FP with strong existing-change flags: {disagreement.get('n_fp_with_strong_existing_change')}",
        f"- FN without strong existing-change flags: {disagreement.get('n_fn_without_strong_existing_change')}",
        f"- {disagreement.get('interpretation')}",
        "",
        "## Full-frame AI classification",
        "",
        f"- Complete coverage: {quality.get('complete_coverage')}",
        f"- n_ai_selected: {n_selected}",
        f"- Label counts: {quality.get('label_counts')}",
        "- This cohort is AI-predicted, not human-confirmed.",
        "",
        "## Integrity",
        "",
        "- Frozen capture, compact index, G scores, locked audit, I archive, and v1 campaign reports were not modified.",
        "- Protected evaluation outcomes were not read.",
        "- No new sentiment analysis and no manuscript writing.",
        "",
        "## Blocker" if blocker else "## Handoff",
        "",
        json.dumps(blocker, indent=2) if blocker else "Independent validation packets are prepared. Human labels for that sample do not yet exist.",
        "",
    ]
    atomic_write_text(WORKSPACE / "AI_RELEVANCE_V3_CAMPAIGN_REPORT_2026-10-29.md", "\n".join(lines))
    brief = [
        "# Brief for Professor Berhe and Codex — AI relevance v3 2026-10-29",
        "",
        f"Gate passed: {gate_obj.get('passed')}. Precision={metrics.get('precision')}, recall={metrics.get('recall')}, F1={metrics.get('f1')}.",
        f"Confusion TP/FP/FN/TN={metrics.get('true_positives')}/{metrics.get('false_positives')}/{metrics.get('false_negatives')}/{metrics.get('true_negatives')}.",
        f"Evidence-validation errors: {metrics.get('evidence_validation_errors')}.",
        f"Spend bound: {spend.get('bound_usd') if isinstance(spend, dict) else spend} of ${PROPOSED_SPEND_CEILING_USD:.2f}.",
        "Compared only to original locked 240 labels. Second-review labels unused. Protected outcomes unread.",
        "No new sentiment runs. No manuscript writing. Historical v1 campaign files were not overwritten.",
        "",
    ]
    if blocker:
        brief.append(f"BLOCKER: {blocker.get('kind')}: {blocker.get('detail')}")
        brief.append(
            f"Scope-vs-label: {disagreement.get('n_fp_with_strong_existing_change')} false positives "
            "have strong existing-change flags under the written primary scope."
        )
    atomic_write_text(WORKSPACE / "AI_RELEVANCE_V3_BRIEF_BERHE_CODEX_2026-10-29.md", "\n".join(brief) + "\n")


def seed_spend() -> None:
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESTRICTED_DIR / "spend_ledger.json"
    prior = 0.0
    if PRIOR_SPEND.exists():
        prior = float(read_json(PRIOR_SPEND).get("actual_usd") or 0.0)
    if dest.exists():
        return
    atomic_write_json(
        dest,
        {
            "ceiling_usd": PROPOSED_SPEND_CEILING_USD,
            "reserve_usd": 0.5,
            "actual_usd": prior,
            "pending_usd": 0.0,
            "entries": [
                {
                    "event": "prior_campaign_balance",
                    "ref": "v1_gate_2026-10-29",
                    "actual_usd": prior,
                    "estimate_usd": 0.0,
                    "utc": utcnow(),
                }
            ],
            "updated_utc": utcnow(),
        },
    )


def run_v3_campaign() -> dict[str, Any]:
    os.environ[PROMPT_VERSION_ENV] = "v3"
    _bind_paths()
    seed_spend()
    freeze = freeze_v3()
    camp.append_log("v3_campaign_start", prompt_sha256=freeze["prompt_sha256"], scope=SCOPE_VERSION)
    integrity = camp.integrity_stage()
    integrity["public"]["prompt_version"] = PROMPT_VERSION_V3
    integrity["public"]["prompt_sha256"] = prompt_v3_sha256()
    integrity["public"]["scope_version"] = SCOPE_VERSION
    atomic_write_json(PUBLIC_DIR / "stage1_integrity.json", integrity["public"])
    audit = integrity["audit"]
    for rec in audit["records"]:
        rec["classifier_run_id"] = GATE_CLASSIFIER_RUN_ID_V3
    if not api_key_present():
        blocker = {"kind": "missing_api_key", "detail": "OPENAI_API_KEY is not set."}
        camp.write_blocker(blocker["kind"], blocker["detail"])
        return {"blocker": blocker, "freeze": freeze}
    if not live_inference_allowed(True):
        blocker = {"kind": "live_locked", "detail": "live_inference_allowed(True) is false."}
        camp.write_blocker(blocker["kind"], blocker["detail"])
        return {"blocker": blocker, "freeze": freeze}

    ledger = SpendLedger(camp.campaign_paths()["spend"])
    try:
        raw_gate = camp.run_gate(audit, ledger)
    except (LiveInferenceForbidden, MissingAPIKeyError, OpenAIRequestError, SpendCeilingError) as exc:
        blocker = {"kind": type(exc).__name__, "detail": str(exc)}
        camp.write_blocker(blocker["kind"], blocker["detail"], {"spend": ledger.snapshot()})
        return {"blocker": blocker, "freeze": freeze, "spend": ledger.snapshot()}

    raw_rows = raw_gate["rows"]
    assert_no_protected_outcomes(raw_rows)
    atomic_write_jsonl(RESTRICTED_DIR / "gate" / "raw_predictions.jsonl", raw_rows)
    passed_rows = apply_pass(raw_rows, audit["records"])
    assert_no_protected_outcomes(passed_rows)
    atomic_write_jsonl(camp.campaign_paths()["gate_predictions"], passed_rows)
    evaluation = evaluate_gate(audit["labels"], passed_rows, require_complete=True)
    fail_codes = evidence_fail_codes(raw_rows)
    evaluation["used_real_reddit_text"] = True
    evaluation["used_second_review_labels"] = False
    evaluation["used_protected_outcomes"] = False
    evaluation["classifier_run_id"] = GATE_CLASSIFIER_RUN_ID_V3
    evaluation["prompt_sha256"] = prompt_v3_sha256()
    evaluation["schema_sha256"] = schema_sha256()
    evaluation["scope_version"] = SCOPE_VERSION
    evaluation["second_pass_applied"] = True
    evaluation["metrics"]["evidence_validation_errors"] = len(fail_codes)
    evaluation["metrics"]["evidence_fail_case_codes"] = fail_codes
    evaluation["spend"] = ledger.snapshot()
    disagreement = disagreement_notes(audit["labels"], passed_rows)
    evaluation["disagreement"] = {
        key: disagreement[key]
        for key in (
            "n_fp_with_strong_existing_change",
            "n_fn_without_strong_existing_change",
            "interpretation",
        )
    }
    atomic_write_json(camp.campaign_paths()["gate_eval"], evaluation)
    public = dict(raw_gate["public"])
    public["metrics"] = {
        key: evaluation["metrics"][key]
        for key in public["metrics"]
    }
    public["metrics"]["evidence_validation_errors"] = len(fail_codes)
    public["gate"] = evaluation["gate"]
    public["false_positive_case_codes"] = evaluation["metrics"]["false_positive_case_codes"]
    public["false_negative_case_codes"] = evaluation["metrics"]["false_negative_case_codes"]
    public["prompt_version"] = PROMPT_VERSION_V3
    public["prompt_sha256"] = prompt_v3_sha256()
    public["scope_version"] = SCOPE_VERSION
    public["used_second_review_labels"] = False
    public["used_protected_outcomes"] = False
    public["spend"] = ledger.snapshot()
    public["disagreement"] = evaluation["disagreement"]
    hits = scan_obj({k: public[k] for k in ("metrics", "gate", "spend") if k in public})
    if hits:
        raise RuntimeError(f"v3 public gate leaked markers: {hits[:8]}")
    atomic_write_json(camp.campaign_paths()["gate_public"], public)
    camp.append_log(
        "v3_gate",
        passed=evaluation["gate"]["passed"],
        precision=evaluation["metrics"]["precision"],
        recall=evaluation["metrics"]["recall"],
        f1=evaluation["metrics"]["f1"],
        evidence_validation_errors=len(fail_codes),
    )
    gate_bundle = {"evaluation": evaluation, "public": public, "rows": passed_rows}

    if not evaluation["gate"]["passed"]:
        blocker = {
            "kind": "cost_control_gate_failed",
            "detail": (
                "v3 development gate failed against locked v1 labels. "
                "Thresholds were not changed. The prompt was not retuned. "
                "Full-frame classification was not started."
            ),
            "precision": evaluation["metrics"]["precision"],
            "recall": evaluation["metrics"]["recall"],
            "false_positive_case_codes": evaluation["metrics"]["false_positive_case_codes"],
            "false_negative_case_codes": evaluation["metrics"]["false_negative_case_codes"],
            "n_fp_with_strong_existing_change": disagreement["n_fp_with_strong_existing_change"],
        }
        camp.write_blocker(blocker["kind"], blocker["detail"], {"gate": public})
        write_v3_reports(
            {
                "freeze": freeze,
                "gate": gate_bundle,
                "blocker": blocker,
                "spend": ledger.snapshot(),
                "disagreement": disagreement,
            }
        )
        return {
            "blocker": blocker,
            "gate": gate_bundle,
            "freeze": freeze,
            "spend": ledger.snapshot(),
            "disagreement": disagreement,
        }

    try:
        full = camp.run_full(ledger)
    except (LiveInferenceForbidden, MissingAPIKeyError, OpenAIRequestError, SpendCeilingError) as exc:
        blocker = {"kind": type(exc).__name__, "detail": str(exc)}
        camp.write_blocker(blocker["kind"], blocker["detail"], {"spend": ledger.snapshot()})
        write_v3_reports(
            {
                "freeze": freeze,
                "gate": gate_bundle,
                "blocker": blocker,
                "spend": ledger.snapshot(),
                "disagreement": disagreement,
            }
        )
        return {"blocker": blocker, "gate": gate_bundle, "freeze": freeze, "spend": ledger.snapshot()}

    passed_full = apply_pass(full["rows"], full["frame"]["records"])
    atomic_write_jsonl(camp.campaign_paths()["full_predictions"], passed_full)
    records_by_id = full["records_by_id"]
    quality = quality_report(
        expected_ids=[str(rec["synthetic_id"]) for rec in full["frame"]["records"]],
        rows=passed_full,
        records_by_id=records_by_id,
        id_field="synthetic_id",
    )
    selected_rows = [
        {
            "reddit_id": row.get("reddit_id") or row.get("synthetic_id"),
            "stratum": row.get("stratum"),
            "predicted_label": row.get("predicted_label"),
            "response_status": row.get("response_status"),
            "protected_membership": row.get("protected_membership"),
            "ai_predicted_not_human_confirmed": True,
            "classifier_run_id": FULL_CLASSIFIER_RUN_ID_V3,
            "second_pass_applied": bool(row.get("second_pass_applied")),
        }
        for row in passed_full
        if row.get("predicted_label") == "relevant" and row.get("response_status") == "ok"
    ]
    atomic_write_jsonl(camp.campaign_paths()["cohort"], selected_rows)
    atomic_write_json(camp.campaign_paths()["quality"], quality)
    full["rows"] = passed_full
    full["quality"] = quality
    full["selected_rows"] = selected_rows

    development_ids = {str(row["reddit_id"]) for row in audit["mapping"]}
    excluded = read_json(WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "excluded_protected_ids.json")
    protected_ids = set(excluded.get("union_ids") or [])
    drawn = camp.prepare_validation(passed_full, records_by_id, development_ids, protected_ids)
    validation_public_path = camp.campaign_paths()["validation_public"]
    validation_public = read_json(validation_public_path)
    validation_public["weighted_estimator_module"] = (
        "full_resource_sentiment_v1.relevance_classify.validation_weighted"
    )
    validation_public["weighted_estimator"] = (
        "Horvitz-Thompson using recorded stratum inclusion_probability from private_id_key.jsonl"
    )
    atomic_write_json(validation_public_path, validation_public)
    bundle = {
        "freeze": freeze,
        "integrity": integrity,
        "gate": gate_bundle,
        "full": full,
        "validation": drawn,
        "spend": ledger.snapshot(),
        "blocker": None,
        "disagreement": disagreement,
    }
    write_v3_reports(bundle)
    camp.append_log("v3_campaign_complete", n_selected=len(selected_rows))
    return bundle
