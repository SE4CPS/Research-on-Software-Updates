"""Authorized AI relevance campaign: 240-case gate, then full frame if the gate passes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_jsonl, atomic_write_text, utcnow
from full_resource_sentiment_v1.io_utils import read_json
from full_resource_sentiment_v1.paths import RESULTS_DIR, WORKSPACE
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.audit_inputs import load_classifier_inputs
from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate
from full_resource_sentiment_v1.relevance_classify.frame_inputs import EXPECTED_FRAME_N, load_frame_classifier_inputs
from full_resource_sentiment_v1.relevance_classify.gate_prepare import verify_frozen_fingerprints
from full_resource_sentiment_v1.relevance_classify.live_client import (
    MissingAPIKeyError,
    OpenAIHttp,
    OpenAIRequestError,
    api_key_present,
    parse_batch_output_line,
)
from full_resource_sentiment_v1.relevance_classify.pipeline import (
    CheckpointStore,
    classify_prepared,
    classify_records,
    ingest_model_response,
)
from full_resource_sentiment_v1.relevance_classify.planner import conservative_cost_usd, plan_batches
from full_resource_sentiment_v1.relevance_classify.preprocess import preprocess_record
from full_resource_sentiment_v1.relevance_classify.providers import (
    LiveInferenceForbidden,
    LiveOpenAIProvider,
    build_openai_batch_line,
    live_inference_allowed,
    serialize_batch_line,
)
from full_resource_sentiment_v1.relevance_classify.quality import quality_report
from full_resource_sentiment_v1.relevance_classify.spec import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    FULL_CLASSIFIER_RUN_ID,
    GATE_CLASSIFIER_RUN_ID,
    MODEL_ALIAS,
    PROPOSED_SPEND_CEILING_USD,
    REASONING_EFFORT,
    TEMPERATURE,
    prompt_sha256,
    schema_sha256,
)
from full_resource_sentiment_v1.relevance_classify.spend import SpendCeilingError, SpendLedger
from full_resource_sentiment_v1.relevance_classify.validation_draw import draw_validation_sample

PUBLIC_DIR = RESULTS_DIR / "ai_relevance_campaign_2026-10-29"
RESTRICTED_DIR = WORKSPACE / "restricted" / "ai_relevance_campaign_2026-10-29"
BATCH_QUEUED_TOKEN_LIMIT = 8_000_000
BATCH_REQUESTS_PER_SHARD = 1500
POLL_INTERVAL_S = 30


def campaign_paths() -> dict[str, Path]:
    return {
        "public": PUBLIC_DIR,
        "restricted": RESTRICTED_DIR,
        "spend": RESTRICTED_DIR / "spend_ledger.json",
        "gate_checkpoint": RESTRICTED_DIR / "gate" / "checkpoint.jsonl",
        "gate_predictions": RESTRICTED_DIR / "gate" / "predictions.jsonl",
        "gate_eval": RESTRICTED_DIR / "gate" / "evaluation.json",
        "gate_public": PUBLIC_DIR / "gate_public_summary.json",
        "full_checkpoint": RESTRICTED_DIR / "full" / "checkpoint.jsonl",
        "full_predictions": RESTRICTED_DIR / "full" / "predictions.jsonl",
        "full_public": PUBLIC_DIR / "full_public_summary.json",
        "cohort": PUBLIC_DIR / "ai_selected_cohort.jsonl",
        "quality": RESTRICTED_DIR / "full" / "quality_report.json",
        "quality_public": PUBLIC_DIR / "quality_public.json",
        "validation_public": PUBLIC_DIR / "validation_design.json",
        "validation_packet_a": RESTRICTED_DIR / "validation" / "reviewer_A_packet.jsonl",
        "validation_packet_b": RESTRICTED_DIR / "validation" / "reviewer_B_packet.jsonl",
        "validation_key": RESTRICTED_DIR / "validation" / "private_id_key.jsonl",
        "execution_log": PUBLIC_DIR / "execution_log.jsonl",
        "blocker": PUBLIC_DIR / "BLOCKER.json",
        "consolidated": WORKSPACE / "AI_RELEVANCE_CAMPAIGN_REPORT_2026-10-29.md",
        "professor": WORKSPACE / "AI_RELEVANCE_CAMPAIGN_BRIEF_BERHE_CODEX_2026-10-29.md",
    }


def append_log(message: str, **fields: Any) -> None:
    paths = campaign_paths()
    paths["public"].mkdir(parents=True, exist_ok=True)
    row = {"utc": utcnow(), "message": message, **fields}
    hits = scan_obj(row)
    if hits:
        row = {"utc": utcnow(), "message": message, "redacted_fields": True}
    with paths["execution_log"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_blocker(kind: str, detail: str, extra: dict[str, Any] | None = None) -> None:
    paths = campaign_paths()
    payload = {"kind": kind, "detail": detail, "utc": utcnow(), **(extra or {})}
    atomic_write_json(paths["blocker"], payload)
    append_log("blocker", kind=kind)


def strip_text(row: dict[str, Any]) -> dict[str, Any]:
    skip = {
        "title",
        "body",
        "title_original",
        "body_original",
        "title_submitted",
        "body_submitted",
        "submitted_text",
        "decision_reason",
        "evidence_phrase",
    }
    return {key: value for key, value in row.items() if key not in skip}


def integrity_stage() -> dict[str, Any]:
    fingerprints = verify_frozen_fingerprints()
    audit = load_classifier_inputs()
    payload = {
        "utc": utcnow(),
        "frozen_fingerprints": fingerprints,
        "prompt_sha256": prompt_sha256(),
        "schema_sha256": schema_sha256(),
        "model": MODEL_ALIAS,
        "temperature": TEMPERATURE,
        "reasoning_effort": REASONING_EFFORT,
        "n_audit": audit["n"],
        "label_counts": audit["label_counts"],
        "stratum_counts": audit["stratum_counts"],
        "protected_union_n": audit["protected_union_n"],
        "api_key_present": api_key_present(),
        "live_allowed_if_requested": live_inference_allowed(True),
        "spend_ceiling_usd": PROPOSED_SPEND_CEILING_USD,
    }
    atomic_write_json(PUBLIC_DIR / "stage1_integrity.json", payload)
    append_log("stage1_integrity", n=audit["n"], labels=audit["label_counts"])
    return {"fingerprints": fingerprints, "audit": audit, "public": payload}


def run_gate(audit: dict[str, Any], ledger: SpendLedger) -> dict[str, Any]:
    paths = campaign_paths()
    (paths["restricted"] / "gate").mkdir(parents=True, exist_ok=True)
    provider = LiveOpenAIProvider(allow_live=True, ledger=ledger, pricing="sync")
    checkpoint = CheckpointStore(paths["gate_checkpoint"])
    records = list(audit["records"])
    for index, record in enumerate(records, start=1):
        classify_records([record], provider, checkpoint=checkpoint)
        if index % 10 == 0 or index == len(records):
            accepted = sum(
                1
                for row in checkpoint.rows.values()
                if row.get("response_status") in {"ok", "skipped_deterministic"}
            )
            append_log("gate_progress", done=index, total=len(records), accepted=accepted)
    rows = [checkpoint.rows[str(record["synthetic_id"])] for record in records]
    atomic_write_jsonl(paths["gate_predictions"], rows)
    eval_payload = evaluate_gate(audit["labels"], rows, mapping_rows=None, require_complete=True)
    eval_payload["used_real_reddit_text"] = True
    eval_payload["classifier_run_id"] = GATE_CLASSIFIER_RUN_ID
    eval_payload["prompt_sha256"] = prompt_sha256()
    eval_payload["schema_sha256"] = schema_sha256()
    eval_payload["spend"] = ledger.snapshot()
    atomic_write_json(paths["gate_eval"], eval_payload)
    public = {
        "n": 240,
        "metrics": {
            key: eval_payload["metrics"][key]
            for key in (
                "true_positives",
                "false_positives",
                "false_negatives",
                "true_negatives",
                "precision",
                "recall",
                "f1",
                "precision_wilson_95",
                "recall_wilson_95",
                "model_unclear",
                "model_unassessable",
                "model_errors",
                "confusion_human_by_prediction",
                "by_stratum",
                "human_counts",
                "prediction_counts",
            )
        },
        "gate": eval_payload["gate"],
        "false_positive_case_codes": eval_payload["metrics"]["false_positive_case_codes"],
        "false_negative_case_codes": eval_payload["metrics"]["false_negative_case_codes"],
        "spend": ledger.snapshot(),
        "ai_predicted_not_human_confirmed": True,
        "development_set_note": (
            "These 240 cases were used as the cost-control development gate. "
            "They are not an untouched final test of a later retuned classifier."
        ),
    }
    hits = scan_obj(public)
    if hits:
        raise RuntimeError(f"public gate summary leaked markers: {hits[:8]}")
    atomic_write_json(paths["gate_public"], public)
    append_log(
        "stage2_gate",
        passed=eval_payload["gate"]["passed"],
        recall=eval_payload["metrics"]["recall"],
        precision=eval_payload["metrics"]["precision"],
        f1=eval_payload["metrics"]["f1"],
    )
    return {"rows": rows, "evaluation": eval_payload, "public": public}


def _write_shard_jsonl(records: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("wb") as handle:
        for record in records:
            prepared = preprocess_record(record)
            if prepared["unassessable_basis"] or prepared["truncated"]:
                continue
            line = build_openai_batch_line(
                synthetic_id=str(record["synthetic_id"]),
                submitted_text=prepared["submitted_text"],
            )
            handle.write(serialize_batch_line(line))
            n += 1
    return n


def run_full(ledger: SpendLedger) -> dict[str, Any]:
    paths = campaign_paths()
    full_dir = paths["restricted"] / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame_classifier_inputs()
    records = frame["records"]
    if frame["n"] != EXPECTED_FRAME_N:
        raise RuntimeError(f"frame n={frame['n']}")
    plan = plan_batches(
        records,
        queued_token_limit=BATCH_QUEUED_TOKEN_LIMIT,
        max_requests_per_file=BATCH_REQUESTS_PER_SHARD,
    )
    estimate = float(plan["estimated_cost_usd_no_cache"])
    if ledger.bound_usd() + estimate > PROPOSED_SPEND_CEILING_USD - ledger.reserve_usd:
        raise SpendCeilingError(
            f"full-frame estimate {estimate:.4f} plus bound {ledger.bound_usd():.4f} exceeds ceiling"
        )
    checkpoint = CheckpointStore(paths["full_checkpoint"])
    # Deterministic skips first, no API.
    remaining: list[dict[str, Any]] = []
    for record in records:
        post_id = str(record["synthetic_id"])
        if checkpoint.accepted(post_id):
            continue
        prepared = preprocess_record(record)
        if prepared["unassessable_basis"] or prepared["truncated"]:
            class _Local:
                name = "deterministic_local"

                def complete(self, request):  # pragma: no cover
                    raise RuntimeError("deterministic path must not call complete")

            row = classify_prepared(prepared, _Local())
            checkpoint.write_if_absent(row)
            continue
        remaining.append(record)

    client = OpenAIHttp.from_env()
    shard_meta = []
    for index in range(0, len(remaining), BATCH_REQUESTS_PER_SHARD):
        shard_records = remaining[index : index + BATCH_REQUESTS_PER_SHARD]
        shard_path = full_dir / f"batch_shard_{index:05d}.jsonl"
        n_lines = _write_shard_jsonl(shard_records, shard_path)
        if n_lines == 0:
            continue
        shard_estimate = conservative_cost_usd(
            n=n_lines,
            input_tokens_each=1500,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        ledger.authorize(shard_estimate, ref=f"batch_shard_{index:05d}")
        submitted = client.submit_batch_file(shard_path)
        batch_id = submitted["batch"]["id"]
        append_log("batch_submitted", shard=index, batch_id=batch_id, n=n_lines)
        status = client.poll_batch(batch_id, interval_s=POLL_INTERVAL_S)
        out_id = status.get("output_file_id")
        err_id = status.get("error_file_id")
        out_path = full_dir / f"batch_output_{index:05d}.jsonl"
        actual_cost = 0.0
        if out_id:
            client.download_file(str(out_id), out_path)
            by_id = {str(rec["synthetic_id"]): rec for rec in shard_records}
            for raw_line in out_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                parsed_line = parse_batch_output_line(json.loads(raw_line))
                custom_id = str(parsed_line.get("custom_id") or "")
                record = by_id.get(custom_id)
                if record is None:
                    continue
                prepared = preprocess_record(record)
                parsed_line["pricing"] = "batch"
                row = ingest_model_response(prepared, parsed_line, provider_name="openai_live_batch")
                actual_cost += float(row.get("cost_usd_actual") or 0.0)
                checkpoint.write_if_absent(row)
        if err_id:
            client.download_file(str(err_id), full_dir / f"batch_error_{index:05d}.jsonl")
        ledger.settle(shard_estimate, actual_cost, ref=f"batch_shard_{index:05d}")
        shard_meta.append(
            {
                "index": index,
                "n": n_lines,
                "batch_id": batch_id,
                "status": status.get("status"),
                "output_file_id": out_id,
                "error_file_id": err_id,
            }
        )

    # Explicit error rows for anything still missing.
    final_rows = []
    by_ckpt = checkpoint.rows
    for record in records:
        post_id = str(record["synthetic_id"])
        row = by_ckpt.get(post_id)
        if row is None:
            prepared = preprocess_record(record)
            row = ingest_model_response(
                prepared,
                {"ok": False, "kind": "missing_response", "parsed": None, "response_id": None},
                provider_name="openai_live_batch",
            )
            checkpoint.write_if_absent(row)
            row = checkpoint.rows[post_id]
        final_rows.append(row)
    checkpoint.flush()
    atomic_write_jsonl(paths["full_predictions"], final_rows)
    records_by_id = {str(rec["synthetic_id"]): rec for rec in records}
    quality = quality_report(
        expected_ids=[str(rec["synthetic_id"]) for rec in records],
        rows=final_rows,
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
            "classifier_run_id": FULL_CLASSIFIER_RUN_ID,
        }
        for row in final_rows
        if row.get("predicted_label") == "relevant" and row.get("response_status") == "ok"
    ]
    atomic_write_jsonl(paths["cohort"], selected_rows)
    public = {
        "n_frame": EXPECTED_FRAME_N,
        "quality": {key: value for key, value in quality.items() if key not in {"missing_ids_head", "extra_ids_head"}},
        "n_ai_selected": len(selected_rows),
        "spend": ledger.snapshot(),
        "shards": shard_meta,
        "ai_predicted_not_human_confirmed": True,
        "removal_notices": "deterministic_unassessable heuristic, not a human judgment",
    }
    hits = scan_obj(public)
    if hits:
        raise RuntimeError(f"public full summary leaked markers: {hits[:8]}")
    atomic_write_json(paths["full_public"], public)
    atomic_write_json(paths["quality"], quality)
    public_quality = {
        key: value
        for key, value in quality.items()
        if key not in {"missing_ids_head", "extra_ids_head", "duplicate_ids_head"}
    }
    atomic_write_json(paths["quality_public"], public_quality)
    append_log("stage3_full", n_selected=len(selected_rows), complete=quality["complete_coverage"])
    return {
        "rows": final_rows,
        "records_by_id": records_by_id,
        "quality": quality,
        "selected_rows": selected_rows,
        "frame": frame,
        "plan": plan,
    }


def prepare_validation(
    full_rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    development_reddit_ids: set[str],
    protected_ids: set[str],
) -> dict[str, Any]:
    paths = campaign_paths()
    (paths["restricted"] / "validation").mkdir(parents=True, exist_ok=True)
    drawn = draw_validation_sample(
        full_rows,
        records_by_id,
        exclude_reddit_ids=development_reddit_ids,
        exclude_protected_ids=protected_ids,
    )
    atomic_write_jsonl(paths["validation_packet_a"], drawn["packet_a"])
    atomic_write_jsonl(paths["validation_packet_b"], drawn["packet_b"])
    atomic_write_jsonl(paths["validation_key"], drawn["private_rows"])
    public = {
        "seed": drawn["seed"],
        "n_drawn": drawn["n_drawn"],
        "n_per_stratum_target": drawn["n_per_stratum_target"],
        "pools_available": drawn["pools_available"],
        "drawn_counts": drawn["drawn_counts"],
        "short_stratum_notes": drawn["short_stratum_notes"],
        "excluded_development_n": drawn["excluded_development_n"],
        "excluded_protected_n": drawn["excluded_protected_n"],
        "human_labels_invented": False,
        "ai_predictions_hidden_from_reviewers": True,
        "analysis_plan": {
            "estimator": (
                "Inverse-probability weighted precision and recall using recorded "
                "stratum inclusion probabilities. Human labels from this sample "
                "are the confirmation layer; AI labels are predictions only."
            ),
            "precision": "Among AI-selected posts in the sample, weighted share that humans mark relevant.",
            "recall": "Among human-relevant posts in the sample, weighted share that AI selected.",
            "false_positives": "AI-selected and human not_relevant, with stratum weights.",
            "false_negatives": "AI-not-selected and human relevant, with stratum weights.",
            "uncertainty": "Wilson or bootstrap intervals on the weighted estimators; report empty-stratum limits.",
        },
    }
    hits = scan_obj(public)
    if hits:
        raise RuntimeError(f"validation public leaked markers: {hits[:8]}")
    atomic_write_json(paths["validation_public"], public)
    append_log("stage5_validation", n=drawn["n_drawn"], counts=drawn["drawn_counts"])
    return drawn


def write_reports(bundle: dict[str, Any]) -> None:
    paths = campaign_paths()
    gate = bundle.get("gate") or {}
    full = bundle.get("full") or {}
    quality = (full or {}).get("quality") or {}
    spend = bundle.get("spend") or {}
    blocker = bundle.get("blocker")
    metrics = ((gate.get("evaluation") or {}).get("metrics") or {})
    gate_obj = ((gate.get("evaluation") or {}).get("gate") or {})
    n_selected = quality.get("n_selected")
    lines = [
        "# AI relevance classification campaign — 2026-10-29",
        "",
        "Status: machine classification and independent-validation handoff. "
        "AI labels are predictions, not human-confirmed ground truth. "
        "The original 240-case audit is a development/cost-control gate, not an untouched final test.",
        "",
        "## 1. Locked 240-case development evaluation",
        "",
        f"- Human labels: 40 relevant, 199 not_relevant, 1 unassessable.",
        f"- Precision: {metrics.get('precision')}",
        f"- Recall: {metrics.get('recall')}",
        f"- F1: {metrics.get('f1')}",
        f"- Wilson 95% precision: {metrics.get('precision_wilson_95')}",
        f"- Wilson 95% recall: {metrics.get('recall_wilson_95')}",
        f"- TP/FP/FN/TN: {metrics.get('true_positives')}/{metrics.get('false_positives')}/{metrics.get('false_negatives')}/{metrics.get('true_negatives')}",
        f"- Unclear / unassessable / errors: {metrics.get('model_unclear')} / {metrics.get('model_unassessable')} / {metrics.get('model_errors')}",
        f"- Cost-control gate passed: {gate_obj.get('passed')}",
        f"- False-positive case codes: {metrics.get('false_positive_case_codes')}",
        f"- False-negative case codes: {metrics.get('false_negative_case_codes')}",
        "",
        "## 2. Full-frame AI classification",
        "",
        f"- Frame size: {EXPECTED_FRAME_N}",
        f"- Complete coverage: {quality.get('complete_coverage')}",
        f"- Label counts: {quality.get('label_counts')}",
        f"- Status counts: {quality.get('status_counts')}",
        f"- Missing / extra / duplicates: {quality.get('n_missing')} / {quality.get('n_extra')} / {quality.get('n_duplicates')}",
        f"- Model errors: {quality.get('n_errors')}",
        f"- Unclear: {quality.get('n_unclear')}; unassessable: {quality.get('n_unassessable')}",
        "",
        "## 3. AI-selected software-update cohort",
        "",
        f"- n_ai_selected: {n_selected}",
        "- Definition: `predicted_label==relevant` and `response_status==ok`.",
        "- This cohort is **AI-predicted**, not human-confirmed.",
        "",
        "## 4. Model error and uncertainty",
        "",
        "- Forty human-relevant development cases make recall high-variance; a 32/40 gate has Wilson 95% about 0.65–0.90.",
        "- Parsing failures remain `error` and are never converted to `not_relevant`.",
        "- Removal notices are a deterministic heuristic (`deterministic_removal_notice`), not a human judgment.",
        "",
        "## 5. Integrity and privacy",
        "",
        f"- Prompt SHA-256: `{prompt_sha256()}`",
        f"- Schema SHA-256: `{schema_sha256()}`",
        f"- Model: `{MODEL_ALIAS}`; temperature {TEMPERATURE}; reasoning.effort={REASONING_EFFORT}.",
        "- Frozen capture, compact index, G scores, locked audit, and I archive were not modified.",
        "- Protected evaluation outcomes were not read.",
        "- Reviewer validation packets hide AI predictions and reddit IDs.",
        "",
        "## 6. API costs and reproducibility",
        "",
        f"- Authorized ceiling: ${PROPOSED_SPEND_CEILING_USD:.2f}",
        f"- Spend snapshot: {spend}",
        f"- Gate checkpoint: `{paths['gate_checkpoint']}`",
        f"- Full checkpoint: `{paths['full_checkpoint']}`",
        "",
        "## 7. Limitations of the AI-selected cohort",
        "",
        "- Independent human validation has **not** been completed.",
        "- Development-set performance is optimistic for later claims if the same 240 cases guided go/no-go.",
        "- Period mix, protected-set exclusion, and AI false negatives can bias any later sentiment trajectory.",
        "",
        "## 8. Human validation still required",
        "",
        "- Two blinded reviewer packets are in `restricted/ai_relevance_campaign_2026-10-29/validation/`.",
        "- Private mapping stays off reviewer machines.",
        "- After dual review and adjudication, compute weighted precision/recall as specified in `results/ai_relevance_campaign_2026-10-29/validation_design.json`.",
        "",
        "## 9. Sentiment trajectory after independent validation",
        "",
        "- Do not rerun sentiment on the AI cohort until the validation sample is labeled.",
        "- If weighted recall/precision meet a pre-registered confirmation rule, restrict trajectory analyses to the AI-selected cohort with selection-bias weights.",
        "- Keep protected evaluation sets untouched. Do not treat AI labels as replacing Workstream G/I human-in-the-loop outcomes.",
        "",
        "## 10. Blocker" if blocker else "## 10. Handoff",
        "",
        json.dumps(blocker, indent=2) if blocker else "Independent human-validation packets are ready. No new sentiment experiment or manuscript was started.",
        "",
    ]
    atomic_write_text(paths["consolidated"], "\n".join(lines) + "\n")
    brief = [
        "# Brief for Professor Berhe and Codex — AI relevance campaign 2026-10-29",
        "",
        f"Gate passed: {gate_obj.get('passed')}. Precision={metrics.get('precision')}, recall={metrics.get('recall')}, F1={metrics.get('f1')}.",
        f"Full-frame coverage complete: {quality.get('complete_coverage')}. AI-selected n={n_selected} (predictions, not human-confirmed).",
        f"Spend bound: {spend.get('bound_usd') if isinstance(spend, dict) else spend} of ${PROPOSED_SPEND_CEILING_USD:.2f}.",
        "Independent dual-reviewer validation packets are prepared; human labels for that sample do not yet exist.",
        "Do not start manuscript writing or new sentiment runs until validation is complete.",
        "",
    ]
    if blocker:
        brief.append(f"BLOCKER: {blocker.get('kind')}: {blocker.get('detail')}")
    atomic_write_text(paths["professor"], "\n".join(brief) + "\n")


def run_campaign() -> dict[str, Any]:
    paths = campaign_paths()
    paths["public"].mkdir(parents=True, exist_ok=True)
    paths["restricted"].mkdir(parents=True, exist_ok=True)
    append_log("campaign_start")
    integrity = integrity_stage()
    audit = integrity["audit"]
    if not api_key_present():
        blocker = {
            "kind": "missing_api_key",
            "detail": "OPENAI_API_KEY is not set in the environment. No live requests were sent.",
        }
        write_blocker(blocker["kind"], blocker["detail"])
        write_reports({"blocker": blocker, "spend": {}})
        return {"blocker": blocker}
    if not live_inference_allowed(True):
        blocker = {
            "kind": "live_locked",
            "detail": "live_inference_allowed(True) is false. Set RELEVANCE_CLASSIFY_RESEARCH_RUN=1 and RELEVANCE_CLASSIFY_ALLOW_LIVE=1.",
        }
        write_blocker(blocker["kind"], blocker["detail"])
        write_reports({"blocker": blocker, "spend": {}})
        return {"blocker": blocker}

    ledger = SpendLedger(paths["spend"])
    try:
        gate = run_gate(audit, ledger)
    except (LiveInferenceForbidden, MissingAPIKeyError, OpenAIRequestError, SpendCeilingError) as exc:
        blocker = {"kind": type(exc).__name__, "detail": str(exc)}
        write_blocker(blocker["kind"], blocker["detail"], {"spend": ledger.snapshot()})
        write_reports({"blocker": blocker, "spend": ledger.snapshot()})
        return {"blocker": blocker, "integrity": integrity}

    if not gate["evaluation"]["gate"]["passed"]:
        blocker = {
            "kind": "cost_control_gate_failed",
            "detail": (
                "Recall or precision missed the operational screening gate. "
                "Full-frame classification was not started. The classifier was not retuned on these 240 cases."
            ),
            "precision": gate["evaluation"]["metrics"]["precision"],
            "recall": gate["evaluation"]["metrics"]["recall"],
        }
        write_blocker(blocker["kind"], blocker["detail"], {"gate": gate["public"]})
        write_reports({"gate": gate, "blocker": blocker, "spend": ledger.snapshot()})
        return {"blocker": blocker, "gate": gate, "integrity": integrity}

    try:
        full = run_full(ledger)
    except (LiveInferenceForbidden, MissingAPIKeyError, OpenAIRequestError, SpendCeilingError) as exc:
        blocker = {"kind": type(exc).__name__, "detail": str(exc)}
        write_blocker(blocker["kind"], blocker["detail"], {"spend": ledger.snapshot()})
        write_reports({"gate": gate, "blocker": blocker, "spend": ledger.snapshot()})
        return {"blocker": blocker, "gate": gate, "integrity": integrity}

    development_ids = {str(row["reddit_id"]) for row in audit["mapping"]}
    excluded = read_json(WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28" / "excluded_protected_ids.json")
    protected_ids = set(excluded.get("union_ids") or [])
    drawn = prepare_validation(full["rows"], full["records_by_id"], development_ids, protected_ids)
    bundle = {
        "integrity": integrity,
        "gate": gate,
        "full": full,
        "validation": drawn,
        "spend": ledger.snapshot(),
        "blocker": None,
    }
    write_reports(bundle)
    append_log("campaign_complete", n_selected=full["quality"]["n_selected"])
    return bundle
