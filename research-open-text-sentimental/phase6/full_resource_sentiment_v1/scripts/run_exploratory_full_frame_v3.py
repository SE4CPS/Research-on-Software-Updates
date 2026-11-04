#!/usr/bin/env python3
"""Exploratory full-frame v3 classification. Does not claim independent validation.

Preserves frozen v3 hashes. Errors stay errors. Does not overwrite historical
audits, A-F sentiment, or protected evaluation artifacts.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_jsonl, atomic_write_text, utcnow
from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.io_utils import read_json, read_jsonl
from full_resource_sentiment_v1.paths import BROAD_SENTIMENT_RUN_DIR, RESULTS_DIR
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.frame_inputs import EXPECTED_FRAME_N, load_frame_classifier_inputs
from full_resource_sentiment_v1.relevance_classify.gate_prepare import verify_frozen_fingerprints
from full_resource_sentiment_v1.relevance_classify.live_client import (
    MissingAPIKeyError,
    OpenAIHttp,
    OpenAIRequestError,
    api_key_present,
    parse_batch_output_line,
)
from full_resource_sentiment_v1.relevance_classify.pipeline import CheckpointStore, classify_prepared, ingest_model_response
from full_resource_sentiment_v1.relevance_classify.planner import conservative_cost_usd, plan_batches
from full_resource_sentiment_v1.relevance_classify.preprocess import (
    evidence_in_title_or_body,
    preprocess_record,
    reject_protected_outcomes,
)
from full_resource_sentiment_v1.relevance_classify.providers import (
    LiveInferenceForbidden,
    TransientProviderError,
    build_openai_batch_line,
    live_inference_allowed,
    serialize_batch_line,
)
from full_resource_sentiment_v1.relevance_classify.quality import quality_report
from full_resource_sentiment_v1.relevance_classify.scope_primary import SCOPE_VERSION, apply_second_pass
from full_resource_sentiment_v1.relevance_classify.spec import (
    ALLOW_LIVE_ENV,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_ALIAS,
    PROMPT_VERSION_ENV,
    PROMPT_VERSION_V3,
    PROPOSED_SPEND_CEILING_USD,
    REASONING_EFFORT,
    RESEARCH_RUN_ENV,
    TEMPERATURE,
    prompt_sha256,
    prompt_v3_sha256,
    schema_sha256,
)
from full_resource_sentiment_v1.relevance_classify.spend import SpendCeilingError, SpendLedger

EXTERNAL_ENV = Path.home() / ".config/reddit-research/openai.env"
PUBLIC_DIR = RESULTS_DIR / "ai_relevance_exploratory_full_2026-10-29"
RESTRICTED_DIR = WORKSPACE / "restricted" / "ai_relevance_exploratory_full_2026-10-29"
PRIOR_SPEND = WORKSPACE / "restricted" / "ai_relevance_v3_2026-10-29" / "spend_ledger.json"
SENTIMENT_DIR = RESULTS_DIR / "exploratory_ai_selected_sentiment_2026-10-29"
RUN_ID = "ai-relevance-exploratory-full-37596-v3-2026-10-29"
BATCH_REQUESTS = 1500
BATCH_QUEUED_TOKEN_LIMIT = 8_000_000
POLL_S = 30
SUBMIT_ATTEMPTS = 5
KNOWN_EXTRA_BATCH_IDS = (
    "batch_6ae3fe168d108190ac4c607caf964a90",  # cancelled duplicate shard-2 upload
)
SHARD4_BROKEN_PIPE_UTC = "2026-10-30T03:31:09+00:00"
EXPECTED_V3_PROMPT = "528ab0b84a97eed4bf457c15d5eed59931a2bd78836a56bd5a93c7c60ad8fb74"
EXPECTED_V1_PROMPT = "fb30113eb732f320018bc78420159feda1298899ac070991ca3b6cc9dcf67f85"
EXPECTED_SCHEMA = "62bead564f30ff68d774a2ecaf766bcdc055d7d92374cda8b23e58735cd1f4bb"
EXPECTED_SCOPE = "fd0db82676f0a6102cf1111d5d878bdb7971d9417bb3d69ad90fa4128319e8b7"
A_F_HASHES = {
    "analysis_A.json": "db6ce51dfba97e3f0eb858d302543188d58e2302ccfe0ecdd2846b18363b6a35",
    "analysis_B.json": "2b628565cab83afa5c5f7705078f782a5951c7a54a076699240f60f650d4511f",
    "analysis_C.json": "d389d35eefd800e97091a6a9a83d05ba23c568ec49166f2699c5d36a5d923e8c",
    "analysis_D.json": "957a6f0a0b574afacdc89ca0943f845b4e96666ae09af3a8ac3ead4111cc16e0",
    "analysis_E.json": "5d194db10f2476e8cd975fe2cb2a1dacd73d5fc4da3d7ef6be4248adb33f74c9",
    "analysis_F.json": "3dc0bc3770be3abda7b374494860cc67ce417f8fa427d0782b56149cf196742a",
    "post_scores.jsonl": "8ae14f59dca38a906b44d0202141f716ff288435d48ccf720f64a672cf5c9771",
}


def load_key() -> bool:
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"):
        return True
    if not EXTERNAL_ENV.is_file():
        return False
    for raw in EXTERNAL_ENV.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip().strip("'").strip('"')
        if key.strip() in {"OPENAI_API_KEY", "OPENAI_KEY"} and value:
            os.environ["OPENAI_API_KEY"] = value
    return bool(os.environ.get("OPENAI_API_KEY"))


def log(message: str, **fields: Any) -> None:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    row = {"utc": utcnow(), "message": message, **fields}
    if scan_obj(row):
        row = {"utc": utcnow(), "message": message, "redacted_fields": True}
    with (PUBLIC_DIR / "execution_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    try:
        print(f"{message} {extra}".strip(), flush=True)
    except BrokenPipeError:
        pass


def assert_freeze() -> None:
    if prompt_v3_sha256() != EXPECTED_V3_PROMPT:
        raise SystemExit("v3 prompt hash drifted; refusing")
    if prompt_sha256() != EXPECTED_V1_PROMPT:
        raise SystemExit("v1 prompt hash drifted; refusing")
    if schema_sha256() != EXPECTED_SCHEMA:
        raise SystemExit("schema hash drifted; refusing")
    scope = WORKSPACE / "src/full_resource_sentiment_v1/relevance_classify/scope_primary.py"
    if sha256_file(scope) != EXPECTED_SCOPE:
        raise SystemExit("scope_primary.py hash drifted; refusing")
    verify_frozen_fingerprints()
    for name, digest in A_F_HASHES.items():
        actual = sha256_file(BROAD_SENTIMENT_RUN_DIR / name)
        if actual != digest:
            raise SystemExit(f"accepted sentiment artifact drifted: {name}")


def seed_spend() -> Path:
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESTRICTED_DIR / "spend_ledger.json"
    prior = 0.0
    if PRIOR_SPEND.exists():
        prior = float(read_json(PRIOR_SPEND).get("actual_usd") or 0.0)
    if dest.exists():
        return dest
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
                    "ref": "v1_gate_plus_v3_gate_plus_validation_pilot",
                    "actual_usd": prior,
                    "utc": utcnow(),
                }
            ],
            "updated_utc": utcnow(),
        },
    )
    return dest


def write_shard(records: list[dict[str, Any]], path: Path) -> int:
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


def load_shard_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["index"]): row for row in read_jsonl(path)}


def save_shard_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    rows = [state[key] for key in sorted(state, key=lambda x: int(x))]
    atomic_write_jsonl(path, rows)


def poll_until_done(client: OpenAIHttp, batch_id: str) -> dict[str, Any]:
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        status = client.get_batch(batch_id)
        st = str(status.get("status") or "")
        log("batch_status", batch_id=batch_id, status=st)
        if st in terminal:
            return status
        time.sleep(POLL_S)


def ingest_output(
    client: OpenAIHttp,
    *,
    out_id: str | None,
    dest: Path,
    shard_records: list[dict[str, Any]],
    checkpoint: CheckpointStore,
) -> float:
    actual = 0.0
    if not out_id:
        return actual
    client.download_file(str(out_id), dest)
    by_id = {str(rec["synthetic_id"]): rec for rec in shard_records}
    for raw_line in dest.read_text(encoding="utf-8").splitlines():
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
        actual += float(row.get("cost_usd_actual") or 0.0)
        if not checkpoint.accepted(str(record["synthetic_id"])):
            checkpoint.rows[str(record["synthetic_id"])] = row
    checkpoint.flush()
    return actual


def classify_full(ledger: SpendLedger) -> dict[str, Any]:
    os.environ[PROMPT_VERSION_ENV] = "v3"
    full_dir = RESTRICTED_DIR / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    state_path = full_dir / "shard_state.jsonl"
    frame = load_frame_classifier_inputs()
    if frame["n"] != EXPECTED_FRAME_N:
        raise RuntimeError(f"frame n={frame['n']}")
    records = frame["records"]
    for rec in records:
        rec["classifier_run_id"] = RUN_ID
        reject_protected_outcomes(rec)
    plan = plan_batches(
        records,
        queued_token_limit=BATCH_QUEUED_TOKEN_LIMIT,
        max_requests_per_file=BATCH_REQUESTS,
    )
    estimate = float(plan["estimated_cost_usd_no_cache"])
    atomic_write_json(PUBLIC_DIR / "batch_plan.json", {k: v for k, v in plan.items() if k != "shards"} | {
        "n_shards": plan["n_shards"],
        "exploratory": True,
        "independent_validation": False,
    })
    log("plan", n_model=plan["n_model_requests"], n_shards=plan["n_shards"], estimate=estimate)
    if ledger.bound_usd() + estimate > PROPOSED_SPEND_CEILING_USD - ledger.reserve_usd:
        raise SpendCeilingError(
            f"full-frame estimate {estimate:.4f} plus bound {ledger.bound_usd():.4f} exceeds ceiling"
        )
    checkpoint = CheckpointStore(full_dir / "checkpoint.jsonl")
    eligible: list[dict[str, Any]] = []
    for record in records:
        post_id = str(record["synthetic_id"])
        prepared = preprocess_record(record)
        if prepared["unassessable_basis"] or prepared["truncated"]:
            if not checkpoint.accepted(post_id):
                class _Local:
                    name = "deterministic_local"

                    def complete(self, request):  # pragma: no cover
                        raise RuntimeError("deterministic path must not call complete")

                checkpoint.rows[post_id] = classify_prepared(prepared, _Local())
            continue
        eligible.append(record)
    checkpoint.flush()
    log("deterministic_done", eligible=len(eligible), checkpointed=len(checkpoint.rows))

    client = OpenAIHttp.from_env()
    state = load_shard_state(state_path)
    max_in_flight = 2
    specs: list[tuple[int, list[dict[str, Any]]]] = []
    for offset in range(0, len(eligible), BATCH_REQUESTS):
        specs.append((offset // BATCH_REQUESTS, eligible[offset : offset + BATCH_REQUESTS]))

    def known_batch_ids() -> set[str]:
        ids = set(KNOWN_EXTRA_BATCH_IDS)
        for rec in state.values():
            for key in ("batch_id", "last_batch_id"):
                val = rec.get(key)
                if val:
                    ids.add(str(val))
        return ids

    def reconcile_stranded_upload() -> None:
        known = known_batch_ids()
        listed = client.list_batches(limit=100)
        unknown = []
        for row in listed:
            bid = str(row.get("id") or "")
            if not bid or bid in known:
                continue
            unknown.append(
                {
                    "id": bid,
                    "status": row.get("status"),
                    "created_at": row.get("created_at"),
                    "request_counts": row.get("request_counts"),
                    "metadata": row.get("metadata"),
                }
            )
        log(
            "batch_reconcile",
            known_n=len(known),
            listed_n=len(listed),
            unknown_n=len(unknown),
            unknown_ids=[row["id"] for row in unknown[:12]],
        )
        meta4 = state.get("4") or {}
        if meta4.get("batch_id"):
            return
        created_after_pipe = []
        for row in unknown:
            created = row.get("created_at")
            if isinstance(created, (int, float)) and float(created) >= 1793331009:
                created_after_pipe.append(row["id"])
        if created_after_pipe:
            raise RuntimeError(
                "unreconciled OpenAI batch after failed shard-4 upload: " + ",".join(created_after_pipe)
            )
        last_event = None
        for entry in ledger.entries:
            if entry.get("ref") == "batch_shard_00004":
                last_event = entry
        if last_event and last_event.get("event") == "authorize":
            ledger.fail(float(last_event.get("estimate_usd") or 0.0), 0.0, ref="batch_shard_00004")
            log("spend_release_stranded", ref="batch_shard_00004", pending=round(ledger.pending_usd, 6), actual=round(ledger.actual_usd, 6))

    def shard_done(index: int, records: list[dict[str, Any]]) -> bool:
        meta = state.get(str(index)) or {}
        if meta.get("status") == "completed" and meta.get("settled"):
            return True
        accepted = all(checkpoint.accepted(str(rec["synthetic_id"])) for rec in records)
        return bool(accepted and meta.get("settled"))

    def submit_shard(index: int, shard_records: list[dict[str, Any]]) -> None:
        key = str(index)
        if (state.get(key) or {}).get("batch_id"):
            return
        shard_path = full_dir / f"batch_shard_{index:05d}.jsonl"
        n_lines = write_shard(shard_records, shard_path)
        if n_lines == 0:
            return
        shard_estimate = conservative_cost_usd(
            n=n_lines,
            input_tokens_each=1500,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        )
        ledger.authorize(shard_estimate, ref=f"batch_shard_{index:05d}")
        submitted = None
        last_exc: Exception | None = None
        for attempt in range(1, SUBMIT_ATTEMPTS + 1):
            try:
                submitted = client.submit_batch_file(
                    shard_path,
                    metadata={"run_id": RUN_ID, "shard_index": str(index)},
                )
                break
            except TransientProviderError as exc:
                last_exc = exc
                log("batch_submit_retry", index=index, attempt=attempt, detail=str(exc)[:120])
                time.sleep(min(60, 5 * attempt))
        if submitted is None:
            ledger.fail(shard_estimate, 0.0, ref=f"batch_shard_{index:05d}")
            raise last_exc or TransientProviderError("submit failed")
        prev = state.get(key) or {}
        state[key] = {
            "index": index,
            "n": n_lines,
            "status": "submitted",
            "batch_id": submitted["batch"]["id"],
            "estimate_usd": shard_estimate,
            "retries": int(prev.get("retries") or 0),
            "last_batch_id": prev.get("last_batch_id"),
        }
        save_shard_state(state_path, state)
        log("batch_submitted", index=index, batch_id=state[key]["batch_id"], n=n_lines)

    def finish_shard(index: int, shard_records: list[dict[str, Any]], status: dict[str, Any]) -> None:
        key = str(index)
        meta = state.get(key) or {"index": index}
        out_id = status.get("output_file_id")
        err_id = status.get("error_file_id")
        actual_cost = ingest_output(
            client,
            out_id=out_id,
            dest=full_dir / f"batch_output_{index:05d}.jsonl",
            shard_records=shard_records,
            checkpoint=checkpoint,
        )
        if err_id:
            client.download_file(str(err_id), full_dir / f"batch_error_{index:05d}.jsonl")
        if not meta.get("settled"):
            ledger.settle(float(meta.get("estimate_usd") or 0.0), actual_cost, ref=f"batch_shard_{index:05d}")
            meta["settled"] = True
        meta["status"] = "completed"
        meta["output_file_id"] = out_id
        meta["actual_cost_usd"] = actual_cost
        state[key] = meta
        save_shard_state(state_path, state)
        log("shard_done", index=index, status=meta["status"], actual=round(actual_cost, 6), bound=round(ledger.bound_usd(), 6))

    reconcile_stranded_upload()
    by_index = {index: recs for index, recs in specs}
    while True:
        unfinished = [(index, recs) for index, recs in specs if not shard_done(index, recs)]
        if not unfinished:
            break
        in_flight = [
            (index, recs)
            for index, recs in unfinished
            if (state.get(str(index)) or {}).get("batch_id")
        ]
        pending = [
            (index, recs)
            for index, recs in unfinished
            if not (state.get(str(index)) or {}).get("batch_id")
            and (state.get(str(index)) or {}).get("status") != "failed_terminal"
            and int((state.get(str(index)) or {}).get("retries") or 0) < 3
        ]
        while len(in_flight) < max_in_flight and pending:
            index, recs = pending.pop(0)
            try:
                submit_shard(index, recs)
            except TransientProviderError as exc:
                log("batch_submit_transient", index=index, detail=str(exc)[:120])
                break
            if (state.get(str(index)) or {}).get("batch_id"):
                in_flight.append((index, recs))
        progressed = False
        for index, recs in list(in_flight):
            meta = state.get(str(index)) or {}
            batch_id = meta.get("batch_id")
            if not batch_id:
                continue
            try:
                status = client.get_batch(str(batch_id))
            except TransientProviderError as exc:
                log("batch_poll_transient", index=index, detail=str(exc)[:120])
                continue
            st = str(status.get("status") or "")
            counts = status.get("request_counts") or {}
            log(
                "batch_status",
                batch_id=batch_id,
                status=st,
                index=index,
                completed_n=counts.get("completed"),
                failed_n=counts.get("failed"),
                total_n=counts.get("total"),
            )
            if st == "completed":
                try:
                    finish_shard(index, recs, status)
                except TransientProviderError as exc:
                    log("shard_ingest_transient", index=index, detail=str(exc)[:120])
                    continue
                progressed = True
            elif st in {"failed", "expired", "cancelled"}:
                retries = int(meta.get("retries") or 0) + 1
                if not meta.get("released"):
                    ledger.fail(float(meta.get("estimate_usd") or 0.0), 0.0, ref=f"batch_shard_{index:05d}")
                    meta["released"] = True
                meta["retries"] = retries
                meta["last_batch_id"] = batch_id
                meta.pop("batch_id", None)
                meta["status"] = "failed_terminal" if retries >= 3 else st
                state[str(index)] = meta
                save_shard_state(state_path, state)
                log("shard_retry", index=index, status=meta["status"], retries=retries)
                progressed = True
        if not progressed:
            time.sleep(POLL_S)

    final_rows = []
    for record in records:
        post_id = str(record["synthetic_id"])
        row = checkpoint.rows.get(post_id)
        if row is None:
            prepared = preprocess_record(record)
            row = ingest_model_response(
                prepared,
                {"ok": False, "kind": "missing_response", "parsed": None, "response_id": None},
                provider_name="openai_live_batch",
            )
            checkpoint.rows[post_id] = row
        final_rows.append(row)
    checkpoint.flush()
    return {"rows": final_rows, "records": records, "plan": plan, "frame": frame, "ledger": ledger}


def apply_pass_and_write(bundle: dict[str, Any]) -> dict[str, Any]:
    records = bundle["records"]
    by_id = {str(rec["synthetic_id"]): rec for rec in records}
    raw = bundle["rows"]
    atomic_write_jsonl(RESTRICTED_DIR / "full" / "raw_predictions.jsonl", raw)
    passed = []
    for row in raw:
        rec = by_id[str(row["synthetic_id"])]
        passed.append(apply_second_pass(row, rec.get("title") or "", rec.get("body") or ""))
    atomic_write_jsonl(RESTRICTED_DIR / "full" / "predictions.jsonl", passed)
    quality = quality_report(
        expected_ids=[str(rec["synthetic_id"]) for rec in records],
        rows=passed,
        records_by_id=by_id,
        id_field="synthetic_id",
    )
    evidence_fail = [row.get("reddit_id") or row.get("synthetic_id") for row in passed if row.get("response_status") == "evidence_fail"]
    n_selected_phrase_ok = 0
    n_selected_phrase_fail = 0
    for row in passed:
        if row.get("predicted_label") != "relevant" or row.get("response_status") != "ok":
            continue
        rec = by_id[str(row["synthetic_id"])]
        if evidence_in_title_or_body(row.get("evidence_phrase"), rec.get("title") or "", rec.get("body") or ""):
            n_selected_phrase_ok += 1
        else:
            n_selected_phrase_fail += 1
    evidence_report = {
        "exploratory": True,
        "independent_validation": False,
        "n_response_status_evidence_fail": len(evidence_fail),
        "n_quality_evidence_fail_on_ok_relevant": quality.get("n_evidence_fail"),
        "n_selected_evidence_phrase_in_title_or_body": n_selected_phrase_ok,
        "n_selected_evidence_phrase_missing": n_selected_phrase_fail,
        "errors_never_coerced_to_not_relevant": True,
        "note": (
            "evidence_fail status rows are errors, not selected. "
            "n_selected_evidence_phrase_missing counts relevant+ok rows whose phrase is not a title/body substring."
        ),
    }
    atomic_write_json(RESTRICTED_DIR / "full" / "evidence_validation_report.json", evidence_report)
    atomic_write_json(PUBLIC_DIR / "evidence_validation_report.json", evidence_report)
    selected = []
    for row in passed:
        if row.get("predicted_label") == "relevant" and row.get("response_status") == "ok":
            selected.append(
                {
                    "reddit_id": row.get("reddit_id") or row.get("synthetic_id"),
                    "stratum": row.get("stratum"),
                    "predicted_label": row.get("predicted_label"),
                    "response_status": row.get("response_status"),
                    "protected_membership": row.get("protected_membership"),
                    "second_pass_applied": bool(row.get("second_pass_applied")),
                    "ai_predicted_not_human_confirmed": True,
                    "exploratory_not_independently_validated": True,
                    "classifier_run_id": RUN_ID,
                }
            )
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(PUBLIC_DIR / "exploratory_ai_selected_cohort.jsonl", selected)
    atomic_write_json(RESTRICTED_DIR / "full" / "quality_report.json", quality)
    period = {}
    for row in passed:
        stratum = str(row.get("stratum") or "unspecified")
        block = period.setdefault(
            stratum,
            {"n": 0, "relevant": 0, "not_relevant": 0, "unclear": 0, "unassessable": 0, "error": 0, "selected": 0},
        )
        block["n"] += 1
        status = row.get("response_status")
        label = row.get("predicted_label")
        if status in {"ok", "skipped_deterministic"} and label in {"relevant", "not_relevant", "unclear", "unassessable"}:
            block[str(label)] += 1
        else:
            block["error"] += 1
        if label == "relevant" and status == "ok":
            block["selected"] += 1
    public = {
        "exploratory": True,
        "independent_validation": False,
        "development_gate_met": False,
        "n_frame": EXPECTED_FRAME_N,
        "label_counts": quality.get("label_counts"),
        "status_counts": quality.get("status_counts"),
        "n_ai_selected": len(selected),
        "n_ai_selected_excluding_protected": sum(1 for row in selected if not row.get("protected_membership")),
        "n_errors": quality.get("n_errors"),
        "n_evidence_fail": quality.get("n_evidence_fail"),
        "n_evidence_fail_status": len(evidence_fail),
        "evidence_validation": evidence_report,
        "complete_coverage": quality.get("complete_coverage"),
        "by_period": period,
        "prompt_version": PROMPT_VERSION_V3,
        "prompt_sha256": prompt_v3_sha256(),
        "schema_sha256": schema_sha256(),
        "scope_version": SCOPE_VERSION,
        "model": MODEL_ALIAS,
        "temperature": TEMPERATURE,
        "reasoning_effort": REASONING_EFFORT,
        "spend": bundle["ledger"].snapshot(),
        "ai_predicted_not_human_confirmed": True,
        "unused_independent_validation_sample": True,
    }
    hits = scan_obj(public)
    if hits:
        raise RuntimeError(f"public leak {hits[:8]}")
    atomic_write_json(PUBLIC_DIR / "full_public_summary.json", public)
    public_quality = {k: v for k, v in quality.items() if k not in {"missing_ids_head", "extra_ids_head", "duplicate_ids_head"}}
    atomic_write_json(PUBLIC_DIR / "quality_public.json", public_quality)
    return {"passed": passed, "selected": selected, "quality": quality, "public": public, "by_id": by_id}


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((p / 100) * (n - 1)))))
        return ordered[idx]

    return {
        "n": n,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p25": pct(25),
        "p75": pct(75),
        "n_negative": sum(1 for x in ordered if x < 0),
        "n_positive": sum(1 for x in ordered if x > 0),
        "n_zero": sum(1 for x in ordered if x == 0),
    }


def sentiment_join(selected: list[dict[str, Any]]) -> dict[str, Any]:
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    selected_ids = {
        str(row["reddit_id"])
        for row in selected
        if not row.get("protected_membership")
    }
    scores_path = BROAD_SENTIMENT_RUN_DIR / "post_scores.jsonl"
    if sha256_file(scores_path) != A_F_HASHES["post_scores.jsonl"]:
        raise SystemExit("post_scores.jsonl hash drifted; refusing to join")
    for name, digest in A_F_HASHES.items():
        if name.startswith("analysis_") and sha256_file(BROAD_SENTIMENT_RUN_DIR / name) != digest:
            raise SystemExit(f"{name} drifted")
    frame_vals: dict[str, list[float]] = {"all": [], "october_2026": [], "older": []}
    cohort_vals: dict[str, list[float]] = {"all": [], "october_2026": [], "older": []}
    n_protected_selected = sum(1 for row in selected if row.get("protected_membership"))
    n_missing_score = 0
    with scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["reddit_id"])
            compound = row.get("combined_compound")
            if compound is None:
                compound = row.get("compound")
            if compound is None:
                continue
            value = float(compound)
            stratum = "october_2026" if row.get("created_month") == "2026-10" else "older"
            if row.get("reserved_membership") or row.get("fresh_membership"):
                continue
            frame_vals["all"].append(value)
            frame_vals[stratum].append(value)
            if rid in selected_ids:
                cohort_vals["all"].append(value)
                cohort_vals[stratum].append(value)
    n_missing_score = max(0, len(selected_ids) - len(cohort_vals["all"]))
    payload = {
        "exploratory": True,
        "independent_validation": False,
        "vader_not_rerun": True,
        "accepted_af_analyses_not_overwritten": True,
        "protected_ids_excluded_from_sentiment_join": True,
        "n_protected_selected_excluded": n_protected_selected,
        "n_selected_missing_vader": n_missing_score,
        "broad_candidate_frame_nonprotected": {k: _summary(v) for k, v in frame_vals.items()},
        "exploratory_ai_selected_nonprotected": {k: _summary(v) for k, v in cohort_vals.items()},
        "accepted_af_hashes": A_F_HASHES,
        "interpretation": (
            "Broad-frame numbers reuse frozen VADER post scores and must be described as "
            "candidate-frame findings. AI-selected numbers are exploratory machine-filtered "
            "subsets, not independently validated software-update posts."
        ),
    }
    hits = scan_obj(payload)
    if hits:
        raise RuntimeError(f"sentiment public leak {hits[:8]}")
    atomic_write_json(SENTIMENT_DIR / "exploratory_sentiment_join.json", payload)
    atomic_write_json(PUBLIC_DIR / "exploratory_sentiment_join.json", payload)
    return payload


def write_manifest(bundle: dict[str, Any], post: dict[str, Any], sentiment: dict[str, Any]) -> None:
    files = {
        "predictions": RESTRICTED_DIR / "full" / "predictions.jsonl",
        "raw_predictions": RESTRICTED_DIR / "full" / "raw_predictions.jsonl",
        "checkpoint": RESTRICTED_DIR / "full" / "checkpoint.jsonl",
        "cohort": PUBLIC_DIR / "exploratory_ai_selected_cohort.jsonl",
        "quality": RESTRICTED_DIR / "full" / "quality_report.json",
        "evidence_validation": PUBLIC_DIR / "evidence_validation_report.json",
        "public": PUBLIC_DIR / "full_public_summary.json",
        "spend": RESTRICTED_DIR / "spend_ledger.json",
        "sentiment": SENTIMENT_DIR / "exploratory_sentiment_join.json",
        "protocol": WORKSPACE / "EXPLORATORY_FULL_FRAME_PROTOCOL_AMENDMENT_2026-10-29.md",
    }
    manifest = {
        "utc": utcnow(),
        "run_id": RUN_ID,
        "exploratory": True,
        "independent_validation": False,
        "development_gate_met": False,
        "prompt_sha256": prompt_v3_sha256(),
        "v1_prompt_sha256_unchanged": prompt_sha256(),
        "schema_sha256": schema_sha256(),
        "scope_version": SCOPE_VERSION,
        "model": MODEL_ALIAS,
        "spend": bundle["ledger"].snapshot(),
        "file_sha256": {name: sha256_file(path) for name, path in files.items() if path.exists()},
        "accepted_af_hashes_verified": A_F_HASHES,
        "used_protected_outcomes": False,
        "historical_audits_unchanged": True,
    }
    atomic_write_json(PUBLIC_DIR / "REPRODUCIBILITY_MANIFEST.json", manifest)
    public = post["public"]
    lines = [
        "# Exploratory full-frame v3 classification — 2026-10-29",
        "",
        "**Exploratory machine classification. Not independently validated.** "
        "The P≥0.75 / R≥0.80 development gate was not met. Thresholds were not changed.",
        "",
        f"- Frame n: {EXPECTED_FRAME_N}; complete coverage: {public.get('complete_coverage')}",
        f"- Label counts: {public.get('label_counts')}",
        f"- Status counts: {public.get('status_counts')}",
        f"- Exploratory AI-selected n: {public.get('n_ai_selected')} (excluding protected: {public.get('n_ai_selected_excluding_protected')})",
        f"- Errors: {public.get('n_errors')}; evidence-fail status rows: {public.get('n_evidence_fail_status')}",
        f"- Period breakdown: {public.get('by_period')}",
        f"- Spend: {public.get('spend')}",
        f"- Prompt SHA-256 `{prompt_v3_sha256()}`; scope `{SCOPE_VERSION}`",
        "",
        "The unseen 240-case validation packets were not used and were not labeled. "
        "No new human-review assignment was created.",
        "",
        "## Sentiment (exploratory join to frozen VADER)",
        "",
        f"- Broad nonprotected frame combined compound: {sentiment['broad_candidate_frame_nonprotected']['all']}",
        f"- Exploratory AI-selected nonprotected combined compound: {sentiment['exploratory_ai_selected_nonprotected']['all']}",
        "- Accepted A–F analyses were not overwritten.",
        "",
        "## Files",
        "",
        f"- Predictions: `{RESTRICTED_DIR / 'full' / 'predictions.jsonl'}`",
        f"- Cohort: `{PUBLIC_DIR / 'exploratory_ai_selected_cohort.jsonl'}`",
        f"- Manifest: `{PUBLIC_DIR / 'REPRODUCIBILITY_MANIFEST.json'}`",
        "",
    ]
    atomic_write_text(WORKSPACE / "EXPLORATORY_FULL_FRAME_REPORT_2026-10-29.md", "\n".join(lines))
    brief = [
        "# Brief for Professor Berhe and Codex — exploratory full-frame v3",
        "",
        "This is an exploratory classification of the frozen 37,596-post frame with unchanged v3. "
        "The development gate was not met. AI-selected posts are not independently validated software-update posts.",
        f"n_ai_selected={public.get('n_ai_selected')}; complete={public.get('complete_coverage')}; "
        f"spend_bound={public.get('spend', {}).get('bound_usd')}.",
        "A–F sentiment files unchanged. Unused independent-validation packets were not labeled.",
        "",
    ]
    atomic_write_text(WORKSPACE / "EXPLORATORY_FULL_FRAME_BRIEF_BERHE_CODEX_2026-10-29.md", "\n".join(brief))


def main() -> int:
    os.environ.setdefault(RESEARCH_RUN_ENV, "1")
    os.environ.setdefault(ALLOW_LIVE_ENV, "1")
    os.environ["RELEVANCE_CLASSIFY_TEST_MODE"] = "0"
    os.environ[PROMPT_VERSION_ENV] = "v3"
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_key()
    log("start", key_present=bool(os.environ.get("OPENAI_API_KEY")), external_env=loaded, prompt=os.environ.get(PROMPT_VERSION_ENV))
    assert_freeze()
    if not api_key_present() or not live_inference_allowed(True):
        log("blocker", kind="live_unavailable")
        return 3
    spend_path = seed_spend()
    ledger = SpendLedger(spend_path)
    try:
        bundle = classify_full(ledger)
    except (LiveInferenceForbidden, MissingAPIKeyError, OpenAIRequestError, SpendCeilingError) as exc:
        log("blocker", kind=type(exc).__name__, detail=str(exc)[:300])
        atomic_write_json(PUBLIC_DIR / "BLOCKER.json", {"kind": type(exc).__name__, "detail": str(exc), "utc": utcnow()})
        return 3
    post = apply_pass_and_write(bundle)
    sentiment = sentiment_join(post["selected"])
    write_manifest(bundle, post, sentiment)
    log("EXPLORATORY_FULL_COMPLETE", n_selected=len(post["selected"]), complete=post["quality"].get("complete_coverage"))
    print("EXPLORATORY_FULL_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
