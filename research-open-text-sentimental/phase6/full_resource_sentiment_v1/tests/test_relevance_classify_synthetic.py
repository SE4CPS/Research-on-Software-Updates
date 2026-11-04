"""Stage 2 synthetic-only tests. Do not load frozen Reddit posts or call a live model."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.pipeline import (
    CheckpointStore,
    DuplicateIdError,
    HumanCorrectionStore,
    assert_no_raw_text,
    classify_prepared,
    classify_records,
    public_row,
)
from full_resource_sentiment_v1.relevance_classify.planner import plan_batches, simulate_full_frame_budget
from full_resource_sentiment_v1.relevance_classify.preprocess import (
    ProtectedOutcomeSuppliedError,
    deterministic_unassessable,
    field_is_whole_removal_notice,
    preprocess_record,
    redact_one_way,
    reject_protected_outcomes,
)
from full_resource_sentiment_v1.relevance_classify.providers import (
    LiveInferenceForbidden,
    LiveOpenAIProvider,
    MockProvider,
    MockTurn,
)
from full_resource_sentiment_v1.relevance_classify.spec import STAGE2_LIVE_LOCKED, prompt_sha256, schema_sha256
from full_resource_sentiment_v1.relevance_classify.synthetic_suite import (
    classification_records,
    load_fixture_records,
    mock_scripts,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def _records_by_id() -> dict[str, dict]:
    return {row["synthetic_id"]: row for row in classification_records()}


def _provider() -> MockProvider:
    return MockProvider(mock_scripts())


def _ok_payload(**overrides):
    payload = {
        "predicted_label": "relevant",
        "decision_reason": "Names a firmware update and a consequence of installing it.",
        "evidence_phrase": "Android 16 QPR1",
        "ambiguity_reasons": [],
    }
    payload.update(overrides)
    return payload


def _classify_payload(tmp_path, payload, *, record_key="synth-version-change", synthetic_id="synth-schema"):
    record = dict(_records_by_id()[record_key])
    record["synthetic_id"] = synthetic_id
    provider = MockProvider({synthetic_id: [MockTurn("ok", payload)]})
    return classify_records([record], provider, checkpoint=CheckpointStore(tmp_path / f"{synthetic_id}.jsonl"))[0]


def test_stage2_live_lock_is_on():
    assert STAGE2_LIVE_LOCKED is True
    with pytest.raises(LiveInferenceForbidden):
        LiveOpenAIProvider(allow_live=True).complete({"synthetic_id": "x"})
    with pytest.raises(LiveInferenceForbidden):
        LiveOpenAIProvider().submit_batch("unused")


def test_no_reachable_live_network_path(monkeypatch, tmp_path):
    def blocked(*_args, **_kwargs):
        raise AssertionError("socket connect is not allowed in Stage 2 tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    assert "openai" not in sys.modules
    records = [_records_by_id()["synth-version-change"]]
    store = CheckpointStore(tmp_path / "ckpt.jsonl")
    rows = classify_records(records, _provider(), checkpoint=store)
    assert rows[0]["response_status"] == "ok"
    assert "openai" not in sys.modules


def test_software_version_change_is_relevant(tmp_path):
    record = _records_by_id()["synth-version-change"]
    rows = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))
    assert rows[0]["predicted_label"] == "relevant"
    assert rows[0]["evidence_phrase"] == "Android 16 QPR1"
    assert "Android 16 QPR1" in preprocess_record(record)["submitted_text"]


def test_news_status_update_is_not_relevant(tmp_path):
    record = _records_by_id()["synth-news-update"]
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["predicted_label"] == "not_relevant"
    assert row["evidence_phrase"] is None
    assert "multiple_update_meanings" in row["ambiguity_reasons"]


def test_bug_without_update_link_is_not_relevant(tmp_path):
    record = _records_by_id()["synth-bug-no-update"]
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["predicted_label"] == "not_relevant"
    assert row["evidence_phrase"] is None


def test_comment_only_update_is_excluded_from_model_input(tmp_path):
    record = _records_by_id()["synth-comment-only-update"]
    prepared = preprocess_record(record)
    assert prepared["comments_ignored"] is True
    assert "iOS 26.1" not in prepared["submitted_text"]
    assert "sourdough" in prepared["submitted_text"]
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["predicted_label"] == "not_relevant"


def test_multiple_updates_one_qualifying_is_relevant(tmp_path):
    record = _records_by_id()["synth-multiple-updates"]
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["predicted_label"] == "relevant"
    assert row["evidence_phrase"] == "Windows 11 KB5046617"


def test_body_removed_intact_title_stays_assessable(tmp_path):
    record = _records_by_id()["synth-body-removed-intact-title"]
    prepared = preprocess_record(record)
    assert prepared["unassessable_basis"] is None
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["predicted_label"] == "relevant"
    assert row["evidence_phrase"] and "iOS 18.2" in row["evidence_phrase"]


def test_both_fields_entirely_removed_are_deterministic_unassessable(tmp_path):
    record = _records_by_id()["synth-both-removed"]
    provider = _provider()
    row = classify_records([record], provider, checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["predicted_label"] == "unassessable"
    assert row["response_status"] == "skipped_deterministic"
    assert row["unassessable_basis"] == "deterministic_removal_notice"
    assert all(call["synthetic_id"] != "synth-both-removed" for call in provider.calls)


def test_quoting_removed_is_not_a_removal_notice():
    record = _records_by_id()["synth-quotes-removed"]
    prepared = preprocess_record(record)
    assert field_is_whole_removal_notice(record["title"]) is False
    assert field_is_whole_removal_notice(record["body"]) is False
    assert prepared["unassessable_basis"] is None
    assert deterministic_unassessable(record["title"], record["body"]) is None


def test_email_phone_url_and_handle_redaction():
    record = _records_by_id()["synth-pii"]
    prepared = preprocess_record(record)
    submitted = prepared["submitted_text"]
    assert "example.test" not in submitted
    assert "https://" not in submitted
    assert "415-555-0100" not in submitted
    assert "u/fake_handle" not in submitted
    assert "/u/other_handle" not in submitted
    assert "reddit.com/user/third_handle" not in submitted
    assert "[REDACTED_EMAIL]" in submitted
    assert "[REDACTED_PHONE]" in submitted
    assert "[REDACTED_URL]" in submitted
    assert "[REDACTED_HANDLE]" in submitted
    assert set(prepared["redaction_types"]) >= {"email", "phone", "url", "handle"}


def test_redaction_is_one_way_without_replacement_map():
    redacted, counts = redact_one_way("mail alex.fake@example.test")
    assert "replacement_map" not in counts
    assert "alex.fake@example.test" not in redacted


def test_positive_evidence_must_be_substring(tmp_path):
    record = _records_by_id()["synth-version-change"]
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    prepared = preprocess_record(record)
    evidence = row["evidence_phrase"]
    assert evidence
    assert evidence in prepared["title_submitted"] or evidence in prepared["body_submitted"]
    assert "TITLE:" not in evidence
    assert "BODY:" not in evidence


def test_negative_and_unclear_allow_null_evidence(tmp_path):
    store = CheckpointStore(tmp_path / "c.jsonl")
    rows = classify_records(
        [_records_by_id()["synth-not-relevant-null-evidence"], _records_by_id()["synth-unclear"]],
        _provider(),
        checkpoint=store,
    )
    assert rows[0]["predicted_label"] == "not_relevant"
    assert rows[0]["evidence_phrase"] is None
    assert rows[1]["predicted_label"] == "unclear"
    assert rows[1]["evidence_phrase"] is None


def test_long_text_sets_truncation_flag_and_does_not_silently_cut():
    record = _records_by_id()["synth-long-text"]
    prepared = preprocess_record(record, max_input_tokens=50)
    assert prepared["truncated"] is True
    assert prepared["submitted_text"].startswith("TITLE:")
    assert record["body"][:40] in prepared["body_submitted"]
    row = classify_prepared(prepared, _provider())
    assert row["response_status"] == "truncated_blocked"
    assert row["predicted_label"] is None
    assert row["predicted_label"] != "not_relevant"


def test_malformed_and_refused_output_are_errors_not_negatives(tmp_path):
    store = CheckpointStore(tmp_path / "c.jsonl")
    rows = classify_records(
        [_records_by_id()["synth-malformed"], _records_by_id()["synth-refusal"], _records_by_id()["synth-missing-field"]],
        _provider(),
        checkpoint=store,
    )
    assert rows[0]["response_status"] == "invalid_json"
    assert rows[1]["response_status"] == "refusal"
    assert rows[2]["response_status"] == "schema_miss"
    for row in rows:
        assert row["predicted_label"] is None
        assert row["review_priority"] == "error"


def test_invalid_positive_evidence_is_error_not_negative(tmp_path):
    record = _records_by_id()["synth-bad-evidence"]
    row = classify_records([record], _provider(), checkpoint=CheckpointStore(tmp_path / "c.jsonl"))[0]
    assert row["response_status"] == "evidence_fail"
    assert row["predicted_label"] is None
    assert row["predicted_label"] != "not_relevant"


def test_duplicate_ids_are_rejected(tmp_path):
    data = load_fixture_records()
    with pytest.raises(DuplicateIdError):
        classify_records(
            [data["synth-duplicate-a"], data["synth-duplicate-b"]],
            _provider(),
            checkpoint=CheckpointStore(tmp_path / "c.jsonl"),
        )


def test_retry_then_success_and_exhausted_timeout(tmp_path):
    store = CheckpointStore(tmp_path / "c.jsonl")
    ok_row, dead_row = classify_records(
        [_records_by_id()["synth-timeout-then-ok"], _records_by_id()["synth-timeout-exhaust"]],
        _provider(),
        checkpoint=store,
        max_retries=3,
    )
    assert ok_row["response_status"] == "ok"
    assert ok_row["retry_count"] == 2
    assert ok_row["predicted_label"] == "relevant"
    assert dead_row["response_status"] == "exhausted"
    assert dead_row["predicted_label"] is None
    assert dead_row["predicted_label"] != "not_relevant"


def test_checkpoint_does_not_overwrite_accepted_rows(tmp_path):
    record = _records_by_id()["synth-version-change"]
    path = tmp_path / "c.jsonl"
    first = classify_records([record], _provider(), checkpoint=CheckpointStore(path))[0]
    mutated = MockProvider(mock_scripts())
    mutated.scripts["synth-version-change"][0].payload = {
        "predicted_label": "not_relevant",
        "decision_reason": "should not overwrite",
        "evidence_phrase": None,
        "ambiguity_reasons": [],
    }
    second = classify_records([record], mutated, checkpoint=CheckpointStore(path))[0]
    assert second["predicted_label"] == "relevant"
    assert second["predicted_label"] == first["predicted_label"]
    assert mutated.calls == []


def test_human_corrections_do_not_overwrite_predictions(tmp_path):
    record = _records_by_id()["synth-news-update"]
    ckpt = CheckpointStore(tmp_path / "c.jsonl")
    row = classify_records([record], _provider(), checkpoint=ckpt)[0]
    corrections = HumanCorrectionStore(tmp_path / "corr.jsonl")
    corrections.add(
        synthetic_id=row["synthetic_id"],
        classifier_run_id=row["classifier_run_id"],
        corrected_label="relevant",
        reason="synthetic reviewer disagreement example",
    )
    reloaded = CheckpointStore(path=ckpt.path).rows[row["synthetic_id"]]
    assert reloaded["predicted_label"] == "not_relevant"
    assert corrections.rows[0]["corrected_label"] == "relevant"
    assert corrections.rows[0]["layer"] == "human_correction_v1"


def test_protected_outcome_fields_are_rejected(tmp_path):
    dirty = load_fixture_records()["synth-protected-outcome"]
    with pytest.raises(ProtectedOutcomeSuppliedError):
        reject_protected_outcomes(dirty)
    with pytest.raises(ProtectedOutcomeSuppliedError):
        preprocess_record(dirty)
    with pytest.raises(ProtectedOutcomeSuppliedError):
        classify_records([dirty], _provider(), checkpoint=CheckpointStore(tmp_path / "unused.jsonl"))


def test_protected_path_strings_are_rejected():
    record = {
        "synthetic_id": "synth-path",
        "title": "x",
        "body": "y",
        "source_path": "/lab/reserved_test.jsonl",
    }
    with pytest.raises(ProtectedOutcomeSuppliedError):
        preprocess_record(record)


def test_title_body_prose_may_contain_reserved_test():
    record = {
        "synthetic_id": "synth-reserved-prose",
        "title": "How reserved_test labels were explained in class",
        "body": "This invented post mentions reserved_test in prose, not as a source path.",
    }
    prepared = preprocess_record(record)
    assert prepared["synthetic_id"] == "synth-reserved-prose"
    assert "reserved_test" in prepared["title_submitted"]


def test_public_summaries_contain_no_raw_text(tmp_path):
    rows = classify_records(
        [_records_by_id()["synth-version-change"], _records_by_id()["synth-pii"]],
        _provider(),
        checkpoint=CheckpointStore(tmp_path / "c.jsonl"),
    )
    public = [public_row(row) for row in rows]
    for item in public:
        assert_no_raw_text(item)
        assert "title" not in item
        assert "body" not in item
        assert "evidence_phrase" not in item
        assert "decision_reason" not in item
        assert "submitted_text" not in item
        assert scan_obj(item) == []


def test_review_priority_is_not_confidence(tmp_path):
    row = classify_records(
        [_records_by_id()["synth-version-change"]],
        _provider(),
        checkpoint=CheckpointStore(tmp_path / "c.jsonl"),
    )[0]
    assert row["review_priority"] == "high"
    assert "not_confidence" in row["review_priority_reason"]
    assert "confidence" not in row


def test_batch_planner_shards_on_byte_request_and_queue_limits():
    records = [_records_by_id()["synth-version-change"] for _ in range(5)]
    for index, record in enumerate(records):
        record = dict(record)
        record["synthetic_id"] = f"synth-plan-{index}"
        records[index] = record
    plan = plan_batches(
        records,
        queued_token_limit=50_000,
        max_requests_per_file=2,
        max_bytes_per_file=50_000,
        max_output_tokens=256,
        proposed_ceiling_usd=25.0,
    )
    assert plan["upload"] is False
    assert plan["spend"] is False
    assert plan["cache_savings_assumed"] is False
    assert plan["n_shards"] >= 2
    assert plan["exceeds_proposed_ceiling"] is False
    assert all(shard["n_requests"] <= 2 for shard in plan["shards"])
    assert all(shard["n_bytes"] <= 50_000 for shard in plan["shards"])
    assert all(shard["queued_input_tokens"] <= 50_000 for shard in plan["shards"])


def test_full_frame_budget_simulation_does_not_use_capture():
    sim = simulate_full_frame_budget(
        n=37596,
        mean_submitted_chars=1146,
        queued_token_limit=20_000_000,
        max_output_tokens=256,
        proposed_ceiling_usd=25.0,
    )
    assert sim["used_frozen_capture"] is False
    assert sim["upload"] is False
    assert sim["spend"] is False
    assert sim["cache_savings_assumed"] is False
    assert sim["chars_per_token_is_tokenizer"] is False
    assert sim["feasible"] is True
    assert sim["n_shards_est"] >= 1
    assert sim["estimated_cost_usd_no_cache"] > 0
    assert sim["exceeds_proposed_ceiling"] is False


def test_planner_requires_account_queue_limit():
    with pytest.raises(ValueError, match="queued_token_limit"):
        plan_batches([_records_by_id()["synth-version-change"]], queued_token_limit=0)


def test_prompt_and_schema_hashes_are_stable():
    assert len(prompt_sha256()) == 64
    assert len(schema_sha256()) == 64
    assert prompt_sha256() == prompt_sha256()
    assert schema_sha256() == schema_sha256()


def test_schema_rejects_extra_key(tmp_path):
    row = _classify_payload(tmp_path, _ok_payload(confidence=0.9), synthetic_id="synth-extra-key")
    assert row["response_status"] == "schema_miss"
    assert row["predicted_label"] is None
    assert row["predicted_label"] != "not_relevant"


def test_schema_rejects_blank_reason(tmp_path):
    row = _classify_payload(tmp_path, _ok_payload(decision_reason="   "), synthetic_id="synth-blank-reason")
    assert row["response_status"] == "schema_miss"
    assert row["predicted_label"] is None


def test_schema_rejects_overlong_reason(tmp_path):
    row = _classify_payload(
        tmp_path,
        _ok_payload(decision_reason="word " * 41),
        synthetic_id="synth-overlong-reason",
    )
    assert row["response_status"] == "schema_miss"
    assert row["predicted_label"] is None


def test_negative_non_null_evidence_is_error(tmp_path):
    row = _classify_payload(
        tmp_path,
        {
            "predicted_label": "not_relevant",
            "decision_reason": "News recap rather than a software version change.",
            "evidence_phrase": "news update",
            "ambiguity_reasons": [],
        },
        record_key="synth-news-update",
        synthetic_id="synth-negative-quote",
    )
    assert row["response_status"] == "evidence_fail"
    assert row["predicted_label"] is None
    assert row["predicted_label"] != "not_relevant"


def test_wrapper_tokens_are_not_valid_positive_evidence(tmp_path):
    for evidence, suffix in (("TITLE:", "title-wrap"), ("BODY:", "body-wrap")):
        row = _classify_payload(
            tmp_path,
            _ok_payload(evidence_phrase=evidence),
            synthetic_id=f"synth-{suffix}",
        )
        assert row["response_status"] == "evidence_fail"
        assert row["predicted_label"] is None


def test_blank_positive_evidence_is_error(tmp_path):
    row = _classify_payload(tmp_path, _ok_payload(evidence_phrase="  "), synthetic_id="synth-blank-evidence")
    assert row["response_status"] == "evidence_fail"
    assert row["predicted_label"] is None


def test_title_body_boundary_span_is_not_valid_evidence(tmp_path):
    record = _records_by_id()["synth-version-change"]
    span = "after install\n\nBODY:\nI installed"
    submitted = preprocess_record(record)["submitted_text"]
    assert span in submitted
    assert span not in record["title"]
    assert span not in record["body"]
    row = _classify_payload(tmp_path, _ok_payload(evidence_phrase=span), synthetic_id="synth-span")
    assert row["response_status"] == "evidence_fail"
    assert row["predicted_label"] is None


def test_valid_title_quote_is_accepted(tmp_path):
    row = _classify_payload(tmp_path, _ok_payload(), synthetic_id="synth-valid-quote")
    assert row["response_status"] == "ok"
    assert row["predicted_label"] == "relevant"
    assert row["evidence_phrase"] == "Android 16 QPR1"


def test_oversized_request_rejected_on_empty_shard():
    record = _records_by_id()["synth-version-change"]
    with pytest.raises(ValueError, match="single request does not fit"):
        plan_batches([record], queued_token_limit=50_000, max_bytes_per_file=10)
    with pytest.raises(ValueError, match="single request does not fit"):
        plan_batches([record], queued_token_limit=1, max_bytes_per_file=50_000)


def test_planner_rejects_nonpositive_limits():
    record = _records_by_id()["synth-version-change"]
    with pytest.raises(ValueError, match="max_bytes_per_file"):
        plan_batches([record], queued_token_limit=50_000, max_bytes_per_file=0)
    with pytest.raises(ValueError, match="max_requests_per_file"):
        plan_batches([record], queued_token_limit=50_000, max_requests_per_file=0)
    with pytest.raises(ValueError, match="max_output_tokens"):
        plan_batches([record], queued_token_limit=50_000, max_output_tokens=0)


def test_full_frame_simulation_reports_infeasible_when_one_request_cannot_fit():
    tiny_queue = simulate_full_frame_budget(
        n=37596,
        mean_submitted_chars=1146,
        queued_token_limit=1,
        max_output_tokens=256,
    )
    assert tiny_queue["feasible"] is False
    assert tiny_queue["n_shards_est"] == 0
    assert tiny_queue["per_shard_request_cap"] == 0
    assert tiny_queue["estimated_cost_usd_no_cache"] is None
    tiny_bytes = simulate_full_frame_budget(
        n=10,
        mean_submitted_chars=1146,
        queued_token_limit=20_000_000,
        max_bytes_per_file=10,
    )
    assert tiny_bytes["feasible"] is False
    assert tiny_bytes["n_shards_est"] == 0


def test_frozen_assets_were_not_modified():
    post_scores = WORKSPACE / "results" / "broad_sentiment_2026-10-22-main" / "post_scores.jsonl"
    compact = WORKSPACE / "results" / "broad_capture_compact_2026-10-22-main.jsonl"
    labels = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_labels.csv"
    stats = WORKSPACE / "restricted" / "relevance_audit_final_2026-10-28" / "final_statistics.json"
    manifest = WORKSPACE / "restricted" / "broad_release_train" / "2026-10-22-main" / "manifest.json"
    i_archive = WORKSPACE / "restricted" / "accepted_workstream_i_2026-10-28" / "ARCHIVE_MANIFEST.json"
    assert sha256_file(post_scores) == "8ae14f59dca38a906b44d0202141f716ff288435d48ccf720f64a672cf5c9771"
    assert sha256_file(compact) == "54448fb09f01f7a8791b23b6f88c442d20e7a78874a2e802f3e40ddb43d35953"
    assert sha256_file(labels) == "07a52b4682844b88cd175da55bc342661310e19eb1a720df6a1ecfc44c9df45e"
    assert sha256_file(stats) == "e61173250bc3270b312a01058b6f8cdf4ecaf18aac91429252fdf23d31dcd1c7"
    assert sha256_file(manifest) == "3239a66564a693c72699dc9c25546f66ed839b05f3c68a26cf43a3db75853b8c"
    assert sha256_file(i_archive) == "4bb657263e471b6672b14c3b34114aa74651872646686b074683ebc73a4ecdff"
