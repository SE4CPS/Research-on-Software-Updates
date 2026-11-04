"""Synthetic-only tests for the 240-case evaluation join. No real audit scoring."""

from __future__ import annotations

import pytest

from full_resource_sentiment_v1.relevance_classify.preprocess import ProtectedOutcomeSuppliedError
from full_resource_sentiment_v1.relevance_classify.evaluate import (
    GATE_MIN_RELEVANT_RETRIEVED,
    EvaluationError,
    evaluate_gate,
    join_case_codes,
    load_human_labels,
    load_predictions,
    to_classifier_inputs,
    wilson_95,
)
from full_resource_sentiment_v1.relevance_classify.gate_prepare import (
    GATE_REAL_RUN_LOCKED,
    GateRunLockedError,
    describe_provider_payload,
    gate_budget_simulation,
    load_real_locked_audit_for_gate,
    presentation_protocol,
    verify_frozen_fingerprints,
)
from full_resource_sentiment_v1.relevance_classify.spec import STAGE2_LIVE_LOCKED


def _label(code: str, final: str, stratum: str = "october_2026") -> dict:
    return {"case_code": code, "stratum": stratum, "final_label": final}


def _pred(code: str, predicted: str | None, status: str = "ok") -> dict:
    return {
        "case_code": code,
        "predicted_label": predicted,
        "response_status": status,
        "review_priority": "low",
    }


def _complete_set():
    labels = [
        _label("C-000001", "relevant", "october_2026"),
        _label("C-000002", "relevant", "october_2026"),
        _label("C-000003", "relevant", "older"),
        _label("C-000004", "relevant", "older"),
        _label("C-000005", "not_relevant", "october_2026"),
        _label("C-000006", "not_relevant", "october_2026"),
        _label("C-000007", "not_relevant", "older"),
        _label("C-000008", "not_relevant", "older"),
        _label("C-000009", "not_relevant", "older"),
        _label("C-000010", "unassessable", "older"),
    ]
    predictions = [
        _pred("C-000001", "relevant"),
        _pred("C-000002", "relevant"),
        _pred("C-000003", "relevant"),
        _pred("C-000004", "relevant"),
        _pred("C-000005", "not_relevant"),
        _pred("C-000006", "not_relevant"),
        _pred("C-000007", "not_relevant"),
        _pred("C-000008", "not_relevant"),
        _pred("C-000009", "not_relevant"),
        _pred("C-000010", "unassessable", "skipped_deterministic"),
    ]
    return labels, predictions


def test_wilson_interval_matches_codex_32_of_40_note():
    interval = wilson_95(32, 40)
    assert interval is not None
    assert interval[0] == pytest.approx(0.6524, abs=0.01)
    assert interval[1] == pytest.approx(0.8958, abs=0.01)
    assert GATE_MIN_RELEVANT_RETRIEVED == 32


def test_perfect_synthetic_join_passes_cost_control_gate():
    labels, predictions = _complete_set()
    result = evaluate_gate(labels, predictions)
    metrics = result["metrics"]
    assert result["join"]["complete"] is True
    assert metrics["true_positives"] == 4
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_negatives"] == 5
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["model_unassessable"] == 1
    assert result["gate"]["passed"] is True
    assert result["used_real_reddit_text"] is False


def test_errors_are_false_negatives_not_not_relevant():
    labels, predictions = _complete_set()
    predictions[3] = _pred("C-000004", None, "invalid_json")
    result = evaluate_gate(labels, predictions)
    metrics = result["metrics"]
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 3
    assert metrics["prediction_counts"]["error"] == 1
    assert metrics["prediction_counts"]["not_relevant"] == 5
    assert metrics["recall"] == 0.75
    assert result["gate"]["passed"] is False


def test_false_positive_and_false_negative_and_unclear():
    labels, predictions = _complete_set()
    predictions[2] = _pred("C-000003", "not_relevant")
    predictions[4] = _pred("C-000005", "relevant")
    predictions[5] = _pred("C-000006", "unclear")
    result = evaluate_gate(labels, predictions)
    metrics = result["metrics"]
    assert metrics["true_positives"] == 3
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["model_unclear"] == 1
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert result["gate"]["recall_ok"] is False


def test_missing_prediction_is_not_a_silent_negative():
    labels, predictions = _complete_set()
    dropped = predictions[0]
    rest = predictions[1:]
    with pytest.raises(EvaluationError, match="incomplete join"):
        evaluate_gate(labels, rest)
    labels_loaded = load_human_labels(labels)
    preds_loaded = load_predictions(rest)
    join = join_case_codes(labels_loaded, preds_loaded)
    assert join["complete"] is False
    assert "C-000001" in join["missing_predictions"]
    incomplete = evaluate_gate(labels, rest, require_complete=False)
    assert incomplete["gate"]["eligible"] is False
    assert incomplete["gate"]["passed"] is False
    assert incomplete["metrics"]["n_missing_predictions"] == 1
    # The remaining three relevant cases are scored; the missing relevant ID is not
    # converted into predicted not_relevant.
    assert incomplete["metrics"]["prediction_counts"].get("not_relevant") == 5
    assert dropped["predicted_label"] == "relevant"


def test_duplicate_prediction_ids_are_rejected():
    labels, predictions = _complete_set()
    predictions.append(_pred("C-000001", "not_relevant"))
    with pytest.raises(EvaluationError, match="duplicate prediction"):
        evaluate_gate(labels, predictions)


def test_duplicate_human_ids_are_rejected():
    labels, predictions = _complete_set()
    labels.append(_label("C-000001", "not_relevant"))
    with pytest.raises(EvaluationError, match="duplicate human"):
        evaluate_gate(labels, predictions)


def test_invalid_human_and_predicted_labels_are_rejected():
    labels, predictions = _complete_set()
    labels[0]["final_label"] = "maybe"
    with pytest.raises(EvaluationError, match="invalid human"):
        evaluate_gate(labels, predictions)
    labels, predictions = _complete_set()
    predictions[0]["predicted_label"] = "sort_of"
    with pytest.raises(EvaluationError, match="invalid predicted_label"):
        evaluate_gate(labels, predictions)


def test_incorrect_join_extra_id_is_rejected():
    labels, predictions = _complete_set()
    predictions.append(_pred("C-999999", "relevant"))
    with pytest.raises(EvaluationError, match="incomplete join"):
        evaluate_gate(labels, predictions)


def test_mapping_join_and_mismatched_reddit_id():
    labels, predictions = _complete_set()
    mapping = [{"case_code": row["case_code"], "reddit_id": f"rid-{row['case_code']}"} for row in labels]
    keyed_by_reddit = []
    for row in predictions:
        keyed_by_reddit.append(
            {
                "reddit_id": f"rid-{row['case_code']}",
                "predicted_label": row["predicted_label"],
                "response_status": row["response_status"],
            }
        )
    result = evaluate_gate(labels, keyed_by_reddit, mapping_rows=mapping)
    assert result["join"]["complete"] is True
    assert result["mapping_used"] is True
    bad = list(keyed_by_reddit)
    bad[0]["reddit_id"] = "rid-unknown"
    with pytest.raises(EvaluationError, match="not in the case_code map"):
        evaluate_gate(labels, bad, mapping_rows=mapping)


def test_predictions_cannot_carry_human_labels_or_protected_outcomes():
    labels, predictions = _complete_set()
    predictions[0]["human_relevant"] = "relevant"
    with pytest.raises(ProtectedOutcomeSuppliedError, match="human_relevant"):
        evaluate_gate(labels, predictions)
    labels, predictions = _complete_set()
    predictions[1]["final_label"] = "relevant"
    with pytest.raises(EvaluationError, match="final_label"):
        evaluate_gate(labels, predictions)


def test_classifier_inputs_drop_labels_and_sort_by_case_code():
    packets = [
        {"case_code": "C-000002", "title": "b", "body": "y", "relevance": ""},
        {"case_code": "C-000001", "title": "a", "body": "x", "relevance": ""},
    ]
    inputs = to_classifier_inputs(packets)
    assert [row["case_code"] for row in inputs] == ["C-000001", "C-000002"]
    assert all(row["synthetic_id"] == row["case_code"] for row in inputs)
    labeled = [{"case_code": "C-000003", "title": "t", "body": "b", "final_label": "relevant"}]
    with pytest.raises(EvaluationError, match="human label"):
        to_classifier_inputs(labeled)


def test_real_gate_loader_is_locked():
    assert GATE_REAL_RUN_LOCKED is True
    assert STAGE2_LIVE_LOCKED is True
    with pytest.raises(GateRunLockedError):
        load_real_locked_audit_for_gate()


def test_gate_budget_and_payload_do_not_upload():
    payload = describe_provider_payload()
    assert payload["upload"] is False
    assert payload["spend"] is False
    assert payload["model_id"] == "gpt-6-luna"
    assert payload["reasoning_effort"] == "none"
    assert payload["temperature"] == 0
    assert "human final_label or reviewer labels" in payload["not_sent_to_provider"]
    budget = gate_budget_simulation(n=240, queued_token_limit=2_000_000)
    assert budget["used_locked_packets"] is False
    assert budget["upload"] is False
    assert budget["n"] == 240
    assert budget["feasible"] is True
    assert budget["estimated_cost_usd_no_cache"] < 1.0
    protocol = presentation_protocol()
    assert protocol["custom_id"] == "case_code"
    assert protocol["join_after_scoring"].startswith("predictions.case_code")


def test_frozen_fingerprints_unchanged():
    observed = verify_frozen_fingerprints()
    assert len(observed) == 6
