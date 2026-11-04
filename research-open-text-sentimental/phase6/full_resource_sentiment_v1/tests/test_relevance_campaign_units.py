"""Unit tests for live-unlock, spend, F1, parsing, and validation draw. No live API."""

from __future__ import annotations

import json

import pytest

from full_resource_sentiment_v1.relevance_classify.evaluate import evaluate_gate, f1_score
from full_resource_sentiment_v1.relevance_classify.live_client import parse_responses_payload
from full_resource_sentiment_v1.relevance_classify.providers import (
    LiveInferenceForbidden,
    LiveOpenAIProvider,
    live_inference_allowed,
)
from full_resource_sentiment_v1.relevance_classify.spec import STAGE2_LIVE_LOCKED
from full_resource_sentiment_v1.relevance_classify.spend import SpendCeilingError, SpendLedger
from full_resource_sentiment_v1.relevance_classify.validation_draw import draw_validation_sample


def test_removal_heuristic_does_not_skip():
    from full_resource_sentiment_v1.relevance_classify.preprocess import (
        deterministic_unassessable,
        preprocess_record,
        removal_notice_heuristic,
    )

    title = "[ Removed by Reddit ]"
    body = "[ Removed by Reddit on account of violating the content policy. ]"
    assert removal_notice_heuristic(title, body) is True
    assert deterministic_unassessable(title, body) is None
    prepared = preprocess_record({"synthetic_id": "synth-heu", "title": title, "body": body})
    assert prepared["removal_notice_heuristic"] is True
    assert prepared["unassessable_basis"] is None


def test_stage2_lock_constant_remains_true():
    assert STAGE2_LIVE_LOCKED is True
    with pytest.raises(LiveInferenceForbidden):
        LiveOpenAIProvider(allow_live=True).complete({"synthetic_id": "x"})


def test_pytest_blocks_research_env(monkeypatch):
    monkeypatch.setenv("RELEVANCE_CLASSIFY_RESEARCH_RUN", "1")
    monkeypatch.setenv("RELEVANCE_CLASSIFY_ALLOW_LIVE", "1")
    assert live_inference_allowed(True) is False


def test_research_override_outside_pytest(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("RELEVANCE_CLASSIFY_RESEARCH_RUN", "1")
    monkeypatch.setenv("RELEVANCE_CLASSIFY_ALLOW_LIVE", "1")
    assert live_inference_allowed(True) is True
    assert live_inference_allowed(False) is False
    monkeypatch.delenv("RELEVANCE_CLASSIFY_RESEARCH_RUN")
    assert live_inference_allowed(True) is False


def test_spend_ledger_enforces_ceiling(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", ceiling_usd=1.0, reserve_usd=0.2)
    ledger.authorize(0.5, ref="a")
    ledger.settle(0.5, 0.4, ref="a")
    with pytest.raises(SpendCeilingError):
        ledger.authorize(0.5, ref="b")
    snap = ledger.snapshot()
    assert snap["actual_usd"] == pytest.approx(0.4)
    assert snap["remaining_usd"] == pytest.approx(0.4)


def test_f1_and_failure_codes():
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.75, 0.8) == pytest.approx(2 * 0.75 * 0.8 / 1.55)
    labels = [{"case_code": f"C-{i:06d}", "stratum": "older", "final_label": "relevant"} for i in range(1, 5)]
    labels += [{"case_code": f"C-{i:06d}", "stratum": "older", "final_label": "not_relevant"} for i in range(5, 10)]
    preds = [
        {"case_code": "C-000001", "predicted_label": "relevant", "response_status": "ok"},
        {"case_code": "C-000002", "predicted_label": "not_relevant", "response_status": "ok"},
        {"case_code": "C-000003", "predicted_label": "relevant", "response_status": "ok"},
        {"case_code": "C-000004", "predicted_label": "relevant", "response_status": "ok"},
        {"case_code": "C-000005", "predicted_label": "relevant", "response_status": "ok"},
        {"case_code": "C-000006", "predicted_label": "not_relevant", "response_status": "ok"},
        {"case_code": "C-000007", "predicted_label": "not_relevant", "response_status": "ok"},
        {"case_code": "C-000008", "predicted_label": "not_relevant", "response_status": "ok"},
        {"case_code": "C-000009", "predicted_label": "not_relevant", "response_status": "ok"},
    ]
    result = evaluate_gate(labels, preds)
    assert result["metrics"]["false_negative_case_codes"] == ["C-000002"]
    assert result["metrics"]["false_positive_case_codes"] == ["C-000005"]
    assert result["metrics"]["f1"] == pytest.approx(f1_score(result["metrics"]["precision"], result["metrics"]["recall"]))


def test_parse_structured_output():
    payload = {
        "id": "resp_1",
        "model": "gpt-6-luna",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(
                            {
                                "predicted_label": "not_relevant",
                                "decision_reason": "News update not a software change.",
                                "evidence_phrase": None,
                                "ambiguity_reasons": [],
                            }
                        ),
                    }
                ],
            }
        ],
    }
    parsed = parse_responses_payload(payload)
    assert parsed["ok"] is True
    assert parsed["parsed"]["predicted_label"] == "not_relevant"
    assert parsed["response_id"] == "resp_1"


def test_validation_draw_four_strata_hides_ai():
    rows = []
    records = {}
    for i in range(80):
        rid = f"t{i:03d}"
        selected = i < 20
        stratum = "october_2026" if i % 2 == 0 else "older"
        rows.append(
            {
                "reddit_id": rid,
                "synthetic_id": rid,
                "predicted_label": "relevant" if selected else "not_relevant",
                "response_status": "ok",
                "stratum": stratum,
                "protected_membership": False,
            }
        )
        records[rid] = {"reddit_id": rid, "title": f"title {i}", "body": f"body {i}", "stratum": stratum}
    drawn = draw_validation_sample(
        rows,
        records,
        exclude_reddit_ids={"t000"},
        exclude_protected_ids=set(),
        n_per_stratum=5,
        seed=20261029,
    )
    assert drawn["n_drawn"] == 20
    assert set(drawn["drawn_counts"]) == {
        "october_ai_selected",
        "october_ai_not_selected",
        "older_ai_selected",
        "older_ai_not_selected",
    }
    assert all(p["relevance"] == "" for p in drawn["packet_a"])
    assert "predicted_label" not in drawn["packet_a"][0]
    assert "reddit_id" not in drawn["packet_a"][0]
    assert drawn["human_labels_invented"] is False
    assert drawn["private_rows"][0]["ai_predicted_not_human_confirmed"] is True
    assert "t000" not in {row["reddit_id"] for row in drawn["private_rows"]}
