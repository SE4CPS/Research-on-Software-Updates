"""Invented posts and mock scripts for Stage 2. Not Reddit capture text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.relevance_classify.providers import MockTurn

FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "synthetic_relevance_posts.json"
)


def load_fixture_records() -> dict[str, dict[str, Any]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    records = {key: dict(value) for key, value in payload.items()}
    records["synth-long-text"]["body"] = ("long synthetic paragraph. " * 400).strip()
    return records


def classification_records() -> list[dict[str, Any]]:
    data = load_fixture_records()
    skip = {"synth-duplicate-b", "synth-protected-outcome"}
    return [data[key] for key in data if key not in skip]


def mock_scripts() -> dict[str, list[MockTurn]]:
    return {
        "synth-version-change": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "Names a firmware update and a consequence of installing it.",
                    "evidence_phrase": "Android 16 QPR1",
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-news-update": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "not_relevant",
                    "decision_reason": "Update means a news recap, not a software version change.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": ["multiple_update_meanings"],
                },
            )
        ],
        "synth-bug-no-update": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "not_relevant",
                    "decision_reason": "Describes a bug without an identifiable version change.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-comment-only-update": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "not_relevant",
                    "decision_reason": "Original title and body are about baking; comments are not visible.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-multiple-updates": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "At least one named software patch is discussed.",
                    "evidence_phrase": "Windows 11 KB5046617",
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-body-removed-intact-title": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "The intact title names an iOS version change and a crash.",
                    "evidence_phrase": "iOS 18.2 made Safari crash",
                    "ambiguity_reasons": ["short_text"],
                },
            )
        ],
        "synth-quotes-removed": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "not_relevant",
                    "decision_reason": "Quotes a placeholder; not a removed original post and not a software update.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-pii": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "Names a macOS version change.",
                    "evidence_phrase": "macOS 15.1",
                    "ambiguity_reasons": ["redaction_affects_key_phrase"],
                },
            )
        ],
        "synth-not-relevant-null-evidence": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "not_relevant",
                    "decision_reason": "Outdoor recap with no software-version content.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-long-text": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "unclear",
                    "decision_reason": "Long synthetic filler without a version change.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": ["short_text"],
                },
            )
        ],
        "synth-malformed": [MockTurn("invalid_json")],
        "synth-refusal": [MockTurn("refusal", message="provider refusal")],
        "synth-missing-field": [
            MockTurn("missing_field", {"predicted_label": "relevant", "decision_reason": "incomplete"})
        ],
        "synth-bad-evidence": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "Positive label with invented evidence.",
                    "evidence_phrase": "this phrase is not in the post",
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-timeout-then-ok": [
            MockTurn("timeout", message="timeout"),
            MockTurn("timeout", message="timeout"),
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "Names a Kindle firmware update that is stuck.",
                    "evidence_phrase": "Kindle firmware 5.18.1",
                    "ambiguity_reasons": [],
                },
            ),
        ],
        "synth-timeout-exhaust": [
            MockTurn("timeout"),
            MockTurn("timeout"),
            MockTurn("timeout"),
            MockTurn("timeout"),
        ],
        "synth-duplicate": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "relevant",
                    "decision_reason": "Mentions a Windows insider build.",
                    "evidence_phrase": "Windows 12 insider build 26100",
                    "ambiguity_reasons": [],
                },
            )
        ],
        "synth-unclear": [
            MockTurn(
                "ok",
                {
                    "predicted_label": "unclear",
                    "decision_reason": "Too thin to decide whether a software update occurred.",
                    "evidence_phrase": None,
                    "ambiguity_reasons": ["short_text"],
                },
            )
        ],
    }
