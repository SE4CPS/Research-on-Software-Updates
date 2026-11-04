"""Prepare, but do not execute, the 240-case development gate.

Real audit text is not loaded into the classifier here. Live inference stays locked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.hashing import sha256_file
from full_resource_sentiment_v1.paths import WORKSPACE
from full_resource_sentiment_v1.relevance_classify.planner import simulate_full_frame_budget
from full_resource_sentiment_v1.relevance_classify.spec import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_ALIAS,
    MODEL_OUTPUT_SCHEMA,
    PROMPT_VERSION,
    PROVIDER_NAME,
    REASONING_EFFORT,
    SCHEMA_VERSION,
    STAGE2_LIVE_LOCKED,
    SYSTEM_PROMPT,
    TEMPERATURE,
    estimate_tokens_from_chars,
    openai_json_schema_format,
    prompt_sha256,
    schema_sha256,
    user_message,
)

# Hard lock: flip only after Codex/PI authorize a real 240-case scoring run.
GATE_REAL_RUN_LOCKED = True

EXPECTED_FROZEN_SHA256 = {
    "results/broad_sentiment_2026-10-22-main/post_scores.jsonl": (
        "8ae14f59dca38a906b44d0202141f716ff288435d48ccf720f64a672cf5c9771"
    ),
    "results/broad_capture_compact_2026-10-22-main.jsonl": (
        "54448fb09f01f7a8791b23b6f88c442d20e7a78874a2e802f3e40ddb43d35953"
    ),
    "restricted/relevance_audit_final_2026-10-28/final_labels.csv": (
        "07a52b4682844b88cd175da55bc342661310e19eb1a720df6a1ecfc44c9df45e"
    ),
    "restricted/relevance_audit_final_2026-10-28/final_statistics.json": (
        "e61173250bc3270b312a01058b6f8cdf4ecaf18aac91429252fdf23d31dcd1c7"
    ),
    "restricted/broad_release_train/2026-10-22-main/manifest.json": (
        "3239a66564a693c72699dc9c25546f66ed839b05f3c68a26cf43a3db75853b8c"
    ),
    "restricted/accepted_workstream_i_2026-10-28/ARCHIVE_MANIFEST.json": (
        "4bb657263e471b6672b14c3b34114aa74651872646686b074683ebc73a4ecdff"
    ),
}

FUTURE_INPUTS = {
    "human_labels": "restricted/relevance_audit_final_2026-10-28/final_labels.csv",
    "id_map": "restricted/relevance_audit_sample_2026-10-28/private_id_key.jsonl",
    "redacted_packets": "restricted/relevance_audit_sample_2026-10-28/reviewer packets (title/body only)",
    "join_key": "case_code",
    "classifier_custom_id": "case_code",
    "reddit_id_use": "provenance after scoring only; not placed in the model user message",
}


class GateRunLockedError(RuntimeError):
    """Raised if a real 240-case scoring path is requested before authorization."""


def assert_real_gate_locked() -> None:
    if GATE_REAL_RUN_LOCKED or STAGE2_LIVE_LOCKED:
        raise GateRunLockedError(
            "Real 240-case scoring is not authorized. GATE_REAL_RUN_LOCKED="
            f"{GATE_REAL_RUN_LOCKED}, STAGE2_LIVE_LOCKED={STAGE2_LIVE_LOCKED}."
        )


def load_real_locked_audit_for_gate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    assert_real_gate_locked()
    raise GateRunLockedError("unreachable")


def verify_frozen_fingerprints(*, workspace: Path = WORKSPACE) -> dict[str, str]:
    """Hash-only integrity check. Does not parse labels into an evaluator or score posts."""
    observed: dict[str, str] = {}
    for relative, expected in EXPECTED_FROZEN_SHA256.items():
        path = workspace / relative
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"frozen hash mismatch {relative}: {digest} != {expected}")
        observed[relative] = digest
    return observed


def describe_provider_payload() -> dict[str, Any]:
    """What a later authorized 240-case Batch request would contain. Nothing is sent now."""
    example_user = user_message("[redacted title]", "[redacted body]")
    return {
        "provider": PROVIDER_NAME,
        "model_id": MODEL_ALIAS,
        "model_snapshot": "alias only; pin the resolved snapshot at live microcheck",
        "endpoint": "OpenAI Batch /v1/responses (planned; not submitted)",
        "temperature": TEMPERATURE,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "schema_version": SCHEMA_VERSION,
        "schema_sha256": schema_sha256(),
        "json_schema_strict": True,
        "response_format": openai_json_schema_format()["name"],
        "schema_required_keys": list(MODEL_OUTPUT_SCHEMA["required"]),
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "system_prompt_tokens_est": estimate_tokens_from_chars(len(SYSTEM_PROMPT)),
        "user_message_template": "TITLE:\\n{redacted_title}\\n\\nBODY:\\n{redacted_body}\\n",
        "user_message_example_chars": len(example_user),
        "sent_to_provider": [
            "frozen system prompt (human guide + output contract)",
            "JSON schema",
            "redacted title and body for one post",
        ],
        "not_sent_to_provider": [
            "human final_label or reviewer labels",
            "comments",
            "subreddit, route, sentiment, stratum",
            "protected evaluation outcomes",
            "raw unredacted PII",
            "reddit_id inside the user message (custom_id is case_code)",
        ],
        "cache_savings_assumed": False,
        "upload": False,
        "spend": False,
        "live_locked": STAGE2_LIVE_LOCKED,
        "real_gate_locked": GATE_REAL_RUN_LOCKED,
    }


def gate_budget_simulation(
    *,
    n: int = 240,
    mean_submitted_chars: int = 1146,
    queued_token_limit: int = 2_000_000,
) -> dict[str, Any]:
    """Length-only cost check for 240 requests. Does not read locked packets."""
    sim = simulate_full_frame_budget(
        n=n,
        mean_submitted_chars=mean_submitted_chars,
        queued_token_limit=queued_token_limit,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        proposed_ceiling_usd=25.0,
    )
    sim["population"] = "synthetic_length_standin_for_possible_240_gate"
    sim["used_locked_packets"] = False
    sim["authorization_required_before_spend"] = True
    return sim


def presentation_protocol() -> dict[str, Any]:
    return {
        "unit": "one locked audit case_code",
        "n": 240,
        "human_labels": EXPECTED_FROZEN_SHA256[
            "restricted/relevance_audit_final_2026-10-28/final_labels.csv"
        ],
        "presentation_order": "lexicographic case_code",
        "model_input_fields": ["title", "body"],
        "text_source_when_authorized": (
            "the same redacted packet title/body the human reviewers saw, "
            "not comments and not live Reddit"
        ),
        "custom_id": "case_code",
        "join_after_scoring": "predictions.case_code == final_labels.case_code",
        "reddit_id_join": "private_id_key.jsonl after scoring for provenance only",
        "protected_outcomes": "never read; the 240 sample already excluded the 693 protected IDs",
        "prediction_table_excludes": ["human_relevant", "final_label", "protected outcomes"],
        "human_table_excludes": "model output; labels stay in the locked audit files",
        "one_pass": True,
        "no_prompt_tuning_on_viewed_scores": (
            "If prompt/model change after viewing the 240 comparison, this set is a "
            "development set and is not an untouched final test."
        ),
        "future_inputs": FUTURE_INPUTS,
    }
