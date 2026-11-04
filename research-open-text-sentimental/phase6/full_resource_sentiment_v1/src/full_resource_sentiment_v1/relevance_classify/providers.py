"""Classifier providers. Pytest stays locked. Authorized research runs use env gates."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from full_resource_sentiment_v1.relevance_classify.spec import (
    ALLOW_LIVE_ENV,
    DEFAULT_MAX_OUTPUT_TOKENS,
    MODEL_ALIAS,
    REASONING_EFFORT,
    RESEARCH_RUN_ENV,
    STAGE2_LIVE_LOCKED,
    TEMPERATURE,
    active_system_prompt,
    openai_json_schema_format,
)


class LiveInferenceForbidden(RuntimeError):
    """Raised when any live/network inference path is requested."""


class TransientProviderError(RuntimeError):
    """Timeout or rate-limit style failure that may be retried."""


class ClassifierProvider(Protocol):
    name: str

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class MockTurn:
    kind: str
    payload: dict[str, Any] | None = None
    message: str = ""


@dataclass
class MockProvider:
    """Scripted responses keyed by synthetic_id. Makes no network calls."""

    scripts: dict[str, list[MockTurn]]
    name: str = "mock"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursors: dict[str, int] = field(default_factory=dict)

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"synthetic_id": request["synthetic_id"], "body": request})
        post_id = request["synthetic_id"]
        turns = self.scripts.get(post_id)
        if not turns:
            raise TransientProviderError(f"no mock script for {post_id}")
        index = self._cursors.get(post_id, 0)
        turn = turns[min(index, len(turns) - 1)]
        self._cursors[post_id] = index + 1
        if turn.kind in {"timeout", "rate_limit"}:
            raise TransientProviderError(turn.message or turn.kind)
        if turn.kind == "invalid_json":
            return {
                "ok": False,
                "kind": "invalid_json",
                "text": "{not-json",
                "parsed": None,
                "response_id": f"mock-invalid-{post_id}",
            }
        if turn.kind == "refusal":
            return {
                "ok": False,
                "kind": "refusal",
                "text": turn.message or "refused",
                "parsed": None,
                "response_id": f"mock-refusal-{post_id}",
            }
        if turn.kind == "missing_field":
            payload = dict(turn.payload or {})
            return {
                "ok": True,
                "kind": "missing_field",
                "text": json.dumps(payload),
                "parsed": payload,
                "response_id": f"mock-missing-{post_id}",
            }
        if turn.kind == "ok":
            payload = dict(turn.payload or {})
            return {
                "ok": True,
                "kind": "ok",
                "text": json.dumps(payload, sort_keys=True),
                "parsed": payload,
                "response_id": f"mock-ok-{post_id}",
            }
        raise TransientProviderError(f"unknown mock kind {turn.kind}")


def pytest_is_running() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def research_run_requested() -> bool:
    return os.environ.get(RESEARCH_RUN_ENV) == "1" and os.environ.get(ALLOW_LIVE_ENV) == "1"


def live_inference_allowed(allow_live: bool) -> bool:
    if pytest_is_running():
        return False
    research = research_run_requested() and allow_live
    if research:
        return True
    if STAGE2_LIVE_LOCKED:
        return False
    if os.environ.get("RELEVANCE_CLASSIFY_TEST_MODE", "1") == "1":
        return False
    if not allow_live:
        return False
    return os.environ.get(ALLOW_LIVE_ENV) == "1"


class LiveOpenAIProvider:
    """Live OpenAI Responses/Batch provider. Blocked under pytest and without env gates."""

    name = "openai_live"

    def __init__(
        self,
        *,
        allow_live: bool = False,
        ledger: Any = None,
        client: Any = None,
        pricing: str = "sync",
    ) -> None:
        self.allow_live = allow_live
        self.ledger = ledger
        self.client = client
        self.pricing = pricing

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        if not live_inference_allowed(self.allow_live):
            raise LiveInferenceForbidden(_live_block_message())
        from full_resource_sentiment_v1.relevance_classify.live_client import (
            OpenAIHttp,
            estimate_sync_request_usd,
            parse_responses_payload,
        )

        client = self.client or OpenAIHttp.from_env()
        submitted = str(request.get("submitted_text") or "")
        estimate = estimate_sync_request_usd(submitted)
        ref = str(request.get("synthetic_id") or request.get("custom_id") or "unknown")
        if self.ledger is not None:
            self.ledger.authorize(estimate, ref=ref)
        try:
            raw = client.create_response(submitted)
            parsed = parse_responses_payload(raw)
            parsed["pricing"] = self.pricing
            if self.ledger is not None:
                actual = float(parsed.get("cost_usd") or 0.0)
                self.ledger.settle(estimate, actual, ref=ref)
            return parsed
        except Exception:
            if self.ledger is not None:
                self.ledger.fail(estimate, 0.0, ref=ref)
            raise

    def submit_batch(self, path: Any) -> dict[str, Any]:
        if not live_inference_allowed(self.allow_live):
            raise LiveInferenceForbidden(_live_block_message())
        from full_resource_sentiment_v1.relevance_classify.live_client import OpenAIHttp

        client = self.client or OpenAIHttp.from_env()
        return client.submit_batch_file(path)

    def download_batch(self, batch_id: str) -> dict[str, Any]:
        if not live_inference_allowed(self.allow_live):
            raise LiveInferenceForbidden(_live_block_message())
        from full_resource_sentiment_v1.relevance_classify.live_client import OpenAIHttp

        client = self.client or OpenAIHttp.from_env()
        return client.get_batch(batch_id)


def _live_block_message() -> str:
    if pytest_is_running():
        return (
            "Live LLM/API/Batch inference is forbidden under pytest. "
            "STAGE2_LIVE_LOCKED remains True; tests never construct a live HTTP client."
        )
    if STAGE2_LIVE_LOCKED and not research_run_requested():
        return (
            "Live LLM/API/Batch inference is locked. Set "
            f"{RESEARCH_RUN_ENV}=1 and {ALLOW_LIVE_ENV}=1 for an authorized research run."
        )
    return "Live inference is not allowed in this process."


def build_openai_batch_line(
    *,
    synthetic_id: str,
    submitted_text: str,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Future Batch JSONL request body. Not uploaded in Stage 2."""
    return {
        "custom_id": synthetic_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL_ALIAS,
            "temperature": TEMPERATURE,
            "reasoning": {"effort": REASONING_EFFORT},
            "max_output_tokens": max_output_tokens,
            "text": {"format": openai_json_schema_format()},
            "input": [
                {"role": "system", "content": active_system_prompt()},
                {"role": "user", "content": submitted_text},
            ],
        },
    }


def serialize_batch_line(line: dict[str, Any]) -> bytes:
    return (json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
