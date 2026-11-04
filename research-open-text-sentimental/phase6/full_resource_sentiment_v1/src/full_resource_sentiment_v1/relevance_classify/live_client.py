"""OpenAI HTTP client for authorized research runs. Never prints API keys or post text."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from full_resource_sentiment_v1.relevance_classify.spec import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    LUNA_SYNC_INPUT_USD_PER_MILLION,
    LUNA_SYNC_OUTPUT_USD_PER_MILLION,
    MODEL_ALIAS,
    REASONING_EFFORT,
    TEMPERATURE,
    active_system_prompt,
    estimate_tokens_from_chars,
    openai_json_schema_format,
)
from full_resource_sentiment_v1.relevance_classify.providers import TransientProviderError

OPENAI_API_BASE = "https://api.openai.com/v1"
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class MissingAPIKeyError(RuntimeError):
    """OPENAI_API_KEY is absent. The key value is never included in the message."""


class OpenAIRequestError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"OpenAI HTTP {status}: {message}")


def api_key_present() -> bool:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    return bool(key and str(key).strip())


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if not key or not str(key).strip():
        raise MissingAPIKeyError("OPENAI_API_KEY is not set in the environment")
    return str(key).strip()


def estimate_sync_request_usd(
    submitted_text: str,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> float:
    input_tokens = estimate_tokens_from_chars(len(active_system_prompt()) + len(submitted_text or ""))
    return (input_tokens / 1_000_000) * LUNA_SYNC_INPUT_USD_PER_MILLION + (
        max_output_tokens / 1_000_000
    ) * LUNA_SYNC_OUTPUT_USD_PER_MILLION


def estimate_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_rate: float,
    output_rate: float,
) -> float:
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def responses_body(submitted_text: str, *, max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS) -> dict[str, Any]:
    return {
        "model": MODEL_ALIAS,
        "temperature": TEMPERATURE,
        "reasoning": {"effort": REASONING_EFFORT},
        "max_output_tokens": max_output_tokens,
        "text": {"format": openai_json_schema_format()},
        "input": [
            {"role": "system", "content": active_system_prompt()},
            {"role": "user", "content": submitted_text},
        ],
    }


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int]:
    usage = usage or {}
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens") or 0
    return int(input_tokens or 0), int(output_tokens or 0)


def parse_responses_payload(data: dict[str, Any]) -> dict[str, Any]:
    response_id = data.get("id")
    model = data.get("model")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    in_tok, out_tok = _usage_tokens(usage)
    cost = estimate_cost_usd(
        input_tokens=in_tok,
        output_tokens=out_tok,
        input_rate=LUNA_SYNC_INPUT_USD_PER_MILLION,
        output_rate=LUNA_SYNC_OUTPUT_USD_PER_MILLION,
    )
    error = data.get("error")
    if error:
        message = error if isinstance(error, str) else str(error.get("message") or error)
        return {
            "ok": False,
            "kind": "error",
            "text": message,
            "parsed": None,
            "response_id": response_id,
            "model": model,
            "usage": usage,
            "cost_usd": cost,
        }
    if data.get("status") in {"failed", "cancelled", "incomplete"} and not data.get("output"):
        return {
            "ok": False,
            "kind": "error",
            "text": str(data.get("status")),
            "parsed": None,
            "response_id": response_id,
            "model": model,
            "usage": usage,
            "cost_usd": cost,
        }

    parsed = data.get("output_parsed")
    text = None
    refused = False
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal" or item.get("refusal"):
            refused = True
            text = str(item.get("refusal") or item.get("content") or "refusal")
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"refusal", "output_refusal"}:
                refused = True
                text = str(content.get("refusal") or content.get("text") or "refusal")
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text") if text is None else text
            if content.get("parsed") is not None:
                parsed = content.get("parsed")
    if refused:
        return {
            "ok": False,
            "kind": "refusal",
            "text": text or "refusal",
            "parsed": None,
            "response_id": response_id,
            "model": model,
            "usage": usage,
            "cost_usd": cost,
        }
    if parsed is None and isinstance(text, str) and text.strip():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "kind": "invalid_json",
                "text": text,
                "parsed": None,
                "response_id": response_id,
                "model": model,
                "usage": usage,
                "cost_usd": cost,
            }
    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "kind": "invalid_json",
            "text": text,
            "parsed": None,
            "response_id": response_id,
            "model": model,
            "usage": usage,
            "cost_usd": cost,
        }
    return {
        "ok": True,
        "kind": "ok",
        "text": text if isinstance(text, str) else json.dumps(parsed, sort_keys=True),
        "parsed": parsed,
        "response_id": response_id,
        "model": model,
        "usage": usage,
        "cost_usd": cost,
    }


def parse_batch_output_line(line: dict[str, Any]) -> dict[str, Any]:
    custom_id = str(line.get("custom_id") or "")
    error = line.get("error")
    response = line.get("response") or {}
    if error:
        return {
            "custom_id": custom_id,
            "ok": False,
            "kind": "error",
            "text": json.dumps(error, sort_keys=True) if not isinstance(error, str) else error,
            "parsed": None,
            "response_id": None,
            "model": None,
            "usage": {},
            "cost_usd": 0.0,
        }
    status_code = response.get("status_code")
    body = response.get("body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return {
                "custom_id": custom_id,
                "ok": False,
                "kind": "invalid_json",
                "text": body,
                "parsed": None,
                "response_id": None,
                "model": None,
                "usage": {},
                "cost_usd": 0.0,
            }
    if not isinstance(body, dict):
        return {
            "custom_id": custom_id,
            "ok": False,
            "kind": "error",
            "text": f"batch status_code={status_code}",
            "parsed": None,
            "response_id": None,
            "model": None,
            "usage": {},
            "cost_usd": 0.0,
        }
    parsed = parse_responses_payload(body)
    parsed["custom_id"] = custom_id
    if status_code not in (None, 200) and parsed.get("ok"):
        parsed["ok"] = False
        parsed["kind"] = "error"
    return parsed


class OpenAIHttp:
    def __init__(self, api_key: str, *, timeout_s: int = 120, base: str = OPENAI_API_BASE) -> None:
        if not api_key:
            raise MissingAPIKeyError("OPENAI_API_KEY is not set in the environment")
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.base = base.rstrip("/")
        self._ssl = ssl.create_default_context()

    @classmethod
    def from_env(cls) -> "OpenAIHttp":
        return cls(_api_key())

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def _open(self, request: urllib.request.Request) -> dict[str, Any] | bytes:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s, context=self._ssl) as handle:
                raw = handle.read()
                content_type = handle.headers.get("Content-Type") or ""
                if "application/json" in content_type or raw[:1] in {b"{", b"["}:
                    return json.loads(raw.decode("utf-8"))
                return raw
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            snippet = body.replace(self._api_key, "[REDACTED]")[:500]
            if exc.code in TRANSIENT_STATUS:
                raise TransientProviderError(f"HTTP {exc.code}") from exc
            raise OpenAIRequestError(exc.code, snippet) from exc
        except TimeoutError as exc:
            raise TransientProviderError("timeout") from exc
        except json.JSONDecodeError as exc:
            raise TransientProviderError("invalid_json") from exc
        except urllib.error.URLError as exc:
            raise TransientProviderError("network") from exc
        except ssl.SSLError as exc:
            raise TransientProviderError("network") from exc
        except (BrokenPipeError, ConnectionError) as exc:
            raise TransientProviderError("network") from exc
        except OSError as exc:
            if getattr(exc, "errno", None) in {32, 54, 104}:
                raise TransientProviderError("network") from exc
            raise

    def create_response(self, submitted_text: str) -> dict[str, Any]:
        payload = json.dumps(responses_body(submitted_text)).encode("utf-8")
        request = urllib.request.Request(
            self.base + "/responses",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        result = self._open(request)
        if not isinstance(result, dict):
            raise OpenAIRequestError(200, "non-JSON responses payload")
        return result

    def upload_file(self, path: Path, *, purpose: str = "batch") -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        filename = path.name
        file_bytes = path.read_bytes()
        chunks = [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="purpose"\r\n\r\n{purpose}\r\n'.encode("utf-8"),
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                "Content-Type: application/jsonl\r\n\r\n"
            ).encode("utf-8"),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        body = b"".join(chunks)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        request = urllib.request.Request(
            self.base + "/files",
            data=body,
            headers=headers,
            method="POST",
        )
        result = self._open(request)
        if not isinstance(result, dict):
            raise OpenAIRequestError(200, "non-JSON file upload payload")
        return result

    def create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: str = "/v1/responses",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": "24h",
        }
        if metadata:
            body["metadata"] = {str(k): str(v)[:512] for k, v in metadata.items()}
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base + "/batches",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        result = self._open(request)
        if not isinstance(result, dict):
            raise OpenAIRequestError(200, "non-JSON batch payload")
        return result

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base + "/batches/" + batch_id,
            headers=self._headers(),
            method="GET",
        )
        result = self._open(request)
        if not isinstance(result, dict):
            raise OpenAIRequestError(200, "non-JSON batch status payload")
        return result

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base + "/batches/" + batch_id + "/cancel",
            data=b"{}",
            headers=self._headers(),
            method="POST",
        )
        result = self._open(request)
        if not isinstance(result, dict):
            raise OpenAIRequestError(200, "non-JSON batch cancel payload")
        return result

    def download_file(self, file_id: str, dest: Path) -> Path:
        request = urllib.request.Request(
            self.base + "/files/" + file_id + "/content",
            headers={"Authorization": f"Bearer {self._api_key}"},
            method="GET",
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s, context=self._ssl) as handle:
                dest.write_bytes(handle.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            snippet = body.replace(self._api_key, "[REDACTED]")[:500]
            if exc.code in TRANSIENT_STATUS:
                raise TransientProviderError(f"HTTP {exc.code}") from exc
            raise OpenAIRequestError(exc.code, snippet) from exc
        except TimeoutError as exc:
            raise TransientProviderError("timeout") from exc
        except urllib.error.URLError as exc:
            raise TransientProviderError("network") from exc
        except ssl.SSLError as exc:
            raise TransientProviderError("network") from exc
        except (BrokenPipeError, ConnectionError) as exc:
            raise TransientProviderError("network") from exc
        except OSError as exc:
            if getattr(exc, "errno", None) in {32, 54, 104}:
                raise TransientProviderError("network") from exc
            raise
        return dest

    def list_batches(self, *, limit: int = 100) -> list[dict[str, Any]]:
        request = urllib.request.Request(
            self.base + "/batches?limit=" + str(int(limit)),
            headers=self._headers(),
            method="GET",
        )
        result = self._open(request)
        if not isinstance(result, dict):
            raise OpenAIRequestError(200, "non-JSON batch list payload")
        data = result.get("data") or []
        return [row for row in data if isinstance(row, dict)]

    def submit_batch_file(self, path: Path, *, metadata: dict[str, str] | None = None) -> dict[str, Any]:
        uploaded = self.upload_file(Path(path), purpose="batch")
        file_id = uploaded.get("id")
        if not file_id:
            raise OpenAIRequestError(200, "upload missing file id")
        batch = self.create_batch(str(file_id), metadata=metadata)
        return {"file": uploaded, "batch": batch}

    def poll_batch(self, batch_id: str, *, interval_s: int = 30, max_wait_s: int = 60 * 60 * 20) -> dict[str, Any]:
        started = time.time()
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            status = self.get_batch(batch_id)
            if status.get("status") in terminal:
                return status
            if time.time() - started > max_wait_s:
                raise TransientProviderError(f"batch {batch_id} poll timeout")
            time.sleep(interval_s)
