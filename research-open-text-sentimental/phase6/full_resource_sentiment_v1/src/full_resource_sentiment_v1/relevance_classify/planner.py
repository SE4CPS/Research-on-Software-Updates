"""Dry-run Batch planner. Counts serialized JSONL bytes. Uploads and spends nothing."""

from __future__ import annotations

from typing import Any

from full_resource_sentiment_v1.relevance_classify.preprocess import preprocess_record
from full_resource_sentiment_v1.relevance_classify.providers import build_openai_batch_line, serialize_batch_line
from full_resource_sentiment_v1.relevance_classify.spec import (
    BATCH_MAX_BYTES_PER_FILE,
    BATCH_MAX_REQUESTS_PER_FILE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    LUNA_BATCH_INPUT_USD_PER_MILLION,
    LUNA_BATCH_OUTPUT_USD_PER_MILLION,
    PROPOSED_SPEND_CEILING_USD,
    active_system_prompt,
    estimate_tokens_from_chars,
)


def conservative_cost_usd(*, n: int, input_tokens_each: int, max_output_tokens: int) -> float:
    """No-cache costing. Bills the configured max output tokens, not a hoped-for short answer."""
    input_cost = (n * input_tokens_each / 1_000_000) * LUNA_BATCH_INPUT_USD_PER_MILLION
    output_cost = (n * max_output_tokens / 1_000_000) * LUNA_BATCH_OUTPUT_USD_PER_MILLION
    return input_cost + output_cost


def validate_positive_limits(
    *,
    queued_token_limit: int,
    max_requests_per_file: int,
    max_bytes_per_file: int,
    max_output_tokens: int,
) -> None:
    if queued_token_limit <= 0:
        raise ValueError("queued_token_limit is an account-specific input and must be > 0")
    if max_requests_per_file <= 0:
        raise ValueError("max_requests_per_file must be > 0")
    if max_bytes_per_file <= 0:
        raise ValueError("max_bytes_per_file must be > 0")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be > 0")


def plan_batches(
    records: list[dict[str, Any]],
    *,
    queued_token_limit: int,
    max_requests_per_file: int = BATCH_MAX_REQUESTS_PER_FILE,
    max_bytes_per_file: int = BATCH_MAX_BYTES_PER_FILE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    proposed_ceiling_usd: float = PROPOSED_SPEND_CEILING_USD,
) -> dict[str, Any]:
    validate_positive_limits(
        queued_token_limit=queued_token_limit,
        max_requests_per_file=max_requests_per_file,
        max_bytes_per_file=max_bytes_per_file,
        max_output_tokens=max_output_tokens,
    )

    prompt_tokens = estimate_tokens_from_chars(len(active_system_prompt()))
    shards: list[dict[str, Any]] = []
    current: dict[str, Any] = _empty_shard(0)
    skipped_deterministic = 0
    truncated_blocked = 0

    def close_shard() -> None:
        nonlocal current
        if current["n_requests"] == 0:
            return
        shards.append(current)
        current = _empty_shard(len(shards))

    for record in records:
        prepared = preprocess_record(record)
        if prepared["unassessable_basis"]:
            skipped_deterministic += 1
            continue
        if prepared["truncated"]:
            truncated_blocked += 1
            continue
        line = build_openai_batch_line(
            synthetic_id=prepared["synthetic_id"],
            submitted_text=prepared["submitted_text"],
            max_output_tokens=max_output_tokens,
        )
        encoded = serialize_batch_line(line)
        n_bytes = len(encoded)
        input_tokens = prompt_tokens + prepared["input_tokens_est"]
        if n_bytes > max_bytes_per_file or input_tokens > queued_token_limit:
            raise ValueError(
                f"{prepared['synthetic_id']} single request does not fit an empty shard "
                f"(bytes={n_bytes}/{max_bytes_per_file}, tokens={input_tokens}/{queued_token_limit})"
            )
        would_exceed = (
            current["n_requests"] + 1 > max_requests_per_file
            or current["n_bytes"] + n_bytes > max_bytes_per_file
            or current["queued_input_tokens"] + input_tokens > queued_token_limit
        )
        if current["n_requests"] > 0 and would_exceed:
            close_shard()
        current["n_requests"] += 1
        current["n_bytes"] += n_bytes
        current["queued_input_tokens"] += input_tokens
        current["ids"].append(prepared["synthetic_id"])
        current["serialized_head_sha256_inputs"].append(prepared["model_input_sha256"])

    close_shard()

    n_model = sum(shard["n_requests"] for shard in shards)
    total_input_tokens = sum(shard["queued_input_tokens"] for shard in shards)
    estimated_cost = (total_input_tokens / 1_000_000) * LUNA_BATCH_INPUT_USD_PER_MILLION + (
        n_model * max_output_tokens / 1_000_000
    ) * LUNA_BATCH_OUTPUT_USD_PER_MILLION

    return {
        "n_input_records": len(records),
        "n_model_requests": n_model,
        "n_skipped_deterministic": skipped_deterministic,
        "n_truncated_blocked": truncated_blocked,
        "n_shards": len(shards),
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "n_requests": shard["n_requests"],
                "n_bytes": shard["n_bytes"],
                "queued_input_tokens": shard["queued_input_tokens"],
                "ids": shard["ids"],
            }
            for shard in shards
        ],
        "max_requests_per_file": max_requests_per_file,
        "max_bytes_per_file": max_bytes_per_file,
        "queued_token_limit": queued_token_limit,
        "prompt_tokens_est": prompt_tokens,
        "max_output_tokens": max_output_tokens,
        "cache_savings_assumed": False,
        "cache_note": "Prompt caching is optional and does not guarantee savings; this plan uses no-cache prices.",
        "input_usd_per_million": LUNA_BATCH_INPUT_USD_PER_MILLION,
        "output_usd_per_million": LUNA_BATCH_OUTPUT_USD_PER_MILLION,
        "total_input_tokens_est": total_input_tokens,
        "total_output_tokens_billed_est": n_model * max_output_tokens,
        "estimated_cost_usd_no_cache": round(estimated_cost, 6),
        "proposed_ceiling_usd": proposed_ceiling_usd,
        "exceeds_proposed_ceiling": estimated_cost > proposed_ceiling_usd,
        "upload": False,
        "spend": False,
        "chars_per_token_is_tokenizer": False,
    }


def simulate_full_frame_budget(
    *,
    n: int,
    mean_submitted_chars: int,
    queued_token_limit: int,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    proposed_ceiling_usd: float = PROPOSED_SPEND_CEILING_USD,
    max_bytes_per_file: int = BATCH_MAX_BYTES_PER_FILE,
    max_requests_per_file: int = BATCH_MAX_REQUESTS_PER_FILE,
) -> dict[str, Any]:
    """Length-only simulation. Does not read frozen Reddit posts."""
    validate_positive_limits(
        queued_token_limit=queued_token_limit,
        max_requests_per_file=max_requests_per_file,
        max_bytes_per_file=max_bytes_per_file,
        max_output_tokens=max_output_tokens,
    )
    if n < 0 or mean_submitted_chars < 0:
        raise ValueError("n and mean_submitted_chars must be >= 0")
    prompt_tokens = estimate_tokens_from_chars(len(active_system_prompt()))
    user_tokens = estimate_tokens_from_chars(mean_submitted_chars)
    input_each = prompt_tokens + user_tokens
    stand_in = build_openai_batch_line(
        synthetic_id="synth-budget-standin",
        submitted_text="x" * mean_submitted_chars,
        max_output_tokens=max_output_tokens,
    )
    bytes_each = len(serialize_batch_line(stand_in))
    request_fits = bytes_each <= max_bytes_per_file and input_each <= queued_token_limit
    if not request_fits:
        reason = (
            f"single request does not fit a shard (bytes={bytes_each}/{max_bytes_per_file}, "
            f"tokens={input_each}/{queued_token_limit})"
        )
        return {
            "n": n,
            "mean_submitted_chars": mean_submitted_chars,
            "prompt_tokens_est": prompt_tokens,
            "user_tokens_est": user_tokens,
            "input_tokens_each_est": input_each,
            "serialized_bytes_each": bytes_each,
            "queued_token_limit": queued_token_limit,
            "max_bytes_per_file": max_bytes_per_file,
            "max_requests_per_file": max_requests_per_file,
            "per_shard_request_cap": 0,
            "n_shards_est": 0,
            "feasible": False,
            "infeasible_reason": reason,
            "estimated_cost_usd_no_cache": None,
            "proposed_ceiling_usd": proposed_ceiling_usd,
            "exceeds_proposed_ceiling": None,
            "cache_savings_assumed": False,
            "chars_per_token_is_tokenizer": False,
            "used_frozen_capture": False,
            "upload": False,
            "spend": False,
        }
    n_by_requests = min(max_requests_per_file, n) if n else 0
    n_by_bytes = max_bytes_per_file // max(bytes_each, 1)
    n_by_queue = queued_token_limit // max(input_each, 1)
    per_shard = min(n_by_requests, n_by_bytes, n_by_queue) if n else 0
    n_shards = 0 if n == 0 or per_shard == 0 else (n + per_shard - 1) // per_shard
    cost = conservative_cost_usd(n=n, input_tokens_each=input_each, max_output_tokens=max_output_tokens)
    return {
        "n": n,
        "mean_submitted_chars": mean_submitted_chars,
        "prompt_tokens_est": prompt_tokens,
        "user_tokens_est": user_tokens,
        "input_tokens_each_est": input_each,
        "serialized_bytes_each": bytes_each,
        "queued_token_limit": queued_token_limit,
        "max_bytes_per_file": max_bytes_per_file,
        "max_requests_per_file": max_requests_per_file,
        "per_shard_request_cap": per_shard,
        "n_shards_est": n_shards,
        "feasible": True,
        "infeasible_reason": None,
        "estimated_cost_usd_no_cache": round(cost, 6),
        "proposed_ceiling_usd": proposed_ceiling_usd,
        "exceeds_proposed_ceiling": cost > proposed_ceiling_usd,
        "cache_savings_assumed": False,
        "chars_per_token_is_tokenizer": False,
        "used_frozen_capture": False,
        "upload": False,
        "spend": False,
    }


def _empty_shard(index: int) -> dict[str, Any]:
    return {
        "shard_index": index,
        "n_requests": 0,
        "n_bytes": 0,
        "queued_input_tokens": 0,
        "ids": [],
        "serialized_head_sha256_inputs": [],
    }
