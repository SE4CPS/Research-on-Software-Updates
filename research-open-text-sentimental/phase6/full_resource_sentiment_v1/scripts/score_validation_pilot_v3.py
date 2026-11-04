#!/usr/bin/env python3
"""Score the unseen validation pilot with frozen v3. Predictions stay off reviewer packets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, atomic_write_jsonl, utcnow
from full_resource_sentiment_v1.io_utils import read_json, read_jsonl
from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_audit import collect_selected_text
from full_resource_sentiment_v1.relevance_classify.evaluate import _is_selected
from full_resource_sentiment_v1.relevance_classify.live_client import MissingAPIKeyError, OpenAIRequestError, api_key_present
from full_resource_sentiment_v1.relevance_classify.pipeline import CheckpointStore, classify_records
from full_resource_sentiment_v1.relevance_classify.providers import LiveInferenceForbidden, LiveOpenAIProvider, live_inference_allowed
from full_resource_sentiment_v1.relevance_classify.scope_primary import apply_second_pass
from full_resource_sentiment_v1.relevance_classify.spec import (
    ALLOW_LIVE_ENV,
    PROMPT_VERSION_ENV,
    RESEARCH_RUN_ENV,
    prompt_v3_sha256,
)
from full_resource_sentiment_v1.relevance_classify.spend import SpendCeilingError, SpendLedger

EXTERNAL_ENV = Path.home() / ".config/reddit-research/openai.env"
PILOT_DIR = WORKSPACE / "restricted" / "independent_validation_pilot_2026-10-29"
PRIVATE_KEY = PILOT_DIR / "validation" / "private_id_key.jsonl"
PACKET_A = PILOT_DIR / "validation" / "reviewer_A_packet.jsonl"
SPEND = WORKSPACE / "restricted" / "ai_relevance_v3_2026-10-29" / "spend_ledger.json"
CHECKPOINT = PILOT_DIR / "v3_checkpoint.jsonl"
PREDICTIONS = PILOT_DIR / "v3_predictions.jsonl"


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


def main() -> int:
    os.environ.setdefault(RESEARCH_RUN_ENV, "1")
    os.environ.setdefault(ALLOW_LIVE_ENV, "1")
    os.environ["RELEVANCE_CLASSIFY_TEST_MODE"] = "0"
    os.environ[PROMPT_VERSION_ENV] = "v3"
    load_key()
    if not api_key_present() or not live_inference_allowed(True):
        print("BLOCKER live scoring unavailable")
        return 3
    private_rows = read_jsonl(PRIVATE_KEY)
    packets = read_jsonl(PACKET_A)
    if any(row.get("relevance") for row in packets):
        raise SystemExit("reviewer packet already has labels; refusing")
    texts = collect_selected_text({row["reddit_id"] for row in private_rows})
    records = []
    for row in private_rows:
        text = texts[row["reddit_id"]]
        records.append(
            {
                "synthetic_id": row["reddit_id"],
                "case_code": row["case_code"],
                "reddit_id": row["reddit_id"],
                "title": text["title"],
                "body": text["body"],
                "stratum": row["stratum"],
                "protected_membership": False,
                "classifier_run_id": "ai-relevance-validation-pilot-240-v3-2026-10-29",
            }
        )
    ledger = SpendLedger(SPEND)
    provider = LiveOpenAIProvider(allow_live=True, ledger=ledger, pricing="sync")
    checkpoint = CheckpointStore(CHECKPOINT)
    try:
        for index, record in enumerate(records, start=1):
            classify_records([record], provider, checkpoint=checkpoint)
            if index % 20 == 0 or index == len(records):
                print(f"pilot_progress {index}/{len(records)} bound_usd={ledger.bound_usd():.6f}", flush=True)
    except (LiveInferenceForbidden, MissingAPIKeyError, OpenAIRequestError, SpendCeilingError) as exc:
        print(f"BLOCKER {type(exc).__name__}: {exc}")
        return 3
    raw = [checkpoint.rows[str(rec["synthetic_id"])] for rec in records]
    by_id = {rec["synthetic_id"]: rec for rec in records}
    passed = [apply_second_pass(row, by_id[row["synthetic_id"]]["title"], by_id[row["synthetic_id"]]["body"]) for row in raw]
    atomic_write_jsonl(PILOT_DIR / "v3_raw_predictions.jsonl", raw)
    atomic_write_jsonl(PREDICTIONS, passed)
    updated = []
    selected = 0
    for priv, pred in zip(private_rows, passed):
        row = dict(priv)
        row["ai_predicted_label"] = pred.get("predicted_label")
        row["ai_response_status"] = pred.get("response_status")
        row["ai_selected"] = _is_selected(pred)
        row["second_pass_applied"] = bool(pred.get("second_pass_applied"))
        row["prompt_sha256"] = prompt_v3_sha256()
        if row["ai_selected"]:
            selected += 1
        updated.append(row)
    atomic_write_jsonl(PRIVATE_KEY, updated)
    public = {
        "n_scored": len(passed),
        "n_ai_selected": selected,
        "prompt_sha256": prompt_v3_sha256(),
        "spend": ledger.snapshot(),
        "ai_predictions_written_to_private_key_only": True,
        "reviewer_packets_unlabeled": True,
        "full_frame_not_classified": True,
        "utc": utcnow(),
    }
    hits = scan_obj(public)
    if hits:
        raise SystemExit(f"public leak {hits[:8]}")
    atomic_write_json(WORKSPACE / "results" / "independent_validation_pilot_2026-10-29" / "v3_pilot_scoring.json", public)
    print(f"n_scored={len(passed)} n_ai_selected={selected} spend_actual={ledger.actual_usd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
