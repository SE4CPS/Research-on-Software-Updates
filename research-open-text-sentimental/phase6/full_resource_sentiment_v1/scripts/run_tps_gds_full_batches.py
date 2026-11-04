#!/usr/bin/env python3
"""Prepare, submit, and checkpoint two model batches for missing TPS/GDS rows.

Uses only redacted title/body packets. Existing labels and protected outcomes
never appear in requests. Requires the existing private research API key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from full_resource_sentiment_v1.relevance_classify.live_client import OpenAIHttp

RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
ENV = Path.home() / ".config/reddit-research/openai.env"
MODELS = {"gpt-6-luna": (0.05, 0.25), "gpt-6.1-terra": (0.50, 2.50)}  # Batch, USD / 1M
SPEND_CAP = 10.00  # this two-model campaign; below prior $25 project ceiling


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if ENV.is_file():
        for raw in ENV.read_text(encoding="utf-8").splitlines():
            line = raw.strip().removeprefix("export ")
            if line.startswith("OPENAI_API_KEY="):
                key = line.partition("=")[2].strip().strip('"').strip("'")
                if key:
                    return key
    raise RuntimeError("Existing private research API key not available")


def paths(model):
    safe = model.replace(".", "_")
    return RUN / f"batch_input_{safe}.jsonl", RUN / f"batch_state_{safe}.json"


def prepare(model):
    prompt = (RUN / "prompt.txt").read_text(encoding="utf-8")
    schema = json.loads((RUN / "schema.json").read_text(encoding="utf-8"))
    packet = RUN / "missing_packet.jsonl"
    out, state = paths(model)
    if not out.exists() and state.exists():
        raise RuntimeError(f"Submitted batch input missing for {model}")
    if not out.exists():
        with out.open("w", encoding="utf-8") as handle:
            for x in rows(packet):
                if x["deterministic_unassessable"]:
                    continue
                body = {
                    "model": model, "temperature": 0, "reasoning": {"effort": "none"},
                    "max_output_tokens": 160,
                    "text": {"format": {"type": "json_schema", "name": "tps_gds_scope", "strict": True, "schema": schema}},
                    "input": [{"role": "system", "content": prompt},
                              {"role": "user", "content": json.dumps({"TITLE": x["title"], "BODY": x["body"]}, ensure_ascii=False)}],
                }
                line = {"custom_id": x["reddit_id"], "method": "POST", "url": "/v1/responses", "body": body}
                handle.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")
    n = sum(1 for _ in out.open(encoding="utf-8"))
    assert n == 4461
    # Deliberately conservative token estimate: 2 chars/token input and
    # maximum 160 output tokens. This must remain under this campaign cap.
    total_chars = sum(len(line) for line in out.open(encoding="utf-8"))
    input_rate, output_rate = MODELS[model]
    conservative_usd = (total_chars / 2 / 1_000_000) * input_rate + (n * 160 / 1_000_000) * output_rate
    return {"model": model, "n": n, "bytes": out.stat().st_size, "sha256": sha(out),
            "conservative_upper_usd": conservative_usd, "path": str(out)}


def submit(model):
    info = prepare(model)
    _, state = paths(model)
    if state.exists():
        return json.loads(state.read_text(encoding="utf-8"))
    infos = [prepare(m) for m in MODELS]
    assert sum(i["conservative_upper_usd"] for i in infos) < SPEND_CAP
    client = OpenAIHttp(load_key())
    uploaded = client.upload_file(Path(info["path"]), purpose="batch")
    file_id = uploaded["id"]
    batch = client.create_batch(file_id, metadata={"research": "tps_gds_full_2026_11_02", "model": model})
    saved = {"model": model, "input_file_id": file_id, "batch_id": batch["id"],
             "submitted_status": batch.get("status"), "input_sha256": info["sha256"],
             "n": info["n"], "conservative_upper_usd": info["conservative_upper_usd"]}
    state.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return saved


def poll(model):
    _, state = paths(model)
    saved = json.loads(state.read_text(encoding="utf-8"))
    client = OpenAIHttp(load_key())
    status = client.get_batch(saved["batch_id"])
    output = {"model": model, "batch_id": saved["batch_id"], "status": status.get("status"),
              "request_counts": status.get("request_counts"),
              "output_file_id": status.get("output_file_id"), "error_file_id": status.get("error_file_id")}
    result_path = RUN / f"batch_result_{model.replace('.', '_')}.jsonl"
    if status.get("status") == "completed" and status.get("output_file_id") and not result_path.exists():
        client.download_file(status["output_file_id"], result_path)
        output["result_sha256"] = sha(result_path)
        output["result_rows"] = sum(1 for _ in result_path.open(encoding="utf-8"))
    elif result_path.exists():
        output["result_sha256"] = sha(result_path)
        output["result_rows"] = sum(1 for _ in result_path.open(encoding="utf-8"))
    (RUN / f"batch_status_{model.replace('.', '_')}.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "poll"))
    parser.add_argument("--model", choices=sorted(MODELS))
    args = parser.parse_args()
    models = [args.model] if args.model else list(MODELS)
    for model in models:
        info = {"prepare": prepare, "submit": submit, "poll": poll}[args.action](model)
        print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
