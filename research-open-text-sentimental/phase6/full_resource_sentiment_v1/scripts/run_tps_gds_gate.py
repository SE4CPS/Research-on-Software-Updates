#!/usr/bin/env python3
"""Resume-safe, capped 90-case live TPS/GDS prompt evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
ENV = Path.home() / ".config/reddit-research/openai.env"
RATES = {"gpt-6-luna": (0.10, 0.50), "gpt-6.1-terra": (1.00, 5.00)}
GATE_SPEND_CEILING = 1.00
LOCK = threading.Lock()


def key_from_env():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if ENV.is_file():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("OPENAI_API_KEY="):
                value = line.partition("=")[2].strip().strip('"').strip("'")
                if value:
                    return value
    raise RuntimeError("API key unavailable in the existing private research setup")


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_output(data):
    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                text += c.get("text") or ""
    return json.loads(text)


def classify(record, key, prompt, schema, model):
    title, body = record["title"], record["body"]
    if record["deterministic_unassessable"]:
        return {"reddit_id": record["reddit_id"], "status": "ok", "label": {
            "relevance": "unassessable", "discussion": None, "confidence": "high",
            "reason": "No original title or body remains."}, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "model": "deterministic"}
    payload = {
        "model": model, "temperature": 0, "reasoning": {"effort": "none"},
        "max_output_tokens": 160,
        "text": {"format": {"type": "json_schema", "name": "tps_gds_scope",
                            "strict": True, "schema": schema}},
        "input": [{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps({"TITLE": title, "BODY": body}, ensure_ascii=False)}],
    }
    last_error = None
    for attempt in range(5):
        try:
            response = requests.post("https://api.openai.com/v1/responses", json=payload,
                                     headers={"Authorization": f"Bearer {key}"}, timeout=120)
            if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                time.sleep(min(20, 2 ** attempt + random.random()))
                continue
            if response.status_code != 200:
                return {"reddit_id": record["reddit_id"], "status": "error",
                        "error": f"HTTP {response.status_code}", "cost_usd": 0.0}
            data = response.json()
            usage = data.get("usage") or {}
            it, ot = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
            input_rate, output_rate = RATES[model]
            cost = it * input_rate / 1_000_000 + ot * output_rate / 1_000_000
            try:
                label = parse_output(data)
                assert set(label) == {"relevance", "discussion", "confidence", "reason"}
                assert label["relevance"] in {"relevant", "not_relevant", "unclear", "unassessable"}
                assert label["discussion"] in {"TPS", "GDS", "Other", "Unclear", None}
                assert label["confidence"] in {"high", "medium", "low"}
                assert 1 <= len(label["reason"].split()) <= 100
            except (ValueError, KeyError, TypeError, AssertionError):
                return {"reddit_id": record["reddit_id"], "status": "invalid_output",
                        "input_tokens": it, "output_tokens": ot, "cost_usd": cost,
                        "response_id": data.get("id"), "model": data.get("model")}
            return {"reddit_id": record["reddit_id"], "status": "ok", "label": label,
                    "input_tokens": it, "output_tokens": ot, "cost_usd": cost,
                    "response_id": data.get("id"), "model": data.get("model")}
        except requests.RequestException as exc:
            last_error = type(exc).__name__
            time.sleep(min(20, 2 ** attempt + random.random()))
    return {"reddit_id": record["reddit_id"], "status": "error", "error": last_error, "cost_usd": 0.0}


def evaluate(preds, gate_name):
    truth = json.loads((RUN / f"{gate_name}_truth.json").read_text(encoding="utf-8"))
    if len(preds) != len(truth):
        return None
    matrix = Counter()
    for rid, expected in truth.items():
        row = preds[rid]
        got = row.get("label", {}).get("discussion") if row["status"] == "ok" and row.get("label", {}).get("relevance") == "relevant" else "NO_VALID_LABEL"
        matrix[(expected, got)] += 1
    metrics = {}
    for label in ("TPS", "GDS", "Other"):
        tp = matrix[(label, label)]
        pred = sum(n for (e, g), n in matrix.items() if g == label)
        actual = sum(n for (e, g), n in matrix.items() if e == label)
        metrics[label] = {"tp": tp, "predicted": pred, "actual": actual,
                          "precision": tp / pred if pred else None,
                          "recall": tp / actual if actual else None}
    return {"matrix": {f"{e}->{g}": n for (e, g), n in sorted(matrix.items())},
            "metrics": metrics,
            "status_counts": dict(Counter(x["status"] for x in preds.values())),
            "cost_usd": sum(x["cost_usd"] for x in preds.values())}


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--limit", type=int, default=90)
    arg.add_argument("--workers", type=int, default=6)
    arg.add_argument("--model", choices=sorted(RATES), default="gpt-6-luna")
    arg.add_argument("--gate", choices=("gate", "gate2"), default="gate")
    args = arg.parse_args()
    packet = list(rows(RUN / f"{args.gate}_packet.jsonl"))
    assert len(packet) == 90 and 1 <= args.limit <= 90
    output = RUN / f"{args.gate}_predictions_{args.model}.jsonl"
    previous = {x["reddit_id"]: x for x in rows(output)} if output.exists() else {}
    assert sum(x.get("cost_usd", 0) for x in previous.values()) < GATE_SPEND_CEILING
    pending = [x for x in packet[:args.limit] if x["reddit_id"] not in previous or previous[x["reddit_id"]]["status"] != "ok"]
    if pending:
        key = key_from_env()
        prompt = (RUN / "prompt.txt").read_text(encoding="utf-8")
        schema = json.loads((RUN / "schema.json").read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(classify, x, key, prompt, schema, args.model): x["reddit_id"] for x in pending}
            with output.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    result = future.result()
                    with LOCK:
                        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                    previous[result["reddit_id"]] = result
                    if sum(x.get("cost_usd", 0) for x in previous.values()) >= GATE_SPEND_CEILING:
                        raise RuntimeError("Gate spending ceiling reached")
    status = evaluate(previous, args.gate)
    if status:
        (RUN / f"{args.gate}_evaluation_{args.model}.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(previous), "status_counts": dict(Counter(x["status"] for x in previous.values())),
                      "cost_usd": sum(x.get("cost_usd", 0) for x in previous.values()),
                      "evaluation": status}, indent=2))


if __name__ == "__main__":
    main()
