#!/usr/bin/env python3
"""Resume-safe full Terra run after Batch queued-token rejection."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_tps_gds_gate import classify, key_from_env, rows

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
PACKET = RUN / "missing_packet.jsonl"
OUTPUT = RUN / "full_predictions_gpt-6_1-terra.jsonl"
MODEL = "gpt-6.1-terra"
SPEND_CAP = 10.00


def main():
    packet = list(rows(PACKET))
    assert len(packet) == 4461
    prior = {x["reddit_id"]: x for x in rows(OUTPUT)} if OUTPUT.exists() else {}
    assert sum(x.get("cost_usd", 0) for x in prior.values()) < SPEND_CAP
    pending = [x for x in packet if x["reddit_id"] not in prior or prior[x["reddit_id"]]["status"] != "ok"]
    if pending:
        key = key_from_env()
        prompt = (RUN / "prompt.txt").read_text(encoding="utf-8")
        schema = json.loads((RUN / "schema.json").read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=12) as pool, OUTPUT.open("a", encoding="utf-8") as handle:
            futures = {pool.submit(classify, x, key, prompt, schema, MODEL): x["reddit_id"] for x in pending}
            for n, future in enumerate(as_completed(futures), 1):
                result = future.result()
                prior[result["reddit_id"]] = result
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                if n % 250 == 0:
                    print(json.dumps({"completed": len(prior), "status_counts": dict(Counter(x["status"] for x in prior.values())),
                                      "cost_usd": round(sum(x.get("cost_usd", 0) for x in prior.values()), 4)}), flush=True)
                if sum(x.get("cost_usd", 0) for x in prior.values()) >= SPEND_CAP:
                    raise RuntimeError("Campaign cost cap reached")
    print(json.dumps({"final_completed": len(prior),
                      "status_counts": dict(Counter(x["status"] for x in prior.values())),
                      "cost_usd": sum(x.get("cost_usd", 0) for x in prior.values())}), flush=True)


if __name__ == "__main__":
    main()
