#!/usr/bin/env python3
"""Score the locked 240-case human relevance audit with the frozen TPS/GDS prompt."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from full_resource_sentiment_v1.relevance_classify.preprocess import redact_one_way, deterministic_unassessable
from run_tps_gds_gate import classify, key_from_env, rows

RUN = BASE / "restricted/tps_gds_full_classification_2026-11-02"
PACKET = BASE / "restricted/relevance_audit_sample_2026-10-28/reviewer_A_packet.jsonl"
TRUTH = BASE / "restricted/relevance_audit_primary_scope_adjudicated_2026-10-29/final_labels.csv"


def main():
    truth = {x["case_code"]: x["final_label"] for x in csv.DictReader(TRUTH.open())}
    source = list(rows(PACKET))
    assert len(source) == len(truth) == 240 and {x["case_code"] for x in source} == set(truth)
    prep = RUN / "relevance_240_packet.jsonl"
    if not prep.exists():
        with prep.open("w", encoding="utf-8") as handle:
            for x in source:
                title, _ = redact_one_way(x["title"])
                body, _ = redact_one_way(x["body"])
                handle.write(json.dumps({"reddit_id": x["case_code"], "title": title, "body": body,
                                         "deterministic_unassessable": deterministic_unassessable(title, body) is not None},
                                        ensure_ascii=False, sort_keys=True) + "\n")
    prompt = (RUN / "prompt.txt").read_text(encoding="utf-8")
    schema = json.loads((RUN / "schema.json").read_text(encoding="utf-8"))
    key = key_from_env()
    for model in ("gpt-6-luna", "gpt-6.1-terra"):
        out = RUN / f"relevance_240_predictions_{model}.jsonl"
        existing = {x["reddit_id"]: x for x in rows(out)} if out.exists() else {}
        pending = [x for x in rows(prep) if x["reddit_id"] not in existing or existing[x["reddit_id"]]["status"] != "ok"]
        if pending:
            with ThreadPoolExecutor(max_workers=8) as pool, out.open("a", encoding="utf-8") as handle:
                futures = [pool.submit(classify, x, key, prompt, schema, model) for x in pending]
                for future in as_completed(futures):
                    y = future.result(); existing[y["reddit_id"]] = y
                    handle.write(json.dumps(y, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
        matrix = Counter()
        for code, gold in truth.items():
            x = existing[code]
            pred = x.get("label", {}).get("relevance") if x["status"] == "ok" else "error"
            matrix[(gold, pred)] += 1
        tp = matrix[("relevant", "relevant")]
        selected = sum(n for (g,p),n in matrix.items() if p == "relevant")
        relevant = sum(n for (g,p),n in matrix.items() if g == "relevant")
        report = {"model": model, "matrix": {f"{g}->{p}":n for (g,p),n in sorted(matrix.items())},
                  "precision": tp/selected if selected else None,"recall":tp/relevant,
                  "status_counts": dict(Counter(x["status"] for x in existing.values())),
                  "cost_usd":sum(x.get("cost_usd",0) for x in existing.values())}
        (RUN / f"relevance_240_evaluation_{model}.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print(json.dumps(report),flush=True)


if __name__ == "__main__": main()
