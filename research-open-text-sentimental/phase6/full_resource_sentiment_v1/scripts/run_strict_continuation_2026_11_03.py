#!/usr/bin/env python3
"""Strict experimental continuation 2026-11-03 — Phases 1–5.

No manuscript, no dataset merge, no paid API, no large model install/download without approval.
Preserves accepted A–F, corrections, Stage-1, 200/240 packets, protected outcomes.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.strict_continuation_2026_11_03.common import (  # noqa: E402
    DATASET_CSV,
    DATASET_JSONL,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    OUT_ROOT,
    RESTRICTED_ROOT,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase1_cohorts import run_phase1  # noqa: E402
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase2_excluded import run_phase2  # noqa: E402
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase3_class_balance import (  # noqa: E402
    run_phase3,
)
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase4_sentiment_readiness import (  # noqa: E402
    run_phase4,
)
from full_resource_sentiment_v1.strict_continuation_2026_11_03.phase5_conversation import (  # noqa: E402
    run_phase5,
)


def _run_tests() -> dict:
    py = BASE / ".venv_master_2026_11_03/bin/python"
    cmd = [str(py), "-m", "pytest", "tests/test_strict_continuation_2026_11_03.py", "-q"]
    proc = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-1000:],
        "passed": proc.returncode == 0,
    }


def main() -> int:
    ensure_dirs()
    ledger = {
        "utc_start": utcnow(),
        "run_id": "strict-continuation-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "phases": {},
    }

    jh, ch = sha256_file(DATASET_JSONL), sha256_file(DATASET_CSV)
    inputs = {
        "jsonl_sha256": jh,
        "csv_sha256": ch,
        "jsonl_ok": jh == EXPECTED_JSONL,
        "csv_ok": ch == EXPECTED_CSV,
        "jsonl_sha_before": jh,
    }
    write_json(OUT_ROOT / "INPUT_VERIFICATION.json", inputs)
    if not inputs["jsonl_ok"] or not inputs["csv_ok"]:
        write_json(OUT_ROOT / "BLOCKER.json", inputs)
        return 2

    print("Tests...", flush=True)
    tests = _run_tests()
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3
    ledger["phases"]["tests"] = {"status": "passed"}

    print("Phase 1: cohorts...", flush=True)
    p1 = run_phase1()
    ledger["phases"]["1"] = {
        "status": "ran",
        "frozen_expected_match": p1.get("all_frozen_expected_match"),
    }

    print("Phase 2: excluded inventory + bounded TF-IDF...", flush=True)
    p2 = run_phase2()
    ledger["phases"]["2"] = {
        "status": "ran_exploratory",
        "inventory_match": p2["inventory"]["match_expected"],
        "pilot_agree": p2["agreement_pilot"],
        "targeted_agree": p2["agreement_targeted"],
        "full_frame": "blocked_awaiting_approval",
    }

    print("Phase 3: class balance...", flush=True)
    p3 = run_phase3()
    ledger["phases"]["3"] = {
        "status": "ran_exploratory",
        "test_ids_match_rq4_v2": p3.get("test_ids_match_rq4_v2"),
        "balanced_macro_f1": p3["models"]["balanced_val_threshold"]["macro_f1"],
        "unweighted_macro_f1": p3["models"]["unweighted_val_threshold"]["macro_f1"],
    }

    print("Phase 4: sentiment readiness...", flush=True)
    p4 = run_phase4()
    ledger["phases"]["4"] = {
        "status": "ran_partial",
        "transformer": "stopped_awaiting_approval",
        "matched_posts": p4["matched_sample"]["n_posts"],
    }

    print("Phase 5: conversation analyses...", flush=True)
    p5 = run_phase5()
    ledger["phases"]["5"] = {
        "status": "ran",
        "author_ge4_gds_n": p5["outcomes"]["author_shift_ge4"]["n_gds"],
    }

    # Decision packet
    packet = f"""# Codex decision packet — strict continuation 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW` (not self-validated)  
**UTC:** {utcnow()}  
**No manuscript. No merge into 6,425. No paid/large download executed.**

## 1. Reconciled cohorts (Phase 1)
- Frozen expected match: **{p1.get('all_frozen_expected_match')}**
- Frozen all TPS/GDS 3028/231; author≥1 956/98; ≥2 437/66; ≥4 115/30; community≥1 1489/140
- Historical 520 kept separate
- Missing replies = unobserved, not neutral
- Tables: `phase1_cohort_reconciliation/`; row reasons: `restricted/.../phase1_row_level_reason_codes.jsonl`

## 2. Excluded screening (Phase 2)
- Inventory n={p2['inventory']['n_inventory']} match_expected={p2['inventory']['match_expected']}
- Author≥1/2/4: {p2['inventory']['observed']}
- Fixed 240 preserved; targeted diagnostic 200 designed (not a rewrite of 240)
- Method: `{p2['model']['model_id']}` train-only TF-IDF; pilot/protected/targeted blocked from training
- Pilot agree/disagree: {p2['agreement_pilot']} — **not accuracy**
- Targeted agree/disagree: {p2['agreement_targeted']}
- NLI/transformer alternative: blocked (no torch)
- Full-frame new classification: **not run**

## 3. Class-balance RQ4 (Phase 3)
- Test IDs match RQ4 v2: **{p3.get('test_ids_match_rq4_v2')}**
- Train TPS/GDS: {p3['class_counts']['train']}; balanced weights ≈ {p3['balanced_weights_formula']}
- Unweighted macro-F1: {p3['models']['unweighted_val_threshold']['macro_f1']:.4f}
- Balanced macro-F1: {p3['models']['balanced_val_threshold']['macro_f1']:.4f}
- Oversample macro-F1: {p3['models']['train_oversample_1to1_unweighted_val_threshold']['macro_f1']:.4f}
- Saved-label agreement only

## 4. Sentiment model status (Phase 4)
- VADER/TextBlob matched posts (author≥1 sample): n={p4['matched_sample']['n_posts']}
- Sign agree/disagree: {p4['matched_sample']['post_sign_agreement_vader_textblob']}
- Cardiff/BERTweet: **not run** — see `phase4_sentiment_readiness/TRANSFORMER_APPROVAL_PROPOSAL.json`

## 5. Conversation analyses (Phase 5)
- Distinct eligibility: post→author ≥1; two-point ≥2; author_shift ≥4; community ≥1
- Author≥4 GDS n={p5['outcomes']['author_shift_ge4']['n_gds']} (small)
- Not resolution; overlapping outcomes not independent evidence
- Recovered-post candidates **not** included

## Limitations
- Mixed-source labels; 200-case human review still empty
- TF-IDF lexical; cue/Luna/TF-IDF disagreements ≠ truth
- Transformer stack not installed
- Comment-body TextBlob for replies not joined this run

## Exact steps needing approval (stop gates)
1. Install torch+transformers + run Cardiff canary/full matched scoring (proposal JSON)
2. Paid API or full-frame new-model classification of 30,998
3. Merge any newly classified posts into the 6,425 dataset
4. Manuscript / PDF / submission package

## Codex validation checklist
1. Input JSONL/CSV hashes unchanged; protected outcomes unread
2. Phase 1 frozen counts match 3028/231 and reply thresholds
3. Phase 2 inventory 30998 / 14918 / 8147 / 2847; pilot 240 SHA unchanged
4. TF-IDF training excluded pilot/protected/targeted; agreement not called accuracy
5. Phase 3 test IDs = RQ4 v2; thresholds from validation only
6. Phase 4 transformer not silently substituted
7. Phase 5 eligibility denominators exact; no resolution claims
8. Final seal hashes match scientific files; no auto-merge

## Reproduce
```bash
.venv_master_2026_11_03/bin/python -m pytest tests/test_strict_continuation_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_strict_continuation_2026_11_03.py
```
Outputs: `{OUT_ROOT}`
"""
    (OUT_ROOT / "CODEX_DECISION_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Strict continuation reproduction

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_strict_continuation_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_strict_continuation_2026_11_03.py
```

Disk free at start: {ledger['disk_free_gib']:.2f} GiB
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "LIMITATIONS.md").write_text(
        """# Limitations

1. Mixed-source saved labels; no completed human validation on 200/240.
2. Missing replies unobserved, not neutral.
3. TF-IDF vs Luna agreement ≠ accuracy/FN/yield.
4. Transformers not run (approval gate).
5. No auto-merge; recovered candidates outside analysis.
6. Overlapping conversation estimands are not independent evidence.
""",
        encoding="utf-8",
    )

    # Hash after all content except seal
    ledger["utc_end"] = utcnow()
    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", ledger)

    # Confirm input hash unchanged after run
    inputs_after = {
        "jsonl_sha_after": sha256_file(DATASET_JSONL),
        "csv_sha_after": sha256_file(DATASET_CSV),
        "unchanged": sha256_file(DATASET_JSONL) == EXPECTED_JSONL and sha256_file(DATASET_CSV) == EXPECTED_CSV,
    }
    write_json(OUT_ROOT / "INPUT_HASH_AFTER.json", inputs_after)

    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    hashes = {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files}
    seal = {
        "utc": utcnow(),
        "n_files": len(hashes),
        "files": hashes,
        "input_verification": {**inputs, **inputs_after},
        "tests": tests,
        "status": "AWAITING_INDEPENDENT_CODEX_REVIEW",
        "note": "Seal excludes itself; not a claim of scientific acceptance.",
    }
    write_json(OUT_ROOT / "OUTPUT_HASHES_FINAL.json", seal)
    write_json(
        OUT_ROOT / "SEAL_META.json",
        {"OUTPUT_HASHES_FINAL_sha256": sha256_file(OUT_ROOT / "OUTPUT_HASHES_FINAL.json"), "n_files": len(hashes)},
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(OUT_ROOT),
                "p1_match": p1.get("all_frozen_expected_match"),
                "p2_match": p2["inventory"]["match_expected"],
                "p3_test_match": p3.get("test_ids_match_rq4_v2"),
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
