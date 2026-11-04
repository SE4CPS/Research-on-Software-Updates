#!/usr/bin/env python3
"""Continuation: cohort fix, full 30998 screen, candidate pool, Cardiff canary/pilot."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.candidate_dataset import (  # noqa: E402
    run_candidate_dataset,
)
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.cardiff_pilot import (  # noqa: E402
    run_cardiff,
)
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.cohort_correction import (  # noqa: E402
    run_cohort_correction,
)
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.common import (  # noqa: E402
    DATASET_CSV,
    DATASET_JSONL,
    EXACT_JOIN_OUT,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    OUT_ROOT,
    RESTRICTED_ROOT,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.continuation_full_screen_cardiff_2026_11_03.full_frame_screen import (  # noqa: E402
    run_full_frame_screen,
)


def main() -> int:
    ensure_dirs()
    ledger = {
        "utc_start": utcnow(),
        "run_id": "continuation-full-screen-cardiff-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "preserves_exact_join_seal": str(EXACT_JOIN_OUT),
        "phases": {},
    }
    jh, ch = sha256_file(DATASET_JSONL), sha256_file(DATASET_CSV)
    if jh != EXPECTED_JSONL or ch != EXPECTED_CSV:
        write_json(OUT_ROOT / "BLOCKER.json", {"jsonl": jh, "csv": ch})
        return 2
    write_json(OUT_ROOT / "INPUT_VERIFICATION.json", {"jsonl_sha256": jh, "csv_sha256": ch, "ok": True})

    print("Tests...", flush=True)
    py = BASE / ".venv_master_2026_11_03/bin/python"
    proc = subprocess.run(
        [str(py), "-m", "pytest", "tests/test_continuation_full_screen_cardiff_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Phase 0: matched-cohort reporting correction...", flush=True)
    cohort = run_cohort_correction()
    fa = cohort.get("first_author_textblob_matched_ids") or {}
    ledger["phases"]["0_cohort"] = {
        "n_common": fa.get("n_common_ids"),
        "n_changed": fa.get("n_textblob_changed_among_common"),
        "join_effect": (fa.get("join_effect_on_contrast") or {}).get("mean_diff_new_minus_old_contrast"),
        "later_author": "INCOMPARABLE",
    }

    print("Phase 1: full-frame screen 30998...", flush=True)
    screen = run_full_frame_screen()
    ledger["phases"]["1_screen"] = {
        "n": screen.get("n_screened"),
        "match": screen.get("match"),
        "candidates": screen.get("n_candidate_tps_gds_ai_only"),
        "tps": screen.get("n_candidate_tps"),
        "gds": screen.get("n_candidate_gds"),
        "confirmed": 0,
    }

    print("Phase 2: versioned candidate dataset...", flush=True)
    cand = run_candidate_dataset()
    ledger["phases"]["2_dataset"] = {
        "version": cand.get("dataset_version"),
        "n_union": cand.get("n_union_rows"),
        "n_ai_added": cand.get("n_ai_assisted_added"),
        "original_untouched": cand.get("original_untouched"),
    }

    print("Phase 3: Cardiff resource + canary + pilot...", flush=True)
    cardiff = run_cardiff()
    ledger["phases"]["3_cardiff"] = {
        "status": cardiff.get("status"),
        "canary_n": (cardiff.get("canary") or {}).get("n"),
        "pilot_n": (cardiff.get("pilot") or {}).get("n"),
    }

    restr_files = sorted(p for p in RESTRICTED_ROOT.rglob("*") if p.is_file())
    write_json(
        OUT_ROOT / "RESTRICTED_OUTPUT_HASHES.json",
        {
            "utc": utcnow(),
            "files": [
                {"path": str(p.relative_to(RESTRICTED_ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in restr_files
            ],
            "n_files": len(restr_files),
        },
    )

    join_effect = (fa.get("join_effect_on_contrast") or {}).get("mean_diff_new_minus_old_contrast")
    packet = f"""# Codex review packet — continuation full screen + Cardiff 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Preserves sealed exact-join tree:** `{EXACT_JOIN_OUT}`  
**No manuscript. Original 6425 not overwritten. AI ≠ confirmed.**

## Checkpoints for Codex
1. Full-frame screen of 30,998  
2. Cardiff canary + bounded pilot  

## 0. Reporting corrections (supplemental; sealed tree untouched)
- First-author TextBlob join effect on **identical** IDs (n={fa.get('n_common_ids')}):  
  scores changed among common: **{fa.get('n_textblob_changed_among_common')}**;  
  GDS−TPS contrast join-effect (new−old on same IDs): **{join_effect}**  
- `later_author_vader` old/new: **INCOMPARABLE** (definition changed: old last-of-≥2 vs new last-of-≥3). VADER values themselves unchanged by text join.  
- File: `phase0_cohort_reporting_correction/MATCHED_COHORT_JOIN_EFFECT.json`

## 1. Full-frame screen (30,998)
- Screened: **{screen.get('n_screened')}** (match={screen.get('match')})  
- AI TPS/GDS candidates: **{screen.get('n_candidate_tps_gds_ai_only')}** (TPS {screen.get('n_candidate_tps')}, GDS {screen.get('n_candidate_gds')})  
- Confirmed / human-validated: **0**  
- Prior 9,882 identifiable; fixed 240 identifiable — **not** independent validation  
- Model card + train ID hashes: `phase1_full_frame_screen/FROZEN_SCREEN_MODEL_CARD.json`  
- Restricted: `full_frame_screen_30998.jsonl` + shards  

## 2. Versioned candidate dataset
- Version: **{cand.get('dataset_version')}**  
- Union rows: **{cand.get('n_union_rows')}** (original {cand.get('n_original_preserved')} + AI {cand.get('n_ai_assisted_added')})  
- Original 6425 untouched: **{cand.get('original_untouched')}**  
- AI GDS/TPS are candidates only — not prevalence  

## 3. Cardiff
- Status: **{cardiff.get('status')}**  
- License: CC-BY-4.0; pinned rev in resource check  
- Canary n: **{(cardiff.get('canary') or {}).get('n')}**; Pilot n: **{(cardiff.get('pilot') or {}).get('n')}**  
- Pilot kept separate from confirmatory results; agreement ≠ truth  
- Details: `phase3_cardiff/`

## Reproduce
```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_continuation_full_screen_cardiff_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_continuation_full_screen_cardiff_2026_11_03.py
```

## Provenance
- Input JSONL: `{jh}`  
- Input CSV: `{ch}`  
- Restricted manifest: `RESTRICTED_OUTPUT_HASHES.json`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Scientific limitations

1. Full-frame TPS/GDS labels are AI-only candidates from mixed-source saved-label training.
2. No human FP/FN rates; fixed 240 and prior 9882 are not independent validation.
3. Candidate dataset does not overwrite original 6425; AI GDS counts ≠ confirmed prevalence.
4. Cardiff pilot is exploratory instrument comparison, not confirmatory truth.
5. later_author old/new contrasts are definitionally incomparable.
6. No manuscript.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce continuation

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_continuation_full_screen_cardiff_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_continuation_full_screen_cardiff_2026_11_03.py
```

Outputs: `{OUT_ROOT}`  
Restricted: `{RESTRICTED_ROOT}`
""",
        encoding="utf-8",
    )

    ledger["utc_end"] = utcnow()
    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", ledger)

    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    seal = {
        "utc": utcnow(),
        "n_files": len(files),
        "files": {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files},
        "input_sha256": {"jsonl": jh, "csv": ch},
        "restricted_manifest": "RESTRICTED_OUTPUT_HASHES.json",
        "status": "AWAITING_INDEPENDENT_CODEX_REVIEW",
    }
    write_json(OUT_ROOT / "OUTPUT_HASHES_FINAL.json", seal)
    write_json(
        OUT_ROOT / "SEAL_META.json",
        {"OUTPUT_HASHES_FINAL_sha256": sha256_file(OUT_ROOT / "OUTPUT_HASHES_FINAL.json"), "n": len(files)},
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(OUT_ROOT),
                "screen_n": screen.get("n_screened"),
                "candidates": screen.get("n_candidate_tps_gds_ai_only"),
                "gds_ai": screen.get("n_candidate_gds"),
                "confirmed": 0,
                "cardiff": cardiff.get("status"),
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
