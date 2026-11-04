#!/usr/bin/env python3
"""Correction checkpoint + Cardiff full + RQ1–RQ4 — 2026-11-03."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.cardiff_full import (  # noqa: E402
    run_cardiff_full,
)
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.common import (  # noqa: E402
    CONTINUATION_OUT,
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
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.rq_analyses import (  # noqa: E402
    run_rq,
)
from full_resource_sentiment_v1.correction_rq_cardiff_2026_11_03.stage1_correct_dataset import (  # noqa: E402
    run_stage1,
)


def main() -> int:
    ensure_dirs()
    ledger = {
        "utc_start": utcnow(),
        "run_id": "correction-rq-cardiff-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "preserves": [str(CONTINUATION_OUT)],
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
        [str(py), "-m", "pytest", "tests/test_correction_rq_cardiff_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Stage 1: removal correction + v2 dataset + dup audit...", flush=True)
    s1 = run_stage1()
    ledger["phases"]["1_correction"] = {
        "overrides": s1.get("n_override_to_unassessable"),
        "candidates": s1.get("n_candidate_tps_gds_ai_only_corrected"),
        "tps": s1.get("n_candidate_tps"),
        "gds": s1.get("n_candidate_gds"),
        "union": s1.get("n_union_rows"),
        "version": s1.get("dataset_version"),
    }
    print("  CORRECTION CHECKPOINT written for Codex", flush=True)

    print("Stage 2: Cardiff full scoring...", flush=True)
    cardiff = run_cardiff_full()
    ledger["phases"]["2_cardiff"] = {
        "status": cardiff.get("status"),
        "n": cardiff.get("n_scored"),
        "truncated": cardiff.get("n_truncated"),
    }

    print("Stage 3: RQ1–RQ4...", flush=True)
    rq = run_rq()
    r13 = rq["rq1_rq3"]
    r4 = rq["rq4"]
    ledger["phases"]["3_rq"] = {
        "rq1_orig_vader_diff": (r13.get("rq1_original_only") or {}).get("rq1_post_vader", {}).get("mean_diff"),
        "rq4_fixed_split": r4.get("fixed_rq4_v2_split"),
        "rq4_macro_f1_balanced": (r4.get("models") or {}).get("text_balanced", {}).get("macro_f1"),
    }

    restr = sorted(p for p in RESTRICTED_ROOT.rglob("*") if p.is_file())
    write_json(
        OUT_ROOT / "RESTRICTED_OUTPUT_HASHES.json",
        {
            "utc": utcnow(),
            "n_files": len(restr),
            "files": [
                {"path": str(p.relative_to(RESTRICTED_ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in restr
            ],
        },
    )

    o = r13.get("rq1_original_only") or {}
    ai = r13.get("rq1_ai_assisted_v2_excl_exact_train_dups") or {}
    rq2 = r13.get("rq2_original_author_ge1_first_minus_post") or {}
    packet = f"""# Codex review packet — correction + Cardiff + RQ1–RQ4 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Preserves:** `{CONTINUATION_OUT}`, original 6425  
**No manuscript. AI ≠ confirmed.**

## Stage 1 — Corrected dataset v2
- Overrides to unassessable: **{s1.get('n_override_to_unassessable')}** (ge_65={s1.get('overrides_ge_65')})
- Effective AI TPS/GDS candidates: **{s1.get('n_candidate_tps_gds_ai_only_corrected')}** (TPS {s1.get('n_candidate_tps')}, GDS {s1.get('n_candidate_gds')})
- Unassessable: **{s1.get('n_unassessable')}**; Relevant/Other: **{s1.get('n_candidate_relevant_other')}**; not candidate: **{s1.get('n_not_candidate')}**
- Dataset version: **{s1.get('dataset_version')}**
- Union rows: **{s1.get('n_union_rows')}** (6425 + {s1.get('n_ai_assisted_added')}); protected in union: **{s1.get('protected_in_union')}**
- Exact train↔screen dups among candidates: **{(s1.get('duplication') or {}).get('exact_normalized_title_body_hash_dups', {}).get('n_candidate_rows')}**
- Near-dup candidates flagged: **{(s1.get('duplication') or {}).get('near_dup_candidates', {}).get('n_candidate_rows_flagged')}**
- Checkpoint: `stage1_dataset_correction/CORRECTION_CHECKPOINT_FOR_CODEX.md`

## Stage 2 — Cardiff full
- Status: **{cardiff.get('status')}**; n scored: **{cardiff.get('n_scored')}**; truncated: **{cardiff.get('n_truncated')}**
- Frozen exact-zero agree C–V / C–TB / V–TB: `{cardiff.get('agreement_exact_zero_frozen')}`
- Neutral .05 agree: `{cardiff.get('agreement_neutral_0_05_frozen')}`
- Truncation sensitivity: `{cardiff.get('truncated_sensitivity')}`
- Agreement ≠ accuracy

## Stage 3 — RQ1–RQ4
### RQ1 original-only post VADER GDS−TPS
`{(o.get('rq1_post_vader') or {})}`
### RQ1 AI-assisted v2 (excl exact train dups)
`{(ai.get('rq1_post_vader') or {})}`
### RQ2 first-author−post
`{(rq2.get('contrast') or {})}`
### RQ4
- Fixed split: **{r4.get('fixed_rq4_v2_split')}**
- text_balanced macro-F1: **{(r4.get('models') or {}).get('text_balanced', {}).get('macro_f1')}**
- text+sentiment balanced: **{(r4.get('models') or {}).get('text_plus_vader_tb_balanced', {}).get('macro_f1')}**
- GDS P/R balanced: **{(r4.get('models') or {}).get('text_balanced', {}).get('gds_precision')} / {(r4.get('models') or {}).get('text_balanced', {}).get('gds_recall')}**

## Limitations
- AI labels unconfirmed; removal correction uses whole-field detector
- No human FP/FN; RQ4 = saved-label agreement
- Sentiment change ≠ issue resolution
- Cardiff instrument disagreement ≠ truth

## Reproduce
```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_correction_rq_cardiff_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_correction_rq_cardiff_2026_11_03.py
```

Input JSONL `{jh}` / CSV `{ch}`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Limitations

1. Corrected AI TPS/GDS counts are still unconfirmed model outputs.
2. Whole-field removal overrides fix known unassessable inclusions; other edge cases may remain.
3. Train↔screen duplication flagged for sensitivity; models not tuned on full-frame outputs.
4. Cardiff/VADER/TextBlob agreement is not accuracy.
5. RQ2/RQ3 must not be read as issue resolution.
6. Protected outcomes unused; no manuscript.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_correction_rq_cardiff_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_correction_rq_cardiff_2026_11_03.py
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
                "overrides": s1.get("n_override_to_unassessable"),
                "candidates_corrected": s1.get("n_candidate_tps_gds_ai_only_corrected"),
                "cardiff": cardiff.get("status"),
                "rq4_fixed": r4.get("fixed_rq4_v2_split"),
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
