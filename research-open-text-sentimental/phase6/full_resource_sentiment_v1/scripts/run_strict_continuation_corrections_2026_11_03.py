#!/usr/bin/env python3
"""Strict continuation corrections/extensions 2026-11-03.

Preserves original strict_continuation tree. No manuscript, merge, paid API, or large installs.
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

from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.common import (  # noqa: E402
    DATASET_CSV,
    DATASET_JSONL,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    ORIG_STRICT,
    OUT_ROOT,
    RESTRICTED_ROOT,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage0_repairs import (  # noqa: E402
    run_stage0,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage1_recovery import (  # noqa: E402
    run_stage1,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage2_sentiment import (  # noqa: E402
    run_stage2,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage3_predictive import (  # noqa: E402
    run_stage3,
)
from full_resource_sentiment_v1.strict_continuation_corrections_2026_11_03.stage4_conversation import (  # noqa: E402
    run_stage4,
)


def main() -> int:
    ensure_dirs()
    ledger = {
        "utc_start": utcnow(),
        "run_id": "strict-continuation-corrections-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "preserves_original_strict": str(ORIG_STRICT),
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
        [str(py), "-m", "pytest", "tests/test_strict_continuation_corrections_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Stage 0: repairs...", flush=True)
    s0 = run_stage0()
    ledger["phases"]["0"] = {
        "sign_matches_309_189": s0["sign"].get("matches_independent_check"),
        "protected_n": s0["protected"].get("n_union_ids"),
    }

    print("Stage 1: recovery / reply depth...", flush=True)
    s1 = run_stage1()
    ledger["phases"]["1"] = {
        "depth_match": s1["depths"].get("match"),
        "n_candidate_tps_gds": s1["candidates"].get("n_candidate_tps_gds_ai_only"),
        "n_confirmed": 0,
    }

    print("Stage 2: matched sentiment + proposals...", flush=True)
    s2 = run_stage2()
    ledger["phases"]["2"] = {
        "post_sign_agree": s2["matched"].get("post_sign_agree"),
        "cardiff": "proposal_only",
        "sentistrength_se": "feasibility_only",
    }

    print("Stage 3: predictive extension...", flush=True)
    s3 = run_stage3()
    ledger["phases"]["3"] = {
        "fixed_split_ok": s3.get("matches_rq4_v2"),
        "text_unweighted_macro_f1": s3["models"]["text_unweighted"]["macro_f1"],
        "text_balanced_macro_f1": s3["models"]["text_balanced"]["macro_f1"],
    }

    print("Stage 4: conversation by depth...", flush=True)
    s4 = run_stage4()
    ledger["phases"]["4"] = {
        "author_ge1_gds": s4["results"]["author_ge1"]["n_gds"],
        "author_ge2_gds": s4["results"]["author_ge2"]["n_gds"],
        "author_ge4_gds": s4["results"]["author_ge4"]["n_gds"],
    }

    # Restricted file hash manifest (public paths+hashes only)
    restr_files = sorted(p for p in RESTRICTED_ROOT.rglob("*") if p.is_file())
    restr_manifest = {
        "utc": utcnow(),
        "files": [
            {"path": str(p.relative_to(RESTRICTED_ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in restr_files
        ],
        "n_files": len(restr_files),
        "note": "Hashes only in public tree; raw text remains restricted 0700",
    }
    write_json(OUT_ROOT / "RESTRICTED_OUTPUT_HASHES.json", restr_manifest)

    depths = s1["depths"]["counts_separate_overlapping_groups"]
    cand = s1["candidates"]
    sign = s0["sign"].get("primary_np_sign") or s0["sign"].get("primary_thr_0_05") or {}
    packet = f"""# Codex review packet — strict continuation corrections 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Original strict tree immutable:** `{ORIG_STRICT}`  
**No manuscript. No merge. No paid/large install executed.**

## Accepted / conditional / rejected
| Item | Tier |
|---|---|
| Frozen cohort counts; reply-depth inventory 30998 | accepted inventory |
| Sign agreement fix 309/189 | correction accepted pending Codex recompute |
| Estimand renames + first/second author outcomes | conditional descriptive |
| TF-IDF/Luna candidate TPS/GDS | **candidates only — rejected as confirmed recoveries** |
| Predictive fixed-split metrics | conditional saved-label agreement |
| Cardiff / SentiStrength-SE / full-frame | proposals only |

## 1. Excluded reply-depth counts (overlapping; reported separately)
{json.dumps(depths, indent=2)}
Priority: author≥2 for conversation-rich recovery; author=1 retained for first-reply sentiment; ≥4 trajectory-only; fixed 240 probability sample retained.

## 2. Candidate vs confirmed recovery
- AI-only TPS+GDS candidates (pilot+targeted bounded): **{cand.get('n_candidate_tps_gds_ai_only')}** (TPS {cand.get('n_candidate_tps')}, GDS {cand.get('n_candidate_gds')})
- Confirmed: **0** (no human reference)
- By author depth: {cand.get('by_author_depth_candidates')}
- Not merged into 6,425

## 3. Corrected sentiment comparison
- Prior buggy report preserved; corrected primary np.sign: agree={sign.get('agree')} disagree={sign.get('disagree')} missing={sign.get('missing')} (Codex cited 309/189; this run {sign.get('agree')}/{sign.get('disagree')}; match_exact={s0['sign'].get('matches_independent_check')}; {s0['sign'].get('off_by_one_note')})
- Full-cohort matched role TextBlob+VADER: see `stage2_matched_role_sentiment/`

## 4. Conversation analyses
- Accurate names: `mean_all_author_replies_minus_post`, `last_minus_first_author_reply`
- New: `first_author_reply_minus_post` (956/98), `second_minus_first_author_reply` (437/66)
- Depth cohorts in `stage4_conversation_by_depth/` — not resolution

## 5. Predictive / class imbalance
- Fixed RQ4 v2 test IDs; unweighted macro-F1={s3['models']['text_unweighted']['macro_f1']:.4f}; balanced={s3['models']['text_balanced']['macro_f1']:.4f}
- PR-AUC/Brier/10k paired bootstrap/multi-split stability included
- Saved-label agreement only

## 6–7. Resource proposals (approval required)
1. `stage2_resource_proposals/CARDIFF_APPROVAL_PROPOSAL.json` — counts all matched post+reply units
2. `stage2_resource_proposals/SENTISTRENGTH_SE_FEASIBILITY.json` — license/Java blocker; not SentiSE
3. `stage1_candidate_recovery/FULL_FRAME_CLASSIFICATION_PROPOSAL.json` — 30,998 two-stage
4. Paid LLM: not prepared as executable; still requires separate budget if requested
5. Cohort merge: **blocked** until confirmed recoveries + explicit approval

## Limitations
- Mixed-source labels; 200-case human review empty
- Candidates ≠ confirmed; AI×AI agreement ≠ truth
- No torch/transformers; MiniLM cache unused
- SentiStrength-SE not locally available under verified license
- Overlapping conversation estimands not independent evidence

## Reproduce
```bash
.venv_master_2026_11_03/bin/python -m pytest tests/test_strict_continuation_corrections_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_strict_continuation_corrections_2026_11_03.py
```
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Scientific limitations report

1. No independent human validation of purpose/relevance for excluded recoveries.
2. TF-IDF positive prevalence and threshold inflate relevant flags on excluded pilot.
3. Purpose stage is AI-only; heuristic cues are diagnostic.
4. Sentiment instruments are non-commensurate.
5. Bootstrap intervals are post-level prediction uncertainty, not label-noise models.
6. Protected outcomes unread (ID-only 693).
7. Manuscript intentionally not started.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce corrections

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_strict_continuation_corrections_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_strict_continuation_corrections_2026_11_03.py
```

Outputs: `{OUT_ROOT}`  
Restricted: `{RESTRICTED_ROOT}`
""",
        encoding="utf-8",
    )

    ledger["utc_end"] = utcnow()
    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", ledger)

    # Final seal last
    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    seal = {
        "utc": utcnow(),
        "n_files": len(files),
        "files": {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files},
        "input_sha256": {"jsonl": jh, "csv": ch},
        "restricted_manifest": "RESTRICTED_OUTPUT_HASHES.json",
        "status": "AWAITING_INDEPENDENT_CODEX_REVIEW",
        "note": "Seal excludes itself",
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
                "sign_ok": s0["sign"].get("matches_independent_check"),
                "depths_ok": s1["depths"].get("match"),
                "candidates": cand.get("n_candidate_tps_gds_ai_only"),
                "confirmed": 0,
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
