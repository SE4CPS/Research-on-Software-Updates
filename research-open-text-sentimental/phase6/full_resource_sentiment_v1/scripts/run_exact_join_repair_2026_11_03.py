#!/usr/bin/env python3
"""Exact comment-join repair + research extensions 2026-11-03.

Preserves prior trees. No manuscript, merge, torch install, or full 30,998 class.
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

from full_resource_sentiment_v1.exact_join_repair_2026_11_03.author_community import (  # noqa: E402
    run_author_community,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.cardiff_proposal import (  # noqa: E402
    run_cardiff_proposal,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.common import (  # noqa: E402
    CORR_OUT,
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
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.exact_join import (  # noqa: E402
    run_exact_join_and_sentiment,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.predictive import (  # noqa: E402
    run_predictive,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.purpose_validation import (  # noqa: E402
    run_purpose_validation,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.recovery_screen import (  # noqa: E402
    run_recovery_screen,
)
from full_resource_sentiment_v1.exact_join_repair_2026_11_03.sign_clarification import (  # noqa: E402
    run_sign_clarification,
)


def main() -> int:
    ensure_dirs()
    ledger = {
        "utc_start": utcnow(),
        "run_id": "exact-join-repair-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "preserves": [str(ORIG_STRICT), str(CORR_OUT)],
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
        [str(py), "-m", "pytest", "tests/test_exact_join_repair_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    # Also re-run prior correction tests for non-regression
    proc2 = subprocess.run(
        [str(py), "-m", "pytest", "tests/test_strict_continuation_corrections_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {
        "exact_join_tests_passed": proc.returncode == 0,
        "prior_corrections_tests_passed": proc2.returncode == 0,
        "exact_stdout": proc.stdout[-2000:],
        "exact_stderr": proc.stderr[-800:],
        "prior_stdout": proc2.stdout[-800:],
    }
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["exact_join_tests_passed"]:
        print(tests)
        return 3

    print("Phase 0: sign estimand clarification...", flush=True)
    s0 = run_sign_clarification()
    ledger["phases"]["0_sign"] = {"estimands": list((s0.get("estimands") or {}).keys())}

    print("Phase 1: exact join + TextBlob recompute...", flush=True)
    join = run_exact_join_and_sentiment()
    ledger["phases"]["1_join"] = {
        "n_score_rows": join.get("n_score_rows_joined"),
        "order_audit_match": (join.get("order_vs_exact_audit") or {}).get("matches_codex_audit"),
        "role_flag_disagreements": join.get("n_role_flag_disagreements"),
        "deltas": list((join.get("delta_vs_prior_role_order_join") or {}).keys()),
    }

    print("Phase 2a: purpose-model validation docs...", flush=True)
    purp = run_purpose_validation()
    ledger["phases"]["2a_purpose"] = {
        "gds_precision": (purp.get("heldout_saved_label_metrics") or {}).get("gds_precision"),
        "gds_recall": (purp.get("heldout_saved_label_metrics") or {}).get("gds_recall"),
    }

    print("Phase 2b: priority+broad recovery screen (not full 30998)...", flush=True)
    rec = run_recovery_screen()
    ledger["phases"]["2b_recovery"] = {
        "screened_n": rec.get("screened_n"),
        "candidates": rec.get("n_candidate_tps_gds_ai_only"),
        "tps": rec.get("n_candidate_tps"),
        "gds": rec.get("n_candidate_gds"),
        "confirmed": 0,
        "inventory_match": rec.get("inventory_match"),
    }

    print("Phase 3: Cardiff tightened proposal (no install)...", flush=True)
    cardiff = run_cardiff_proposal(join)
    ledger["phases"]["3_cardiff"] = {"status": cardiff.get("status"), "free_gib": (cardiff.get("environment") or {}).get("free_disk_gib")}

    print("Phase 4: author/community chronological analyses...", flush=True)
    auth = run_author_community()
    ledger["phases"]["4_author"] = {
        "cohorts": list((auth.get("cohort_results") or {}).keys()),
        "title_body_n": (auth.get("title_vs_body") or {}).get("n"),
    }

    print("Phase 5: predictive comparisons...", flush=True)
    pred = run_predictive()
    ledger["phases"]["5_pred"] = {
        "fixed_split_ok": pred.get("fixed_rq4_v2_split"),
        "macro_f1_unweighted": (pred.get("models") or {}).get("text_unweighted", {}).get("macro_f1"),
        "macro_f1_balanced": (pred.get("models") or {}).get("text_balanced", {}).get("macro_f1"),
        "gds_p": (pred.get("models") or {}).get("text_balanced", {}).get("gds_precision"),
        "gds_r": (pred.get("models") or {}).get("text_balanced", {}).get("gds_recall"),
    }

    # Restricted hashes
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
            "note": "Hashes only; raw text remains restricted 0700",
        },
    )

    deltas = join.get("delta_vs_prior_role_order_join") or {}
    fa_tb = deltas.get("first_author_textblob") or {}
    fc_tb = deltas.get("first_community_textblob") or {}
    packet = f"""# Codex review packet — exact join repair 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Preserves immutable:** `{ORIG_STRICT}`, `{CORR_OUT}`  
**No manuscript. No merge. No torch install. Full 30,998 not run.**

## Acceptance tiers
| Item | Tier |
|---|---|
| Exact original_index join + reply TextBlob recompute | **repair for validation** |
| Order-vs-exact audit 10983/44/2 | reproducibility check |
| Sign estimand clarification 308/190 vs 309/189 vs 232/266 | documentation |
| Priority author≥2 + broad-sample recovery screen | **AI candidates only** |
| Purpose held-out saved-label metrics | conditional; ≠ human accuracy |
| Cardiff / SentiStrength-SE / full-frame | **proposals only** |
| Author/community chronological contrasts | descriptive; ≠ resolution |
| Predictive fixed-split metrics | conditional saved-label agreement |

## 1. Exact join repair
- Score rows joined: **{join.get('n_score_rows_joined')}**
- Join status: `{json.dumps(join.get('join_status_counts'))}`
- Role-flag disagreements (adapter retained): **{join.get('n_role_flag_disagreements')}**
- Order-join audit matches Codex: **{(join.get('order_vs_exact_audit') or {}).get('matches_codex_audit')}**
- Original-post TextBlob join unaffected: **True**
- First-author TextBlob GDS−TPS mean_diff: prior `{fa_tb.get('prior_role_order_mean_diff')}` → exact `{fa_tb.get('exact_join_mean_diff')}` (Δ `{fa_tb.get('delta_exact_minus_prior')}`)
- First-community TextBlob GDS−TPS: prior `{fc_tb.get('prior_role_order_mean_diff')}` → exact `{fc_tb.get('exact_join_mean_diff')}` (Δ `{fc_tb.get('delta_exact_minus_prior')}`)
- Details: `phase1_exact_join/EXACT_JOIN_SENTIMENT.json`

## 2. Dataset recovery (candidates; not merged)
- Excluded inventory match 30998/8147: **{rec.get('inventory_match')}**
- Screened (priority ge2 + broad 0/1 samples): **{rec.get('screened_n')}**
- AI TPS/GDS candidates: **{rec.get('n_candidate_tps_gds_ai_only')}** (TPS {rec.get('n_candidate_tps')}, GDS {rec.get('n_candidate_gds')})
- Confirmed / human-validated: **0**
- Full 30,998: proposal only — `phase2_recovery_screen/FULL_FRAME_30998_PROPOSAL.json`
- Purpose validation / FN-FP plan: `phase2_purpose_validation/`

## 3. Cardiff
- Status: **{cardiff.get('status')}**
- Pinned rev: `{cardiff.get('pinned_revision')}`
- Weights cached MiB: `{(cardiff.get('dependency_size_estimates') or {}).get('weights_already_cached_mib')}`
- Free disk GiB: `{(cardiff.get('environment') or {}).get('free_disk_gib')}`
- Torch/transformers installed: `{(cardiff.get('environment') or {}).get('torch_installed')}` / `{(cardiff.get('environment') or {}).get('transformers_installed')}`
- Separate requests A canary 20 / B bounded pilot / C full matched units in proposal JSON

## 4. Author / community
- Event0=post (title+body segments); event1=first author; event2=second author
- Cohorts: all / author≥1 / ≥2 / ≥4 / community≥1
- Sentiment change **≠** issue resolution
- `phase4_author_community/AUTHOR_COMMUNITY_ANALYSES.json`

## 5. Predictive
- RQ4 v2 test hash match: **{pred.get('fixed_rq4_v2_split')}**
- Unweighted macro-F1: **{(pred.get('models') or {}).get('text_unweighted', {}).get('macro_f1')}**
- Balanced macro-F1: **{(pred.get('models') or {}).get('text_balanced', {}).get('macro_f1')}**
- GDS P/R (balanced): **{(pred.get('models') or {}).get('text_balanced', {}).get('gds_precision')} / {(pred.get('models') or {}).get('text_balanced', {}).get('gds_recall')}**
- Bootstrap CI balanced−unweighted: `{(pred.get('paired_bootstrap_balanced_minus_unweighted') or {}).get('ci95')}`

## 6. Resource gates (stop until approved)
1. Cardiff install + canary/pilot/full (`phase3_cardiff_proposal/`)
2. SentiStrength-SE license/jar
3. Full-frame 30,998 classification
4. Cohort merge of any recovered posts
5. Paid LLM (not prepared)

## Limitations
- Mixed-source saved labels; no completed human 200/240 purpose truth
- Recovery outputs are AI candidates only
- Reply TextBlob now exact-joined; prior role-order reply TextBlob superseded for reply units only
- Cardiff lengths are char/word proxies until tokenizer approved
- Predictive metrics ≠ field accuracy

## Reproduce
```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_exact_join_repair_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_exact_join_repair_2026_11_03.py
```

## Provenance
- Input JSONL sha256: `{jh}`
- Input CSV sha256: `{ch}`
- Restricted manifest: `RESTRICTED_OUTPUT_HASHES.json`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Scientific limitations — exact join repair 2026-11-03

1. Exact join repairs reply TextBlob pairing; it does not create human-validated purpose labels.
2. Recovery screening (priority author≥2 + broad samples) yields AI candidates only; full 30,998 not classified.
3. Purpose/relevance models use mixed-source saved labels; held-out metrics ≠ independent human accuracy.
4. Fixed 240 probability sample remains unlabeled for confirmation rates.
5. Cardiff/SentiStrength-SE/full-frame/merge remain gated.
6. Sentiment trajectory ≠ issue resolution.
7. No manuscript.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "DATASET_AND_LABEL_PROVENANCE.md").write_text(
        f"""# Dataset and label provenance

| Artifact | Role | SHA-256 / note |
|---|---|---|
| `full_dataset.jsonl` | Frozen analysis frame | `{jh}` |
| `full_dataset.csv` | Frozen companion | `{ch}` |
| `comment_scores.jsonl` | Frozen VADER reply scores + original_index | see join audit |
| Capture `2026-10-22-main` | Raw comment bodies for exact join | restricted |
| Protected IDs | 693 ID-only isolation | outcomes unread |
| Recovery candidates | AI-only; this tree | not merged |
| Prior trees | Immutable | `{ORIG_STRICT}`, `{CORR_OUT}` |
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce exact-join repair

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_exact_join_repair_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_exact_join_repair_2026_11_03.py
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
                "join_audit_ok": (join.get("order_vs_exact_audit") or {}).get("matches_codex_audit"),
                "role_flag_disagreements": join.get("n_role_flag_disagreements"),
                "recovery_candidates": rec.get("n_candidate_tps_gds_ai_only"),
                "confirmed": 0,
                "cardiff": "proposal_only",
                "pred_fixed_split": pred.get("fixed_rq4_v2_split"),
                "tests": tests["exact_join_tests_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
