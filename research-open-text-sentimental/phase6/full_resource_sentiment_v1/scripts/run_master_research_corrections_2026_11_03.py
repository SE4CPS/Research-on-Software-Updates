#!/usr/bin/env python3
"""Master A–F scientific corrections — new versioned outputs only.

Does not modify results/master_research_2026-11-03/, Stage-1, frozen data, or review packets.
No Cardiff/torch download, paid API, full-frame reclass, or protected outcomes.
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

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.cadence import (  # noqa: E402
    run_cadence_completion,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.common import (  # noqa: E402
    DATASET_CSV,
    DATASET_JSONL,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    MANUSCRIPT_ROOT,
    ORIG_OUT,
    OUT_ROOT,
    PILOT_IDS,
    RESTRICTED_ROOT,
    STAGE1_DIR,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.pilot_report import (  # noqa: E402
    run_pilot_reporting,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.rq4 import (  # noqa: E402
    run_rq4_correction,
)
from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (  # noqa: E402
    run_textblob_checks,
)
from full_resource_sentiment_v1.paths import FRESH_500, RESERVED_TEST  # noqa: E402


def _protected_overlap() -> int:
    from full_resource_sentiment_v1.master_research_corrections_2026_11_03.common import load_jsonl

    prot = set()
    for path in (RESERVED_TEST, FRESH_500):
        for row in load_jsonl(path):
            rid = row.get("reddit_id") or row.get("id")
            if rid:
                prot.add(str(rid))
    ids = {str(r["reddit_id"]) for r in load_jsonl(DATASET_JSONL)}
    return len(ids & prot)


def _run_tests() -> dict:
    py = BASE / ".venv_master_2026_11_03/bin/python"
    cmd = [str(py), "-m", "pytest", "tests/test_master_research_corrections_2026_11_03.py", "-q"]
    proc = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-1000:],
        "passed": proc.returncode == 0,
    }


def _write_manuscript_map(rq4: dict, tb: dict, cadence: dict, pilot: dict) -> None:
    MANUSCRIPT_ROOT.mkdir(parents=True, exist_ok=True)
    v2_models = rq4.get("models") or {}
    paired = (rq4.get("paired_bootstrap") or {}).get("text_plus_vader_minus_text_only") or {}
    tb_comb = (tb.get("independent_contrasts") or {}).get("textblob_combined_polarity") or {}
    rows = [
        {
            "claim_id": "C1",
            "claim": "Frozen combined dataset has 6,425 unique IDs; protected overlap 0",
            "status": "accepted_prior_A",
            "evidence": str(ORIG_OUT / "phase_a_inventory/INVENTORY.json"),
            "citations_needed": [],
            "unresolved": False,
        },
        {
            "claim_id": "C2",
            "claim": "Stage-1 VADER post-level GDS−TPS not clearly separated (CI includes 0)",
            "status": "accepted_stage1",
            "evidence": str(STAGE1_DIR / "FINDINGS.md"),
            "citations_needed": ["prior Stage-1 Codex review"],
            "unresolved": False,
        },
        {
            "claim_id": "C3",
            "claim": "TextBlob matched-ID combined polarity GDS−TPS negative (independent interval)",
            "status": "conditional_descriptive",
            "evidence": str(OUT_ROOT / "textblob_scientific_checks/TEXTBLOB_CHECKS.json"),
            "point": tb_comb.get("mean_diff"),
            "ci95": tb_comb.get("mean_diff_ci95"),
            "citations_needed": ["TextBlob PatternAnalyzer citation", "scale non-commensurability note"],
            "unresolved": "No human sentiment reference; not accuracy",
        },
        {
            "claim_id": "C4",
            "claim": "Author-shift GDS−TPS (n_GDS=30) interval includes 0 — exploratory",
            "status": "accepted_point_estimate_uncertain_ci",
            "evidence": str(ORIG_OUT / "phase_d_cadence/gds_minus_tps_cadence.csv"),
            "citations_needed": [],
            "unresolved": "Small GDS≥4 cell",
        },
        {
            "claim_id": "C5",
            "claim": "RQ4 text vs text+sentiment incremental value",
            "status": "unresolved_exploratory",
            "evidence": str(OUT_ROOT / "rq4_near_dup_split/RQ4_RESULTS_V2.json"),
            "v2_text_macro_f1": (v2_models.get("tfidf_text_only") or {}).get("macro_f1"),
            "v2_text_vader_macro_f1": (v2_models.get("text_plus_vader") or {}).get("macro_f1"),
            "paired_delta": paired,
            "citations_needed": ["TF-IDF/logreg", "leakage/grouping method"],
            "unresolved": "Do not claim sentiment helps/hurts unless stable across v1/v2/holdouts; labels mixed-source",
        },
        {
            "claim_id": "C6",
            "claim": "Cue-heuristic weighted GDS 66.8 as recoverable yield",
            "status": "rejected",
            "evidence": str(OUT_ROOT / "pilot_reporting/PILOT_REPORTING.json"),
            "citations_needed": [],
            "unresolved": False,
            "note": pilot.get("yield_claim_status"),
        },
        {
            "claim_id": "C7",
            "claim": "Cadence waiting times / censoring coverage for TPS vs GDS",
            "status": "new_descriptive",
            "evidence": str(OUT_ROOT / "cadence_timing/CADENCE_TIMING.json"),
            "citations_needed": ["survival/censoring framing if used"],
            "unresolved": "Not resolution; right-censoring substantial",
        },
    ]
    write_json(MANUSCRIPT_ROOT / "EVIDENCE_MAP.json", {"utc": utcnow(), "claims": rows})
    lines = [
        "# Manuscript evidence map (draft source) — 2026-11-03 corrections",
        "",
        "**Not final.** Marked for Codex check before any PDF presentation/submission.",
        "",
        "| ID | Claim | Status | Unresolved |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['claim_id']} | {r['claim']} | `{r['status']}` | {r.get('unresolved') or '—'} |"
        )
    lines.extend(
        [
            "",
            "## Draft narrative stubs (accepted / conditional only)",
            "",
            "1. **Data provenance.** The analysis frame comprises 6,425 Reddit posts with mixed human/AI decision sources; "
            "protected evaluation IDs were excluded (overlap 0). Independent human validation of the 200-case packet remains incomplete (0 decisions).",
            "",
            "2. **Sentiment instruments (conditional).** On the frozen comparable TPS/GDS cohort (3,028/231), VADER and TextBlob "
            "are reported as separate instruments. Raw score subtraction across instruments is not interpreted as disagreement about ground truth.",
            "",
            "3. **Conversation roles (conditional).** Author-shift (≥4 author replies) remains underpowered for GDS (n=30) with an interval spanning zero; "
            "post-to-author is a separate estimand and does not establish technical resolution.",
            "",
            "4. **RQ4 (unresolved).** Predictive agreement with saved labels under a connected-component near-duplicate split is exploratory; "
            "sentiment features are not declared helpful or harmful pending stability across designs.",
            "",
            "5. **Rejected.** Keyword-cue IPW GDS yield (66.8) is not a population recovery estimate.",
            "",
            "## Citations to fill before PDF",
            "- VADER (Hutto & Gilbert)",
            "- TextBlob / Pattern sentiment",
            "- Software-update / user dissatisfaction literature (TBD by authors)",
            "- Reddit research ethics / data use statement (TBD)",
            "",
        ]
    )
    (MANUSCRIPT_ROOT / "DRAFT_SOURCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (MANUSCRIPT_ROOT / "README.md").write_text(
        "Draft manuscript evidence map only. Not a submission-ready PDF.\n",
        encoding="utf-8",
    )


def main() -> int:
    ensure_dirs()
    ledger: dict = {
        "utc_start": utcnow(),
        "run_id": "master-research-corrections-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "preserves": {
            "original_af": str(ORIG_OUT),
            "stage1": str(STAGE1_DIR),
            "frozen_jsonl": str(DATASET_JSONL),
        },
        "phases": {},
    }

    # Input verification
    jh, ch = sha256_file(DATASET_JSONL), sha256_file(DATASET_CSV)
    input_check = {
        "jsonl_sha256": jh,
        "csv_sha256": ch,
        "jsonl_ok": jh == EXPECTED_JSONL,
        "csv_ok": ch == EXPECTED_CSV,
        "protected_overlap": _protected_overlap(),
        "stage1_exists": STAGE1_DIR.exists(),
        "orig_af_exists": ORIG_OUT.exists(),
        "pilot_ids_sha256": sha256_file(PILOT_IDS),
    }
    write_json(OUT_ROOT / "INPUT_VERIFICATION.json", input_check)
    if not input_check["jsonl_ok"] or not input_check["csv_ok"] or input_check["protected_overlap"] != 0:
        write_json(OUT_ROOT / "BLOCKER.json", input_check)
        print("STOP input verification failed", input_check)
        return 2

    print("Tests (corrections fixtures only)...", flush=True)
    test_res = _run_tests()
    write_json(OUT_ROOT / "TEST_RESULTS.json", test_res)
    if not test_res["passed"]:
        print("STOP tests failed", test_res)
        return 3
    ledger["phases"]["tests"] = {"status": "passed"}

    print("RQ4 near-dup correction...", flush=True)
    rq4 = run_rq4_correction()
    ledger["phases"]["rq4_v2"] = {
        "status": "ran_exploratory",
        "n_components": rq4["split"]["n_components"],
        "test_macro_f1_text": rq4["models"]["tfidf_text_only"]["macro_f1"],
        "test_macro_f1_text_vader": rq4["models"]["text_plus_vader"]["macro_f1"],
        "component_overlap_train_test": rq4["split"]["component_overlap"]["train_test"],
    }

    print("TextBlob scientific checks...", flush=True)
    tb = run_textblob_checks()
    ledger["phases"]["textblob_checks"] = {
        "status": "ran",
        "hash_mismatches": tb["title_body_hash_mismatches_vs_restricted_manifest"],
        "tb_combined_mean_diff": tb["independent_contrasts"]["textblob_combined_polarity"]["mean_diff"],
    }

    print("Cadence timing completion...", flush=True)
    cadence = run_cadence_completion()
    ledger["phases"]["cadence_timing"] = {
        "status": "ran",
        "gds_n_post_to_first_reply": cadence["coverage"]["GDS"]["n_post_to_first_reply"],
        "tps_n_post_to_first_reply": cadence["coverage"]["TPS"]["n_post_to_first_reply"],
    }

    print("Pilot reporting correction...", flush=True)
    pilot = run_pilot_reporting()
    ledger["phases"]["pilot_reporting"] = {
        "status": "ran",
        "yield_claim": "rejected",
        "ids_redrawn": False,
    }

    print("Manuscript evidence map (draft)...", flush=True)
    _write_manuscript_map(rq4, tb, cadence, pilot)
    ledger["phases"]["manuscript_map"] = {"status": "draft_only_not_final"}

    # Codex packet (before final seal)
    packet = f"""# Codex review packet — master A–F corrections 2026-11-03

**Status:** `STOP_FOR_CODEX_INDEPENDENT_REVIEW`  
**UTC:** {utcnow()}  
**Original A–F tree:** preserved immutable at `{ORIG_OUT}`

## What changed vs original A–F
| Item | Original | Correction |
|---|---|---|
| RQ4 grouping | title+body[:500] hash | Connected components: exact full text + shared title (≥20 chars) & body Jaccard ≥0.55 |
| RQ4 metrics | v1 exploratory | v2 rerun on new freeze; v1 retained |
| TextBlob intervals | single impl/seed | Independent numpy bootstrap seed 20261103+777; length/source/period |
| Cadence | role/trajectory + elapsed among replies | + post→first reply/author, gaps, duration, censoring coverage |
| Pilot | cue outputs readable as second model | Relabeled keyword heuristic; 66.8 yield claim **rejected** |
| Packaging | root hashes before Phase F rewrite | Final seal written **last**, excludes self hash |

## Key corrected numbers (see machine tables)
- RQ4 v2 text-only macro-F1: **{rq4['models']['tfidf_text_only']['macro_f1']:.4f}**; text+VADER: **{rq4['models']['text_plus_vader']['macro_f1']:.4f}**; component overlap train∩test: **{rq4['split']['component_overlap']['train_test']}**
- Paired Δ text+VADER−text: {rq4['paired_bootstrap']['text_plus_vader_minus_text_only']}
- TextBlob combined independent mean_diff: {tb['independent_contrasts']['textblob_combined_polarity'].get('mean_diff')}, CI {tb['independent_contrasts']['textblob_combined_polarity'].get('mean_diff_ci95')}
- Title/body hash mismatches vs restricted manifest: **{tb['title_body_hash_mismatches_vs_restricted_manifest']}**
- Cadence GDS/TPS n with post→first reply: {cadence['coverage']['GDS']['n_post_to_first_reply']}/{cadence['coverage']['TPS']['n_post_to_first_reply']}
- Pilot: cue GDS raw count {pilot.get('cue_predicted_gds_raw_count')}; yield claim rejected

## Scientific tier
- **Accepted (prior):** Phase A inventory; Stage-1 VADER reuse; Phase B sample integrity
- **Conditional/descriptive:** TextBlob/VADER within-model contrasts; cadence coverage/timing
- **Exploratory unresolved:** RQ4 incremental sentiment value
- **Rejected:** cue IPW GDS yield as recovery estimate; any human accuracy/FN from 240

## Paths
- Root: `{OUT_ROOT}`
- `rq4_near_dup_split/`, `textblob_scientific_checks/`, `cadence_timing/`, `pilot_reporting/`
- Draft manuscript map: `{MANUSCRIPT_ROOT}`
- Restricted (if any new): `{RESTRICTED_ROOT}`

## Verify
1. Original A–F hashes for phase_* files still match prior seal (unchanged).
2. Input JSONL/CSV hashes unchanged; protected overlap 0.
3. RQ4 v2 component overlap 0; policy thresholds predeclared.
4. Final `OUTPUT_HASHES_FINAL.json` written after all artifacts; does not hash itself.
5. No Cardiff/paid/full-frame/manuscript-final PDF.
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")

    (OUT_ROOT / "RUN.md").write_text(
        f"""# Corrections reproduction

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_master_research_corrections_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_master_research_corrections_2026_11_03.py
```

Outputs: `{OUT_ROOT}`  
Does not write to `{ORIG_OUT}`.
""",
        encoding="utf-8",
    )

    (OUT_ROOT / "FINDINGS.md").write_text(
        f"""# Corrections consolidated findings

**UTC:** {utcnow()}  
Original A–F preserved. See per-phase FINDINGS and `CODEX_REVIEW_PACKET.md`.

- RQ4 v2 components={rq4['split']['n_components']}; text macro-F1={rq4['models']['tfidf_text_only']['macro_f1']:.4f}; text+VADER={rq4['models']['text_plus_vader']['macro_f1']:.4f}
- TextBlob independent combined GDS−TPS={tb['independent_contrasts']['textblob_combined_polarity'].get('mean_diff')}
- Pilot yield 66.8 claim rejected; heuristic only
- Cadence timing tables added; not resolution
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "LIMITATIONS.md").write_text(
        """# Corrections limitations

1. Labels still mixed-source; 200-case human review empty.
2. Near-dup policy misses different-title paraphrases.
3. TextBlob/VADER scales not commensurate.
4. Cadence right-censoring and count gaps.
5. Manuscript map is draft evidence only — not a final PDF.
6. Cardiff still blocked pending approval.
""",
        encoding="utf-8",
    )

    ledger["utc_end"] = utcnow()
    ledger["phases"]["packaging"] = {"status": "sealing"}
    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", ledger)

    write_json(
        OUT_ROOT / "FINAL_MANIFEST.json",
        {
            "utc": utcnow(),
            "out_root": str(OUT_ROOT),
            "status": "STOP_FOR_CODEX_INDEPENDENT_REVIEW",
            "original_af_preserved": str(ORIG_OUT),
            "seal_file": "OUTPUT_HASHES_FINAL.json",
            "seal_note": "OUTPUT_HASHES_FINAL.json is written after this manifest and is excluded from its own file list.",
        },
    )

    # FINAL SEAL LAST — hash all closed artifacts except the seal file itself
    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    hashes = {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files}
    seal = {
        "utc": utcnow(),
        "n_files": len(hashes),
        "files": hashes,
        "input_verification": input_check,
        "code": {
            "script": "scripts/run_master_research_corrections_2026_11_03.py",
            "package": "src/full_resource_sentiment_v1/master_research_corrections_2026_11_03/",
            "tests": "tests/test_master_research_corrections_2026_11_03.py",
            "test_results": test_res,
        },
        "privacy_protected": {
            "protected_overlap": input_check["protected_overlap"],
            "protected_outcomes_read": False,
        },
        "scientific_tier_summary": {
            "accepted_prior": ["phase_A", "stage1_vader", "pilot_sample_integrity"],
            "conditional": ["textblob_within_model", "cadence_timing_descriptive"],
            "exploratory_unresolved": ["rq4_sentiment_incremental"],
            "rejected": ["cue_ipw_gds_yield_66_8"],
        },
        "note": "Seal written after all other outputs; this file is not listed in files.",
    }
    write_json(OUT_ROOT / "OUTPUT_HASHES_FINAL.json", seal)
    write_json(
        OUT_ROOT / "SEAL_META.json",
        {
            "utc": utcnow(),
            "OUTPUT_HASHES_FINAL_sha256": sha256_file(OUT_ROOT / "OUTPUT_HASHES_FINAL.json"),
            "FINAL_MANIFEST_sha256": hashes.get("FINAL_MANIFEST.json"),
            "n_files_in_seal": len(hashes),
        },
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(OUT_ROOT),
                "rq4_text_f1": rq4["models"]["tfidf_text_only"]["macro_f1"],
                "rq4_text_vader_f1": rq4["models"]["text_plus_vader"]["macro_f1"],
                "tests_passed": test_res["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
