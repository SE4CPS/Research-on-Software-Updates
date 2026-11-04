#!/usr/bin/env python3
"""Senti4SD comparison on length-sensitivity units. Preserves sealed trees. No paid APIs."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03 import SEED  # noqa: E402
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.analysis import analyze_cohort  # noqa: E402
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import (  # noqa: E402
    DATASET_JSONL,
    EXPECTED_JSONL,
    JAVA_BIN,
    LENGTH_OUT,
    OUT_ROOT,
    PYSENTI,
    RESTRICTED_ROOT,
    SEALED_AUDIT,
    VENDOR,
    VENV_SENTI,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.score import score_all  # noqa: E402
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.units import (  # noqa: E402
    build_ai_units,
    build_frozen_units,
)


def _fmt(c: dict | None) -> str:
    if not c or c.get("mean_diff") is None:
        return "n/a"
    ci = c.get("mean_diff_ci95") or [None, None]
    return f"n={c.get('n_a')}/{c.get('n_b')} mean={c['mean_diff']:.6f} CI95=[{ci[0]:.6f}, {ci[1]:.6f}]"


def _record_versions() -> dict:
    py = VENV_SENTI / "bin/python"
    pkgs = subprocess.check_output(
        [str(py), "-c", "import numpy,pandas,sklearn,joblib,scipy; print(numpy.__version__, pandas.__version__, sklearn.__version__, joblib.__version__, scipy.__version__)"],
        text=True,
    ).strip()
    java_v = subprocess.check_output([str(JAVA_BIN), "-version"], stderr=subprocess.STDOUT, text=True).splitlines()[0]
    pysenti_commit = subprocess.check_output(["git", "-C", str(PYSENTI), "rev-parse", "HEAD"], text=True).strip()
    s4_commit = subprocess.check_output(
        ["git", "-C", str(VENDOR / "Senti4SD"), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "utc": utcnow(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "java": java_v,
        "java_bin": str(JAVA_BIN),
        "python": subprocess.check_output([str(py), "-V"], text=True).strip(),
        "python_packages": pkgs,
        "liblinear_dylib": "/opt/homebrew/opt/liblinear/lib/liblinear.dylib",
        "pySenti4SD_commit": pysenti_commit,
        "Senti4SD_commit": s4_commit,
        "Senti4SD_model": str(PYSENTI / "Senti4SD.model"),
        "Senti4SD_model_sha256": sha256_file(PYSENTI / "Senti4SD.model"),
        "dsm_bin_sha256": sha256_file(PYSENTI / "java/dsm.bin"),
        "Senti4SD_fast_jar_sha256": sha256_file(PYSENTI / "java/Senti4SD-fast.jar"),
        "license": "MIT (collab-uniba/Senti4SD and collab-uniba/pySenti4SD)",
        "citation": "Calefato et al. 2018 EMSE — Sentiment Polarity Detection for Software Development",
        "mac_runtime_notes": [
            "Used authors' pySenti4SD + pretrained Senti4SD.model and ClassificationTask dsm.bin/Senti4SD-fast.jar",
            "Homebrew OpenJDK 25 + Homebrew liblinear 2.49 dylib",
            "Minimal Mac patches: prefer liblinear.dylib; guard missing find_parameter_C; numpy APIs on scipy module",
        ],
        "disk_free_gib_at_start": shutil.disk_usage("/").free / (1024**3),
    }


def main() -> int:
    ensure_dirs()
    jh = sha256_file(DATASET_JSONL)
    if jh != EXPECTED_JSONL:
        write_json(OUT_ROOT / "BLOCKER.json", {"reason": "dataset hash mismatch", "jsonl": jh})
        return 2

    length_seal = sha256_file(LENGTH_OUT / "OUTPUT_HASHES_FINAL.json")
    audit_seal = sha256_file(SEALED_AUDIT / "OUTPUT_HASHES_FINAL.json") if (SEALED_AUDIT / "OUTPUT_HASHES_FINAL.json").exists() else None
    write_json(
        OUT_ROOT / "SEALED_PRIOR_HASHES.json",
        {
            "length_sensitivity_OUTPUT_HASHES_FINAL_sha256": length_seal,
            "audit_corrections_OUTPUT_HASHES_FINAL_sha256": audit_seal,
            "jsonl": jh,
        },
    )
    versions = _record_versions()
    write_json(OUT_ROOT / "SETUP_VERIFICATION.json", versions)

    print("Tests...", flush=True)
    master_py = BASE / ".venv_master_2026_11_03/bin/python"
    proc = subprocess.run(
        [str(master_py), "-m", "pytest", "tests/test_senti4sd_comparison_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    # Canary already produced; copy reference
    canary = PYSENTI / "predictions/canary_predictions.csv"
    if canary.exists():
        (OUT_ROOT / "CANARY_PREDICTIONS.csv").write_text(canary.read_text(encoding="utf-8"), encoding="utf-8")

    print("Building AI units (same 5942 pairs)...", flush=True)
    ai_units = build_ai_units()
    print(f"  AI units: {len(ai_units)}", flush=True)
    print("Building frozen units...", flush=True)
    fr_units = build_frozen_units()
    print(f"  Frozen units: {len(fr_units)}", flush=True)
    write_json(
        OUT_ROOT / "UNIT_COUNTS.json",
        {
            "ai_units": len(ai_units),
            "ai_posts": sum(1 for u in ai_units if u["unit"] == "original_post"),
            "frozen_units": len(fr_units),
            "frozen_posts": sum(1 for u in fr_units if u["unit"] == "original_post"),
        },
    )

    # Persist unit manifests without text (hashes only) for audit
    from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import write_jsonl

    write_jsonl(
        RESTRICTED_ROOT / "ai_units_manifest.jsonl",
        [{k: v for k, v in u.items() if k != "text"} for u in ai_units],
    )
    write_jsonl(
        RESTRICTED_ROOT / "frozen_units_manifest.jsonl",
        [{k: v for k, v in u.items() if k != "text"} for u in fr_units],
    )

    print("Scoring AI with Senti4SD...", flush=True)
    ai_score_meta = score_all(ai_units, "ai_assisted_v2", batch_size=80)
    write_json(OUT_ROOT / "AI_SCORE_META.json", ai_score_meta)
    print("Scoring frozen with Senti4SD...", flush=True)
    fr_score_meta = score_all(fr_units, "original_frozen", batch_size=80)
    write_json(OUT_ROOT / "FROZEN_SCORE_META.json", fr_score_meta)

    print("Analyzing...", flush=True)
    ai_rep = analyze_cohort("ai_assisted_v2", SEED + 300)
    fr_rep = analyze_cohort("original_frozen", SEED + 400)

    findings = f"""# Findings — Senti4SD comparison (2026-11-03)

## Setup (verified)
- Implementation: authors' **pySenti4SD** commit `{versions['pySenti4SD_commit']}` + pretrained `Senti4SD.model`
- DSM/JAR: authors' **Senti4SD** commit `{versions['Senti4SD_commit']}` (`dsm.bin`, `Senti4SD-fast.jar`)
- License: MIT. Citation: Calefato et al. 2018 EMSE.
- Runtime: `{versions['java']}`; Python `{versions['python']}`; packages `{versions['python_packages']}`
- Mac patches documented in `SETUP_VERIFICATION.json` (liblinear dylib; scipy/numpy shim). Not a different model.
- Canary on authors' Sample.csv: see `CANARY_PREDICTIONS.csv`.
- Sealed priors preserved: length-sensitivity `{length_seal}`; audit `{audit_seal}`.

## Cohorts (separate)
- AI: {ai_rep['n_pairs']} pairs ({ai_rep['n_gds']} GDS / {ai_rep['n_tps']} TPS); failures recorded: {ai_rep['n_failures_recorded']}
- Frozen: {fr_rep['n_pairs']} pairs ({fr_rep['n_gds']} GDS / {fr_rep['n_tps']} TPS); failures: {fr_rep['n_failures_recorded']}
- Join: exact `reddit_id` + per-unit `text_hash`.

## Senti4SD GDS−TPS (polarity sign −1/0/+1)
| Contrast | AI | Frozen |
|---|---|---|
| Original post | {_fmt(ai_rep['senti4sd_post_gds_minus_tps'])} | {_fmt(fr_rep['senti4sd_post_gds_minus_tps'])} |
| First-author reply − post | {_fmt(ai_rep['senti4sd_first_author_minus_post_gds_minus_tps'])} | {_fmt(fr_rep['senti4sd_first_author_minus_post_gds_minus_tps'])} |

## Classification agreement vs other instruments (not accuracy)
Agreement of discrete classes on identical units. VADER uses ±0.05 thresholds; TextBlob uses polarity sign; Cardiff uses saved polarity signs.

### AI posts
{json.dumps(ai_rep['classification_agreement_on_posts'], indent=2)}

### AI replies
{json.dumps(ai_rep['classification_agreement_on_replies'], indent=2)}

### AI delta direction agreement (sign of reply−post only)
{json.dumps(ai_rep['delta_direction_agreement'], indent=2)}

### Frozen (separate)
Posts: {json.dumps(fr_rep['classification_agreement_on_posts'], indent=2)}
Delta directions: {json.dumps(fr_rep['delta_direction_agreement'], indent=2)}

## Guardrails
- Do **not** compare raw score magnitudes across models.
- Agreement ≠ accuracy.
- Sentiment change ≠ issue resolution.
- AI purpose labels unconfirmed.
"""
    (OUT_ROOT / "FINDINGS.md").write_text(findings, encoding="utf-8")

    # Manuscript evidence table + outline (verified vs limitations)
    evidence = f"""# Manuscript evidence table and outline — after Senti4SD comparison

Status: evidence preparation only. No unsupported claims. Cohorts separate.

## Verified results (safe to cite with stated scope)

| ID | Claim scope | Evidence | Denominator / CI | Caveat |
|---|---|---|---|---|
| E1 | AI first-author−post VADER GDS−TPS (full text) | length-sensitivity FINDINGS | n=610/5332; mean≈−0.115; CI excludes 0 | AI labels unconfirmed; ≠ issue resolution |
| E2 | Same contrast, TextBlob | length-sensitivity | ≈0; CI includes 0 | Instruments disagree |
| E3 | Same contrast, Cardiff sign | length-sensitivity | ≈−0.174; CI excludes 0 | Different scale than VADER |
| E4 | Equal-length prefix VADER attenuates | length-sensitivity | ≈−0.022; CI includes 0 | Prefix = different text portion |
| E5 | Length/source/date-adjusted VADER GDS coef | length-sensitivity | ≈−0.088; CI excludes 0 | Association within AI labels |
| E6 | Senti4SD post GDS−TPS (sign) | this tree AI | `{_fmt(ai_rep['senti4sd_post_gds_minus_tps'])}` | Class→sign mapping; not magnitude |
| E7 | Senti4SD reply−post GDS−TPS (sign) | this tree AI | `{_fmt(ai_rep['senti4sd_first_author_minus_post_gds_minus_tps'])}` | Directional class-sign delta only |
| E8 | Senti4SD vs VADER/TB/Cardiff class agreement | this tree | see FINDINGS JSON blocks | Agreement ≠ accuracy |
| E9 | Frozen cohort kept separate | all trees | frozen n={fr_rep['n_pairs']} | Smaller; many CIs include 0 |
| E10 | Exact-join repair; Cardiff truncation fix; Cliff’s δ sign fix | sealed audit/correction trees | hashes in SEALED_PRIOR_HASHES | Do not re-open seals |

## Limitations (must remain in manuscript)

1. Expanded AI TPS/GDS labels are model predictions, not human-validated.
2. 200-case independent review still has zero saved human decisions (as of prior audit).
3. Sentiment instruments disagree; disagreement is not accuracy.
4. Prefix and full-text estimands answer different questions.
5. Sentiment change is not issue resolution.
6. Senti4SD is an SE-domain classifier; polarity signs are coarse (−1/0/+1).
7. Do not pool frozen and AI cohorts for inference.
8. No paid-API / ChatGPT labels in sealed analyses.

## Outline (suggested section order)

1. **Data and labels** — frozen human-comparable vs AI-assisted v2; separate tables.
2. **Sentiment instruments** — VADER, TextBlob, Cardiff, Senti4SD (versions/licenses).
3. **RQ-linked verified contrasts** — only E-rows above; report denominators and CIs.
4. **Length sensitivity** — full vs prefix vs adjusted; state prefix ≠ full text.
5. **Cross-instrument agreement** — classifications/directions only; forbid magnitude pooling.
6. **Limitations** — copy the list above without softening.
7. **Claims not supported** — “GDS authors become more negative over time” as a strong causal/trajectory claim remains unsupported given instrument disagreement and unvalidated AI labels.
"""
    (OUT_ROOT / "MANUSCRIPT_EVIDENCE_TABLE_AND_OUTLINE.md").write_text(evidence, encoding="utf-8")

    packet = f"""# Codex review packet — Senti4SD comparison 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Preserves** length-sensitivity seal `{length_seal}` and audit seal `{audit_seal}`.  
**No manuscript prose beyond evidence table/outline. No label changes. No paid APIs.**

## Ask
1. Confirm authors' pySenti4SD + pretrained model/DSM/JAR were used (commits/hashes in SETUP_VERIFICATION.json).
2. Confirm canary succeeded and full scoring used exact reddit_id + text_hash joins.
3. Confirm AI and frozen cohorts remain separate.
4. Confirm FINDINGS report post and reply−post GDS−TPS with denominators/CIs, plus classification/direction agreement without calling it accuracy or comparing magnitudes.
5. Confirm sealed prior trees were not overwritten.

## Headline numbers (AI)
- Post GDS−TPS: `{_fmt(ai_rep['senti4sd_post_gds_minus_tps'])}`
- Reply−post GDS−TPS: `{_fmt(ai_rep['senti4sd_first_author_minus_post_gds_minus_tps'])}`
- Failures: AI `{ai_score_meta.get('n_failed')}` / Frozen `{fr_score_meta.get('n_failed')}`

## Reproduce
```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_senti4sd_comparison_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_senti4sd_comparison_2026_11_03.py
```

Outputs: `{OUT_ROOT}`  
Restricted scores: `{RESTRICTED_ROOT}`  
Vendor: `{VENDOR}`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Limitations

1. AI TPS/GDS labels unconfirmed.
2. Instrument disagreement ≠ accuracy.
3. No cross-model magnitude comparisons.
4. Senti4SD signs are coarse class encodings.
5. Sentiment change ≠ issue resolution.
6. Frozen and AI cohorts not pooled.
7. Mac runtime required documented binding shims; model weights unchanged.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_senti4sd_comparison_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_senti4sd_comparison_2026_11_03.py
```

Requires vendor assets under `{VENDOR}` and Homebrew OpenJDK + liblinear.
""",
        encoding="utf-8",
    )

    write_json(
        OUT_ROOT / "PROGRESS_LEDGER.json",
        {
            "utc": utcnow(),
            "ai_pairs": ai_rep["n_pairs"],
            "frozen_pairs": fr_rep["n_pairs"],
            "ai_failed": ai_score_meta.get("n_failed"),
            "frozen_failed": fr_score_meta.get("n_failed"),
            "tests": tests["passed"],
        },
    )

    restr = sorted(p for p in RESTRICTED_ROOT.rglob("*") if p.is_file() and "_work_" not in str(p))
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

    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    seal = {
        "utc": utcnow(),
        "n_files": len(files),
        "files": {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files},
        "input_sha256": {"jsonl": jh},
        "sealed_prior": {"length_sensitivity": length_seal, "audit": audit_seal},
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
                "ai_pairs": ai_rep["n_pairs"],
                "frozen_pairs": fr_rep["n_pairs"],
                "ai_post": _fmt(ai_rep["senti4sd_post_gds_minus_tps"]),
                "ai_delta": _fmt(ai_rep["senti4sd_first_author_minus_post_gds_minus_tps"]),
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
