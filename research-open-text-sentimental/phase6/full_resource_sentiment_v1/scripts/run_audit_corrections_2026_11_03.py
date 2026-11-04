#!/usr/bin/env python3
"""Audit corrections 1–5 + Codex packet. Preserves sealed trees. No manuscript."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.audit_corrections_2026_11_03.cliffs_and_truncation import (  # noqa: E402
    run_cliffs_regeneration,
    run_truncation_correction,
)
from full_resource_sentiment_v1.audit_corrections_2026_11_03.common import (  # noqa: E402
    DATASET_CSV,
    DATASET_JSONL,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    OUT_ROOT,
    RESTRICTED_ROOT,
    SEALED_CORR,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.audit_corrections_2026_11_03.expanded_conversation import (  # noqa: E402
    run_expanded_conversation,
)
from full_resource_sentiment_v1.audit_corrections_2026_11_03.rq1_adjusted import (  # noqa: E402
    run_rq1_adjusted,
)
from full_resource_sentiment_v1.audit_corrections_2026_11_03.rq4_component_bootstrap import (  # noqa: E402
    run_rq4,
)


def main() -> int:
    ensure_dirs()
    ledger = {
        "utc_start": utcnow(),
        "run_id": "audit-corrections-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "preserves_sealed": str(SEALED_CORR),
        "phases": {},
    }
    jh, ch = sha256_file(DATASET_JSONL), sha256_file(DATASET_CSV)
    if jh != EXPECTED_JSONL or ch != EXPECTED_CSV:
        write_json(OUT_ROOT / "BLOCKER.json", {"jsonl": jh, "csv": ch})
        return 2
    write_json(OUT_ROOT / "INPUT_VERIFICATION.json", {"jsonl_sha256": jh, "csv_sha256": ch, "ok": True})
    sealed_hash = sha256_file(SEALED_CORR / "OUTPUT_HASHES_FINAL.json") if (SEALED_CORR / "OUTPUT_HASHES_FINAL.json").exists() else None
    write_json(OUT_ROOT / "SEALED_PRIOR_HASHES.json", {
        "correction_rq_cardiff_OUTPUT_HASHES_FINAL_sha256": sealed_hash,
        "dataset_jsonl": jh,
        "dataset_csv": ch,
    })

    print("Tests...", flush=True)
    py = BASE / ".venv_master_2026_11_03/bin/python"
    proc = subprocess.run(
        [str(py), "-m", "pytest", "tests/test_audit_corrections_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Item 1: Cliff's delta correction...", flush=True)
    cliffs = run_cliffs_regeneration()
    ledger["phases"]["1_cliffs"] = {
        "ai_vader_cliffs": (cliffs.get("ai_assisted_v2_excl_exact_train_dups") or {}).get("vader", {}).get("cliffs_delta"),
        "frozen_vader_cliffs": (cliffs.get("frozen_original_posts") or {}).get("vader", {}).get("cliffs_delta"),
    }

    print("Item 2: Cardiff truncation correction...", flush=True)
    trunc = run_truncation_correction()
    ledger["phases"]["2_trunc"] = {
        "frozen_trunc_n": (trunc.get("frozen_all_units_complete_scores") or {}).get("truncated", {}).get("n_complete_scores"),
        "frozen_nontrunc_n": (trunc.get("frozen_all_units_complete_scores") or {}).get("nontruncated", {}).get("n_complete_scores"),
    }

    print("Item 3: expanded AI conversation RQ2/RQ3...", flush=True)
    conv = run_expanded_conversation()
    ledger["phases"]["3_conversation"] = {
        "ai_ge1": (conv.get("ai_eligibility") or {}).get("author_ge1"),
        "ai_both": (conv.get("ai_eligibility") or {}).get("both_roles"),
        "cardiff_replies": (conv.get("cardiff_reply_scoring") or {}).get("status"),
    }

    print("Item 4: RQ1 adjusted sensitivities...", flush=True)
    rq1 = run_rq1_adjusted()
    ledger["phases"]["4_rq1"] = {
        "ai_vader_adj": (
            (rq1.get("ai_assisted_v2_excl_exact_train_dups") or {})
            .get("vader", {})
            .get("regression_adjusted_log_length", {})
            .get("gds_coef_adjusted")
        ),
    }

    print("Item 5: RQ4 component bootstrap...", flush=True)
    rq4 = run_rq4()
    ledger["phases"]["5_rq4"] = {
        "fixed_split": rq4.get("fixed_rq4_v2_split"),
        "boot_two_minus_text": (rq4.get("component_bootstrap_paired_macro_f1") or {})
        .get("two_feature_minus_text_balanced", {})
        .get("ci95"),
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

    ai_v = (cliffs.get("ai_assisted_v2_excl_exact_train_dups") or {}).get("vader") or {}
    ft = (trunc.get("frozen_all_units_complete_scores") or {})
    packet = f"""# Codex review packet — audit corrections 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Preserves sealed:** `{SEALED_CORR}` (OUTPUT_HASHES_FINAL sha256 `{sealed_hash}`)  
**Original 6425 / protected / strict GDS / submission PDF unchanged. No manuscript.**

## Item 1 — Cliff's δ sign
- Convention fixed to **P(GDS>TPS)−P(GDS<TPS)** for `GDS−TPS` contrasts.
- AI VADER cliffs_delta now **{ai_v.get('cliffs_delta')}** (was ≈ −0.2137 reversed).
- Means/CIs definitionally unchanged; regenerated tables in `item1_cliffs_delta_correction/`.
- Unit test: `test_cliffs_delta_sign_matches_gds_minus_tps`.

## Item 2 — Cardiff truncation (same-stratum arms)
- Frozen truncated complete-score n=**{(ft.get('truncated') or {}).get('n_complete_scores')}** vs nontruncated n=**{(ft.get('nontruncated') or {}).get('n_complete_scores')}**
- Frozen trunc C–V / C–TB: **{(ft.get('truncated') or {}).get('cardiff_vs_vader')}** / **{(ft.get('truncated') or {}).get('cardiff_vs_textblob')}**
- Frozen nontrunc C–V / C–TB: **{(ft.get('nontruncated') or {}).get('cardiff_vs_vader')}** / **{(ft.get('nontruncated') or {}).get('cardiff_vs_textblob')}**
- AI-only and by-unit tables + length-band matched sensitivity included.
- Original Cardiff score file preserved (not overwritten).

## Item 3 — Expanded AI conversation
- AI eligibility: ge1=**{(conv.get('ai_eligibility') or {}).get('author_ge1')}**, ge2=**{(conv.get('ai_eligibility') or {}).get('author_ge2')}**, both roles=**{(conv.get('ai_eligibility') or {}).get('both_roles')}**
- RQ2/RQ3 estimands computed separately for original vs AI (VADER; exact-join TextBlob on first/second/first-community).
- Cardiff reply scoring status: **{(conv.get('cardiff_reply_scoring') or {}).get('status')}**
- Join missingness: `{conv.get('ai_join_missingness')}`
- Sentiment change ≠ resolution.

## Item 4 — RQ1 length/source/provenance
- Unadjusted vs log-length regression-adjusted vs length-band standardized contrasts for VADER/TextBlob/Cardiff.
- Source/date/decision-origin VADER sensitivities; nontruncated-only sensitivity.
- Labels unchanged.

## Item 5 — RQ4 component bootstrap
- Fixed split match: **{rq4.get('fixed_rq4_v2_split')}**
- Models: text unweighted/balanced, +VADER/TB, +Cardiff on same test IDs.
- Component-resampled paired macro-F1 CIs: `{rq4.get('component_bootstrap_paired_macro_f1')}`
- Five-split stability includes sentiment models.
- Saved-label agreement ≠ human accuracy.

## Reproduce
```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_audit_corrections_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_audit_corrections_2026_11_03.py
```

Input JSONL `{jh}` / CSV `{ch}`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Limitations

1. Cliff's δ corrected prospectively; sealed prior trees retain historical reversed values.
2. AI TPS/GDS labels remain unconfirmed.
3. Expanded RQ2/RQ3 use AI-selected labels; not field accuracy or issue resolution.
4. RQ1 adjusted contrasts are sensitivity analyses under possible selection bias.
5. RQ4 metrics are saved-label agreement only.
6. No manuscript; submission PDF untouched.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce audit corrections

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_audit_corrections_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_audit_corrections_2026_11_03.py
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
        "sealed_prior_OUTPUT_HASHES_FINAL_sha256": sealed_hash,
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
                "ai_vader_cliffs": ledger["phases"]["1_cliffs"]["ai_vader_cliffs"],
                "cardiff_replies": ledger["phases"]["3_conversation"]["cardiff_replies"],
                "rq4_fixed": rq4.get("fixed_rq4_v2_split"),
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
