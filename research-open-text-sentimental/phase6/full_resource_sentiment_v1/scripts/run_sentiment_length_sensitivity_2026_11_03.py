#!/usr/bin/env python3
"""First-author−post length sensitivity (full / prefix / adjusted). No manuscript."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03.analysis import (  # noqa: E402
    run_analysis,
)
from full_resource_sentiment_v1.sentiment_length_sensitivity_2026_11_03.common import (  # noqa: E402
    DATASET_CSV,
    DATASET_JSONL,
    EXPECTED_CSV,
    EXPECTED_JSONL,
    OUT_ROOT,
    RESTRICTED_ROOT,
    SEALED_AUDIT,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)


def _fmt(c: dict | None) -> str:
    if not c:
        return "n/a"
    return (
        f"n={c.get('n_a')}/{c.get('n_b')} mean={c.get('mean_diff')} "
        f"CI95={c.get('mean_diff_ci95')} cliffs={c.get('cliffs_delta')}"
    )


def main() -> int:
    ensure_dirs()
    jh, ch = sha256_file(DATASET_JSONL), sha256_file(DATASET_CSV)
    if jh != EXPECTED_JSONL or ch != EXPECTED_CSV:
        write_json(OUT_ROOT / "BLOCKER.json", {"jsonl": jh, "csv": ch})
        return 2
    write_json(OUT_ROOT / "INPUT_VERIFICATION.json", {"jsonl_sha256": jh, "csv_sha256": ch, "ok": True})
    sealed = sha256_file(SEALED_AUDIT / "OUTPUT_HASHES_FINAL.json") if (SEALED_AUDIT / "OUTPUT_HASHES_FINAL.json").exists() else None
    write_json(
        OUT_ROOT / "SEALED_PRIOR_HASHES.json",
        {"audit_corrections_OUTPUT_HASHES_FINAL_sha256": sealed, "jsonl": jh, "csv": ch},
    )

    print("Tests...", flush=True)
    py = BASE / ".venv_master_2026_11_03/bin/python"
    proc = subprocess.run(
        [str(py), "-m", "pytest", "tests/test_sentiment_length_sensitivity_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Length sensitivity analyses...", flush=True)
    results = run_analysis()
    ai = results["ai"]
    fr = results["frozen"]
    summ = results["summary"]

    findings = f"""# Findings — first-author−post length sensitivity (2026-11-03)

## Scope
- **AI cohort:** exactly **{ai['n_pairs']}** posts with a scored first original-author reply ({ai['n_gds']} GDS / {ai['n_tps']} TPS).
- **Frozen cohort:** **{fr['n_pairs']}** pairs ({fr['n_gds']} GDS / {fr['n_tps']} TPS), reported separately.
- Labels unchanged. Sealed prior trees untouched. No manuscript.

## Length context (AI)
{json.dumps(ai.get('length_summary'), indent=2)}

## Analysis 1 — full saved text (reply − post)
| Instrument | AI GDS−TPS | Frozen GDS−TPS |
|---|---|---|
| VADER | `{_fmt(ai['analysis_1_full_saved_text']['vader'])}` | `{_fmt(fr['analysis_1_full_saved_text']['vader'])}` |
| TextBlob | `{_fmt(ai['analysis_1_full_saved_text']['textblob'])}` | `{_fmt(fr['analysis_1_full_saved_text']['textblob'])}` |
| Cardiff sign | `{_fmt(ai['analysis_1_full_saved_text']['cardiff_sign'])}` | `{_fmt(fr['analysis_1_full_saved_text']['cardiff_sign'])}` |

## Analysis 2 — equal-length character prefixes
Prefix length = `min(post_chars, reply_chars)` for **both** texts. This scores a **different portion** of the longer document (typically its opening), not the full post/reply. Prefix and full-text estimands are not interchangeable.

| Instrument | AI GDS−TPS | Frozen GDS−TPS |
|---|---|---|
| VADER | `{_fmt(ai['analysis_2_equal_length_prefixes']['vader'])}` | `{_fmt(fr['analysis_2_equal_length_prefixes']['vader'])}` |
| TextBlob | `{_fmt(ai['analysis_2_equal_length_prefixes']['textblob'])}` | `{_fmt(fr['analysis_2_equal_length_prefixes']['textblob'])}` |
| Cardiff sign | `{_fmt(ai['analysis_2_equal_length_prefixes']['cardiff_sign'])}` | `{_fmt(fr['analysis_2_equal_length_prefixes']['cardiff_sign'])}` |

## Analysis 3 — paired model (full-text delta)
`delta ~ GDS + log1p(post_chars) + log1p(reply_chars) + source + date_band`

| Instrument | AI GDS coef (CI95) | Frozen GDS coef (CI95) |
|---|---|---|
| VADER | `{ai['analysis_3_paired_length_source_date_adjusted']['vader']}` | `{fr['analysis_3_paired_length_source_date_adjusted']['vader']}` |
| TextBlob | `{ai['analysis_3_paired_length_source_date_adjusted']['textblob']}` | `{fr['analysis_3_paired_length_source_date_adjusted']['textblob']}` |
| Cardiff sign | `{ai['analysis_3_paired_length_source_date_adjusted']['cardiff_sign']}` | `{fr['analysis_3_paired_length_source_date_adjusted']['cardiff_sign']}` |

## Truncation
- AI: `{ai.get('truncation')}`
- Frozen: `{fr.get('truncation')}`

## Limits
- AI purpose labels are unconfirmed.
- Instrument disagreement is expected; not accuracy.
- Sentiment change ≠ issue resolution.
"""
    (OUT_ROOT / "FINDINGS.md").write_text(findings, encoding="utf-8")

    packet = f"""# Codex review packet — sentiment-length sensitivity 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Preserves sealed audit tree:** `{SEALED_AUDIT}` (sha256 `{sealed}`)  
**No manuscript. Labels unchanged. Cohorts separate.**

## Design
1. Full saved text: first-author reply − original post  
2. Equal-length character prefixes (`min(post, reply)` on both) — **different text portion than full-text**  
3. Paired OLS adjusting for post length, reply length, source, date  

Instruments: VADER, TextBlob, Cardiff sign — reported separately.

## AI cohort (n={ai['n_pairs']}; {ai['n_gds']}/{ai['n_tps']} GDS/TPS)
- Full VADER: `{_fmt(ai['analysis_1_full_saved_text']['vader'])}`
- Prefix VADER: `{_fmt(ai['analysis_2_equal_length_prefixes']['vader'])}`
- Adjusted VADER GDS coef: `{(ai['analysis_3_paired_length_source_date_adjusted']['vader'] or {}).get('gds_coef_adjusted')}` CI `{(ai['analysis_3_paired_length_source_date_adjusted']['vader'] or {}).get('gds_coef_ci95')}`
- Full TextBlob: `{_fmt(ai['analysis_1_full_saved_text']['textblob'])}`
- Prefix TextBlob: `{_fmt(ai['analysis_2_equal_length_prefixes']['textblob'])}`
- Full Cardiff: `{_fmt(ai['analysis_1_full_saved_text']['cardiff_sign'])}`
- Prefix Cardiff: `{_fmt(ai['analysis_2_equal_length_prefixes']['cardiff_sign'])}`

## Frozen cohort (n={fr['n_pairs']}; {fr['n_gds']}/{fr['n_tps']})
- Full VADER: `{_fmt(fr['analysis_1_full_saved_text']['vader'])}`
- Prefix VADER: `{_fmt(fr['analysis_2_equal_length_prefixes']['vader'])}`
- Adjusted VADER: `{(fr['analysis_3_paired_length_source_date_adjusted']['vader'] or {}).get('gds_coef_adjusted')}`

## Reproduce
```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_sentiment_length_sensitivity_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_sentiment_length_sensitivity_2026_11_03.py
```

Details: `length_sensitivity/`, `FINDINGS.md`  
Input JSONL `{jh}` / CSV `{ch}`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Limitations

1. AI TPS/GDS labels are model predictions, not human-confirmed.
2. Prefix analysis uses a different text span than full-document scoring.
3. Cardiff polarity signs are coarse (−1/0/+1) and not commensurate with VADER/TextBlob.
4. Sentiment change is not issue resolution.
5. Frozen and AI cohorts are not pooled for inference.
6. No manuscript in this run.
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_sentiment_length_sensitivity_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_sentiment_length_sensitivity_2026_11_03.py
```

Outputs: `{OUT_ROOT}`  
Restricted: `{RESTRICTED_ROOT}`
""",
        encoding="utf-8",
    )

    ledger = {
        "utc": utcnow(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": shutil.disk_usage("/").free / (1024**3),
        "ai_n": ai["n_pairs"],
        "frozen_n": fr["n_pairs"],
        "tests": tests["passed"],
        "summary_keys": list(summ.keys()),
    }
    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", ledger)

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

    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    seal = {
        "utc": utcnow(),
        "n_files": len(files),
        "files": {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files},
        "input_sha256": {"jsonl": jh, "csv": ch},
        "sealed_prior_audit_sha256": sealed,
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
                "ai_n": ai["n_pairs"],
                "frozen_n": fr["n_pairs"],
                "ai_full_vader_mean": (ai["analysis_1_full_saved_text"]["vader"] or {}).get("mean_diff"),
                "ai_prefix_vader_mean": (ai["analysis_2_equal_length_prefixes"]["vader"] or {}).get("mean_diff"),
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
