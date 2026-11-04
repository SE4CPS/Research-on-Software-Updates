#!/usr/bin/env python3
"""Pre-manuscript evidence reconciliation. No manuscript; preserve seals."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.pre_manuscript_evidence_2026_11_03.build import run_all  # noqa: E402
from full_resource_sentiment_v1.pre_manuscript_evidence_2026_11_03.common import (  # noqa: E402
    AUDIT_OUT,
    CORR_OUT,
    DATASET_JSONL,
    EXPECTED_JSONL,
    LENGTH_OUT,
    OUT_ROOT,
    SENTI4SD_OUT,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)


def main() -> int:
    ensure_dirs()
    jh = sha256_file(DATASET_JSONL)
    if jh != EXPECTED_JSONL:
        write_json(OUT_ROOT / "BLOCKER.json", {"reason": "dataset hash mismatch", "jsonl": jh})
        return 2

    prior = {
        "audit": sha256_file(AUDIT_OUT / "OUTPUT_HASHES_FINAL.json"),
        "length": sha256_file(LENGTH_OUT / "OUTPUT_HASHES_FINAL.json"),
        "senti4sd": sha256_file(SENTI4SD_OUT / "OUTPUT_HASHES_FINAL.json"),
        "correction": sha256_file(CORR_OUT / "OUTPUT_HASHES_FINAL.json"),
        "jsonl": jh,
    }
    write_json(OUT_ROOT / "INPUT_VERIFICATION.json", {"ok": True, **prior})

    print("Tests...", flush=True)
    py = BASE / ".venv_master_2026_11_03/bin/python"
    proc = subprocess.run(
        [str(py), "-m", "pytest", "tests/test_pre_manuscript_evidence_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Building ledger, four-model match, figures...", flush=True)
    summary = run_all()

    # Confirm seals unchanged after run
    after = {
        "audit": sha256_file(AUDIT_OUT / "OUTPUT_HASHES_FINAL.json"),
        "length": sha256_file(LENGTH_OUT / "OUTPUT_HASHES_FINAL.json"),
        "senti4sd": sha256_file(SENTI4SD_OUT / "OUTPUT_HASHES_FINAL.json"),
        "correction": sha256_file(CORR_OUT / "OUTPUT_HASHES_FINAL.json"),
    }
    seals_ok = after == {k: prior[k] for k in after}
    write_json(OUT_ROOT / "SEAL_PRESERVATION_CHECK.json", {"ok": seals_ok, "before": prior, "after": after})
    if not seals_ok:
        write_json(OUT_ROOT / "BLOCKER.json", {"reason": "sealed prior hash changed unexpectedly"})
        return 4

    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", {"utc": utcnow(), "summary": summary, "tests": tests["passed"]})
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_pre_manuscript_evidence_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_pre_manuscript_evidence_reconciliation_2026_11_03.py
```

Outputs: `{OUT_ROOT}`
""",
        encoding="utf-8",
    )
    (OUT_ROOT / "SCIENTIFIC_LIMITATIONS.md").write_text(
        """# Limitations (carry into manuscript)

1. AI purpose labels unconfirmed; no completed human validation packet.
2. Four-model table is reply-eligible (6,996), not full RQ1 (15,312).
3. Instrument disagreement is expected; agreement ≠ accuracy; no cross-scale magnitudes.
4. Sentiment change ≠ issue resolution.
5. RQ4 metrics are saved-label agreement, not field accuracy.
6. Author decisions on claim strength / Senti4SD expansion remain open.
""",
        encoding="utf-8",
    )

    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    seal = {
        "utc": utcnow(),
        "n_files": len(files),
        "files": {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files},
        "sealed_priors_unchanged": seals_ok,
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
                "n_ledger_rows": summary["n_ledger_rows"],
                "matched_pairs": summary["match"]["total_matched_pairs"],
                "ai_fully_matched": summary["match"]["cohorts"]["ai_assisted_v2"]["fully_matched"],
                "ai_rq2_dirs": summary["four_model_ai_rq2_dirs"],
                "seals_ok": seals_ok,
                "tests": tests["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
