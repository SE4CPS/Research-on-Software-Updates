#!/usr/bin/env python3
"""Apply Codex CONDITIONAL PASS corrections. Preserve sealed trees. No manuscript."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.pre_manuscript_evidence_corrections_2026_11_03.build import (  # noqa: E402
    run_all,
)
from full_resource_sentiment_v1.pre_manuscript_evidence_corrections_2026_11_03.common import (  # noqa: E402
    DATASET_JSONL,
    EXPECTED_JSONL,
    OUT_ROOT,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)


def main() -> int:
    ensure_dirs()
    if sha256_file(DATASET_JSONL) != EXPECTED_JSONL:
        write_json(OUT_ROOT / "BLOCKER.json", {"reason": "dataset hash mismatch"})
        return 2

    print("Tests...", flush=True)
    py = BASE / ".venv_master_2026_11_03/bin/python"
    proc = subprocess.run(
        [str(py), "-m", "pytest", "tests/test_pre_manuscript_evidence_corrections_2026_11_03.py", "-q"],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    tests = {"passed": proc.returncode == 0, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
    write_json(OUT_ROOT / "TEST_RESULTS.json", tests)
    if not tests["passed"]:
        print(tests)
        return 3

    print("Building corrections...", flush=True)
    summary = run_all()
    if not summary["seals_ok"]:
        return 4
    if not summary["prov_match"] or not summary["flow_ok"]:
        write_json(OUT_ROOT / "BLOCKER.json", {"summary": summary})
        return 5

    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", {"utc": utcnow(), "summary": summary, "tests": True})
    (OUT_ROOT / "RUN.md").write_text(
        f"""# Reproduce

```bash
cd phase6/full_resource_sentiment_v1
.venv_master_2026_11_03/bin/python -m pytest tests/test_pre_manuscript_evidence_corrections_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_pre_manuscript_evidence_corrections_2026_11_03.py
```

Outputs: `{OUT_ROOT}`
""",
        encoding="utf-8",
    )

    exclude = {"OUTPUT_HASHES_FINAL.json", "SEAL_META.json"}
    files = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file() and p.name not in exclude)
    seal = {
        "utc": utcnow(),
        "n_files": len(files),
        "files": {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in files},
        "status": "AWAITING_FINAL_CODEX_CHECK",
        "codex_conditions_addressed": {
            "1_label_provenance": True,
            "2_vader_rq2_canonical": True,
            "3_dataset_flow": True,
            "4_senti4sd_uncertainty": True,
        },
    }
    write_json(OUT_ROOT / "OUTPUT_HASHES_FINAL.json", seal)
    write_json(
        OUT_ROOT / "SEAL_META.json",
        {"OUTPUT_HASHES_FINAL_sha256": sha256_file(OUT_ROOT / "OUTPUT_HASHES_FINAL.json"), "n": len(files)},
    )

    print(json.dumps({"ok": True, "out": str(OUT_ROOT), **summary, "tests": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
