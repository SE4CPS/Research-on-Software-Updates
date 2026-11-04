#!/usr/bin/env python3
"""Stage 1: corrected VADER TPS/GDS analysis (new versioned outputs only).

Stops after Stage 1. No model installs, no Stage 2, no manuscript, no 200-case changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.tps_gds_sentiment_v2.analyze import (  # noqa: E402
    run_stage1,
    sensitivity_suite,
)
from full_resource_sentiment_v1.tps_gds_sentiment_v2.dictionary import (  # noqa: E402
    EXPECTED,
    MEASURE_DICTIONARY,
    N_BOOT,
    RUN_ID,
    SEED,
)
from full_resource_sentiment_v1.tps_gds_sentiment_v2.load import verify_and_load  # noqa: E402

OUT = BASE / "results/tps_gds_sentiment_v2_2026-11-03/stage1_corrected_vader"


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def flatten_contrast(block: dict[str, Any] | dict, name: str) -> list[dict]:
    rows = []
    for field, stats in (block or {}).items():
        if not isinstance(stats, dict) or "mean_diff" not in stats:
            continue
        ci = stats.get("mean_diff_ci95") or [None, None]
        rows.append(
            {
                "block": name,
                "field": field,
                "difference_name": stats.get("difference_name"),
                "n_gds": stats.get("n_a"),
                "n_tps": stats.get("n_b"),
                "mean_gds": stats.get("mean_a"),
                "mean_tps": stats.get("mean_b"),
                "mean_diff": stats.get("mean_diff"),
                "mean_diff_ci95_low": ci[0],
                "mean_diff_ci95_high": ci[1],
                "median_diff": stats.get("median_diff"),
                "cliffs_delta": stats.get("cliffs_delta"),
                "n_bootstrap": stats.get("n_bootstrap"),
                "seed": stats.get("seed"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(stats: dict | None) -> str:
    if not stats or stats.get("mean_diff") is None:
        return "n/a"
    ci = stats.get("mean_diff_ci95") or [None, None]
    return (
        f"{stats['mean_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] "
        f"(n_gds={stats.get('n_a')}, n_tps={stats.get('n_b')}; δ={stats.get('cliffs_delta')})"
    )


def fmt_std(block: dict | None) -> str:
    if not block or block.get("standardized_mean_diff") is None:
        return "n/a"
    ci = block.get("mean_diff_ci95") or [None, None]
    return (
        f"{block['standardized_mean_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] "
        f"(eligible TPS/GDS={block.get('n_tps_eligible')}/{block.get('n_gds_eligible')}; "
        f"excluded bands={block.get('bands_missing_a_class')})"
    )


def build_findings(verify: dict, primary: dict, sensitivity: dict) -> str:
    g = primary["gds_minus_tps_unadjusted"]
    la = primary["length_adjusted"]
    lines = [
        "# Stage 1 findings — corrected VADER TPS vs GDS (v2)",
        "",
        f"**UTC:** {utcnow()}  ",
        f"**Run ID:** `{RUN_ID}`  ",
        f"**Bootstrap:** {N_BOOT} post-level draws, seed {SEED}.  ",
        "**Validation:** 200-case review still has **zero** saved decisions; not a completed independent validation.",
        "",
        "## Input verification",
        f"- Frame N={verify['n']}; protected overlap={verify['protected_overlap']}; ok={verify['ok']}",
        f"- Frozen relevant TPS/GDS: **{verify['n_frozen_relevant_tps']} / {verify['n_frozen_relevant_gds']}**",
        f"- JSONL SHA-256: `{verify['jsonl_sha256']}`",
        "",
        "## Corrected definitions",
        "- **`author_shift`**: later-half − earlier-half of nonempty **author** replies; requires **≥4** author replies.",
        "- **`original_post_to_author_mean`**: `author_mean − combined_compound`; requires **≥1** author reply.",
        "- These are **different estimands**. Prior v1 dictionary wording for author_shift was wrong; treat the small-n author_shift GDS contrast as **exploratory**.",
        "",
        "## Eligibility (frozen relevant)",
        "```json",
        json.dumps(primary["eligibility"], indent=2, sort_keys=True),
        "```",
        "",
        f"Author-shift provenance: `{json.dumps(primary['author_shift_provenance'], sort_keys=True)}`",
        "",
        "## Unadjusted GDS − TPS (frozen relevant)",
    ]
    for field in (
        "title_compound",
        "body_compound",
        "combined_compound",
        "author_mean",
        "community_mean",
        "author_shift",
        "original_post_to_author_mean",
        "last_minus_first",
        "order_slope_normalized",
        "elapsed_hours",
    ):
        lines.append(f"- **{field}**: {fmt(g.get(field))}")

    lines.extend(
        [
            "",
            "## Length-adjusted GDS − TPS (prespecified bands; standardized over common-support bands)",
            f"- **body_compound by body band**: {fmt_std(la.get('body_compound_by_body_band'))}",
            f"- **title_compound by title band**: {fmt_std(la.get('title_compound_by_title_band'))}",
            f"- **combined_compound by body band**: {fmt_std(la.get('combined_compound_by_body_band'))}",
            "",
            "### Unadjusted on the same length-eligible rows (companion, not the adjusted estimand)",
            f"- body: {fmt((la.get('body_compound_by_body_band') or {}).get('unadjusted_on_same_eligible'))}",
            f"- title: {fmt((la.get('title_compound_by_title_band') or {}).get('unadjusted_on_same_eligible'))}",
            f"- combined: {fmt((la.get('combined_compound_by_body_band') or {}).get('unadjusted_on_same_eligible'))}",
            "",
            "## Secondary OLS (not primary)",
            f"- combined ~ GDS + log1p(body_len) + title_len + oct2026 + existing: "
            f"`{json.dumps(primary.get('secondary_ols_combined'), sort_keys=True, default=str)[:500]}...`",
            "",
            "## Sensitivity (combined_compound GDS − TPS)",
        ]
    )
    for sl in sensitivity["frozen_slices"]:
        lines.append(
            f"- **{sl['slice']}**: TPS={sl['n_tps']}, GDS={sl['n_gds']}; "
            f"{fmt((sl.get('contrast') or {}).get('combined_compound'))}"
        )
    hist = sensitivity["historical_separate"]
    lines.extend(
        [
            "",
            "## Historical definition (separate)",
            f"- Relevant historical TPS/GDS: {hist['n_tps']}/{hist['n_gds']} of {hist['n_relevant']}",
            f"- Combined: {fmt((hist.get('contrast_post_level') or {}).get('combined_compound'))}",
            "",
            "## Stopped before",
            "Stage 2 model install/pilot; Stage 3–6; manuscript; paid APIs; changes to frozen labels or 200-case files.",
            "",
        ]
    )
    return "\n".join(lines)


def build_limitations() -> str:
    return """# Limitations — Stage 1 corrected VADER (v2)

| Limitation | Implication |
|---|---|
| Deferred 200-case human review (0 saved decisions) | No independent label accuracy or sampling-weighted P/R. |
| Assembled selected 6,425 table | Not inference to all 37,596 ReleaseTrain candidates. |
| Mixed provenance (1,964 existing + 4,461 Terra) | Label error may differ by source; sensitivities required. |
| Author-shift small GDS n | Exploratory; prior independent large bootstrap was unstable near zero. |
| Length adjustment | Band standardization uses common-support bands only; OLS is secondary. |
| Historical 520 separate | Different score construction; not pooled into frozen cadence/author_shift. |
| Missing replies are null | Absence ≠ observed neutral. |
| Count-gap retained | Incomplete capture is a sensitivity stratum, not completeness proof. |
| VADER lexical scores | Not human affect labels; length-sensitive. |
| Sentiment change ≠ resolution | Author-shift / last−first do not establish issue fix. |
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    verify = verify_and_load()
    verify_public = {k: v for k, v in verify.items() if k != "rows"}
    write_json(OUT / "INPUT_VERIFICATION.json", verify_public)
    write_json(OUT / "ANALYSIS_DICTIONARY.json", MEASURE_DICTIONARY)
    write_json(OUT / "EXPECTED_COUNTS.json", EXPECTED)
    if not verify["ok"]:
        write_json(OUT / "BLOCKER.json", {"utc": utcnow(), "discrepancies": verify["discrepancies"]})
        print("STOP: verification failed", verify["discrepancies"])
        return 2

    print("Running primary Stage-1 analysis (10k bootstrap)...", flush=True)
    primary = run_stage1(verify["rows"])
    print("Running sensitivity suite...", flush=True)
    sensitivity = sensitivity_suite(verify["rows"])

    write_json(OUT / "primary_analysis.json", primary)
    write_json(OUT / "sensitivity_analysis.json", sensitivity)

    write_csv(
        OUT / "tables/gds_minus_tps_unadjusted.csv",
        flatten_contrast(primary["gds_minus_tps_unadjusted"], "unadjusted"),
    )
    write_csv(
        OUT / "tables/other_minus_tps_context.csv",
        flatten_contrast(primary["other_minus_tps_context"], "other_context"),
    )

    length_rows = []
    for key, block in primary["length_adjusted"].items():
        if key == "note" or not isinstance(block, dict):
            continue
        ci = block.get("mean_diff_ci95") or [None, None]
        length_rows.append(
            {
                "estimand": block.get("estimand"),
                "score_field": block.get("score_field"),
                "band_key": block.get("band_key"),
                "standardized_mean_diff": block.get("standardized_mean_diff"),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "n_tps_eligible": block.get("n_tps_eligible"),
                "n_gds_eligible": block.get("n_gds_eligible"),
                "bands_missing_a_class": json.dumps(block.get("bands_missing_a_class")),
                "unadjusted_mean_diff": (block.get("unadjusted_on_same_eligible") or {}).get("mean_diff"),
            }
        )
        for band in block.get("bands") or []:
            length_rows.append(
                {
                    "estimand": f"band_detail:{block.get('estimand')}",
                    "score_field": block.get("score_field"),
                    "band_key": band.get("band"),
                    "standardized_mean_diff": band.get("mean_gds_minus_tps"),
                    "ci95_low": None,
                    "ci95_high": None,
                    "n_tps_eligible": band.get("n_tps"),
                    "n_gds_eligible": band.get("n_gds"),
                    "bands_missing_a_class": None if band.get("common_support") else "missing_class",
                    "unadjusted_mean_diff": band.get("mean_gds_minus_tps"),
                }
            )
    write_csv(OUT / "tables/length_adjusted_gds_minus_tps.csv", length_rows)

    sens_rows = []
    for sl in sensitivity["frozen_slices"] + sensitivity["reply_count_slices"]:
        for field, stats in (sl.get("contrast") or {}).items():
            ci = stats.get("mean_diff_ci95") or [None, None]
            sens_rows.append(
                {
                    "slice": sl["slice"],
                    "field": field,
                    "n_tps": sl["n_tps"],
                    "n_gds": sl["n_gds"],
                    "mean_diff": stats.get("mean_diff"),
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "cliffs_delta": stats.get("cliffs_delta"),
                    "n_obs_tps": (sl.get("eligibility") or {}).get(field, {}).get("TPS"),
                    "n_obs_gds": (sl.get("eligibility") or {}).get(field, {}).get("GDS"),
                }
            )
    write_csv(OUT / "tables/sensitivity_gds_minus_tps.csv", sens_rows)

    (OUT / "FINDINGS.md").write_text(build_findings(verify_public, primary, sensitivity), encoding="utf-8")
    (OUT / "LIMITATIONS.md").write_text(build_limitations(), encoding="utf-8")
    (OUT / "RUN.md").write_text(
        f"""# Stage 1 reproduction

```bash
cd phase6/full_resource_sentiment_v1
.venv/bin/python -m pytest tests/test_tps_gds_sentiment_v2_stage1.py -q
.venv/bin/python scripts/run_tps_gds_sentiment_v2_stage1_2026-11-03.py
```

Outputs: `results/tps_gds_sentiment_v2_2026-11-03/stage1_corrected_vader/`

Bootstrap: n={N_BOOT}, seed={SEED}. Does not modify frozen inputs or prior v1 analysis directory.
""",
        encoding="utf-8",
    )

    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "OUTPUT_HASHES.json")
    hashes = {str(p.relative_to(OUT)): sha256_file(p) for p in outputs}
    write_json(
        OUT / "MANIFEST.json",
        {
            "utc": utcnow(),
            "run_id": RUN_ID,
            "stage": 1,
            "status": "STOP_FOR_CODEX_REVIEW",
            "input_jsonl_sha256": verify["jsonl_sha256"],
            "input_csv_sha256": verify["csv_sha256"],
            "n_boot": N_BOOT,
            "seed": SEED,
            "prior_analysis_preserved": "results/tps_gds_sentiment_analysis_2026-11-02/",
            "validation_200_untouched": True,
            "next": "Codex independent review; do not start Stage 2 without disk gate + approval",
        },
    )
    write_json(OUT / "OUTPUT_HASHES.json", {"utc": utcnow(), "files": hashes})

    print(
        json.dumps(
            {
                "ok": True,
                "out": str(OUT),
                "combined_unadjusted": primary["gds_minus_tps_unadjusted"]["combined_compound"],
                "author_shift": primary["gds_minus_tps_unadjusted"]["author_shift"],
                "post_to_author": primary["gds_minus_tps_unadjusted"]["original_post_to_author_mean"],
                "length_adj_body": primary["length_adjusted"]["body_compound_by_body_band"].get(
                    "standardized_mean_diff"
                ),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
