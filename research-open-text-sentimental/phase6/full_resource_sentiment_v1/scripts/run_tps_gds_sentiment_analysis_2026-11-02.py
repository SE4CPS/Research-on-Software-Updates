#!/usr/bin/env python3
"""Descriptive TPS vs GDS sentiment analysis on the frozen 6,425-ID combined table.

Does not modify labels, call APIs, unblind the deferred 200-case sample, or write a manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.tps_gds_sentiment_analysis.analyze import (  # noqa: E402
    eligibility_table,
    run_primary_analysis,
    sensitivity_suite,
)
from full_resource_sentiment_v1.tps_gds_sentiment_analysis.dictionary import (  # noqa: E402
    EXPECTED_COUNTS,
    MEASURE_DICTIONARY,
)
from full_resource_sentiment_v1.tps_gds_sentiment_analysis.load import verify_and_load  # noqa: E402

OUT = BASE / "results/tps_gds_sentiment_analysis_2026-11-02"


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


def flatten_contrast_rows(block: dict, contrast_key: str = "gds_minus_tps") -> list[dict]:
    rows = []
    for field, stats in (block.get(contrast_key) or {}).items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "field": field,
                "difference_name": stats.get("difference_name"),
                "n_gds_or_a": stats.get("n_a"),
                "n_tps_or_b": stats.get("n_b"),
                "mean_a": stats.get("mean_a"),
                "mean_b": stats.get("mean_b"),
                "mean_diff": stats.get("mean_diff"),
                "mean_diff_ci95_low": (stats.get("mean_diff_ci95") or [None, None])[0],
                "mean_diff_ci95_high": (stats.get("mean_diff_ci95") or [None, None])[1],
                "median_diff": stats.get("median_diff"),
                "cliffs_delta": stats.get("cliffs_delta"),
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


def fmt_ci(stats: dict | None) -> str:
    if not stats or stats.get("mean_diff") is None:
        return "n/a"
    ci = stats.get("mean_diff_ci95") or [None, None]
    return f"{stats['mean_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] (n_a={stats.get('n_a')}, n_b={stats.get('n_b')}; δ={stats.get('cliffs_delta')})"


def build_findings(primary: dict, sensitivity: dict, eligibility: dict, verify: dict) -> str:
    g = primary["gds_minus_tps"]
    cad = primary["cadence"]
    lines = [
        "# Findings — TPS vs GDS sentiment (descriptive; validation deferred)",
        "",
        f"**UTC:** {utcnow()}  ",
        "**Scope:** Combined 6,425-ID research table. Primary contrasts use `final_relevance==relevant` and "
        "`sentiment_definition==frozen_broad_release_train_package_scores`. Other is a separate context group. "
        "The 200-case blinded review was deferred; these results are **not** independently human-validated accuracy claims.",
        "",
        "## Input verification",
        f"- Frame N: **{verify['n']}**; protected overlap: **{verify['protected_overlap']}**",
        f"- Relevance / discussion counts match frozen validation report: **{'yes' if verify['ok'] else 'NO'}**",
        f"- Sentiment definitions: `{json.dumps(verify['sentiment_definition_counts'], sort_keys=True)}`",
        f"- Existing / new model decisions: **{verify['n_existing_decisions']} / {verify['n_new_model_decisions']}**",
        f"- JSONL SHA-256: `{verify['jsonl_sha256']}`",
        "",
        "## Analysis population (frozen definition)",
        f"- Relevant frozen rows: **{primary['n_relevant_frozen']}**",
        f"- By discussion: `{json.dumps(primary['n_by_discussion'], sort_keys=True)}`",
        "",
        "## Primary GDS − TPS contrasts (mean difference, 95% bootstrap CI, Cliff's δ)",
    ]
    for field in (
        "title_compound",
        "body_compound",
        "combined_compound",
        "body_minus_title",
        "author_mean",
        "community_mean",
        "author_shift",
        "last_minus_first",
        "order_slope_normalized",
        "elapsed_hours",
    ):
        lines.append(f"- **{field}**: {fmt_ci(g.get(field))}")

    lines.extend(
        [
            "",
            "## Cadence eligibility (frozen; not resolution)",
            f"- TPS author-shift n / order last−first n / elapsed n: "
            f"**{cad['TPS']['n_author_shift']} / {cad['TPS']['n_order_eligible_last_minus_first']} / {cad['TPS']['n_elapsed']}**",
            f"- GDS author-shift n / order last−first n / elapsed n: "
            f"**{cad['GDS']['n_author_shift']} / {cad['GDS']['n_order_eligible_last_minus_first']} / {cad['GDS']['n_elapsed']}**",
            "",
            "## Length caution (Workstream I approach, full-text only)",
            "- Title/body contrasts are stratified by body-length bands where available.",
            "- Equal-length prefix estimands from Workstream I are **not** mixed with full-text estimates.",
            f"- GDS − TPS body length: {fmt_ci(primary['length_diagnostics'].get('gds_minus_tps_body_len'))}",
            f"- GDS − TPS body−title: {fmt_ci(primary['length_diagnostics'].get('gds_minus_tps_body_minus_title'))}",
            "",
            "## Sensitivity (selected combined_compound GDS − TPS)",
        ]
    )
    for sl in sensitivity["frozen_slices"]:
        stats = (sl.get("contrast") or {}).get("combined_compound")
        lines.append(
            f"- **{sl['slice']}**: TPS={sl['n_tps']}, GDS={sl['n_gds']}; combined {fmt_ci(stats)}"
        )
    hist = sensitivity["historical_relevant_separate"]
    lines.extend(
        [
            "",
            "## Historical-542 definition (separate; not pooled into cadence)",
            f"- Relevant historical rows: TPS={hist['n_tps']}, GDS={hist['n_gds']} (of {hist['n_relevant_historical']})",
            f"- Combined compound: {fmt_ci((hist.get('contrast_post_level') or {}).get('combined_compound'))}",
            "",
            "## Eligibility counts (observed denominators)",
            "```json",
            json.dumps(eligibility, indent=2, sort_keys=True),
            "```",
            "",
            "## Stopped before",
            "Manuscript writing; label/threshold changes; paid API calls; any claim that the deferred 200-case review passed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_limitations() -> str:
    return """# Limitations table — TPS/GDS sentiment analysis 2026-11-02

| Limitation | Implication |
|---|---|
| Deferred 200-case blinded human review | No fresh independent accuracy, sampling-weighted P/R, or confirmation of all 6,425 labels. |
| Mixed provenance (1,964 existing + 4,461 Terra) | Label error may differ by `decision_source`; sensitivities are required, not optional. |
| Selected 6,425 frame, not all 37,596 candidates | Findings do not generalize to the full ReleaseTrain candidate snapshot without further sampling. |
| Two sentiment definitions | Cadence/reply-order/elapsed measures exist only on frozen broad package scores; historical-542 is reported separately. |
| Missing replies are null | Absence is not observed neutral; denominators shrink for author/community/order/elapsed measures. |
| Count-gap retention | Incomplete comment capture remains in the table as a sensitivity stratum, not proof of completeness. |
| VADER lexical scores | Not human affect labels; length-sensitive for title/body (Workstream I). |
| Sentiment change ≠ resolution | Author/community shifts and last−first slopes do not establish whether an update issue was fixed. |
| Single-table descriptive design | Effect sizes and CIs describe this assembled dataset; they are not causal estimates. |
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    verify = verify_and_load()
    # Drop rows from JSON export of verify
    verify_public = {k: v for k, v in verify.items() if k != "rows"}
    write_json(OUT / "INPUT_VERIFICATION.json", verify_public)
    if not verify["ok"]:
        write_json(OUT / "BLOCKER.json", {"utc": utcnow(), "discrepancies": verify["discrepancies"]})
        print("STOP: verification failed", verify["discrepancies"])
        return 2

    rows = verify["rows"]
    write_json(OUT / "ANALYSIS_DICTIONARY.json", MEASURE_DICTIONARY)
    write_json(OUT / "EXPECTED_COUNTS.json", EXPECTED_COUNTS)

    primary = run_primary_analysis(rows)
    sensitivity = sensitivity_suite(rows)
    eligibility = eligibility_table(rows)

    # Strip nothing from machine tables; they are aggregates only (no raw Reddit text).
    write_json(OUT / "primary_analysis.json", primary)
    write_json(OUT / "sensitivity_analysis.json", sensitivity)
    write_json(OUT / "eligibility_counts.json", eligibility)

    write_csv(OUT / "tables/gds_minus_tps_primary.csv", flatten_contrast_rows(primary, "gds_minus_tps"))
    write_csv(OUT / "tables/other_minus_tps_context.csv", flatten_contrast_rows(primary, "other_minus_tps"))

    sens_rows = []
    for sl in sensitivity["frozen_slices"] + sensitivity["reply_count_slices"]:
        for field, stats in (sl.get("contrast") or {}).items():
            sens_rows.append(
                {
                    "slice": sl["slice"],
                    "field": field,
                    "n_tps": sl["n_tps"],
                    "n_gds": sl["n_gds"],
                    "mean_diff": stats.get("mean_diff"),
                    "mean_diff_ci95_low": (stats.get("mean_diff_ci95") or [None, None])[0],
                    "mean_diff_ci95_high": (stats.get("mean_diff_ci95") or [None, None])[1],
                    "cliffs_delta": stats.get("cliffs_delta"),
                    "n_obs_tps": (sl.get("observed_n") or {}).get(field, {}).get("TPS"),
                    "n_obs_gds": (sl.get("observed_n") or {}).get(field, {}).get("GDS"),
                }
            )
    write_csv(OUT / "tables/sensitivity_gds_minus_tps.csv", sens_rows)

    findings = build_findings(primary, sensitivity, eligibility, verify_public)
    (OUT / "FINDINGS.md").write_text(findings, encoding="utf-8")
    (OUT / "LIMITATIONS.md").write_text(build_limitations(), encoding="utf-8")

    (OUT / "RUN.md").write_text(
        """# TPS/GDS sentiment analysis — commands

```bash
cd phase6/full_resource_sentiment_v1
.venv/bin/python -m pytest tests/test_tps_gds_sentiment_analysis.py -q
.venv/bin/python scripts/run_tps_gds_sentiment_analysis_2026-11-02.py
```

Outputs land in `results/tps_gds_sentiment_analysis_2026-11-02/`.
Does not modify frozen datasets, labels, or the deferred 200-case review files.
""",
        encoding="utf-8",
    )

    # Manifest + hashes of outputs (not of restricted review packet).
    outputs = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "OUTPUT_HASHES.json")
    hashes = {str(p.relative_to(OUT)): sha256_file(p) for p in outputs}
    manifest = {
        "utc": utcnow(),
        "run_id": "tps-gds-sentiment-analysis-2026-11-02",
        "validation_status": "deferred_200_case_review_not_completed",
        "input_jsonl_sha256": verify["jsonl_sha256"],
        "input_csv_sha256": verify["csv_sha256"],
        "n_master_rows_retained": len(rows),
        "commands": [
            ".venv/bin/python -m pytest tests/test_tps_gds_sentiment_analysis.py -q",
            ".venv/bin/python scripts/run_tps_gds_sentiment_analysis_2026-11-02.py",
        ],
        "scientific_rule": (
            "Descriptive analysis of the assembled selected dataset; provenance remains visible; "
            "do not claim independent validation from a skim or deferred review."
        ),
    }
    write_json(OUT / "MANIFEST.json", manifest)
    write_json(OUT / "OUTPUT_HASHES.json", {"utc": utcnow(), "files": hashes})

    headline = {
        "ok": True,
        "n": verify["n"],
        "n_relevant_frozen": primary["n_relevant_frozen"],
        "n_by_discussion": primary["n_by_discussion"],
        "combined_gds_minus_tps": primary["gds_minus_tps"]["combined_compound"],
        "out": str(OUT),
    }
    print(json.dumps(headline, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
