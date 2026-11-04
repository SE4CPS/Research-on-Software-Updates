#!/usr/bin/env python3
"""Freeze manifest, draw the approved 200-case sample once, and build blinded packets.

Does not collect human labels, unblind predictions, call paid APIs, or modify frozen sources.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.validation_200.constants import (  # noqa: E402
    EXPECTED_DRAW_COVERED,
    EXPECTED_DRAW_FRESH,
    EXPECTED_DRAW_TOTAL,
    RUN_ID,
    STRATA,
)
from full_resource_sentiment_v1.validation_200.frame_audit import build_frame_audit  # noqa: E402
from full_resource_sentiment_v1.validation_200.sampler import (  # noqa: E402
    assert_no_forbidden_overlap,
    draw_validation_sample,
)
from full_resource_sentiment_v1.validation_200.text_resolve import (  # noqa: E402
    build_full_text_index,
    redact_packet_text,
)

PUBLIC = BASE / "results/validation_200_2026-11-02"
RESTRICTED = BASE / "restricted/validation_200_2026-11-02"


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_practice_set() -> None:
    """Invented purpose exercises; no frame IDs."""
    practice = [
        {
            "practice_id": "P-01",
            "title": "Kernel panic after jumping from 6.8 to 6.11 on my laptop",
            "body": "After upgrading the Linux kernel from 6.8 to 6.11 the machine panics on boot. Journal shows a NULL deref in the wifi driver. Has anyone seen a known regression or a workaround?",
            "answer_a": "relevant",
            "answer_b": "TPS",
            "why": "Named version-to-version kernel upgrade with diagnostic detail and a fix request.",
        },
        {
            "practice_id": "P-02",
            "title": "This update ruined a perfectly good app",
            "body": "The latest release looks worse, feels slower, and the company clearly does not care about long-time users. I am done recommending it.",
            "answer_a": "relevant",
            "answer_b": "GDS",
            "why": "Primary purpose is unfavorable evaluation of a software change, not troubleshooting.",
        },
        {
            "practice_id": "P-03",
            "title": "Release notes for Widgetory 4.2",
            "body": "Widgetory 4.2 is out. Changelog: dark mode polish, export CSV, and dependency bumps. Download links below.",
            "answer_a": "relevant",
            "answer_b": "Other",
            "why": "Announcement/release information without troubleshooting or discontent as the main aim.",
        },
        {
            "practice_id": "P-04",
            "title": "Chrome keeps crashing on PDF files",
            "body": "No idea when it started. Opening some PDFs freezes the tab. I have not changed anything recently that I know of.",
            "answer_a": "unclear",
            "answer_b": "not_applicable",
            "why": "Crash report without an established software-update link.",
        },
        {
            "practice_id": "P-05",
            "title": "[removed]",
            "body": "[deleted]",
            "answer_a": "unassessable",
            "answer_b": "not_applicable",
            "why": "No substantive original text remains.",
        },
        {
            "practice_id": "P-06",
            "title": "Should I switch from Fedora to NixOS?",
            "body": "Thinking about leaving Fedora for NixOS for reproducibility. Curious about the migration pain.",
            "answer_a": "not_relevant",
            "answer_b": "not_applicable",
            "why": "Proposed OS switch/planning, not an established update of existing software.",
        },
        {
            "practice_id": "P-07",
            "title": "Angry but need the stacktrace fixed after 2.4.1",
            "body": "I hate this update. Since 2.4.1 the importer throws ValueError on UTF-16 files. Repro steps and traceback attached. Looking for a patch or config flag.",
            "answer_a": "relevant",
            "answer_b": "TPS",
            "why": "Dominant purpose is diagnosis/fix despite angry tone.",
        },
        {
            "practice_id": "P-08",
            "title": "Update: I found my keys",
            "body": "Earlier post was about losing house keys. Update: they were in the laundry basket.",
            "answer_a": "not_relevant",
            "answer_b": "not_applicable",
            "why": "Status edit word 'Update:' is not a software update.",
        },
        {
            "practice_id": "P-09",
            "title": "Firmware 1.0.9 for the desk lamp is disappointing",
            "body": "They pushed firmware 1.0.9 and the lamp now randomly dims. Support is useless. Product feels abandoned.",
            "answer_a": "relevant",
            "answer_b": "GDS",
            "why": "Firmware change with primary discontent; no repair objective.",
        },
        {
            "practice_id": "P-10",
            "title": "Love the new polish in Notes 3.5",
            "body": "Upgraded to Notes 3.5 today. The new outline view is excellent and sync feels snappier.",
            "answer_a": "relevant",
            "answer_b": "Other",
            "why": "Praise/positive evaluation of a version change, not TPS or GDS.",
        },
        {
            "practice_id": "P-11",
            "title": "Database UPDATE query times out",
            "body": "My SQL UPDATE on the orders table hangs after 30s. Index advice welcome.",
            "answer_a": "not_relevant",
            "answer_b": "not_applicable",
            "why": "SQL UPDATE is not a software-release update.",
        },
        {
            "practice_id": "P-12",
            "title": "Something changed after last week and settings look different",
            "body": "Menus moved around. Not sure if this was an automatic update or a remote config experiment.",
            "answer_a": "unclear",
            "answer_b": "not_applicable",
            "why": "Possible change, but software-update relationship unresolved.",
        },
        {
            "practice_id": "P-13",
            "title": "Planning the rollout of patch Tuesday in our lab",
            "body": "We will stage Windows cumulative updates to a pilot ring first. Checklist and ownership notes inside.",
            "answer_a": "relevant",
            "answer_b": "Other",
            "why": "Update-process planning/description, not troubleshooting or discontent.",
        },
        {
            "practice_id": "P-14",
            "title": "First public launch of NebulaDraw",
            "body": "NebulaDraw v1.0 is our first release. Looking for early adopters.",
            "answer_a": "not_relevant",
            "answer_b": "not_applicable",
            "why": "First-ever launch boundary, not a change to existing software.",
        },
    ]
    answers = {row["practice_id"]: {k: row[k] for k in ("answer_a", "answer_b", "why")} for row in practice}
    public_rows = [
        {k: row[k] for k in ("practice_id", "title", "body")}
        for row in practice
    ]
    practice_dir = RESTRICTED / "practice"
    practice_dir.mkdir(parents=True, exist_ok=True)
    write_json(practice_dir / "practice_cases_public.json", {"n": len(public_rows), "cases": public_rows})
    write_json(practice_dir / "practice_answer_key_private.json", {"n": len(answers), "answers": answers})
    (PUBLIC / "PRACTICE_CASES.md").write_text(
        "# Invented practice cases (decision A + B)\n\n"
        "These are fabricated. They are not in the 6,425 frame and are not validation cases.\n\n"
        + "\n".join(
            f"## {row['practice_id']}\n\n**Title:** {row['title']}\n\n{row['body']}\n\n"
            "Your A: ________  Your B: ________\n"
            for row in practice
        ),
        encoding="utf-8",
    )


def main() -> int:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    RESTRICTED.mkdir(parents=True, exist_ok=True)

    audit = build_frame_audit()
    audit_public = {k: v for k, v in audit.items() if k not in {"cells", "by_id", "fresh_ids", "covered_ids"}}
    write_json(PUBLIC / "FRAME_AUDIT.json", audit_public)
    if not audit["ok"]:
        write_json(
            PUBLIC / "BLOCKER.json",
            {"utc": utcnow(), "kind": "frame_count_discrepancy", "discrepancies": audit["discrepancies"]},
        )
        print("STOP: frame audit discrepancies:")
        for item in audit["discrepancies"]:
            print(" -", item)
        return 2

    # Freeze manifest before draw.
    manifest = {
        "utc": utcnow(),
        "run_id": RUN_ID,
        "approved_design": "160_fresh_plus_40_covered",
        "base_seed": "20261102",
        "algorithm": (
            "For each stratum, lexicographically sort Reddit IDs; "
            "seed = int.from_bytes(sha256('20261102|' + stratum_name)[:8], 'big'); "
            "random.Random(seed).sample(sorted_ids, n_h); "
            "pi=n_h/N_h; weight=N_h/n_h; "
            "presentation shuffled with sha256('20261102|presentation_order')[:8] big-endian."
        ),
        "source_hashes": audit["source_hashes"],
        "stratum_expected_N": {k: v["expected_N"] for k, v in STRATA.items()},
        "stratum_n": {k: v["n"] for k, v in STRATA.items()},
        "stratum_observed_N": audit["stratum_counts"],
        "n_fresh_frame": audit["n_fresh"],
        "n_covered_frame": audit["n_covered"],
        "protected_overlap": audit["protected_overlap"],
        "code_paths": {
            "frame_audit": "src/full_resource_sentiment_v1/validation_200/frame_audit.py",
            "sampler": "src/full_resource_sentiment_v1/validation_200/sampler.py",
            "runner": "scripts/run_validation_200_draw_2026-11-02.py",
        },
        "drawn": False,
    }
    manifest_path = RESTRICTED / "DRAW_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("drawn"):
            print("STOP: draw already executed; refusing second draw")
            print("existing selected hash:", existing.get("selected_ids_sha256"))
            return 3
    write_json(manifest_path, manifest)
    write_json(PUBLIC / "DRAW_MANIFEST_PUBLIC.json", {k: v for k, v in manifest.items() if k != "code_paths"})

    # Execute the one approved draw.
    draw = draw_validation_sample(audit["cells"])
    if draw["n_selected"] != EXPECTED_DRAW_TOTAL:
        raise SystemExit(f"selected {draw['n_selected']} != {EXPECTED_DRAW_TOTAL}")
    if draw["n_fresh"] != EXPECTED_DRAW_FRESH or draw["n_covered"] != EXPECTED_DRAW_COVERED:
        raise SystemExit(f"fresh/covered draw counts {draw['n_fresh']}/{draw['n_covered']}")

    # Forbidden overlaps.
    exposed_union = set()
    for key, n in audit["exposure_component_total"].items():
        # rebuild exposure for overlap checks via cells domains
        pass
    fresh_ids = set(audit["fresh_ids"])
    covered_ids = set(audit["covered_ids"])
    protected_union = set()
    # protected overlap already zero on frame; still assert sample
    from full_resource_sentiment_v1.validation_200.frame_audit import load_protected_ids, load_exposure_ids

    protected = load_protected_ids()
    exposure = load_exposure_ids()
    prior_exposure = set().union(*exposure.values())
    assert_no_forbidden_overlap(
        draw["selected_ids_sorted"],
        protected_ids=protected["union"] | protected["reserved"] | protected["fresh_protected"],
        fresh_component_ids={row["reddit_id"] for row in draw["selected"] if row["domain"] == "fresh"},
        prior_exposure_ids=prior_exposure,
        practice_ids=set(audit["training_ids"]),
    )
    if set(draw["selected_ids_sorted"]) & protected["union"]:
        raise SystemExit("protected union overlap after draw")

    # Temp reproducibility check.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        redo = draw_validation_sample(audit["cells"])
        if redo["selected_ids_sorted"] != draw["selected_ids_sorted"]:
            raise SystemExit("reproducibility check failed: selected IDs differ")
        if redo["selected_ids_sha256"] != draw["selected_ids_sha256"]:
            raise SystemExit("reproducibility check failed: hash differs")
        (tmp_path / "selected_ids.txt").write_text(
            "\n".join(redo["selected_ids_sorted"]) + "\n", encoding="utf-8"
        )

    selected_ids_path = RESTRICTED / "selected_ids_sorted.txt"
    selected_ids_path.write_text("\n".join(draw["selected_ids_sorted"]) + "\n", encoding="utf-8")
    selected_hash_path = RESTRICTED / "selected_ids.sha256"
    selected_hash_path.write_text(draw["selected_ids_sha256"] + "\n", encoding="utf-8")

    # Resolve full uncapped text for selected IDs only.
    needed = set(draw["selected_ids_sorted"])
    docs = build_full_text_index(needed_ids=needed)
    missing_text = sorted(needed - set(docs))
    if missing_text:
        write_json(PUBLIC / "BLOCKER.json", {"kind": "missing_original_text", "ids": missing_text})
        print("STOP: missing original text for", len(missing_text), "ids")
        return 4

    # Compare against capped model packet silently-not-used: flag if body exceeds 8000.
    private_rows = []
    blinded_rows = []
    for row in draw["selected"]:
        rid = row["reddit_id"]
        meta = audit["by_id"][rid]
        doc = docs[rid]
        packet = redact_packet_text(doc["title"], doc["body"])
        if packet["body_chars_original"] > 8000:
            packet["exceeds_model_packet_cap"] = True
        else:
            packet["exceeds_model_packet_cap"] = False
        private = {
            **row,
            **meta,
            "snapshot": doc["snapshot"],
            "original_title_sha256": packet["original_title_sha256"],
            "original_body_sha256": packet["original_body_sha256"],
            "display_title_sha256": packet["display_title_sha256"],
            "display_body_sha256": packet["display_body_sha256"],
            "redaction_counts_title": packet["redaction_counts_title"],
            "redaction_counts_body": packet["redaction_counts_body"],
            "body_chars_original": packet["body_chars_original"],
            "title_chars_original": packet["title_chars_original"],
            "exceeds_model_packet_cap": packet["exceeds_model_packet_cap"],
            # Keep original text only in restricted private key for audit; reviewer UI uses display text file.
            "original_title": packet["original_title"],
            "original_body": packet["original_body"],
        }
        private_rows.append(private)
        blinded_rows.append(
            {
                "case_code": row["case_code"],
                "presentation_order": row["presentation_order"],
                "title": packet["display_title"],
                "body": packet["display_body"],
                "title_sha256": packet["display_title_sha256"],
                "body_sha256": packet["display_body_sha256"],
                # Empty human fields for export templates.
                "decision_a": "",
                "decision_b": "",
                "confidence": "",
                "evidence_phrase": "",
                "reason": "",
                "boundary_note": "",
                "recognized_prior_case": "",
            }
        )

    # Ensure blinded rows contain no forbidden analyst fields.
    forbidden = {
        "reddit_id",
        "final_relevance",
        "final_discussion",
        "weight",
        "inclusion_probability",
        "stratum",
        "domain",
        "category",
        "model_reason",
        "decision_source",
        "sentiment",
        "compound",
        "N_h",
        "n_h",
        "snapshot",
        "original_title",
        "original_body",
    }
    for row in blinded_rows:
        leak = set(row) & forbidden
        if leak:
            raise SystemExit(f"blinded packet leak fields {leak}")

    private_path = RESTRICTED / "private_id_key.jsonl"
    with private_path.open("w", encoding="utf-8") as handle:
        for row in sorted(private_rows, key=lambda r: r["presentation_order"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    blinded_path = RESTRICTED / "blinded_cases.jsonl"
    with blinded_path.open("w", encoding="utf-8") as handle:
        for row in sorted(blinded_rows, key=lambda r: r["presentation_order"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # CSV packet for offline reading (still blinded).
    import csv

    csv_path = RESTRICTED / "blinded_review_packet.csv"
    fields = [
        "presentation_order",
        "case_code",
        "title",
        "body",
        "decision_a",
        "decision_b",
        "confidence",
        "evidence_phrase",
        "reason",
        "boundary_note",
        "recognized_prior_case",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(blinded_rows, key=lambda r: r["presentation_order"]):
            writer.writerow({k: row.get(k, "") for k in fields})

    # Reviewer sheet templates (public names, restricted content stays restricted).
    for name, reviewer in (("reviewer_sheet_template.csv", ""),):
        pass
    for reviewer in ("A", "B"):
        path = RESTRICTED / f"reviewer_{reviewer}_sheet.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "presentation_order",
                    "case_code",
                    "decision_a",
                    "decision_b",
                    "confidence",
                    "evidence_phrase",
                    "reason",
                    "boundary_note",
                    "recognized_prior_case",
                    "reviewer_id",
                    "reviewed_utc",
                ],
            )
            writer.writeheader()
            for row in sorted(blinded_rows, key=lambda r: r["presentation_order"]):
                writer.writerow(
                    {
                        "presentation_order": row["presentation_order"],
                        "case_code": row["case_code"],
                        "decision_a": "",
                        "decision_b": "",
                        "confidence": "",
                        "evidence_phrase": "",
                        "reason": "",
                        "boundary_note": "",
                        "recognized_prior_case": "",
                        "reviewer_id": reviewer,
                        "reviewed_utc": "",
                    }
                )

    write_practice_set()

    # Update manifest as drawn (immutable archive).
    manifest.update(
        {
            "drawn": True,
            "drawn_utc": utcnow(),
            "selected_ids_sha256": draw["selected_ids_sha256"],
            "n_selected": draw["n_selected"],
            "n_fresh_selected": draw["n_fresh"],
            "n_covered_selected": draw["n_covered"],
            "n_exceeding_model_cap": sum(1 for row in private_rows if row["exceeds_model_packet_cap"]),
            "private_id_key_sha256": sha256_file(private_path),
            "blinded_cases_sha256": sha256_file(blinded_path),
            "selected_ids_file_sha256": sha256_file(selected_ids_path),
        }
    )
    write_json(manifest_path, manifest)
    # Freeze a copy that should not be rewritten by later app exports.
    shutil.copy2(manifest_path, RESTRICTED / "DRAW_MANIFEST_FROZEN.json")
    shutil.copy2(selected_ids_path, RESTRICTED / "selected_ids_sorted_FROZEN.txt")
    shutil.copy2(selected_hash_path, RESTRICTED / "selected_ids_FROZEN.sha256")

    # Verification report (public: counts/hashes only).
    period_counts = Counter(
        str(audit["by_id"][rid].get("selection_stratum") or "unspecified")
        for rid in draw["selected_ids_sorted"]
    )
    commentish = {
        "n_with_body": sum(1 for row in private_rows if row["body_chars_original"] > 0),
        "n_empty_body": sum(1 for row in private_rows if row["body_chars_original"] == 0),
        "n_empty_title": sum(1 for row in private_rows if row["title_chars_original"] == 0),
    }
    report = f"""# Verification report — validation 200 draw (`{RUN_ID}`)

**UTC:** {utcnow()}  
**Status:** Sample drawn once. No human decisions entered. Predictions not unblinded.

## Frame audit
- Frame N: **{audit['n_frame']}**
- Fresh / covered: **{audit['n_fresh']} / {audit['n_covered']}**
- Protected overlap: **{audit['protected_overlap']}**
- Category counts: `{json.dumps(audit['category_counts'], sort_keys=True)}`
- All ten stratum N matched expected values: **yes**

## Draw
- Selected: **{draw['n_selected']}** (fresh {draw['n_fresh']}, covered {draw['n_covered']})
- Selected IDs SHA-256: `{draw['selected_ids_sha256']}`
- Temp reproducibility re-draw: **exact match**
- Second draw refused once manifest marked drawn: enforced by runner

## Sample composition (not prevalence estimates)
- Selection periods (`selection_stratum` on frame): `{dict(period_counts)}`
- Text availability: `{commentish}`
- Bodies longer than the old 8,000-char model packet cap: **{manifest['n_exceeding_model_cap']}** (full text retained for human review)

## Blinded packet checks
- Blinded case file: `restricted/validation_200_2026-11-02/blinded_cases.jsonl`
- Private ID key (weights, Reddit IDs, labels): `restricted/validation_200_2026-11-02/private_id_key.jsonl`
- No Reddit IDs / weights / predictions in blinded cases: **checked**
- Protected IDs in sample: **0**

## Review interface
Start the localhost app:

```bash
cd phase6/full_resource_sentiment_v1
.venv/bin/python scripts/validation_200_review_app.py
```

Then open: **http://127.0.0.1:8765/**

Autosave journal/DB: `restricted/validation_200_2026-11-02/review_state.sqlite`  
Exports: `restricted/validation_200_2026-11-02/exports/`

## Stopped before
Human labeling, unblinding, weighted metrics on real labels, sentiment analyses, manuscript writing, paid API calls, and any changes to frozen classifications.
"""
    (PUBLIC / "VERIFICATION_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        PUBLIC / "OUTPUT_HASHES.json",
        {
            "utc": utcnow(),
            "selected_ids_sha256": draw["selected_ids_sha256"],
            "private_id_key_sha256": sha256_file(private_path),
            "blinded_cases_sha256": sha256_file(blinded_path),
            "frame_audit_sha256": sha256_file(PUBLIC / "FRAME_AUDIT.json"),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )

    (PUBLIC / "RUN.md").write_text(
        """# Validation 200 — commands

## Synthetic tests
```bash
cd phase6/full_resource_sentiment_v1
.venv/bin/python -m pytest tests/test_validation_200_synthetic.py -q
```

## Frame audit + single draw (already executed if DRAW_MANIFEST says drawn=true)
```bash
.venv/bin/python scripts/run_validation_200_draw_2026-11-02.py
```

## Review app
```bash
.venv/bin/python scripts/validation_200_review_app.py
```
Open http://127.0.0.1:8765/

Do not open `restricted/validation_200_2026-11-02/private_id_key.jsonl` in the reviewer browser session.
""",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "selected": draw["n_selected"],
                "fresh": draw["n_fresh"],
                "covered": draw["n_covered"],
                "selected_ids_sha256": draw["selected_ids_sha256"],
                "review_app": "scripts/validation_200_review_app.py",
                "interface": "http://127.0.0.1:8765/",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
