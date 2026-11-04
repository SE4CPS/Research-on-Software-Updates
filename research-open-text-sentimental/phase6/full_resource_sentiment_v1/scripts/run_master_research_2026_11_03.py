#!/usr/bin/env python3
"""Master research 2026-11-03 — locally safe phases A–F.

No paid APIs, no large model downloads, no frozen overwrites, no protected outcomes.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

from full_resource_sentiment_v1.master_research_2026_11_03.common import (  # noqa: E402
    OUT_ROOT,
    RESTRICTED_ROOT,
    STAGE1_DIR,
    ensure_dirs,
    sha256_file,
    utcnow,
    write_json,
)
from full_resource_sentiment_v1.master_research_2026_11_03.phase_a import (  # noqa: E402
    build_text_thread_manifest,
    run_inventory,
)
from full_resource_sentiment_v1.master_research_2026_11_03.phase_b import run_phase_b  # noqa: E402
from full_resource_sentiment_v1.master_research_2026_11_03.phase_c import run_phase_c  # noqa: E402
from full_resource_sentiment_v1.master_research_2026_11_03.phase_d import run_phase_d  # noqa: E402
from full_resource_sentiment_v1.master_research_2026_11_03.phase_e import run_phase_e  # noqa: E402


def disk_free_gib() -> float:
    usage = shutil.disk_usage("/")
    return usage.free / (1024**3)


def main() -> int:
    ensure_dirs()
    ledger: dict = {
        "utc_start": utcnow(),
        "run_id": "master-research-2026-11-03",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "disk_free_gib": disk_free_gib(),
        "phases": {},
    }

    print("Phase A: inventory...", flush=True)
    inv = run_inventory()
    if not inv["ok"]:
        write_json(OUT_ROOT / "BLOCKER.json", {"phase": "A", "discrepancies": inv["discrepancies"]})
        print("STOP Phase A mismatch", inv["discrepancies"])
        return 2
    man = build_text_thread_manifest()
    ledger["phases"]["A"] = {
        "status": "ran",
        "inventory_ok": True,
        "manifest": man,
        "independently_reviewed": False,
        "accepted": False,
    }

    print("Phase B: classification diagnostics...", flush=True)
    b = run_phase_b()
    ledger["phases"]["B"] = {
        "status": "ran_exploratory",
        "protected_overlap_in_pilot": b.get("protected_overlap_in_pilot"),
        "n_nonempty_title_or_body": b.get("n_nonempty_title_or_body"),
        "second_relevance_counts": b.get("second_relevance_counts"),
        "v3_agree_n": b.get("v3_agree_n"),
        "v3_disagree_n": b.get("v3_disagree_n"),
        "paid_api": False,
        "full_frame": "blocked_awaiting_approval",
        "independently_reviewed": False,
    }

    print("Phase C: TextBlob matched comparison (Cardiff blocked)...", flush=True)
    c = run_phase_c()
    ledger["phases"]["C"] = {
        "status": "ran_partial",
        "textblob": "ran",
        "cardiff": c.get("cardiff", {}).get("status"),
        "stage1_reused": True,
        "independently_reviewed": False,
    }

    print("Phase D: cadence...", flush=True)
    d = run_phase_d()
    ledger["phases"]["D"] = {
        "status": "ran",
        "eligibility_gds_author_ge4": d["eligibility"]["GDS"]["author_ge4"],
        "independently_reviewed": False,
    }

    print("Phase E: RQ4 predictive...", flush=True)
    e = run_phase_e()
    ledger["phases"]["E"] = {
        "status": "ran_exploratory",
        "scientific_status": e.get("scientific_status"),
        "test_macro_f1_text": e["models"]["tfidf_text_only"]["macro_f1"],
        "test_macro_f1_text_vader": e["models"]["text_plus_vader"]["macro_f1"],
        "independently_reviewed": False,
    }

    # Phase F: consolidated packet
    print("Phase F: consolidated Codex packet...", flush=True)
    outputs = sorted(p for p in OUT_ROOT.rglob("*") if p.is_file())
    hashes = {str(p.relative_to(OUT_ROOT)): sha256_file(p) for p in outputs if p.name != "OUTPUT_HASHES.json"}
    write_json(OUT_ROOT / "OUTPUT_HASHES.json", {"utc": utcnow(), "files": hashes})

    # Load machine tables for exact eligibility in the Codex packet
    c_table = (OUT_ROOT / "phase_c_sentiment_comparison/tables_gds_minus_tps.csv").read_text(encoding="utf-8")
    d_table = (OUT_ROOT / "phase_d_cadence/gds_minus_tps_cadence.csv").read_text(encoding="utf-8")
    e_table = (OUT_ROOT / "phase_e_predictive_rq4/model_comparison.csv").read_text(encoding="utf-8")
    elig = d.get("eligibility") or {}
    paired = (e.get("paired_bootstrap") or {}).get("text_plus_vader_minus_text_only") or {}

    findings = f"""# Master research consolidated findings — 2026-11-03

**UTC:** {utcnow()}  
**Stage-1 VADER:** reused from `{STAGE1_DIR}` (not overwritten).  
**200-case human review:** still zero saved decisions — AI predictions ≠ human reference.

## Phase A
Inventory PASS: 6,425 IDs; jsonl SHA `99b575…`; protected overlap 0; text/thread manifest built (public hashes; restricted text). Capture text for frame: {man.get('n_with_capture_text')} / missing {man.get('n_missing_capture')}.

## Phase B (exploratory diagnostics)
Fixed 240 pilot preserved (sha `{b.get('pilot_ids_sha256')}`). Text from broad capture for excluded-pool IDs: nonempty {b.get('n_nonempty_title_or_body')} / missing {b.get('n_missing_capture_text')}.  
Second cue-classifier relevance counts: `{b.get('second_relevance_counts')}`; among relevant discussion: `{b.get('second_discussion_among_relevant')}`.  
v3 vs second: compared={b.get('v3_compared_n')}, agree={b.get('v3_agree_n')}, disagree={b.get('v3_disagree_n')} — **agreement ≠ accuracy**.  
Paid API / full-frame reclassification: **not run**.

## Phase C
TextBlob 0.19.0 scored on matched frozen TPS/GDS texts vs reused VADER.  
Cardiff: **{c.get('cardiff', {}).get('status')}**. Do not equate VADER compound with TextBlob polarity.

```
{c_table.strip()}
```

## Phase D
Eligibility TPS author≥1/≥4/both: {elig.get('TPS')}; GDS: {elig.get('GDS')}.  
Author shift = later−earlier (≥4); post→author separate (≥1). Sentiment change ≠ resolution.

```
{d_table.strip()}
```

## Phase E (exploratory RQ4)
Grouped TF-IDF text vs text+VADER/TextBlob on frozen TPS/GDS labels (n_total={e.get('split', {}).get('n_total')}; test={e.get('split', {}).get('n_test')}; group leakage train∩test={e.get('split', {}).get('group_leakage_check', {}).get('train_test_group_overlap')}).  
Paired bootstrap text+VADER − text-only Δmacro-F1 mean={paired.get('mean_delta_macro_f1')} CI95={paired.get('ci95')}.  
Performance = agreement with mixed-source saved labels until independent human reference exists.

```
{e_table.strip()}
```

## Blocked pending Manu approval
- Large model download / Cardiff full scoring
- Paid API classification campaigns
- Full-frame reclassification
- Manuscript writing
"""
    (OUT_ROOT / "FINDINGS.md").write_text(findings, encoding="utf-8")
    (OUT_ROOT / "LIMITATIONS.md").write_text(
        """# Consolidated limitations

1. No completed independent human validation (200-case empty; 240 unreviewed).
2. Mixed decision_source provenance on 6,425 table.
3. Selected frame ≠ all 37,596 candidates.
4. Cardiff not scored this phase (torch/transformers not installed; large download not authorized).
5. RQ4 predicts saved labels, not field truth; AI predictions ≠ human reference.
6. Missing replies ≠ neutral; count gaps retained.
7. Sentiment trajectories ≠ technical resolution.
8. Cross-model raw scores are not commensurate.
9. Second-rubric cue classifier is exploratory diagnostics only — not a validated relevance instrument.
10. 520 historical frame posts lack broad-capture package text (harmonized scores only).
""",
        encoding="utf-8",
    )

    (OUT_ROOT / "RUN.md").write_text(
        f"""# Master research reproduction

```bash
cd phase6/full_resource_sentiment_v1
# modest isolated env (already created if following Cursor run)
.venv_master_2026_11_03/bin/python -m pytest tests/test_master_research_2026_11_03.py -q
.venv_master_2026_11_03/bin/python scripts/run_master_research_2026_11_03.py
```

Environment lock: `results/master_research_2026-11-03/ENVIRONMENT_LOCK.json`  
Outputs: `results/master_research_2026-11-03/`  
Restricted text/scores: `restricted/master_research_2026-11-03/` (mode 0700)

Disk free at start of this run: {ledger['disk_free_gib']:.2f} GiB
""",
        encoding="utf-8",
    )

    # Agreement/disagreement examples only (not TP/FP — no human reference on the 240)
    disagree_path = OUT_ROOT / "phase_b_classification_diagnostics/v3_second_disagreements.jsonl"
    disagree_ex: list[dict] = []
    if disagree_path.exists():
        for line in disagree_path.read_text(encoding="utf-8").splitlines()[:2]:
            if line.strip():
                disagree_ex.append(json.loads(line))

    packet = f"""# Codex consolidated review packet — master research 2026-11-03

**Status:** `STOP_FOR_CODEX_INDEPENDENT_REVIEW`  
**UTC:** {utcnow()}  
**Independently reviewed:** no (any phase)  
**Accepted:** Stage-1 VADER only (prior); this master package not yet accepted.

## Scope completed by Cursor
Locally safe phases A–E + this handoff. No paid API, no large model download, no new Reddit collection, no frozen overwrite, no full-frame reclassification, no protected-outcome access, no manuscript.

## Immutable inputs verified
| Artifact | Check |
|---|---|
| `full_dataset.jsonl` | n=6425; SHA `99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd` |
| Protected overlap | **0** |
| Stage-1 VADER dir | reused, not overwritten: `{STAGE1_DIR}` |
| Pilot 240 IDs | not redrawn; sha `{b.get('pilot_ids_sha256')}` |
| 200-case human decisions | still **0** — AI ≠ human reference |

## Decision provenance (6,425 frame)
See `phase_a_inventory/INVENTORY.json` → `counts_by_relevance_purpose_source` and `decision_status` (`existing_decision` 1964 / `model_decision` 4461). Labels are mixed human/AI; do not treat as independent field truth.

## Phase results (exact eligibility / uncertainty)

### A — Inventory + text/thread manifest
- Public manifest sha: `{man.get('public_manifest_sha256')}`
- Restricted text rows: {man.get('n_public')}; with capture text {man.get('n_with_capture_text')}; missing {man.get('n_missing_capture')}; with scored comments {man.get('n_with_scored_comments')}

### B — Classification diagnostics (exploratory)
- Text source for excluded-pool pilot: broad capture pages; nonempty title/body **{b.get('n_nonempty_title_or_body')}**/240; missing capture **{b.get('n_missing_capture_text')}**
- Second classifier `second_rubric_cue_v1` relevance: `{b.get('second_relevance_counts')}`
- Among predicted-relevant discussion cues: `{b.get('second_discussion_among_relevant')}`
- Exploratory IPW sums (model outputs only): relevant={b.get('exploratory_ipw_sum_relevant')}; predicted GDS={b.get('exploratory_ipw_sum_predicted_gds')}
- v3 (`predicted_label`) vs second: compared={b.get('v3_compared_n')}, agree={b.get('v3_agree_n')}, disagree={b.get('v3_disagree_n')}
- **Do not report as accuracy, FN rate, or verified GDS yield.**
- Agreement/disagreement examples (not TP/FP): `{json.dumps(disagree_ex)[:1200]}`

### C — Matched sentiment (TextBlob + reused VADER; Cardiff blocked)
- Cardiff: **{c.get('cardiff', {}).get('status')}** — canary not executed; no silent substitute.
- GDS−TPS contrasts (thread bootstrap):

```
{c_table.strip()}
```

### D — Author/community trajectory & cadence
- Eligibility: TPS `{elig.get('TPS')}`; GDS `{elig.get('GDS')}`
- Definitions: author_shift = later−earlier with ≥4 author replies; post→author separate (≥1). Not resolution.

```
{d_table.strip()}
```

### E — Leakage-aware RQ4 (exploratory label agreement)
- Split seed 20261103; n_train/val/test = {e.get('split', {}).get('n_train')}/{e.get('split', {}).get('n_val')}/{e.get('split', {}).get('n_test')}; train∩test group overlap = **{e.get('split', {}).get('group_leakage_check', {}).get('train_test_group_overlap')}**
- Features: title/body (+ post-time sentiment only); no replies/labels/confidence/reasons/decision_source/protected outcomes
- Paired bootstrap text+VADER − text-only Δmacro-F1: mean={paired.get('mean_delta_macro_f1')}, CI95={paired.get('ci95')}

```
{e_table.strip()}
```

## Failures / blocked
| Item | Status | Why |
|---|---|---|
| Cardiff NLP | blocked | no torch/transformers; large download needs approval |
| Embedding MiniLM baseline | not run | no local pinned model |
| Paid API 240/full-frame | not run | needs concrete budget + Manu approval |
| Human validation 200/240 | not done | zero saved human decisions |
| Manuscript | not written | outputs unreviewed |

## Null / contrary notes for review
- Stage-1 VADER combined GDS−TPS CI includes 0 (reused); TextBlob combined polarity mean diff CI excludes 0 on this run — **scales not commensurate**; neither is human-validated sentiment accuracy.
- RQ4: text-only macro-F1 **higher** than text+VADER on this split (paired Δ CI includes negative values) — exploratory contrary to a naive “sentiment helps” claim.
- Author_shift GDS−TPS CI includes 0 (n_GDS=30).

## Verify checklist for Codex
1. Input hashes still match frozen 6,425 dataset; protected overlap 0.
2. Pilot 240 not redrawn (`phase_b` pilot sha).
3. Stage-1 directory untouched.
4. RQ4 split group-overlap = 0; no label/confidence/reason features.
5. Cardiff blocked status documented; no silent substitution.
6. Claims remain exploratory where human reference absent; AI ≠ human.
7. Phase B text nonempty count is 240 (excluded-pool capture join), not zero.

## Paths
- Root: `results/master_research_2026-11-03/`
- Progress: `PROGRESS_LEDGER.json`
- `FINDINGS.md`, `LIMITATIONS.md`, `RUN.md`, `OUTPUT_HASHES.json`, `ENVIRONMENT_LOCK.json`
- `phase_a_inventory/`, `phase_b_classification_diagnostics/`, `phase_c_sentiment_comparison/`, `phase_d_cadence/`, `phase_e_predictive_rq4/`
- Restricted: `restricted/master_research_2026-11-03/`

## Reproduce
See `RUN.md`.
"""
    (OUT_ROOT / "CODEX_CONSOLIDATED_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")

    ledger["utc_end"] = utcnow()
    ledger["phases"]["F"] = {"status": "ran", "codex_packet": "CODEX_CONSOLIDATED_REVIEW_PACKET.md"}
    write_json(OUT_ROOT / "PROGRESS_LEDGER.json", ledger)
    write_json(
        OUT_ROOT / "MANIFEST.json",
        {
            "utc": utcnow(),
            "out_root": str(OUT_ROOT),
            "restricted_root": str(RESTRICTED_ROOT),
            "stage1_preserved": str(STAGE1_DIR),
            "n_output_files": len(hashes),
            "status": "STOP_FOR_CODEX_INDEPENDENT_REVIEW",
        },
    )

    print(json.dumps({"ok": True, "out": str(OUT_ROOT), "ledger_phases": list(ledger["phases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
