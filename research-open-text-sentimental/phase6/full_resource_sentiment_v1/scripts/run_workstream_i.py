"""Workstream I robustness checks. Does not modify G outputs or mark findings accepted."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from full_resource_sentiment_v1.broad_sentiment import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    git_revision,
    utcnow,
)
from full_resource_sentiment_v1.hashing import sha256_file  # noqa: E402
from full_resource_sentiment_v1.io_utils import read_json  # noqa: E402
from full_resource_sentiment_v1.paths import (  # noqa: E402
    BROAD_CAPTURE_DIR,
    BROAD_SENTIMENT_RUN_DIR,
    LOCAL_COMPACT_RECORDS,
    WORKSPACE,
)
from full_resource_sentiment_v1.privacy import scan_paths  # noqa: E402
from full_resource_sentiment_v1.robustness_i_cadence import analysis_i_c, analysis_i_d, analysis_i_e  # noqa: E402
from full_resource_sentiment_v1.robustness_i_common import (  # noqa: E402
    CAPTURE_ID,
    EXPECTED_N,
    G_OUTPUT_NAMES,
    I_N_BOOT,
    I_RESTRICTED_DIR,
    I_RUN_DIR,
    I_SEED,
    assert_g_guard,
    collect_text_measures,
    hash_g_outputs,
    load_comments_by_id,
    load_jsonl_by_id,
    package_and_code_meta,
    recompute_g_points,
    verify_f_membership,
)
from full_resource_sentiment_v1.robustness_i_length import analysis_i_a, analysis_i_b, analysis_i_f  # noqa: E402


def write_docs(run_dir: Path, manifest: dict, analyses: dict[str, dict], *, workspace_copies: bool) -> None:
    a = analyses["I_A"]
    b = analyses["I_B"]
    f = analyses["I_F"]
    c = analyses["I_C"]
    d = analyses["I_D"]
    e = analyses["I_E"]
    run_md = f"""# Workstream I robustness run

Capture: `{manifest['capture_id']}`
Status: **Awaiting Codex review** (not self-validated)

All commands are run from:

```
{WORKSPACE}
```

## Command

```bash
.venv/bin/python -m pytest -q tests/test_workstream_i.py tests/test_package_pin.py
.venv/bin/python scripts/run_workstream_i.py
```

Elapsed seconds: {manifest['elapsed_seconds']:.1f}
Package: releasetrain-sentiment=={manifest['package_version']}
Seed: {manifest['seed']}; thread bootstrap resamples: {manifest['n_bootstrap']}
G outputs modified: {manifest['g_outputs_modified']}

## Outputs

Directory: `{run_dir}`

- I_A_length.json, I_B_length.json, I_F_length.json, I_F_length_bins.jsonl
- I_C_roles.json, I_D_order.json, I_E_elapsed.json
- manifest.json, hashes.json
"""
    prefix = a["prefix_diagnostic"]
    bprefix = b["prefix_diagnostic"]
    report = f"""# Workstream I execution report

Date: {manifest['run_finished_utc'][:10]}. Executor: Cursor. Status for Codex: **Awaiting Codex review**. This report does not mark Workstream I completed or findings accepted.

Workspace: `{WORKSPACE}`
Run directory: `{run_dir}`

## What was executed

Prespecified robustness checks I-A, I-B, I-F, I-C and I-D/E on the frozen 37,596-post Reddit-URL frame. Workstream G score tables were read-only. Package 0.2.1 was used only for equal-word-prefix diagnostics. Count-gap posts were retained. Protected reserved/fresh outcomes were not read. No manuscript writing or model training.

## Commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_workstream_i.py
```

Elapsed: {manifest['elapsed_seconds']:.1f} seconds.

## Frozen G guard

Recomputed G point estimates from immutable score tables matched the handoff targets within 1e-8. Primary Ns: posts {EXPECTED_N['posts']}, comments {EXPECTED_N['comments']}, A {EXPECTED_N['A']}, B {EXPECTED_N['B']}, C {EXPECTED_N['C']}, D {EXPECTED_N['D']}, E {EXPECTED_N['E']}, F {EXPECTED_N['F']} ({EXPECTED_N['F_troubleshooting']}/{EXPECTED_N['F_other']}). Count-gap Reddit-URL posts retained: {EXPECTED_N['count_gap']}.

## Result summary (descriptive only)

- I-A full-text reference mean {a['g_reference'].get('mean')}; prefix n={prefix['n_k_ge_3']} fail_k={prefix['n_fail_k_lt_3']}; prefix mean {prefix['prefix'].get('mean')} CI {prefix['prefix'].get('mean_ci95')}; same-subset full-text mean {prefix['full_text_on_same_subset'].get('mean')}.
- I-B mean-of-replies reference {b['g_reference'].get('mean')}; first-reply prefix n={bprefix['n_k_ge_3']} fail_k={bprefix['n_fail_k_lt_3']}; prefix mean {bprefix['prefix'].get('mean')}; first-reply full-text on same subset {bprefix['full_first_reply_on_same_subset'].get('mean')}.
- I-F unadjusted mean_diff {f['unadjusted_combined_post'].get('mean_diff')}; pooled length-adjusted reported={f['length_bins']['pooled_adjusted_estimate_reported']}; support_failure={f['length_bins']['support_failure']}; OLS s2_target {f['ols_s2_target'].get('s2_target_coefficient')} CI {f['ols_s2_target'].get('s2_target_ci95')}.
- I-C overall mean {c['overall'].get('mean')} CI {c['overall'].get('mean_ci95')}; N={c['eligible_n']}.
- I-D overall mean {d['overall'].get('mean')}; share |last-first|>=0.10 {d['overall'].get('share_abs_ge_0.10')}; >=0.30 {d['overall'].get('share_abs_ge_0.30')}.
- I-E Spearman {e['overall_spearman'].get('spearman')} CI {e['overall_spearman'].get('spearman_ci95')}; D-E N difference {e.get('n_d_minus_n_e')}; timestamp coverage {e['timestamp_coverage'].get('total_coverage')}.

These are not publication claims. Codex must independently validate before any scientific classification.

## Integrity

- G output hashes before/after: unchanged={not manifest['g_outputs_modified']}
- Output SHA-256: `hashes.json`
- Privacy scan hits: {manifest['privacy_scan_hits']}
- Git revision: `{manifest.get('git_revision')}`
- F independent development join: {manifest['f_membership']}
- Sparse/suppressed results: F pooled adjusted reported={f['length_bins']['pooled_adjusted_estimate_reported']}

## Limitations

A/B prefix diagnostics are truncated-text estimands. F remains a selected 1,864 reviewed overlap, not a broad-frame group effect. D and E are not independent confirmations. Count-gap does not imply recoverable missing replies. No causal language.
"""
    atomic_write_text(run_dir / "RUN.md", run_md)
    atomic_write_text(run_dir / "EXECUTION_REPORT.md", report)
    if workspace_copies:
        atomic_write_text(WORKSPACE / "WORKSTREAM_I_RUN.md", run_md)
        atomic_write_text(WORKSPACE / "WORKSTREAM_I_EXECUTION_REPORT.md", report)


def run_workstream_i(
    *,
    capture_dir: Path = BROAD_CAPTURE_DIR,
    g_run_dir: Path = BROAD_SENTIMENT_RUN_DIR,
    local_path: Path = LOCAL_COMPACT_RECORDS,
    run_dir: Path = I_RUN_DIR,
    write_workspace_copies: bool = True,
    skip_text_pass: bool = False,
    measures: dict | None = None,
) -> dict:
    started = time.monotonic()
    started_utc = utcnow()
    g_hashes_before = hash_g_outputs(g_run_dir)
    if version("releasetrain-sentiment") != "0.2.1":
        raise ValueError("package version is not 0.2.1")
    posts = load_jsonl_by_id(g_run_dir / "post_scores.jsonl")
    threads = load_jsonl_by_id(g_run_dir / "thread_metrics.jsonl")
    if len(posts) != EXPECTED_N["posts"] or len(threads) != EXPECTED_N["posts"]:
        raise ValueError("primary post/thread counts drifted")
    comments_by_id = load_comments_by_id(g_run_dir / "comment_scores.jsonl")
    n_gap = sum(1 for row in posts.values() if row.get("count_gap"))
    if n_gap != EXPECTED_N["count_gap"]:
        raise ValueError(f"count-gap posts {n_gap} != {EXPECTED_N['count_gap']}")
    points = recompute_g_points(posts, threads)
    assert_g_guard(points)
    f_membership = verify_f_membership(threads, local_path)
    g_a = read_json(g_run_dir / "analysis_A.json")
    g_b = read_json(g_run_dir / "analysis_B.json")
    g_c = read_json(g_run_dir / "analysis_C.json")
    g_d = read_json(g_run_dir / "analysis_D.json")
    g_e = read_json(g_run_dir / "analysis_E.json")
    g_f = read_json(g_run_dir / "analysis_F.json")
    if measures is None and not skip_text_pass:
        need = {
            rid
            for rid, post in posts.items()
            if post.get("title_compound") is not None and post.get("body_compound") is not None
        }
        need.update(rid for rid, thread in threads.items() if thread.get("author_reply_eligible"))
        measures = collect_text_measures(
            capture_dir=capture_dir,
            threads=threads,
            comments_by_id=comments_by_id,
            need_prefix_ids=need,
        )
    if measures is None:
        raise ValueError("text measures are required")
    i_a = analysis_i_a(posts, threads, measures, g_a)
    i_b = analysis_i_b(threads, comments_by_id, measures, g_b)
    i_f, bin_rows = analysis_i_f(threads, measures, g_f)
    i_c = analysis_i_c(threads, g_c)
    i_d = analysis_i_d(threads, comments_by_id, g_d)
    i_e = analysis_i_e(threads, comments_by_id, g_e)
    analyses = {"I_A": i_a, "I_B": i_b, "I_F": i_f, "I_C": i_c, "I_D": i_d, "I_E": i_e}
    run_dir.mkdir(parents=True, exist_ok=True)
    I_RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(I_RESTRICTED_DIR, 0o700)
    prefix_rows = []
    for rid, measure in measures.items():
        prefix_rows.append(
            {
                "reddit_id": rid,
                "title_words": measure.get("title_words"),
                "body_words": measure.get("body_words"),
                "original_post_words": measure.get("original_post_words"),
                "original_post_chars": measure.get("original_post_chars"),
                "k_title_body": measure.get("k_title_body"),
                "title_prefix_compound": measure.get("title_prefix_compound"),
                "body_prefix_compound": measure.get("body_prefix_compound"),
                "k_post_first_reply": measure.get("k_post_first_reply"),
                "post_prefix_compound": measure.get("post_prefix_compound"),
                "first_reply_prefix_compound": measure.get("first_reply_prefix_compound"),
                "mean_author_reply_words": measure.get("mean_author_reply_words"),
                "first_author_reply_words": measure.get("first_author_reply_words"),
            }
        )
    prefix_rows.sort(key=lambda row: row["reddit_id"])
    atomic_write_jsonl(I_RESTRICTED_DIR / "prefix_diagnostics.jsonl", prefix_rows)
    atomic_write_json(run_dir / "I_A_length.json", i_a)
    atomic_write_json(run_dir / "I_B_length.json", i_b)
    atomic_write_json(run_dir / "I_F_length.json", i_f)
    atomic_write_jsonl(run_dir / "I_F_length_bins.jsonl", bin_rows)
    atomic_write_json(run_dir / "I_C_roles.json", i_c)
    atomic_write_json(run_dir / "I_D_order.json", i_d)
    atomic_write_json(run_dir / "I_E_elapsed.json", i_e)
    output_names = [
        "I_A_length.json",
        "I_B_length.json",
        "I_F_length.json",
        "I_F_length_bins.jsonl",
        "I_C_roles.json",
        "I_D_order.json",
        "I_E_elapsed.json",
    ]
    output_hashes = {name: sha256_file(run_dir / name) for name in output_names}
    privacy_hits = scan_paths([run_dir / name for name in output_names])
    if privacy_hits:
        raise ValueError(f"public I outputs contain forbidden fields: {privacy_hits[:20]}")
    g_hashes_after = hash_g_outputs(g_run_dir)
    if g_hashes_after != g_hashes_before:
        raise ValueError("Workstream G files changed during I")
    elapsed = time.monotonic() - started
    meta = package_and_code_meta()
    manifest = {
        "status": "awaiting_codex_review",
        "complete": True,
        "capture_id": CAPTURE_ID,
        "run_started_utc": started_utc,
        "run_finished_utc": utcnow(),
        "elapsed_seconds": elapsed,
        "seed": I_SEED,
        "n_bootstrap": I_N_BOOT,
        "package_version": meta["package_version"],
        "vader_version": meta["vader_version"],
        "package_tarball_sha256": meta["package_tarball_sha256"],
        "code_fingerprint": meta["code_fingerprint"],
        "git_revision": git_revision(),
        "g_input_hashes": g_hashes_before,
        "output_sha256": output_hashes,
        "privacy_scan_hits": privacy_hits,
        "g_outputs_modified": False,
        "live_api_called": False,
        "protected_outcomes_read": False,
        "validated_by_cursor": False,
        "g_guard_points": points,
        "f_membership": f_membership,
        "denominators": EXPECTED_N,
        "restricted_prefix_diagnostics": str(I_RESTRICTED_DIR / "prefix_diagnostics.jsonl"),
        "notes": [
            "Workstream I robustness only. Status is Awaiting Codex review.",
            "Prefix diagnostics are truncated-text estimands, not corrections of full-text sentiment.",
            "F cannot estimate broad troubleshooting prevalence.",
        ],
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "hashes.json", output_hashes)
    write_docs(run_dir, manifest, analyses, workspace_copies=write_workspace_copies)
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir), "elapsed_seconds": elapsed}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Workstream I robustness checks")
    parser.add_argument("--capture-dir", type=Path, default=BROAD_CAPTURE_DIR)
    parser.add_argument("--g-run-dir", type=Path, default=BROAD_SENTIMENT_RUN_DIR)
    parser.add_argument("--local-records", type=Path, default=LOCAL_COMPACT_RECORDS)
    parser.add_argument("--run-dir", type=Path, default=I_RUN_DIR)
    parser.add_argument("--no-workspace-copies", action="store_true")
    args = parser.parse_args()
    run_workstream_i(
        capture_dir=args.capture_dir,
        g_run_dir=args.g_run_dir,
        local_path=args.local_records,
        run_dir=args.run_dir,
        write_workspace_copies=not args.no_workspace_copies,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
