# Workstream 1 reproducibility

All commands are run from:

```
/Users/manumathewjiss/Documents/Research-on-Software-Updates/research-open-text-sentimental/phase6/full_resource_sentiment_v1
```

Original data, annotations, historical scripts, saved C1-prime tables, models, and the vendor package source are read-only. Outputs stay in `results/`.

## Environment

Python 3.14.7. Create a local virtualenv and install the pinned archive of `releasetrain-sentiment==0.2.1` (do not install a different PyPI build):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install ../../phase2/new_sources/package/releasetrain_sentiment-0.2.1.tar.gz
```

Expected tarball SHA-256:

```
251b36225f53772bb5cdd3bd126e9142b00bd89f228f175be89c53e82ed94b77
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

This execution after 01B: **35 passed, 0 failed** (26 from 01 plus 9 joined-record tests). VADER `codecs.open` deprecation warnings on Python 3.14 are from the upstream package, not this workspace.

## Inventory, adapter check inputs, and 542 comparison

```bash
.venv/bin/python scripts/run_workstream1.py
```

Writes:

- `results/source_manifest.json`
- `results/overlap_matrix.json`
- `results/normalized_schema.json`
- `results/hand_checked_cases.json`
- `results/historical_542_original_vs_package.jsonl`
- `results/historical_542_summary.json`
- `results/environment.json`

The script stops with exit code 2 if unique-ID or overlap counts disagree with `EXPERIMENT_SPEC.md`. It does not fetch live API data, does not score reserved/fresh tests, and does not call `build_c1_prime()` (which would write into `data/c1_prime`). Do not rerun this script for the 01B record-builder correction.

## Joined normalized records (01B)

```bash
.venv/bin/python scripts/run_normalized_records.py
```

Writes:

- `results/normalized_records_compact.jsonl` (one joined record per unique Reddit ID; hashes/flags only)
- `results/normalized_records_audit.json`
- `results/normalized_schema.json` (refreshed documentation)

Codex can call `iter_normalized_records(loaded)` or `build_normalized_record(loaded, reddit_id)` from `full_resource_sentiment_v1.records`. `choose_analysis_snapshot(record, "historical_reproduction" | "master_source_conversations" | "engagement_selected_cohort")` selects the documented snapshot without a new join. The script does not fetch live API data, score protected tests, or rewrite the 542 comparison.

## Public vs local text

Public result files store IDs, hashes, flags, and metrics. They do not store usernames or full title/body/comment text. Tests read raw snapshots in memory from the original files. Optional local-only dumps, if ever created, belong in `local_normalized/` (gitignored).
