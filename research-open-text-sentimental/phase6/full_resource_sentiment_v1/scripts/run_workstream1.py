#!/usr/bin/env python3
"""Workstream 1: inventory, adapter validation inputs, and historical 542 comparison.

Does not fetch live API data, does not score protected tests, and does not
write into original data directories.
"""

from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKSPACE.parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(REPO_ROOT))

from full_resource_sentiment_v1.compare_historical import run_historical_542  # noqa: E402
from full_resource_sentiment_v1.inventory import (  # noqa: E402
    build_overlap_matrix,
    load_all_sources,
    write_inventory_outputs,
)
from full_resource_sentiment_v1.io_utils import write_json  # noqa: E402
from full_resource_sentiment_v1.paths import PACKAGE_TARBALL, RESULTS_DIR  # noqa: E402
from full_resource_sentiment_v1.hashing import sha256_file  # noqa: E402


def environment_record() -> dict:
    import releasetrain_sentiment
    import vaderSentiment

    return {
        "python": sys.version,
        "executable": sys.executable,
        "releasetrain_sentiment": version("releasetrain-sentiment"),
        "releasetrain_sentiment_file": releasetrain_sentiment.__file__,
        "vaderSentiment": version("vaderSentiment"),
        "vaderSentiment_file": vaderSentiment.__file__,
        "pytest": version("pytest"),
        "pandas": version("pandas"),
        "numpy": version("numpy"),
        "package_tarball_path": str(PACKAGE_TARBALL),
        "package_tarball_sha256": sha256_file(PACKAGE_TARBALL),
        "package_source_modified": False,
        "protected_tests_scored": False,
        "broad_api_fetched": False,
    }


def main() -> int:
    loaded = load_all_sources()
    overlap = build_overlap_matrix(loaded)
    write_inventory_outputs(loaded, overlap)
    write_json(RESULTS_DIR / "environment.json", environment_record())

    if not overlap["matches_expected"]:
        print("STOP: unique-ID or overlap counts disagree with EXPERIMENT_SPEC.md")
        print(overlap["mismatches"])
        return 2

    comparison = run_historical_542(loaded)
    summary = comparison["summary"]
    print(f"inventory written under {RESULTS_DIR}")
    print(f"historical 542 rows: {comparison['n']}")
    print(
        "original vs saved mismatches:",
        summary["original_vs_saved"]["n_mismatches"],
        "max_abs_diff",
        summary["original_vs_saved"]["max_abs_float_diff"],
    )
    print(
        "package author_avg unobserved zeros:",
        summary["package_native"]["n_author_avg_unobserved_zero"],
    )
    print(
        "harmonized author-comment means observed:",
        summary["harmonized"]["n_author_comment_mean_observed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
