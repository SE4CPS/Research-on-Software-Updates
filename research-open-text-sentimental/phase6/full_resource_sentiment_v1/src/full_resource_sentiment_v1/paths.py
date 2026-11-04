"""Read-only source paths. All writes stay under this workspace."""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
REPO_ROOT = WORKSPACE.parents[1]
RESULTS_DIR = WORKSPACE / "results"
LOCAL_NORMALIZED_DIR = WORKSPACE / "local_normalized"

HISTORICAL_LABELS = REPO_ROOT / "tps_gds_classification" / "data" / "updated_labeled_dataset_unique.csv"
HISTORICAL_RAW_DIR = REPO_ROOT / "tps_gds_classification" / "data" / "raw"
HISTORICAL_RAW_TPS = HISTORICAL_RAW_DIR / "tps_response.json"
HISTORICAL_RAW_GDS = HISTORICAL_RAW_DIR / "gds_response.json"

C1_PRIME_METRICS = REPO_ROOT / "data" / "c1_prime" / "c1_prime_metrics.csv"
C1_PRIME_METADATA = REPO_ROOT / "data" / "c1_prime" / "build_metadata.json"

COHORT_324 = REPO_ROOT / "data" / "enhanced_automated_sentiment_results.json"

MASTER = REPO_ROOT / "phase2" / "combined_dataset" / "master_records.json"

OCT5_API_DIR = REPO_ROOT / "phase2" / "new_sources" / "api"
OCT5_PAGINATION_MANIFEST = OCT5_API_DIR / "pagination_manifest.json"
OCT5_PAGE_FILES = (
    "tps_response.json",
    "tps_page_2.json",
    "tps_page_3.json",
    "tps_page_4.json",
    "tps_page_5.json",
    "gds_response.json",
)

DEVELOPMENT_V2 = REPO_ROOT / "phase3" / "revised_development_v2" / "development_v2.jsonl"
REDDIT_DEVELOPMENT = REPO_ROOT / "phase3" / "reddit_only_sensitivity" / "development_reddit_only.jsonl"
RESERVED_TEST = REPO_ROOT / "phase3" / "experiment_v1" / "data" / "reserved_test.jsonl"
FRESH_500 = REPO_ROOT / "phase3" / "fresh_test_20261007" / "frozen_joint_labels" / "all_500_reviewed.jsonl"

PACKAGE_TARBALL = (
    REPO_ROOT / "phase2" / "new_sources" / "package" / "releasetrain_sentiment-0.2.1.tar.gz"
)

BROAD_CAPTURE_ID = "2026-10-22-main"
BROAD_CAPTURE_DIR = WORKSPACE / "restricted" / "broad_release_train" / BROAD_CAPTURE_ID
BROAD_COMPACT_INDEX = RESULTS_DIR / f"broad_capture_compact_{BROAD_CAPTURE_ID}.jsonl"
BROAD_RELEVANCE_DENOMS = RESULTS_DIR / "broad_relevance_denominators.json"
BROAD_SENTIMENT_RUN_DIR = RESULTS_DIR / f"broad_sentiment_{BROAD_CAPTURE_ID}"
LOCAL_COMPACT_RECORDS = RESULTS_DIR / "normalized_records_compact.jsonl"
BOOTSTRAP_SEED = 20261022
N_BOOTSTRAP = 2000

EXPECTED_COUNTS = {
    "historical_labels_unique": 542,
    "master_unique": 3362,
    "development_unique": 3151,
    "reddit_development_unique": 3121,
    "reserved_unique": 200,
    "fresh_unique": 500,
    "cohort_unique": 324,
    "historical_master_overlap": 542,
    "cohort_master_overlap": 18,
    "fresh_master_overlap": 0,
    "fresh_reserved_overlap": 0,
}

HISTORICAL_LABEL_COUNTS = {"tps_corrected_1": 86, "gds_corrected_0": 456}


def all_input_files() -> list[Path]:
    files = [
        HISTORICAL_LABELS,
        HISTORICAL_RAW_TPS,
        HISTORICAL_RAW_GDS,
        C1_PRIME_METRICS,
        C1_PRIME_METADATA,
        COHORT_324,
        MASTER,
        OCT5_PAGINATION_MANIFEST,
        DEVELOPMENT_V2,
        REDDIT_DEVELOPMENT,
        RESERVED_TEST,
        FRESH_500,
        PACKAGE_TARBALL,
    ]
    files.extend(OCT5_API_DIR / name for name in OCT5_PAGE_FILES)
    return files
