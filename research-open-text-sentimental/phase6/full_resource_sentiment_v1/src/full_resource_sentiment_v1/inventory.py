"""Source counts, overlaps, and compact normalized membership records."""

from __future__ import annotations

from typing import Any

from .io_utils import write_json
from .loaders import (
    load_c1_prime,
    load_cohort_324,
    load_historical_labels,
    load_historical_raw,
    load_jsonl_ids,
    load_master,
    load_oct5_pages,
    source_file_manifest,
    snapshot_compact,
)
from .paths import (
    DEVELOPMENT_V2,
    EXPECTED_COUNTS,
    FRESH_500,
    HISTORICAL_LABEL_COUNTS,
    REDDIT_DEVELOPMENT,
    RESERVED_TEST,
    RESULTS_DIR,
)
from .schema import schema_document


def _set(ids: list[str]) -> set[str]:
    return set(ids)


def build_overlap_matrix(loaded: dict[str, Any]) -> dict[str, Any]:
    historical = _set(loaded["historical_labels"]["inventory"]["ids"])
    master = _set(loaded["master"]["ids"])
    development = _set(loaded["development"]["ids"])
    reddit_dev = _set(loaded["reddit_development"]["ids"])
    reserved = _set(loaded["reserved"]["ids"])
    fresh = _set(loaded["fresh"]["ids"])
    cohort = _set(loaded["cohort"]["ids"])
    oct5 = _set(loaded["oct5"]["ids"])
    historical_raw_union = _set(loaded["historical_raw"]["union_ids"])

    actual = {
        "historical_labels_unique": len(historical),
        "master_unique": len(master),
        "development_unique": len(development),
        "reddit_development_unique": len(reddit_dev),
        "reserved_unique": len(reserved),
        "fresh_unique": len(fresh),
        "cohort_unique": len(cohort),
        "historical_master_overlap": len(historical & master),
        "cohort_master_overlap": len(cohort & master),
        "fresh_master_overlap": len(fresh & master),
        "fresh_reserved_overlap": len(fresh & reserved),
    }
    mismatches = {
        key: {"expected": EXPECTED_COUNTS[key], "actual": actual[key]}
        for key in EXPECTED_COUNTS
        if actual[key] != EXPECTED_COUNTS[key]
    }
    return {
        "expected": EXPECTED_COUNTS,
        "actual": actual,
        "matches_expected": not mismatches,
        "mismatches": mismatches,
        "additional_counts": {
            "historical_raw_tps_unique": loaded["historical_raw"]["tps"]["inventory"]["n_unique_ids"],
            "historical_raw_gds_unique": loaded["historical_raw"]["gds"]["inventory"]["n_unique_ids"],
            "historical_raw_union": len(historical_raw_union),
            "historical_raw_tps_gds_overlap": loaded["historical_raw"]["n_tps_gds_id_overlap"],
            "historical_labels_in_raw_union": len(historical & historical_raw_union),
            "october5_unique": len(oct5),
            "october5_master_overlap": len(oct5 & master),
            "historical_oct5_overlap": len(historical & oct5),
            "reserved_master_overlap": len(reserved & master),
            "development_master_overlap": len(development & master),
            "reddit_development_in_development": len(reddit_dev & development),
            "cohort_historical_overlap": len(cohort & historical),
            "master_uncertain_review": loaded["master"]["n_uncertain_review"],
            "historical_corrected_label_counts": loaded["historical_labels"]["corrected_label_counts"],
            "expected_historical_label_counts": HISTORICAL_LABEL_COUNTS,
        },
        "overlap_id_lists": {
            "cohort_master": sorted(cohort & master),
            "fresh_master": sorted(fresh & master),
            "fresh_reserved": sorted(fresh & reserved),
            "historical_not_in_master": sorted(historical - master),
            "master_uncertain_review": loaded["master"]["uncertain_review_ids"],
        },
        "notes": [
            "Protected reserved/fresh files were read for ID membership only.",
            "API TPS/GDS routes are not human labels.",
            "Counts are unique nonempty reddit IDs; overlapping sources are not added together.",
        ],
    }


def load_all_sources() -> dict[str, Any]:
    return {
        "file_manifest": source_file_manifest(),
        "historical_labels": load_historical_labels(),
        "historical_raw": load_historical_raw(),
        "c1_prime": load_c1_prime(),
        "cohort": load_cohort_324(),
        "master": load_master(),
        "oct5": load_oct5_pages(),
        "development": load_jsonl_ids(DEVELOPMENT_V2, "development_v2", ids_only=False),
        "reddit_development": load_jsonl_ids(REDDIT_DEVELOPMENT, "reddit_development", ids_only=False),
        "reserved": load_jsonl_ids(RESERVED_TEST, "reserved_test", ids_only=True),
        "fresh": load_jsonl_ids(FRESH_500, "fresh_500", ids_only=True),
    }


def public_manifest(loaded: dict[str, Any]) -> dict[str, Any]:
    def strip_by_id(block: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in block.items() if k not in {"by_id", "merged_by_id_original_rule", "ids", "candidate_route_by_id", "metadata_predicted_by_id", "metadata_labeled_by_id", "occurrences_by_id"}}
        if "inventory" in out and isinstance(out["inventory"], dict):
            inv = dict(out["inventory"])
            inv.pop("ids", None)
            out["inventory"] = inv
        if "tps" in out:
            out["tps"] = {k: v for k, v in out["tps"].items() if k != "by_id"}
            if "inventory" in out["tps"]:
                inv = dict(out["tps"]["inventory"])
                inv.pop("ids", None)
                out["tps"]["inventory"] = inv
        if "gds" in out:
            out["gds"] = {k: v for k, v in out["gds"].items() if k != "by_id"}
            if "inventory" in out["gds"]:
                inv = dict(out["gds"]["inventory"])
                inv.pop("ids", None)
                out["gds"]["inventory"] = inv
        if "pages" in out:
            pages = []
            for page in out["pages"]:
                page_pub = dict(page)
                if "inventory" in page_pub and isinstance(page_pub["inventory"], dict):
                    inv = dict(page_pub["inventory"])
                    inv.pop("ids", None)
                    page_pub["inventory"] = inv
                pages.append(page_pub)
            out["pages"] = pages
        return out

    return {
        "files": loaded["file_manifest"],
        "sources": {
            "historical_labels": strip_by_id(loaded["historical_labels"]),
            "historical_raw": strip_by_id(loaded["historical_raw"]),
            "c1_prime": strip_by_id(loaded["c1_prime"]),
            "cohort": strip_by_id(loaded["cohort"]),
            "master": strip_by_id(loaded["master"]),
            "oct5": strip_by_id(loaded["oct5"]),
            "development": strip_by_id(loaded["development"]),
            "reddit_development": strip_by_id(loaded["reddit_development"]),
            "reserved": strip_by_id(loaded["reserved"]),
            "fresh": strip_by_id(loaded["fresh"]),
        },
    }


def hand_checked_cases(loaded: dict[str, Any]) -> dict[str, Any]:
    march_id = "1o0nzld"
    oct_id = "1wyg8ze"
    march = loaded["historical_raw"]["merged_by_id_original_rule"][march_id]
    oct_post = loaded["oct5"]["by_id"][oct_id]
    march_snap = snapshot_compact(
        march,
        "historical_raw_merged_original_rule",
        str(loaded["historical_raw"]["tps"]["file"]),
        loaded["historical_raw"]["tps"]["capture_time"],
    )
    oct_snap = snapshot_compact(
        oct_post,
        "october5_api_pages",
        str(loaded["oct5"]["pages"][0]["file"]),
        loaded["oct5"]["pagination_manifest"].get("collected_at_utc"),
    )
    return {
        "policy": "Hashes and flags only; no usernames or full text.",
        "march_1o0nzld": {
            "reddit_id": march_id,
            "snapshot": march_snap,
            "raw_keys_present": sorted(march.keys()),
            "has_author_description": "author_description" in march,
            "has_selftext": "selftext" in march,
            "has_description": "description" in march,
            "comments_are_list": isinstance(march.get("comments"), list),
        },
        "october_1wyg8ze": {
            "reddit_id": oct_id,
            "snapshot": oct_snap,
            "raw_keys_present": sorted(oct_post.keys()),
            "has_author_description": "author_description" in oct_post,
            "author_description_nonempty": bool(str(oct_post.get("author_description") or "").strip()),
            "has_selftext": "selftext" in oct_post,
            "has_description": "description" in oct_post,
            "comments_are_list": isinstance(oct_post.get("comments"), list),
            "api_candidate_route": loaded["oct5"]["candidate_route_by_id"].get(oct_id),
            "metadata_predicted_is_not_human_label": True,
        },
    }


def write_inventory_outputs(loaded: dict[str, Any], overlap: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_DIR / "source_manifest.json", public_manifest(loaded))
    write_json(RESULTS_DIR / "overlap_matrix.json", overlap)
    write_json(RESULTS_DIR / "normalized_schema.json", schema_document())
    write_json(RESULTS_DIR / "hand_checked_cases.json", hand_checked_cases(loaded))
