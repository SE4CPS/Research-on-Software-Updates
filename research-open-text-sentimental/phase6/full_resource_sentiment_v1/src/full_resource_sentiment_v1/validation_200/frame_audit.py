"""Read-only audit of the frozen 6,425-ID validation frame."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.validation_200.constants import (
    CATEGORIES,
    EXPECTED_COVERED_N,
    EXPECTED_FRAME_N,
    EXPECTED_FRESH_N,
    STRATA,
    phase6_root,
    repo_root,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def predicted_category(row: dict[str, Any]) -> str:
    """Derive sampling category from final_relevance / final_discussion only."""
    fr = str(row.get("final_relevance") or "")
    fd = str(row.get("final_discussion") or "")
    if fr == "relevant" and fd == "TPS":
        return "relevant_tps"
    if fr == "relevant" and fd == "GDS":
        return "relevant_gds"
    if fr == "relevant" and fd == "Other":
        return "relevant_other"
    if fr == "not_relevant":
        return "not_relevant"
    if fr in {"unclear", "unassessable"} or (fr == "relevant" and fd in {"Unclear", "unclear"}):
        return "unclear_unassessable"
    raise ValueError(f"unmapped final labels reddit_id={row.get('reddit_id')} fr={fr!r} fd={fd!r}")


def stratum_name(domain: str, category: str) -> str:
    return f"{domain}_{category}"


def parse_training_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\*\*Source ID:\*\*\s*`([0-9a-z]+)`", text, flags=re.I))


def load_protected_ids() -> dict[str, set[str]]:
    root = repo_root()
    phase6 = phase6_root()
    reserved = {
        str(row.get("reddit_id") or row.get("id"))
        for row in load_jsonl(root / "phase3/experiment_v1/data/reserved_test.jsonl")
    }
    fresh = {
        str(row.get("reddit_id") or row.get("id"))
        for row in load_jsonl(root / "phase3/fresh_test_20261007/frozen_joint_labels/all_500_reviewed.jsonl")
    }
    payload = json.loads(
        (phase6 / "restricted/relevance_audit_sample_2026-10-28/excluded_protected_ids.json").read_text(
            encoding="utf-8"
        )
    )
    union = {str(x) for x in (payload.get("union_ids") or [])}
    return {"reserved": reserved, "fresh_protected": fresh, "union": union}


def load_exposure_ids() -> dict[str, set[str]]:
    root = repo_root()
    phase6 = phase6_root()
    out: dict[str, set[str]] = {}
    out["combined_pool"] = {
        str(row["reddit_id"])
        for row in load_jsonl(phase6 / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl")
    }
    with (root / "tps_gds_classification/data/updated_labeled_dataset_unique.csv").open(encoding="utf-8") as handle:
        out["historical_542"] = {str(row["reddit_id"]) for row in csv.DictReader(handle)}
    classif = phase6 / "restricted/tps_gds_full_classification_2026-11-02"
    out["gate_packet"] = {str(row["reddit_id"]) for row in load_jsonl(classif / "gate_packet.jsonl")}
    out["gate2_packet"] = {str(row["reddit_id"]) for row in load_jsonl(classif / "gate2_packet.jsonl")}
    out["relevance_240_packet"] = {
        str(row["reddit_id"]) for row in load_jsonl(classif / "relevance_240_packet.jsonl")
    }
    out["relevance_audit_240"] = {
        str(row["reddit_id"])
        for row in load_jsonl(phase6 / "restricted/relevance_audit_sample_2026-10-28/private_id_key.jsonl")
    }
    out["validation_pilot_240"] = {
        str(row["reddit_id"])
        for row in load_jsonl(
            phase6 / "restricted/independent_validation_pilot_2026-10-29/validation/private_id_key.jsonl"
        )
    }
    out["reviewer_training_18"] = parse_training_ids(
        phase6 / "reviewer_training/06_REVIEWER_TRAINING_EXERCISES.md"
    )
    return out


def build_frame_audit() -> dict[str, Any]:
    phase6 = phase6_root()
    frame_path = phase6 / "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl"
    pool_path = phase6 / "results/tps_gds_combined_pool_2026-11-02/combined_pool.jsonl"
    rows = load_jsonl(frame_path)
    frame_ids = [str(row["reddit_id"]) for row in rows]
    frame_set = set(frame_ids)
    if len(frame_ids) != len(frame_set):
        raise RuntimeError("duplicate reddit_id in full_dataset.jsonl")
    if len(frame_set) != EXPECTED_FRAME_N:
        raise RuntimeError(f"frame n={len(frame_set)} != {EXPECTED_FRAME_N}")

    protected = load_protected_ids()
    exposure = load_exposure_ids()
    exposed_union: set[str] = set()
    for ids in exposure.values():
        exposed_union |= ids
    covered = exposed_union & frame_set
    fresh = frame_set - covered

    category_counts = Counter(predicted_category(row) for row in rows)
    cells: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        rid = str(row["reddit_id"])
        category = predicted_category(row)
        domain = "covered" if rid in covered else "fresh"
        name = stratum_name(domain, category)
        cells[name].append(rid)
        by_id[rid] = {
            "reddit_id": rid,
            "domain": domain,
            "category": category,
            "stratum": name,
            "final_relevance": row.get("final_relevance"),
            "final_discussion": row.get("final_discussion"),
            # Predictions retained only for private analyst key after draw.
            "model_version": row.get("model_version"),
            "decision_source": row.get("decision_source"),
            "selection_stratum": row.get("selection_stratum"),
        }

    discrepancies: list[str] = []
    if len(fresh) != EXPECTED_FRESH_N:
        discrepancies.append(f"fresh N={len(fresh)} expected {EXPECTED_FRESH_N}")
    if len(covered) != EXPECTED_COVERED_N:
        discrepancies.append(f"covered N={len(covered)} expected {EXPECTED_COVERED_N}")
    for name, meta in STRATA.items():
        got = len(cells.get(name, []))
        expected = int(meta["expected_N"])
        if got != expected:
            discrepancies.append(f"stratum {name}: N={got} expected {expected}")

    for key, ids in protected.items():
        overlap = sorted(frame_set & ids)
        if overlap:
            discrepancies.append(f"protected overlap ({key}) n={len(overlap)} head={overlap[:5]}")

    source_hashes = {
        "full_dataset.jsonl": sha256_file(frame_path),
        "combined_pool.jsonl": sha256_file(pool_path),
        "historical_542_csv": sha256_file(
            repo_root() / "tps_gds_classification/data/updated_labeled_dataset_unique.csv"
        ),
        "gate_packet.jsonl": sha256_file(
            phase6 / "restricted/tps_gds_full_classification_2026-11-02/gate_packet.jsonl"
        ),
        "gate2_packet.jsonl": sha256_file(
            phase6 / "restricted/tps_gds_full_classification_2026-11-02/gate2_packet.jsonl"
        ),
        "relevance_240_packet.jsonl": sha256_file(
            phase6 / "restricted/tps_gds_full_classification_2026-11-02/relevance_240_packet.jsonl"
        ),
        "relevance_audit_private_id_key.jsonl": sha256_file(
            phase6 / "restricted/relevance_audit_sample_2026-10-28/private_id_key.jsonl"
        ),
        "validation_pilot_private_id_key.jsonl": sha256_file(
            phase6
            / "restricted/independent_validation_pilot_2026-10-29/validation/private_id_key.jsonl"
        ),
        "reviewer_training_exercises.md": sha256_file(
            phase6 / "reviewer_training/06_REVIEWER_TRAINING_EXERCISES.md"
        ),
        "excluded_protected_ids.json": sha256_file(
            phase6 / "restricted/relevance_audit_sample_2026-10-28/excluded_protected_ids.json"
        ),
    }

    return {
        "ok": not discrepancies,
        "discrepancies": discrepancies,
        "n_frame": len(frame_set),
        "n_fresh": len(fresh),
        "n_covered": len(covered),
        "category_counts": {k: int(category_counts[k]) for k in CATEGORIES},
        "stratum_counts": {name: len(cells.get(name, [])) for name in STRATA},
        "exposure_component_in_frame": {k: len(v & frame_set) for k, v in exposure.items()},
        "exposure_component_total": {k: len(v) for k, v in exposure.items()},
        "protected_overlap": {k: len(frame_set & v) for k, v in protected.items()},
        "source_hashes": source_hashes,
        "cells": {k: sorted(v) for k, v in cells.items()},
        "by_id": by_id,
        "fresh_ids": sorted(fresh),
        "covered_ids": sorted(covered),
        "training_ids": sorted(exposure["reviewer_training_18"]),
    }
