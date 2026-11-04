"""Deterministic stratified sampler for the 200-case validation."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Iterable

from full_resource_sentiment_v1.validation_200.constants import BASE_SEED, STRATA


def stratum_seed(stratum_name: str, base_seed: str = BASE_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}|{stratum_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def presentation_seed(base_seed: str = BASE_SEED) -> int:
    digest = hashlib.sha256(f"{base_seed}|presentation_order".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def case_code_for_reddit_id(reddit_id: str, base_seed: str = BASE_SEED) -> str:
    digest = hashlib.sha256(f"{base_seed}|case_code|{reddit_id}".encode("utf-8")).hexdigest()
    return "V-" + digest[:10].upper()


def draw_stratum(sorted_ids: list[str], n: int, stratum_name: str, *, base_seed: str = BASE_SEED) -> list[str]:
    if n > len(sorted_ids):
        raise ValueError(f"stratum {stratum_name}: n={n} > N={len(sorted_ids)}")
    if sorted_ids != sorted(sorted_ids):
        raise ValueError(f"stratum {stratum_name}: ids must be lexicographically sorted")
    if len(sorted_ids) != len(set(sorted_ids)):
        raise ValueError(f"stratum {stratum_name}: duplicate ids")
    rng = random.Random(stratum_seed(stratum_name, base_seed=base_seed))
    return rng.sample(list(sorted_ids), n)


def draw_validation_sample(
    cells: dict[str, list[str]],
    *,
    base_seed: str = BASE_SEED,
    strata: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Draw without replacement from each stratum. Returns selected rows and weights."""
    strata = strata or STRATA
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, meta in strata.items():
        expected_n = int(meta["expected_N"])
        n_h = int(meta["n"])
        ids = list(cells.get(name) or [])
        if len(ids) != expected_n:
            raise RuntimeError(f"refusing draw: stratum {name} N={len(ids)} expected {expected_n}")
        if n_h > expected_n:
            raise RuntimeError(f"refusing draw: stratum {name} n={n_h} > N={expected_n}")
        sorted_ids = sorted(ids)
        drawn = draw_stratum(sorted_ids, n_h, name, base_seed=base_seed)
        pi = n_h / expected_n
        weight = expected_n / n_h
        for rid in drawn:
            if rid in seen:
                raise RuntimeError(f"duplicate selected id {rid}")
            seen.add(rid)
            domain, category = name.split("_", 1)
            # category retains underscores: relevant_tps etc. domain is fresh|covered
            # name is f"{domain}_{category}" where category itself has underscores.
            if name.startswith("fresh_"):
                domain = "fresh"
                category = name[len("fresh_") :]
            elif name.startswith("covered_"):
                domain = "covered"
                category = name[len("covered_") :]
            else:
                raise RuntimeError(f"bad stratum name {name}")
            selected.append(
                {
                    "reddit_id": rid,
                    "stratum": name,
                    "domain": domain,
                    "category": category,
                    "N_h": expected_n,
                    "n_h": n_h,
                    "inclusion_probability": pi,
                    "weight": weight,
                    "stratum_seed": stratum_seed(name, base_seed=base_seed),
                    "case_code": case_code_for_reddit_id(rid, base_seed=base_seed),
                }
            )

    # Presentation order independent of category.
    order_ids = [row["reddit_id"] for row in selected]
    rng = random.Random(presentation_seed(base_seed=base_seed))
    presentation = list(order_ids)
    rng.shuffle(presentation)
    rank = {rid: i + 1 for i, rid in enumerate(presentation)}
    for row in selected:
        row["presentation_order"] = rank[row["reddit_id"]]

    codes = [row["case_code"] for row in selected]
    if len(codes) != len(set(codes)):
        raise RuntimeError("case_code collision")

    selected_sorted = sorted(selected, key=lambda row: row["presentation_order"])
    return {
        "base_seed": base_seed,
        "n_selected": len(selected_sorted),
        "n_fresh": sum(1 for row in selected_sorted if row["domain"] == "fresh"),
        "n_covered": sum(1 for row in selected_sorted if row["domain"] == "covered"),
        "selected": selected_sorted,
        "selected_ids_sorted": sorted(row["reddit_id"] for row in selected_sorted),
        "selected_ids_sha256": hashlib.sha256(
            ("\n".join(sorted(row["reddit_id"] for row in selected_sorted)) + "\n").encode("utf-8")
        ).hexdigest(),
    }


def assert_no_forbidden_overlap(
    selected_ids: Iterable[str],
    *,
    protected_ids: set[str],
    fresh_component_ids: set[str],
    prior_exposure_ids: set[str],
    practice_ids: set[str],
) -> None:
    selected = set(selected_ids)
    if selected & protected_ids:
        raise RuntimeError(f"protected overlap in sample: {sorted(selected & protected_ids)[:10]}")
    bad_fresh = (selected & fresh_component_ids) & (prior_exposure_ids | practice_ids)
    # Fresh component must not include prior exposure or practice IDs.
    if bad_fresh:
        raise RuntimeError(f"fresh sample contains exposed/practice ids: {sorted(bad_fresh)[:10]}")
