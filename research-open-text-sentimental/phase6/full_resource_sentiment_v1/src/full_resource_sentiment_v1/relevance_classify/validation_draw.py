"""Deterministic four-stratum blinded validation sample. Does not invent human labels."""

from __future__ import annotations

import random
from typing import Any

from full_resource_sentiment_v1.relevance_audit import PACKET_FIELDS, packet_is_blinded
from full_resource_sentiment_v1.relevance_classify.evaluate import _is_selected
from full_resource_sentiment_v1.relevance_classify.spec import VALIDATION_N_PER_STRATUM, VALIDATION_SEED


class ValidationDrawError(ValueError):
    pass


STRATA = (
    ("october_2026", True, "october_ai_selected"),
    ("october_2026", False, "october_ai_not_selected"),
    ("older", True, "older_ai_selected"),
    ("older", False, "older_ai_not_selected"),
)


def _case_code(index: int) -> str:
    return f"V-{index:06d}"


def draw_validation_sample(
    rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    *,
    exclude_reddit_ids: set[str],
    exclude_protected_ids: set[str],
    seed: int = VALIDATION_SEED,
    n_per_stratum: int = VALIDATION_N_PER_STRATUM,
) -> dict[str, Any]:
    pools: dict[str, list[str]] = {name: [] for _, _, name in STRATA}
    for row in rows:
        rid = str(row.get("reddit_id") or row.get("synthetic_id") or "")
        if not rid or rid in exclude_reddit_ids or rid in exclude_protected_ids:
            continue
        rec = records_by_id.get(rid) or {}
        if rec.get("protected_membership") or row.get("protected_membership"):
            continue
        stratum = str(rec.get("stratum") or row.get("stratum") or "")
        selected = _is_selected(row)
        for want_stratum, want_selected, name in STRATA:
            if stratum == want_stratum and selected is want_selected:
                pools[name].append(rid)
                break
    for name, ids in pools.items():
        ids.sort()
        pools[name] = ids

    rng = random.Random(seed)
    drawn: dict[str, list[str]] = {}
    inclusion: dict[str, float] = {}
    notes: dict[str, str] = {}
    for _, _, name in STRATA:
        available = pools[name]
        take = min(n_per_stratum, len(available))
        if take < n_per_stratum:
            notes[name] = f"only {len(available)} eligible; taking all"
        sampled = rng.sample(available, take) if take else []
        sampled.sort()
        drawn[name] = sampled
        prob = 0.0 if not available else take / len(available)
        for rid in sampled:
            inclusion[rid] = prob

    ordered_ids: list[str] = []
    for _, _, name in STRATA:
        ordered_ids.extend(sorted(drawn[name]))
    # Stable case codes on lexicographic IDs within the concatenated stratum order.
    by_id_code = {rid: _case_code(index) for index, rid in enumerate(ordered_ids, start=1)}

    private_rows = []
    packets = []
    for rid in ordered_ids:
        rec = records_by_id[rid]
        code = by_id_code[rid]
        packet = {
            "case_code": code,
            "title": rec.get("title") or "",
            "body": rec.get("body") or "",
            "relevance": "",
            "evidence_phrase": "",
            "uncertainty_note": "",
        }
        hits = packet_is_blinded(packet)
        if hits:
            raise ValidationDrawError(f"packet blinding failure {code}: {hits}")
        extra = [key for key in packet if key not in PACKET_FIELDS]
        if extra:
            raise ValidationDrawError(f"unexpected packet keys {extra}")
        packets.append(packet)
        pred = next(row for row in rows if str(row.get("reddit_id") or row.get("synthetic_id")) == rid)
        private_rows.append(
            {
                "case_code": code,
                "reddit_id": rid,
                "stratum": rec.get("stratum"),
                "ai_selected": _is_selected(pred),
                "ai_predicted_label": pred.get("predicted_label"),
                "ai_response_status": pred.get("response_status"),
                "inclusion_probability": inclusion[rid],
                "validation_seed": seed,
                "ai_predicted_not_human_confirmed": True,
            }
        )

    order_a = list(range(len(packets)))
    order_b = list(range(len(packets)))
    rng_a = random.Random(seed + 1)
    rng_b = random.Random(seed + 2)
    rng_a.shuffle(order_a)
    rng_b.shuffle(order_b)
    packet_a = [packets[i] for i in order_a]
    packet_b = [packets[i] for i in order_b]
    return {
        "seed": seed,
        "n_per_stratum_target": n_per_stratum,
        "n_drawn": len(packets),
        "pools_available": {name: len(ids) for name, ids in pools.items()},
        "drawn_counts": {name: len(ids) for name, ids in drawn.items()},
        "drawn_ids": drawn,
        "short_stratum_notes": notes,
        "excluded_development_n": len(exclude_reddit_ids),
        "excluded_protected_n": len(exclude_protected_ids),
        "packets": packets,
        "packet_a": packet_a,
        "packet_b": packet_b,
        "private_rows": private_rows,
        "reviewer_a_order": order_a,
        "reviewer_b_order": order_b,
        "rng_algorithm": (
            "Python random.Random.sample on sorted eligible IDs per stratum, "
            "then V-###### codes on concatenated sorted stratum draws, "
            "then independent shuffles with seed+1 and seed+2"
        ),
        "human_labels_invented": False,
        "ai_predictions_hidden_from_reviewers": True,
    }
