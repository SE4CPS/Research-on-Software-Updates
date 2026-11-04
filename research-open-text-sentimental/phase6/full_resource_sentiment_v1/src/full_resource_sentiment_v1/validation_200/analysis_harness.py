"""Synthetic-only Horvitz–Thompson weighted validation harness."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def weighted_confusion(
    rows: Iterable[dict[str, Any]],
    *,
    predicted_key: str,
    human_key: str,
    weight_key: str = "weight",
) -> dict[tuple[Any, Any], float]:
    counts: dict[tuple[Any, Any], float] = defaultdict(float)
    for row in rows:
        counts[(row[predicted_key], row[human_key])] += float(row[weight_key])
    return dict(counts)


def precision_recall_f1(confusion: dict[tuple[Any, Any], float], positive: Any) -> dict[str, float | None]:
    tp = float(confusion.get((positive, positive), 0.0))
    pred_pos = sum(v for (p, _h), v in confusion.items() if p == positive)
    human_pos = sum(v for (_p, h), v in confusion.items() if h == positive)
    precision = (tp / pred_pos) if pred_pos > 0 else None
    recall = (tp / human_pos) if human_pos > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred_pos": pred_pos, "human_pos": human_pos}


def macro_f1(scores: list[dict[str, float | None]]) -> float | None:
    vals = [s["f1"] for s in scores if s.get("f1") is not None]
    if not vals:
        return None
    return sum(float(v) for v in vals) / len(vals)


def split_fresh_vs_all(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    fresh = [row for row in rows if row.get("domain") == "fresh"]
    return {"fresh": fresh, "all": list(rows)}
