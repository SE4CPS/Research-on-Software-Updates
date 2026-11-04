"""Inverse-probability weighted precision/recall for the independent validation sample.

This module never invents human labels and never reads protected outcomes.
It scores only after dual review of the blinded V-###### packets.
"""

from __future__ import annotations

from typing import Any

from full_resource_sentiment_v1.relevance_classify.evaluate import (
    HUMAN_BINARY_EXCLUDED,
    HUMAN_BINARY_NEGATIVE,
    HUMAN_BINARY_POSITIVE,
    EvaluationError,
    f1_score,
    wilson_95,
)
from full_resource_sentiment_v1.relevance_classify.preprocess import reject_protected_outcomes

ASSESSABLE = frozenset({HUMAN_BINARY_POSITIVE, HUMAN_BINARY_NEGATIVE})


def _weight(row: dict[str, Any]) -> float:
    p = row.get("inclusion_probability")
    if p is None:
        raise EvaluationError(f"{row.get('case_code')} missing inclusion_probability")
    prob = float(p)
    if prob <= 0.0:
        raise EvaluationError(f"{row.get('case_code')} inclusion_probability must be > 0")
    return 1.0 / prob


def score_weighted_validation(
    labeled_rows: list[dict[str, Any]],
    *,
    human_label_field: str = "final_label",
) -> dict[str, Any]:
    """Horvitz–Thompson weighted precision and recall.

    Each row must already join a human label to the private validation key
    (`ai_selected`, `inclusion_probability`, `stratum`). Packets themselves
    must stay unlabeled until reviewers finish.
    """
    n = 0
    n_excluded = 0
    w_selected = 0.0
    w_selected_relevant = 0.0
    w_human_relevant = 0.0
    tp = fp = fn = tn = 0
    by_stratum: dict[str, dict[str, float]] = {}
    for row in labeled_rows:
        reject_protected_outcomes(row)
        if row.get("final_label") and human_label_field != "final_label":
            pass
        human = str(row.get(human_label_field) or "").strip()
        if not human:
            raise EvaluationError(f"{row.get('case_code')} has no human label")
        n += 1
        if human in HUMAN_BINARY_EXCLUDED:
            n_excluded += 1
            continue
        if human not in ASSESSABLE:
            raise EvaluationError(f"{row.get('case_code')} invalid human label {human!r}")
        selected = bool(row.get("ai_selected"))
        weight = _weight(row)
        stratum = str(row.get("stratum") or "unspecified")
        block = by_stratum.setdefault(
            stratum,
            {"n": 0, "w_selected": 0.0, "w_selected_relevant": 0.0, "w_human_relevant": 0.0},
        )
        block["n"] += 1
        if selected:
            w_selected += weight
            block["w_selected"] += weight
        if human == HUMAN_BINARY_POSITIVE:
            w_human_relevant += weight
            block["w_human_relevant"] += weight
            if selected:
                tp += 1
                w_selected_relevant += weight
                block["w_selected_relevant"] += weight
            else:
                fn += 1
        elif selected:
            fp += 1
        else:
            tn += 1

    precision = None if w_selected == 0 else w_selected_relevant / w_selected
    recall = None if w_human_relevant == 0 else w_selected_relevant / w_human_relevant
    for block in by_stratum.values():
        block["precision"] = None if block["w_selected"] == 0 else block["w_selected_relevant"] / block["w_selected"]
        block["recall"] = None if block["w_human_relevant"] == 0 else block["w_selected_relevant"] / block["w_human_relevant"]
    return {
        "n_labeled": n,
        "n_excluded_unclear_or_unassessable": n_excluded,
        "unweighted_tp": tp,
        "unweighted_fp": fp,
        "unweighted_fn": fn,
        "unweighted_tn": tn,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1_score(precision, recall),
        "unweighted_precision": None if (tp + fp) == 0 else tp / (tp + fp),
        "unweighted_recall": None if (tp + fn) == 0 else tp / (tp + fn),
        "unweighted_precision_wilson_95": None if (tp + fp) == 0 else wilson_95(tp, tp + fp),
        "unweighted_recall_wilson_95": None if (tp + fn) == 0 else wilson_95(tp, tp + fn),
        "by_stratum": by_stratum,
        "estimator": (
            "Horvitz-Thompson: weight = 1 / recorded stratum inclusion_probability. "
            "Precision is the weighted share of AI-selected posts that humans mark relevant. "
            "Recall is the weighted share of human-relevant posts that AI selected. "
            "Do not report unweighted sample accuracy as full-frame accuracy."
        ),
        "ai_predicted_not_human_confirmed": True,
        "used_protected_outcomes": False,
    }
