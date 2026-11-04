"""Join AI predictions to human labels and score the 240-case cost-control gate.

Predictions and human labels stay in separate tables. This module never loads
frozen Reddit text, never calls a model, and never reads protected outcomes.
"""

from __future__ import annotations

import math
from typing import Any

from full_resource_sentiment_v1.privacy import scan_obj
from full_resource_sentiment_v1.relevance_classify.preprocess import (
    PROTECTED_OUTCOME_KEYS,
    ProtectedOutcomeSuppliedError,
    reject_protected_outcomes,
)
from full_resource_sentiment_v1.relevance_classify.spec import LABELS

EXPECTED_N = 240
EXPECTED_HUMAN_COUNTS = {
    "relevant": 40,
    "not_relevant": 199,
    "unassessable": 1,
    "unclear": 0,
}
GATE_RECALL_MIN = 0.80
GATE_PRECISION_MIN = 0.75
GATE_MIN_RELEVANT_RETRIEVED = 32
WILSON_Z = 1.959963984540054
SELECTED_STATUSES = frozenset({"ok"})
HUMAN_BINARY_POSITIVE = "relevant"
HUMAN_BINARY_NEGATIVE = "not_relevant"
HUMAN_BINARY_EXCLUDED = frozenset({"unclear", "unassessable"})
PREDICTION_FORBIDDEN_KEYS = PROTECTED_OUTCOME_KEYS | {
    "final_label",
    "reviewer_A_label",
    "reviewer_B_label",
    "adjudicated",
    "human_label",
}
LABEL_ID_KEYS = ("case_code",)
PREDICTION_ID_KEYS = ("case_code", "synthetic_id")
MAPPING_ID_KEYS = ("case_code", "reddit_id")


class EvaluationError(ValueError):
    """Raised when labels, predictions, or the join cannot be scored."""


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return None
    return 2.0 * precision * recall / denom


def wilson_95(k: int, n: int) -> list[float] | None:
    """Wilson score interval for a binomial proportion. k successes in n trials."""
    if n <= 0:
        return None
    if k < 0 or k > n:
        raise EvaluationError(f"wilson successes k={k} outside 0..{n}")
    p = k / n
    z2 = WILSON_Z * WILSON_Z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = WILSON_Z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _require_dict_rows(rows: list[dict[str, Any]], *, name: str) -> None:
    if not isinstance(rows, list):
        raise EvaluationError(f"{name} must be a list of objects")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EvaluationError(f"{name}[{index}] is not an object")


def _case_code(row: dict[str, Any], *, keys: tuple[str, ...] = LABEL_ID_KEYS) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    raise EvaluationError(f"row missing case_code: keys={keys}")


def _reject_forbidden_prediction_fields(row: dict[str, Any]) -> None:
    reject_protected_outcomes(row)
    hits = [key for key in row if str(key).lower() in {item.lower() for item in PREDICTION_FORBIDDEN_KEYS}]
    if hits:
        raise EvaluationError(
            "prediction table must not contain human labels or protected outcomes: " + ", ".join(sorted(hits))
        )


def load_human_labels(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _require_dict_rows(rows, name="human_labels")
    by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        reject_protected_outcomes(row)
        code = _case_code(row)
        if code in by_code:
            raise EvaluationError(f"duplicate human label case_code {code}")
        label = str(row.get("final_label") or "").strip()
        if label not in LABELS:
            raise EvaluationError(f"invalid human final_label {label!r} for {code}")
        stratum = str(row.get("stratum") or "").strip() or None
        by_code[code] = {
            "case_code": code,
            "final_label": label,
            "stratum": stratum,
        }
    return by_code


def load_predictions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _require_dict_rows(rows, name="predictions")
    by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        _reject_forbidden_prediction_fields(row)
        code = _case_code(row, keys=PREDICTION_ID_KEYS)
        if code in by_code:
            raise EvaluationError(f"duplicate prediction case_code {code}")
        status = str(row.get("response_status") or "").strip()
        if not status:
            raise EvaluationError(f"prediction {code} missing response_status")
        predicted = row.get("predicted_label")
        if predicted is not None:
            predicted = str(predicted).strip()
            if predicted not in LABELS:
                raise EvaluationError(f"invalid predicted_label {predicted!r} for {code}")
        by_code[code] = {
            "case_code": code,
            "predicted_label": predicted,
            "response_status": status,
            "review_priority": row.get("review_priority"),
            "unassessable_basis": row.get("unassessable_basis"),
        }
    return by_code


def load_id_mapping(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Optional case_code → reddit_id map. Used after scoring, not as model input."""
    _require_dict_rows(rows, name="id_mapping")
    mapping: dict[str, str] = {}
    seen_reddit: set[str] = set()
    for row in rows:
        reject_protected_outcomes(row)
        code = str(row.get("case_code") or "").strip()
        reddit_id = str(row.get("reddit_id") or "").strip()
        if not code or not reddit_id:
            raise EvaluationError("id mapping rows need case_code and reddit_id")
        if code in mapping:
            raise EvaluationError(f"duplicate mapping case_code {code}")
        if reddit_id in seen_reddit:
            raise EvaluationError(f"duplicate mapping reddit_id {reddit_id}")
        mapping[code] = reddit_id
        seen_reddit.add(reddit_id)
    return mapping


def apply_mapping_to_predictions(
    predictions: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """If a prediction is keyed only by reddit_id, attach case_code from the map."""
    reverse = {reddit_id: code for code, reddit_id in mapping.items()}
    out: list[dict[str, Any]] = []
    for row in predictions:
        copied = dict(row)
        code = str(copied.get("case_code") or copied.get("synthetic_id") or "").strip()
        reddit_id = str(copied.get("reddit_id") or "").strip()
        if code and reddit_id and mapping.get(code) not in {None, reddit_id}:
            raise EvaluationError(f"prediction {code} reddit_id does not match mapping")
        if not code and reddit_id:
            if reddit_id not in reverse:
                raise EvaluationError(f"prediction reddit_id {reddit_id} is not in the case_code map")
            copied["case_code"] = reverse[reddit_id]
        out.append(copied)
    return out


def join_case_codes(
    labels: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    label_ids = set(labels)
    pred_ids = set(predictions)
    missing = sorted(label_ids - pred_ids)
    extra = sorted(pred_ids - label_ids)
    joined = []
    for code in sorted(label_ids & pred_ids):
        joined.append({**labels[code], "prediction": predictions[code]})
    return {
        "joined": joined,
        "missing_predictions": missing,
        "extra_predictions": extra,
        "n_labels": len(labels),
        "n_predictions": len(predictions),
        "complete": not missing and not extra,
    }


def _is_selected(prediction: dict[str, Any]) -> bool:
    return (
        prediction.get("response_status") in SELECTED_STATUSES
        and prediction.get("predicted_label") == HUMAN_BINARY_POSITIVE
    )


def _four_way_key(prediction: dict[str, Any]) -> str:
    status = prediction.get("response_status")
    label = prediction.get("predicted_label")
    if status in SELECTED_STATUSES and label in LABELS:
        return str(label)
    if status == "skipped_deterministic" and label == "unassessable":
        return "unassessable"
    return "error"


def compute_metrics(join: dict[str, Any]) -> dict[str, Any]:
    joined: list[dict[str, Any]] = join["joined"]
    missing = list(join["missing_predictions"])
    confusion: dict[str, dict[str, int]] = {}
    human_counts = {label: 0 for label in LABELS}
    pred_counts = {key: 0 for key in (*LABELS, "error")}
    by_stratum: dict[str, dict[str, int]] = {}

    def bump(stratum: str | None, key: str, amount: int = 1) -> None:
        name = stratum or "unspecified"
        block = by_stratum.setdefault(
            name,
            {"n": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "human_relevant": 0, "selected": 0},
        )
        block[key] = block.get(key, 0) + amount

    tp = fp = fn = tn = 0
    selected_excluded = 0
    model_unclear = 0
    model_unassessable = 0
    model_errors = 0
    false_positive_case_codes: list[str] = []
    false_negative_case_codes: list[str] = []

    for row in joined:
        human = row["final_label"]
        pred = row["prediction"]
        human_counts[human] = human_counts.get(human, 0) + 1
        pred_key = _four_way_key(pred)
        pred_counts[pred_key] = pred_counts.get(pred_key, 0) + 1
        confusion.setdefault(human, {})
        confusion[human][pred_key] = confusion[human].get(pred_key, 0) + 1
        if pred_key == "unclear":
            model_unclear += 1
        elif pred_key == "unassessable":
            model_unassessable += 1
        elif pred_key == "error":
            model_errors += 1
        selected = _is_selected(pred)
        stratum = row.get("stratum")
        bump(stratum, "n")
        if human == HUMAN_BINARY_POSITIVE:
            bump(stratum, "human_relevant")
            if selected:
                tp += 1
                bump(stratum, "tp")
                bump(stratum, "selected")
            else:
                fn += 1
                bump(stratum, "fn")
                false_negative_case_codes.append(str(row.get("case_code") or pred.get("case_code") or ""))
        elif human == HUMAN_BINARY_NEGATIVE:
            if selected:
                fp += 1
                bump(stratum, "fp")
                bump(stratum, "selected")
                false_positive_case_codes.append(str(row.get("case_code") or pred.get("case_code") or ""))
            else:
                tn += 1
                bump(stratum, "tn")
        else:
            if selected:
                selected_excluded += 1
                bump(stratum, "selected")

    n_pos = human_counts.get("relevant", 0)
    n_neg = human_counts.get("not_relevant", 0)
    precision_den = tp + fp
    recall = None if n_pos == 0 else tp / n_pos
    precision = None if precision_den == 0 else tp / precision_den
    for block in by_stratum.values():
        s_tp = block.get("tp", 0)
        s_fp = block.get("fp", 0)
        s_fn = block.get("fn", 0)
        s_pos = block.get("human_relevant", 0)
        s_prec_den = s_tp + s_fp
        block["precision"] = None if s_prec_den == 0 else s_tp / s_prec_den
        block["recall"] = None if s_pos == 0 else s_tp / s_pos
        block["f1"] = f1_score(block["precision"], block["recall"])
        block["precision_wilson_95"] = None if s_prec_den == 0 else wilson_95(s_tp, s_prec_den)
        block["recall_wilson_95"] = None if s_pos == 0 else wilson_95(s_tp, s_pos)
    # Conservative bound: count human excluded cases that were selected as FP.
    precision_if_excluded_are_fp = None if (precision_den + selected_excluded) == 0 else tp / (
        precision_den + selected_excluded
    )
    # If missing relevant labels exist they cannot be retrieved.
    # They are reported separately; primary recall uses joined human positives only
    # unless the caller treats incompleteness as a failed gate.
    return {
        "n_joined": len(joined),
        "n_missing_predictions": len(missing),
        "n_extra_predictions": len(join["extra_predictions"]),
        "complete_join": bool(join["complete"]),
        "human_counts": human_counts,
        "prediction_counts": pred_counts,
        "confusion_human_by_prediction": confusion,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "model_unclear": model_unclear,
        "model_unassessable": model_unassessable,
        "model_errors": model_errors,
        "selected_human_excluded": selected_excluded,
        "n_human_relevant_joined": n_pos,
        "n_human_not_relevant_joined": n_neg,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "precision_wilson_95": None if precision_den == 0 else wilson_95(tp, precision_den),
        "recall_wilson_95": None if n_pos == 0 else wilson_95(tp, n_pos),
        "false_positive_case_codes": [code for code in false_positive_case_codes if code],
        "false_negative_case_codes": [code for code in false_negative_case_codes if code],
        "precision_if_selected_excluded_count_as_fp": precision_if_excluded_are_fp,
        "by_stratum": by_stratum,
        "binary_rule": {
            "selected": "predicted_label==relevant and response_status==ok",
            "human_positive": "final_label==relevant",
            "human_negative": "final_label==not_relevant",
            "human_excluded_from_binary": sorted(HUMAN_BINARY_EXCLUDED),
            "missing_or_error_on_relevant": "counts as false_negative, never as not_relevant",
        },
        "small_n_note": (
            "Forty human-relevant cases make recall and precision high-variance. "
            f"Retrieving {GATE_MIN_RELEVANT_RETRIEVED}/40 is observed recall {GATE_RECALL_MIN:.2f}, "
            "but the Wilson 95% interval is about 0.65-0.90. These thresholds are a "
            "cost-control gate, not publication-quality accuracy."
        ),
    }


def apply_cost_control_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    recall = metrics["recall"]
    precision = metrics["precision"]
    tp = metrics["true_positives"]
    n_pos = metrics["n_human_relevant_joined"]
    complete = metrics["complete_join"]
    recall_ok = recall is not None and recall >= GATE_RECALL_MIN
    if n_pos == EXPECTED_HUMAN_COUNTS["relevant"]:
        recall_ok = bool(recall_ok and tp >= GATE_MIN_RELEVANT_RETRIEVED)
    precision_ok = precision is not None and precision >= GATE_PRECISION_MIN
    eligible = bool(complete)
    passed = eligible and recall_ok and precision_ok
    return {
        "gate_name": "cost_control_240_not_publication_accuracy",
        "eligible": eligible,
        "passed": passed,
        "recall_ok": recall_ok,
        "precision_ok": precision_ok,
        "recall_min": GATE_RECALL_MIN,
        "precision_min": GATE_PRECISION_MIN,
        "min_relevant_retrieved": GATE_MIN_RELEVANT_RETRIEVED,
        "ineligible_reason": None if eligible else "join is incomplete or has extra/missing IDs",
        "interpretation": (
            "A pass only authorizes considering a later full-frame run. It does not "
            "make AI labels human-confirmed or justify treating the 240 as an untouched "
            "final test if the prompt or model is changed afterward."
        ),
    }


def evaluate_gate(
    human_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    mapping_rows: list[dict[str, Any]] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    labels = load_human_labels(human_rows)
    predictions_in = list(prediction_rows)
    mapping = None
    if mapping_rows is not None:
        mapping = load_id_mapping(mapping_rows)
        predictions_in = apply_mapping_to_predictions(predictions_in, mapping)
    predictions = load_predictions(predictions_in)
    join = join_case_codes(labels, predictions)
    if require_complete and not join["complete"]:
        raise EvaluationError(
            "incomplete join: missing="
            + ",".join(join["missing_predictions"][:12])
            + (" extra=" + ",".join(join["extra_predictions"][:12]) if join["extra_predictions"] else "")
        )
    metrics = compute_metrics(join)
    gate = apply_cost_control_gate(metrics)
    payload = {
        "expected_n_if_real_audit": EXPECTED_N,
        "expected_human_counts_if_real_audit": EXPECTED_HUMAN_COUNTS,
        "observed_n_human": len(labels),
        "observed_n_predictions": len(predictions),
        "join": {
            "complete": join["complete"],
            "missing_predictions": join["missing_predictions"],
            "extra_predictions": join["extra_predictions"],
        },
        "metrics": metrics,
        "gate": gate,
        "mapping_used": mapping is not None,
        "used_real_reddit_text": False,
        "used_protected_outcomes": False,
    }
    public_hits = scan_obj(
        {
            "metrics": {key: value for key, value in metrics.items() if key != "confusion_human_by_prediction"},
            "gate": gate,
            "join_counts": {
                "missing": len(join["missing_predictions"]),
                "extra": len(join["extra_predictions"]),
            },
        }
    )
    if public_hits:
        raise EvaluationError(f"public evaluation object leaked raw-text markers: {public_hits[:8]}")
    return payload


def to_classifier_inputs(packets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip packets to title/body plus case_code. Labels never enter the model input."""
    _require_dict_rows(packets, name="packets")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for packet in packets:
        reject_protected_outcomes(packet)
        code = _case_code(packet, keys=("case_code", "synthetic_id"))
        if code in seen:
            raise EvaluationError(f"duplicate packet case_code {code}")
        seen.add(code)
        if packet.get("final_label") or packet.get("human_relevant") or packet.get("relevance"):
            raise EvaluationError(f"packet {code} still contains a human label")
        title = "" if packet.get("title") is None else str(packet.get("title") or "")
        body = "" if packet.get("body") is None else str(packet.get("body") or "")
        out.append(
            {
                "synthetic_id": code,
                "case_code": code,
                "title": title,
                "body": body,
            }
        )
    return sorted(out, key=lambda row: row["case_code"])
