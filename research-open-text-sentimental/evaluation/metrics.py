"""
Classification metrics and bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class FoldMetrics:
    accuracy: float
    f1_macro: float
    f1_tps: float
    precision_tps: float
    recall_tps: float
    pr_auc_tps: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_tps: np.ndarray | None = None,
) -> FoldMetrics:
    """Binary TPS=1 metrics; y_score_tps used for PR-AUC when provided."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_tps = float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    pr_auc = 0.0
    if y_score_tps is not None and len(np.unique(y_true)) > 1:
        try:
            pr_auc = float(average_precision_score(y_true, y_score_tps, pos_label=1))
        except ValueError:
            pr_auc = 0.0
    return FoldMetrics(
        accuracy=acc,
        f1_macro=f1_macro,
        f1_tps=f1_tps,
        precision_tps=float(prec),
        recall_tps=float(rec),
        pr_auc_tps=pr_auc,
    )


def aggregate_fold_metrics(folds: list[FoldMetrics]) -> dict[str, Any]:
    """Mean ± std across folds."""
    keys = list(FoldMetrics.__dataclass_fields__.keys())
    out: dict[str, Any] = {}
    for key in keys:
        vals = [getattr(f, key) for f in folds]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score_tps: np.ndarray | None,
    metric: str,
    n_samples: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI on (y_true, y_pred) pairs.
    metric: accuracy | f1_macro | f1_tps | pr_auc_tps
    Returns (point_estimate, lower, upper).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    def point() -> float:
        m = compute_metrics(y_true, y_pred, y_score_tps)
        return getattr(m, metric)

    est = point()
    boots: list[float] = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score_tps[idx] if y_score_tps is not None else None
        if len(np.unique(yt)) < 2 and metric == "pr_auc_tps":
            continue
        boots.append(getattr(compute_metrics(yt, yp, ys), metric))

    if not boots:
        return est, est, est
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return est, lo, hi


def confusion_matrix_labels() -> list[str]:
    return ["GDS", "TPS"]


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])
