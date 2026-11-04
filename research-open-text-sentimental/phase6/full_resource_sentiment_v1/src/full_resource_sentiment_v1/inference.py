"""Thread-bootstrap inference for the frozen A–F scoring contract."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .paths import BOOTSTRAP_SEED, N_BOOTSTRAP


def _as_float_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def _py_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return float(value)


def summarize_numeric(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = _as_float_array(values)
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "n_zero": 0,
        }
    return {
        "n": n,
        "mean": _py_float(arr.mean()),
        "median": _py_float(np.median(arr)),
        "std": _py_float(arr.std(ddof=1) if n > 1 else 0.0),
        "min": _py_float(arr.min()),
        "max": _py_float(arr.max()),
        "p05": _py_float(np.percentile(arr, 5)),
        "p25": _py_float(np.percentile(arr, 25)),
        "p75": _py_float(np.percentile(arr, 75)),
        "p95": _py_float(np.percentile(arr, 95)),
        "n_zero": int(np.count_nonzero(arr == 0.0)),
    }


def percentile_ci(samples: list[float] | np.ndarray, alpha: float = 0.05) -> list[float | None] | None:
    arr = _as_float_array(samples)
    if arr.size == 0:
        return None
    lo = _py_float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = _py_float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    if lo is None or hi is None:
        return None
    return [lo, hi]


def bootstrap_stat_samples(
    values: list[float] | np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    arr = _as_float_array(values)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, arr.size, size=(n_boot, arr.size))
    drawn = arr[index]
    return np.asarray([stat_fn(row) for row in drawn], dtype=float)


def bootstrap_mean_median_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    arr = _as_float_array(values)
    summary = summarize_numeric(arr)
    if arr.size < 1:
        summary["mean_ci95"] = None
        summary["median_ci95"] = None
        summary["n_bootstrap"] = n_boot
        summary["seed"] = seed
        summary["bootstrap_unit"] = "thread"
        return summary
    rng = np.random.default_rng(seed)
    index = rng.integers(0, arr.size, size=(n_boot, arr.size))
    drawn = arr[index]
    means = drawn.mean(axis=1)
    medians = np.median(drawn, axis=1)
    summary["mean_ci95"] = percentile_ci(means)
    summary["median_ci95"] = percentile_ci(medians)
    summary["n_bootstrap"] = n_boot
    summary["seed"] = seed
    summary["bootstrap_unit"] = "thread"
    return summary


def paired_difference_summary(
    first: list[float] | np.ndarray,
    second: list[float] | np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    difference_name: str = "second_minus_first",
) -> dict[str, Any]:
    a = _as_float_array(first)
    b = _as_float_array(second)
    if a.size != b.size:
        raise ValueError("paired arrays must have the same length")
    diff = b - a
    out = bootstrap_mean_median_ci(diff, n_boot=n_boot, seed=seed)
    out["difference_name"] = difference_name
    out["mean_first"] = _py_float(a.mean()) if a.size else None
    out["mean_second"] = _py_float(b.mean()) if b.size else None
    out["median_first"] = _py_float(np.median(a)) if a.size else None
    out["median_second"] = _py_float(np.median(b)) if b.size else None
    if diff.size > 1:
        sd = float(diff.std(ddof=1))
        out["cohens_dz"] = None if sd == 0.0 else _py_float(diff.mean() / sd)
    else:
        out["cohens_dz"] = None
    out["first"] = summarize_numeric(a)
    out["second"] = summarize_numeric(b)
    return out


def stratified_mean_diff(
    group_a: list[float] | np.ndarray,
    group_b: list[float] | np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    difference_name: str = "a_minus_b",
) -> dict[str, Any]:
    a = _as_float_array(group_a)
    b = _as_float_array(group_b)
    if a.size == 0 or b.size == 0:
        return {
            "difference_name": difference_name,
            "n_a": int(a.size),
            "n_b": int(b.size),
            "mean_a": _py_float(a.mean()) if a.size else None,
            "mean_b": _py_float(b.mean()) if b.size else None,
            "mean_diff": None,
            "mean_diff_ci95": None,
            "median_a": _py_float(np.median(a)) if a.size else None,
            "median_b": _py_float(np.median(b)) if b.size else None,
            "median_diff": None,
            "median_diff_ci95": None,
            "cliffs_delta": cliffs_delta(a, b) if a.size and b.size else None,
            "n_bootstrap": n_boot,
            "seed": seed,
            "bootstrap_unit": "thread_stratified",
            "a": summarize_numeric(a),
            "b": summarize_numeric(b),
        }
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, a.size, size=(n_boot, a.size))
    idx_b = rng.integers(0, b.size, size=(n_boot, b.size))
    mean_diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    median_diffs = np.median(a[idx_a], axis=1) - np.median(b[idx_b], axis=1)
    mean_diff = float(a.mean() - b.mean())
    median_diff = float(np.median(a) - np.median(b))
    return {
        "difference_name": difference_name,
        "n_a": int(a.size),
        "n_b": int(b.size),
        "mean_a": _py_float(a.mean()),
        "mean_b": _py_float(b.mean()),
        "mean_diff": mean_diff,
        "mean_diff_ci95": percentile_ci(mean_diffs),
        "median_a": _py_float(np.median(a)),
        "median_b": _py_float(np.median(b)),
        "median_diff": median_diff,
        "median_diff_ci95": percentile_ci(median_diffs),
        "cliffs_delta": cliffs_delta(a, b),
        "n_bootstrap": n_boot,
        "seed": seed,
        "bootstrap_unit": "thread_stratified",
        "a": bootstrap_mean_median_ci(a, n_boot=n_boot, seed=seed),
        "b": bootstrap_mean_median_ci(b, n_boot=n_boot, seed=seed),
    }


def cliffs_delta(group_a: list[float] | np.ndarray, group_b: list[float] | np.ndarray) -> float | None:
    a = _as_float_array(group_a)
    b = _as_float_array(group_b)
    if a.size == 0 or b.size == 0:
        return None
    diff = np.subtract.outer(a, b)
    n_gt = int(np.count_nonzero(diff > 0))
    n_lt = int(np.count_nonzero(diff < 0))
    return float((n_gt - n_lt) / (a.size * b.size))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if den == 0.0:
        return None
    return float(np.sum(x * y) / den)


def spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float | None:
    xa = _as_float_array(x)
    ya = _as_float_array(y)
    if xa.size != ya.size or xa.size < 2:
        return None
    return pearson(rankdata_average(xa), rankdata_average(ya))


def bootstrap_spearman(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    xa = _as_float_array(x)
    ya = _as_float_array(y)
    point = spearman(xa, ya)
    if xa.size < 2:
        return {
            "n": int(xa.size),
            "spearman": point,
            "spearman_ci95": None,
            "n_bootstrap": n_boot,
            "seed": seed,
            "bootstrap_unit": "thread",
        }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, xa.size, size=xa.size)
        value = spearman(xa[idx], ya[idx])
        if value is not None:
            samples.append(value)
    return {
        "n": int(xa.size),
        "spearman": point,
        "spearman_ci95": percentile_ci(samples) if samples else None,
        "n_bootstrap": n_boot,
        "seed": seed,
        "bootstrap_unit": "thread",
        "n_defined_resamples": len(samples),
    }
