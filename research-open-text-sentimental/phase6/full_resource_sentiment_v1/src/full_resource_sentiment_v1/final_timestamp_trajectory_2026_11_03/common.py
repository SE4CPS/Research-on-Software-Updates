"""Shared helpers for timestamp trajectory analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def stable_seed(*parts: object, base: int = 0) -> int:
    """Process-independent positive seed derived from stable SHA-256."""
    payload = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(base + (int(digest[:8], 16) % 100_000))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_created_utc(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_boot: int,
    chunk: int = 500,
) -> dict[str, float | int | None]:
    """Thread-level bootstrap mean CI (values already one-per-thread)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_boot": n_boot,
            "seed": seed,
        }
    mean = float(np.mean(values))
    median = float(np.median(values))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        take = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(take, n))
        boots[done : done + take] = values[idx].mean(axis=1)
        done += take
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n_boot": n_boot,
        "seed": seed,
    }


def bootstrap_paired_mean_ci(
    deltas: np.ndarray,
    *,
    seed: int,
    n_boot: int,
) -> dict[str, float | int | None]:
    return bootstrap_mean_ci(deltas, seed=seed, n_boot=n_boot)


def bootstrap_gds_minus_tps(
    gds: np.ndarray,
    tps: np.ndarray,
    *,
    seed: int,
    n_boot: int,
    chunk: int = 500,
) -> dict[str, float | int | None]:
    """Independent stratified bootstrap of mean(GDS) − mean(TPS)."""
    gds = np.asarray(gds, dtype=float)
    tps = np.asarray(tps, dtype=float)
    gds = gds[np.isfinite(gds)]
    tps = tps[np.isfinite(tps)]
    ng, nt = int(gds.size), int(tps.size)
    if ng == 0 or nt == 0:
        return {
            "n_gds": ng,
            "n_tps": nt,
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_boot": n_boot,
            "seed": seed,
        }
    mean = float(gds.mean() - tps.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        take = min(chunk, n_boot - done)
        g_idx = rng.integers(0, ng, size=(take, ng))
        t_idx = rng.integers(0, nt, size=(take, nt))
        boots[done : done + take] = gds[g_idx].mean(axis=1) - tps[t_idx].mean(axis=1)
        done += take
    return {
        "n_gds": ng,
        "n_tps": nt,
        "mean": mean,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n_boot": n_boot,
        "seed": seed,
    }


def bin_label(elapsed_hours: float, bins: list[tuple[float, float, str]]) -> str | None:
    if elapsed_hours is None or not np.isfinite(elapsed_hours) or elapsed_hours < 0:
        return None
    for lo, hi, lab in bins:
        if lo <= elapsed_hours < hi:
            return lab
    return None
