"""Bounded timestamp-based VADER trajectory analysis — 2026-11-03.

Uses existing saved scores/timestamps only. No new model scoring or manuscript."""

RUN_ID = "final-timestamp-trajectory-2026-11-03"
SEED = 20261103
N_BOOT = 10000

# Predeclared elapsed bins (hours); may merge only by documented count rule before inspecting sentiment.
ELAPSED_BINS_HOURS = [
    (0.0, 6.0, "0-6h"),
    (6.0, 24.0, "6-24h"),
    (24.0, 72.0, "1-3d"),
    (72.0, float("inf"), ">3d"),
]
MIN_THREADS_PER_CELL = 20  # merge rule threshold (documented)

__all__ = ["RUN_ID", "SEED", "N_BOOT", "ELAPSED_BINS_HOURS", "MIN_THREADS_PER_CELL"]
