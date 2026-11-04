"""Persist a cumulative spend ledger. Never logs API keys or post text."""

from __future__ import annotations

from typing import Any

from full_resource_sentiment_v1.broad_sentiment import atomic_write_json, utcnow
from full_resource_sentiment_v1.io_utils import read_json
from full_resource_sentiment_v1.relevance_classify.spec import (
    PROPOSED_SPEND_CEILING_USD,
    SPEND_RESERVE_USD,
)


class SpendCeilingError(RuntimeError):
    """Raised when a new paid request would exceed the authorized ceiling."""


class SpendLedger:
    def __init__(
        self,
        path,
        *,
        ceiling_usd: float = PROPOSED_SPEND_CEILING_USD,
        reserve_usd: float = SPEND_RESERVE_USD,
    ) -> None:
        self.path = path
        self.ceiling_usd = float(ceiling_usd)
        self.reserve_usd = float(reserve_usd)
        if path.exists():
            payload = read_json(path)
            self.actual_usd = float(payload.get("actual_usd") or 0.0)
            self.pending_usd = float(payload.get("pending_usd") or 0.0)
            self.entries = list(payload.get("entries") or [])
        else:
            self.actual_usd = 0.0
            self.pending_usd = 0.0
            self.entries = []

    def bound_usd(self) -> float:
        return self.actual_usd + self.pending_usd

    def remaining_usd(self) -> float:
        return self.ceiling_usd - self.reserve_usd - self.bound_usd()

    def authorize(self, estimate_usd: float, *, ref: str) -> None:
        estimate = max(0.0, float(estimate_usd))
        if self.bound_usd() + estimate > self.ceiling_usd - self.reserve_usd:
            raise SpendCeilingError(
                f"refusing {ref}: bound {self.bound_usd():.6f} + estimate {estimate:.6f} "
                f"exceeds ceiling {self.ceiling_usd:.2f} minus reserve {self.reserve_usd:.2f}"
            )
        self.pending_usd += estimate
        self.entries.append(
            {
                "event": "authorize",
                "ref": ref,
                "estimate_usd": estimate,
                "utc": utcnow(),
            }
        )
        self._save()

    def settle(self, estimate_usd: float, actual_usd: float, *, ref: str) -> None:
        estimate = max(0.0, float(estimate_usd))
        actual = max(0.0, float(actual_usd))
        self.pending_usd = max(0.0, self.pending_usd - estimate)
        self.actual_usd += actual
        self.entries.append(
            {
                "event": "settle",
                "ref": ref,
                "estimate_usd": estimate,
                "actual_usd": actual,
                "utc": utcnow(),
            }
        )
        self._save()

    def fail(self, estimate_usd: float, actual_usd: float, *, ref: str) -> None:
        estimate = max(0.0, float(estimate_usd))
        actual = max(0.0, float(actual_usd))
        self.pending_usd = max(0.0, self.pending_usd - estimate)
        self.actual_usd += actual
        self.entries.append(
            {
                "event": "fail",
                "ref": ref,
                "estimate_usd": estimate,
                "actual_usd": actual,
                "utc": utcnow(),
            }
        )
        self._save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "ceiling_usd": self.ceiling_usd,
            "reserve_usd": self.reserve_usd,
            "actual_usd": self.actual_usd,
            "pending_usd": self.pending_usd,
            "bound_usd": self.bound_usd(),
            "remaining_usd": self.remaining_usd(),
            "n_entries": len(self.entries),
        }

    def _save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "ceiling_usd": self.ceiling_usd,
                "reserve_usd": self.reserve_usd,
                "actual_usd": self.actual_usd,
                "pending_usd": self.pending_usd,
                "entries": self.entries[-5000:],
                "updated_utc": utcnow(),
            },
        )
