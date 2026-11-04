"""Strict continuation corrections/extensions 2026-11-03.

Immutable: results/strict_continuation_2026-11-03/, frozen data, Stage-1, 240 packet.
No manuscript, merge, paid API, or large installs without approval.
"""

RUN_ID = "strict-continuation-corrections-2026-11-03"
SEED = 20261103
N_BOOT = 5000
FROZEN_DEF = "frozen_broad_release_train_package_scores"

# Predeclared polarity thresholds for three-way agreement sensitivity
POLARITY_NEUTRAL_ABS = (0.05, 0.10, 0.05)  # VADER abs, TextBlob abs variants documented in outputs

__all__ = ["RUN_ID", "SEED", "N_BOOT", "FROZEN_DEF"]
