"""Master research corrections 2026-11-03 (Codex-required extensions).

Does not modify results/master_research_2026-11-03/ or Stage-1 VADER.
"""

RUN_ID = "master-research-corrections-2026-11-03"
SEED = 20261103
N_BOOT = 5000
# Independent TextBlob interval seed (distinct from original Phase C seed)
TB_BOOT_SEED = 20261103 + 777
# Predeclared near-duplicate thresholds (chosen before seeing test metrics / labels)
# Exact normalized full text always merges.
# Shared nontrivial exact title + body word-set Jaccard >= TITLE_BODY_JACCARD merges.
TITLE_MIN_CHARS = 20
TITLE_BODY_JACCARD = 0.55
# Full-text word-set Jaccard for rare same-prefix hash collisions audit only (not primary)
FULLTEXT_JACCARD_AUDIT = 0.85

__all__ = [
    "RUN_ID",
    "SEED",
    "N_BOOT",
    "TB_BOOT_SEED",
    "TITLE_MIN_CHARS",
    "TITLE_BODY_JACCARD",
    "FULLTEXT_JACCARD_AUDIT",
]
