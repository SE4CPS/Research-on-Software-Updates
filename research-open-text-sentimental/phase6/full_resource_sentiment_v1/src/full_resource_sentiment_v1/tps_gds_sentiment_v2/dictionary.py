"""Corrected score dictionary for Stage-1 VADER TPS/GDS analysis."""

from __future__ import annotations

from typing import Any

RUN_ID = "tps-gds-sentiment-v2-stage1-2026-11-03"
FROZEN_DEF = "frozen_broad_release_train_package_scores"
HISTORICAL_DEF = "historical_542_harmonized_package_comparison"
SEED = 20261103
N_BOOT = 10000

# Prespecified length bands (characters). Do not change after looking at contrasts.
BODY_BANDS = (
    ("body_lt_100", lambda n: n < 100),
    ("body_100_to_500", lambda n: 100 <= n <= 500),
    ("body_gt_500", lambda n: n > 500),
)
TITLE_BANDS = (
    ("title_lt_20", lambda n: n < 20),
    ("title_20_to_50", lambda n: 20 <= n <= 50),
    ("title_gt_50", lambda n: n > 50),
)

MEASURE_DICTIONARY: dict[str, dict[str, Any]] = {
    "title_compound": {
        "source": "sentiment.title_compound / post_scores.title_compound",
        "construction": "VADER compound of saved title (releasetrain-sentiment 0.2.1).",
        "eligible_when": "title_nonempty and value not null",
        "missingness": "Empty title → null, never treated as observed zero-neutral for missing text.",
    },
    "body_compound": {
        "source": "sentiment.body_compound / post_scores.body_compound",
        "construction": "VADER compound of saved body (capture author_description).",
        "eligible_when": "body_nonempty and value not null",
        "missingness": "Empty body → null; excluded from body and length-adjusted body contrasts.",
    },
    "combined_compound": {
        "source": "sentiment.combined_compound",
        "construction": "Package combined original-post score from title+body.",
        "eligible_when": "value not null on frozen definition",
        "missingness": "Historical-definition combined uses a separate construction; not pooled here.",
    },
    "author_mean": {
        "source": "sentiment.author_mean",
        "construction": "Mean VADER compound of nonempty original-author replies.",
        "eligible_when": "n_author_replies_nonempty >= 1 and author_mean not null",
        "missingness": "No author replies → null (not zero).",
    },
    "author_shift": {
        "source": "sentiment.author_shift via scoring.early_late_or_none",
        "construction": (
            "Later-half mean minus earlier-half mean of nonempty author-reply compounds "
            "in chronological/package order. Requires >= 4 nonempty author replies."
        ),
        "eligible_when": "n_author_replies_nonempty >= 4 and author_shift not null",
        "missingness": "Fewer than four author replies → null.",
        "not_equal_to": "original_post_to_author_mean",
        "scientific_status_note": (
            "Prior v1 report mislabeled this estimand in the dictionary. "
            "Numeric values already matched early/late; treat small-n GDS contrast as exploratory."
        ),
    },
    "original_post_to_author_mean": {
        "source": "derived: author_mean - combined_compound",
        "construction": (
            "Mean of observed author replies minus combined original-post compound. "
            "Distinct from author_shift."
        ),
        "eligible_when": "author_mean and combined_compound both observed (>=1 author reply)",
        "missingness": "Null when author replies absent.",
        "not_equal_to": "author_shift",
    },
    "community_mean": {
        "source": "sentiment.community_mean",
        "construction": "Mean VADER compound of nonempty non-author replies.",
        "eligible_when": "n_community_replies_nonempty >= 1",
        "missingness": "Null when no community replies.",
    },
    "last_minus_first": {
        "source": "sentiment.last_minus_first",
        "construction": "Last timed nonempty reply compound minus first; order_eligible.",
        "eligible_when": "order_eligible and value not null",
        "missingness": "Not a resolution measure.",
    },
    "order_slope_normalized": {
        "source": "sentiment.order_slope_normalized",
        "construction": "Normalized OLS slope of reply compounds vs position; needs >=3 timed replies.",
        "eligible_when": "order_eligible and value not null",
        "missingness": "Not a resolution measure.",
    },
    "elapsed_hours": {
        "source": "sentiment.elapsed_hours",
        "construction": "Hours spanning distinct timed replies when elapsed_eligible.",
        "eligible_when": "elapsed_eligible and value not null",
        "missingness": "Invalid timestamps → ineligible.",
    },
    "length_adjusted_gds_minus_tps": {
        "source": "derived standardized contrast",
        "construction": (
            "Within predeclared length bands, compute mean(GDS)-mean(TPS); "
            "combine with weights = pooled eligible-post share in each band that has both classes. "
            "Bands without both classes are reported, not extrapolated."
        ),
        "body_bands": ["body_lt_100", "body_100_to_500", "body_gt_500"],
        "title_bands": ["title_lt_20", "title_20_to_50", "title_gt_50"],
        "not_a_substitute": "within-post body_minus_title band summaries",
    },
}

EXPECTED = {
    "n_frame": 6425,
    "jsonl_sha256": "99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd",
    "csv_sha256": "39a500f2332c17bc041000db7a66e7bc360b14cd6ecde70ef2a7720288867124",
    "frozen_relevant_tps": 3028,
    "frozen_relevant_gds": 231,
    "sentiment_frozen": 5905,
    "sentiment_historical": 520,
    "protected_overlap": 0,
}
