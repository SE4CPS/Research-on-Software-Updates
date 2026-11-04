"""Analysis dictionary for TPS/GDS sentiment measures on the 6,425-ID table."""

from __future__ import annotations

from typing import Any

# Two sentiment definitions must not be pooled for cadence/reply measures.
FROZEN_DEF = "frozen_broad_release_train_package_scores"
HISTORICAL_DEF = "historical_542_harmonized_package_comparison"

MEASURE_DICTIONARY: dict[str, dict[str, Any]] = {
    "title_compound": {
        "source_field": "sentiment.title_compound",
        "construction": "VADER compound of saved title text (package post score).",
        "eligible_when": "title_nonempty is true and value is not null",
        "missingness": "Empty/missing title → null (not VADER 0). Never treat package placeholders as observed neutral.",
        "definitions": [FROZEN_DEF, HISTORICAL_DEF],
        "pool_across_definitions": "only_if_subset_labeled_and_construction_checked; prefer separate or frozen-primary",
    },
    "body_compound": {
        "source_field": "sentiment.body_compound",
        "construction": "VADER compound of saved body/selftext.",
        "eligible_when": "body_nonempty is true and value is not null",
        "missingness": "Empty body → null, excluded from body and body−title paired contrasts.",
        "definitions": [FROZEN_DEF, HISTORICAL_DEF],
        "pool_across_definitions": "prefer frozen-primary; historical body often sparse",
    },
    "combined_compound": {
        "source_field": "sentiment.combined_compound",
        "construction": (
            f"{FROZEN_DEF}: package combined original-post score. "
            f"{HISTORICAL_DEF}: harmonized package_post_concat_compound. "
            "Constructions are related but not proven identical; report frozen as primary."
        ),
        "eligible_when": "value is not null",
        "missingness": "Null if unavailable; zeros may be true VADER zeros when title/body nonempty—do not invent zeros for missing text.",
        "definitions": [FROZEN_DEF, HISTORICAL_DEF],
        "pool_across_definitions": False,
    },
    "body_minus_title": {
        "source_field": "sentiment.body_minus_title (frozen) or derived body−title when both observed",
        "construction": "Paired difference body_compound − title_compound on posts with both observed.",
        "eligible_when": "title and body compounds both non-null",
        "missingness": "Unpaired when body empty.",
        "definitions": [FROZEN_DEF],
        "pool_across_definitions": False,
        "workstream_i_note": (
            "Workstream I showed full-text body−title contrasts are length-sensitive; "
            "equal-length prefix estimands are a different outcome and are not mixed here."
        ),
    },
    "author_mean": {
        "source_field": "sentiment.author_mean",
        "construction": "Mean VADER compound of nonempty original-author replies.",
        "eligible_when": "n_author_replies_nonempty > 0 and author_mean not null",
        "missingness": "No saved author replies → null (not 0).",
        "definitions": [FROZEN_DEF, HISTORICAL_DEF],
        "pool_across_definitions": False,
    },
    "community_mean": {
        "source_field": "sentiment.community_mean",
        "construction": "Mean VADER compound of nonempty non-author replies.",
        "eligible_when": "n_community_replies_nonempty > 0 and community_mean not null",
        "missingness": "No saved community replies → null.",
        "definitions": [FROZEN_DEF, HISTORICAL_DEF],
        "pool_across_definitions": False,
    },
    "author_shift": {
        "source_field": "sentiment.author_shift",
        "construction": "Author-reply mean minus combined original-post compound (frozen package).",
        "eligible_when": "author_mean and combined_compound both observed",
        "missingness": "Null when author replies absent.",
        "definitions": [FROZEN_DEF, HISTORICAL_DEF],
        "pool_across_definitions": False,
    },
    "first_reply_compound": {
        "source_field": "sentiment.first_reply_compound",
        "construction": "VADER compound of first nonempty saved reply in stored order.",
        "eligible_when": "first_reply_compound not null",
        "missingness": "Unavailable on historical_542 definition in this table.",
        "definitions": [FROZEN_DEF],
        "pool_across_definitions": False,
    },
    "last_reply_compound": {
        "source_field": "sentiment.last_reply_compound",
        "construction": "VADER compound of last nonempty saved reply.",
        "eligible_when": "last_reply_compound not null",
        "missingness": "Unavailable on historical_542 definition.",
        "definitions": [FROZEN_DEF],
        "pool_across_definitions": False,
    },
    "last_minus_first": {
        "source_field": "sentiment.last_minus_first",
        "construction": "last_reply_compound − first_reply_compound among order-eligible threads.",
        "eligible_when": "order_eligible and last_minus_first not null",
        "missingness": "Does not measure issue resolution.",
        "definitions": [FROZEN_DEF],
        "pool_across_definitions": False,
    },
    "order_slope_normalized": {
        "source_field": "sentiment.order_slope_normalized",
        "construction": "Normalized reply-order slope of nonempty reply compounds.",
        "eligible_when": "order_eligible and order_slope_normalized not null",
        "missingness": "Unavailable historically; not a resolution proxy.",
        "definitions": [FROZEN_DEF],
        "pool_across_definitions": False,
    },
    "elapsed_hours": {
        "source_field": "sentiment.elapsed_hours",
        "construction": "Hours from post timestamp to last timed nonempty reply when elapsed_eligible.",
        "eligible_when": "elapsed_eligible is true and elapsed_hours not null",
        "missingness": "Invalid/missing timestamps → ineligible.",
        "definitions": [FROZEN_DEF],
        "pool_across_definitions": False,
    },
}

EXPECTED_COUNTS = {
    "n_frame": 6425,
    "relevance": {"relevant": 4944, "not_relevant": 1352, "unclear": 50, "unassessable": 79},
    "relevant_discussion": {"TPS": 3052, "GDS": 242, "Other": 1642, "Unclear": 8},
    "sentiment_definition": {FROZEN_DEF: 5905, HISTORICAL_DEF: 520},
    "existing_decisions": 1964,
    "new_model_decisions": 4461,
    "protected_overlap": 0,
    "full_dataset_jsonl_sha256": "99b575277fb3c9acc4a116ceb2ea9ec44efb2a0875ebdb088240cc2cf2d5aefd",
    "full_dataset_csv_sha256": "39a500f2332c17bc041000db7a66e7bc360b14cd6ecde70ef2a7720288867124",
}
