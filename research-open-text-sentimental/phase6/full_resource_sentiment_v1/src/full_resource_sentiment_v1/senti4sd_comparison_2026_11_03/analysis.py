"""GDS−TPS contrasts and cross-instrument direction/classification agreement."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.senti4sd_comparison_2026_11_03.common import (
    OUT_ROOT,
    RESTRICTED_ROOT,
    load_jsonl,
    write_json,
    write_jsonl,
)


def _vader_class(v: float | None) -> str | None:
    if v is None:
        return None
    if v >= 0.05:
        return "positive"
    if v <= -0.05:
        return "negative"
    return "neutral"


def _textblob_class(v: float | None) -> str | None:
    if v is None:
        return None
    if v > 0.0:
        return "positive"
    if v < 0.0:
        return "negative"
    return "neutral"


def _cardiff_class(v: float | None) -> str | None:
    if v is None:
        return None
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def _sign_of_delta(d: float | None) -> str | None:
    if d is None:
        return None
    if d > 0:
        return "positive"
    if d < 0:
        return "negative"
    return "zero"


def _pair_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for s in scores:
        by[s["reddit_id"]][s["unit"]] = s
    pairs = []
    for rid, roles in by.items():
        post = roles.get("original_post")
        reply = roles.get("first_author_reply")
        if not post or not reply:
            continue
        if post.get("text_hash") is None or reply.get("text_hash") is None:
            continue
        # Exact join integrity: reddit_id shared; each role keeps its text_hash
        pairs.append(
            {
                "reddit_id": rid,
                "label": post["label"],
                "cohort": post["cohort"],
                "post_text_hash": post["text_hash"],
                "reply_text_hash": reply["text_hash"],
                "post_senti4sd_label": post["senti4sd_label"],
                "reply_senti4sd_label": reply["senti4sd_label"],
                "post_senti4sd_sign": post["senti4sd_sign"],
                "reply_senti4sd_sign": reply["senti4sd_sign"],
                "delta_senti4sd_sign": float(reply["senti4sd_sign"]) - float(post["senti4sd_sign"]),
                "post_vader": post.get("vader"),
                "reply_vader": reply.get("vader"),
                "post_textblob": post.get("textblob"),
                "reply_textblob": reply.get("textblob"),
                "post_cardiff_sign": post.get("cardiff_sign"),
                "reply_cardiff_sign": reply.get("cardiff_sign"),
                "delta_vader": (
                    float(reply["vader"]) - float(post["vader"])
                    if reply.get("vader") is not None and post.get("vader") is not None
                    else None
                ),
                "delta_textblob": (
                    float(reply["textblob"]) - float(post["textblob"])
                    if reply.get("textblob") is not None and post.get("textblob") is not None
                    else None
                ),
                "delta_cardiff_sign": (
                    float(reply["cardiff_sign"]) - float(post["cardiff_sign"])
                    if reply.get("cardiff_sign") is not None and post.get("cardiff_sign") is not None
                    else None
                ),
            }
        )
    return pairs


def _contrast_sign(pairs: list[dict[str, Any]], key: str, seed: int) -> dict[str, Any]:
    g = [float(p[key]) for p in pairs if p["label"] == "GDS" and p.get(key) is not None]
    t = [float(p[key]) for p in pairs if p["label"] == "TPS" and p.get(key) is not None]
    return independent_stratified_mean_diff(g, t, seed=seed, n_boot=N_BOOT)


def _class_agreement(pairs: list[dict[str, Any]], role: str) -> dict[str, Any]:
    """Compare Senti4SD class labels to VADER/TextBlob/Cardiff classes on identical units."""
    out: dict[str, Any] = {}
    for other, mapper in (
        ("vader", lambda p: _vader_class(p.get(f"{role}_vader"))),
        ("textblob", lambda p: _textblob_class(p.get(f"{role}_textblob"))),
        ("cardiff", lambda p: _cardiff_class(p.get(f"{role}_cardiff_sign"))),
    ):
        s4_key = f"{role}_senti4sd_label"
        n = 0
        agree = 0
        conf = Counter()
        missing = 0
        for p in pairs:
            a = p.get(s4_key)
            b = mapper(p)
            if a is None or b is None:
                missing += 1
                continue
            n += 1
            if a == b:
                agree += 1
            conf[(a, b)] += 1
        out[other] = {
            "n_comparable": n,
            "n_missing_other_or_s4": missing,
            "n_agree": agree,
            "pct_agree": (agree / n) if n else None,
            "note": "Classification agreement only — not accuracy; instruments differ in training domain/scale.",
            "confusion_senti4sd_vs_other_top": [
                {"senti4sd": a, "other": b, "n": c} for (a, b), c in conf.most_common(12)
            ],
        }
    return out


def _delta_direction_agreement(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare sign of reply−post change; not magnitude."""
    out: dict[str, Any] = {}
    for other, key in (
        ("vader", "delta_vader"),
        ("textblob", "delta_textblob"),
        ("cardiff_sign", "delta_cardiff_sign"),
    ):
        n = 0
        agree = 0
        missing = 0
        conf = Counter()
        for p in pairs:
            a = _sign_of_delta(p.get("delta_senti4sd_sign"))
            b = _sign_of_delta(p.get(key))
            if a is None or b is None:
                missing += 1
                continue
            n += 1
            if a == b:
                agree += 1
            conf[(a, b)] += 1
        out[other] = {
            "n_comparable": n,
            "n_missing": missing,
            "n_same_direction_or_zero": agree,
            "pct_same_direction_or_zero": (agree / n) if n else None,
            "note": "Direction/zero agreement of deltas only — not magnitude and not accuracy.",
            "confusion_senti4sd_vs_other_top": [
                {"senti4sd_dir": a, "other_dir": b, "n": c} for (a, b), c in conf.most_common(12)
            ],
        }
    return out


def analyze_cohort(cohort_tag: str, seed: int) -> dict[str, Any]:
    scores = load_jsonl(RESTRICTED_ROOT / f"{cohort_tag}_senti4sd_scores.jsonl")
    fail_path = RESTRICTED_ROOT / f"{cohort_tag}_senti4sd_failures.jsonl"
    failures = load_jsonl(fail_path) if fail_path.exists() else []
    pairs = _pair_rows(scores)
    write_jsonl(RESTRICTED_ROOT / f"{cohort_tag}_paired_rows.jsonl", pairs)

    post_rows = [{"label": p["label"], "post_senti4sd_sign": p["post_senti4sd_sign"]} for p in pairs]
    delta_rows = [{"label": p["label"], "delta_senti4sd_sign": p["delta_senti4sd_sign"]} for p in pairs]

    report = {
        "cohort": cohort_tag,
        "n_score_rows": len(scores),
        "n_pairs": len(pairs),
        "n_gds": sum(1 for p in pairs if p["label"] == "GDS"),
        "n_tps": sum(1 for p in pairs if p["label"] == "TPS"),
        "n_failures_recorded": len(failures),
        "join_rule": "exact reddit_id + per-unit text_hash (post and reply joined on reddit_id)",
        "senti4sd_post_gds_minus_tps": _contrast_sign(post_rows, "post_senti4sd_sign", seed),
        "senti4sd_first_author_minus_post_gds_minus_tps": _contrast_sign(
            delta_rows, "delta_senti4sd_sign", seed + 1
        ),
        "classification_agreement_on_posts": _class_agreement(pairs, "post"),
        "classification_agreement_on_replies": _class_agreement(pairs, "reply"),
        "delta_direction_agreement": _delta_direction_agreement(pairs),
        "label_distribution_posts": dict(Counter(p["post_senti4sd_label"] for p in pairs)),
        "label_distribution_replies": dict(Counter(p["reply_senti4sd_label"] for p in pairs)),
        "interpretation_guardrails": [
            "Do not compare raw score magnitudes across instruments.",
            "Classification/direction agreement is not accuracy.",
            "Senti4SD labels are SE-domain polarity classes mapped to signs (−1/0/+1) for contrasts only.",
            "AI labels unconfirmed where applicable; cohorts kept separate.",
        ],
    }
    write_json(OUT_ROOT / f"{cohort_tag}_ANALYSIS.json", report)
    return report
