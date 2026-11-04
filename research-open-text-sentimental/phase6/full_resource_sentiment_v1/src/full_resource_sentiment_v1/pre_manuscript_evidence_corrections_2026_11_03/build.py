"""Implement Codex CONDITIONAL PASS corrections only."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.pre_manuscript_evidence_corrections_2026_11_03.common import (
    AI_NEW_CANDIDATES,
    ANALYTICAL_UNION,
    AUDIT_OUT,
    BROAD_CAPTURE,
    CAPTURE_LINKED_ORIGINAL,
    CORR_OUT,
    DATASET_JSONL,
    EXPECTED_JSONL,
    FROZEN_DEF,
    HIST_DEF,
    HISTORICAL_IN_ORIGINAL,
    LENGTH_OUT,
    ORIGINAL_SELECTED,
    OUT_ROOT,
    PRIOR_RECON,
    PROTECTED_IDS,
    PROTECTED_IDS_N,
    SCREENED_EXCLUDED,
    SENTI4SD_OUT,
    load_json,
    load_jsonl,
    sha256_file,
    utcnow,
    write_json,
)

MIXED_ORIGIN_LABEL = "mixed-origin saved purpose labels"
AI_UNCONFIRMED = "AI_purpose_labels_unconfirmed_exploratory"


def verify_frozen_provenance() -> dict[str, Any]:
    """Verify Codex breakdown against decision_source on frozen comparable 3,259."""
    expected = {
        "TPS": {
            "original_f_review": 1628,
            "gpt-6.1-terra_ai_classification": 1346,
            "manu_review_ai_assisted_entry": 31,
            "codex_ai_assisted_pilot": 23,
        },
        "GDS": {
            "original_f_review": 57,
            "gpt-6.1-terra_ai_classification": 155,
            "manu_review_ai_assisted_entry": 6,
            "codex_ai_assisted_pilot": 13,
        },
    }
    observed: dict[str, Counter] = {"TPS": Counter(), "GDS": Counter()}
    n = 0
    for r in load_jsonl(DATASET_JSONL):
        if r.get("final_relevance") != "relevant":
            continue
        if r.get("final_discussion") not in {"TPS", "GDS"}:
            continue
        if r.get("sentiment_definition") != FROZEN_DEF:
            continue
        n += 1
        observed[r["final_discussion"]][r.get("decision_source")] += 1

    observed_plain = {lab: dict(cnt) for lab, cnt in observed.items()}
    match = observed_plain == expected and n == 3259
    totals = {
        "n_frozen_comparable": n,
        "n_tps": sum(observed["TPS"].values()),
        "n_gds": sum(observed["GDS"].values()),
        "human_f_review_total": observed["TPS"]["original_f_review"] + observed["GDS"]["original_f_review"],
        "ai_assisted_or_model_total": n
        - (observed["TPS"]["original_f_review"] + observed["GDS"]["original_f_review"]),
    }
    report = {
        "utc": utcnow(),
        "field_used": "decision_source",
        "cohort": "frozen comparable RQ1 posts (sentiment_definition=frozen_broad_release_train_package_scores; relevant TPS/GDS)",
        "expected_codex_breakdown": expected,
        "observed_breakdown": observed_plain,
        "match_expected": match,
        "totals": totals,
        "wording_rule": (
            "Do not call this cohort entirely human-reviewed. Use "
            f"'{MIXED_ORIGIN_LABEL}'. Human F-review is only a subset "
            f"({totals['human_f_review_total']}/{n})."
        ),
        "display_labels": {
            "original_f_review": "F review (human)",
            "gpt-6.1-terra_ai_classification": "Terra AI classification",
            "manu_review_ai_assisted_entry": "Manu AI-assisted entry",
            "codex_ai_assisted_pilot": "Codex AI pilot",
        },
    }
    write_json(OUT_ROOT / "tables/FROZEN_COMPARABLE_LABEL_PROVENANCE.json", report)
    return report


def build_corrected_ledger(prov: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild ledger from sealed sources with provenance + RQ2 ID fixes."""
    cliffs = load_json(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json")
    adj = load_json(AUDIT_OUT / "item4_rq1_length_source_sensitivity/RQ1_ADJUSTED_SENSITIVITY.json")
    conv = load_json(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json")
    rq4 = load_json(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json")
    length = load_json(LENGTH_OUT / "length_sensitivity/SUMMARY.json")
    s4_ai = load_json(SENTI4SD_OUT / "ai_assisted_v2_ANALYSIS.json")
    s4_fr = load_json(SENTI4SD_OUT / "original_frozen_ANALYSIS.json")
    dataset = load_json(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json")

    fr = cliffs["frozen_original_posts"]
    ai = cliffs["ai_assisted_v2_excl_exact_train_dups"]
    ai_adj = adj["ai_assisted_v2_excl_exact_train_dups"]
    rows: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        rows.append(dict(kwargs))

    def fmt(c: dict[str, Any]) -> tuple[str, str]:
        lo, hi = c["mean_diff_ci95"]
        return f"{c['mean_diff']:.6f}", f"[{lo:.6f}, {hi:.6f}]"

    add(
        evidence_id="D1",
        rq="DATA",
        cohort="analytical_union_v2",
        label_provenance="mixed: original_6425 (mixed-origin purpose labels) + AI_screen_v2 unconfirmed",
        analysis_unit="reddit_id",
        instrument="n/a",
        score_scale="count",
        n_gds="",
        n_tps="",
        estimand="analytical_union_rows",
        estimate=dataset["n_union_rows"],
        ci95="",
        adjustment_or_sensitivity="removal-corrected v2",
        authority_status="AUTHORITATIVE",
        source_artifact=str(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json"),
        safe_wording="Union is not human-reviewed end-to-end.",
        notes=f"broad={BROAD_CAPTURE}; screened={SCREENED_EXCLUDED}; original={ORIGINAL_SELECTED}; AI new={AI_NEW_CANDIDATES}",
    )

    # Provenance machine row
    add(
        evidence_id="PROV-F-3259",
        rq="META",
        cohort="frozen_score_comparable_RQ1",
        label_provenance=MIXED_ORIGIN_LABEL,
        analysis_unit="original_post",
        instrument="n/a",
        score_scale="counts",
        n_gds=231,
        n_tps=3028,
        estimand="decision_source_breakdown",
        estimate=json_dumps_compact(prov["observed_breakdown"]),
        ci95="",
        adjustment_or_sensitivity="verified against full_dataset.jsonl decision_source",
        authority_status="AUTHORITATIVE",
        source_artifact=str(DATASET_JSONL),
        safe_wording=prov["wording_rule"],
        notes="TPS F1628/Terra1346/Manu31/Codex23; GDS F57/Terra155/Manu6/Codex13",
    )

    for inst, key, scale in (
        ("VADER", "vader", "compound[-1,1]"),
        ("TextBlob", "textblob", "polarity[-1,1]"),
        ("Cardiff", "cardiff_sign", "class_sign{-1,0,+1}"),
    ):
        c = fr[key]
        est, ci = fmt(c)
        add(
            evidence_id=f"RQ1-F-{inst}",
            rq="RQ1",
            cohort="frozen_score_comparable",
            label_provenance=MIXED_ORIGIN_LABEL,
            analysis_unit="original_post (title+body one event)",
            instrument=inst,
            score_scale=scale,
            n_gds=c["n_a"],
            n_tps=c["n_b"],
            estimand="GDS_minus_TPS_mean",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="unadjusted; Cliff's δ corrected sign",
            authority_status="AUTHORITATIVE",
            source_artifact=str(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json"),
            safe_wording="Conditional frozen-score cohort; mixed-origin purpose labels; not all human-reviewed.",
            notes=f"cliffs_delta={c.get('cliffs_delta'):.6f}",
        )

    for inst, key, scale in (
        ("VADER", "vader", "compound[-1,1]"),
        ("TextBlob", "textblob", "polarity[-1,1]"),
        ("Cardiff", "cardiff_sign", "class_sign{-1,0,+1}"),
    ):
        c = ai[key]
        est, ci = fmt(c)
        add(
            evidence_id=f"RQ1-AI-{inst}",
            rq="RQ1",
            cohort="ai_assisted_v2_excl_exact_train_dups",
            label_provenance=AI_UNCONFIRMED,
            analysis_unit="original_post (title+body one event)",
            instrument=inst,
            score_scale=scale,
            n_gds=c["n_a"],
            n_tps=c["n_b"],
            estimand="GDS_minus_TPS_mean",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="unadjusted; 3 exact train-text TPS dups excluded",
            authority_status="AUTHORITATIVE",
            source_artifact=str(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json"),
            safe_wording="Association within AI-predicted purpose labels only.",
            notes=f"cliffs_delta={c.get('cliffs_delta'):.6f}",
        )

    va = ai_adj["vader"]["regression_adjusted_log_length"]
    add(
        evidence_id="RQ1-AI-VADER-adj",
        rq="RQ1",
        cohort="ai_assisted_v2_excl_exact_train_dups",
        label_provenance=AI_UNCONFIRMED,
        analysis_unit="original_post",
        instrument="VADER",
        score_scale="compound[-1,1]",
        n_gds=va["n_gds"],
        n_tps=va["n_tps"],
        estimand="GDS_coef_log1p_chars_adjusted",
        estimate=f"{va['gds_coef_adjusted']:.6f}",
        ci95=f"[{va['gds_coef_ci95'][0]:.6f}, {va['gds_coef_ci95'][1]:.6f}]",
        adjustment_or_sensitivity="OLS: score ~ GDS + log1p(chars)",
        authority_status="AUTHORITATIVE_SENSITIVITY",
        source_artifact=str(AUDIT_OUT / "item4_rq1_length_source_sensitivity/RQ1_ADJUSTED_SENSITIVITY.json"),
        safe_wording="Length-adjusted association; not causal.",
        notes="",
    )

    for cohort, rep, eid, lab in (
        ("ai_assisted_v2_reply_eligible", s4_ai, "RQ1-AI-Senti4SD-replyelig", AI_UNCONFIRMED),
        ("frozen_score_reply_eligible", s4_fr, "RQ1-F-Senti4SD-replyelig", MIXED_ORIGIN_LABEL),
    ):
        c = rep["senti4sd_post_gds_minus_tps"]
        est, ci = fmt(c)
        add(
            evidence_id=eid,
            rq="RQ1",
            cohort=cohort,
            label_provenance=lab,
            analysis_unit="original_post | conditioned on first-author reply existing",
            instrument="Senti4SD",
            score_scale="class_sign{-1,0,+1}",
            n_gds=c["n_a"],
            n_tps=c["n_b"],
            estimand="GDS_minus_TPS_mean_sign",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="REPLY-ELIGIBLE SUBSET ONLY — not full RQ1 population",
            authority_status="AUTHORITATIVE_FOR_REPLY_ELIGIBLE_FOUR_MODEL",
            source_artifact=str(SENTI4SD_OUT / ("ai_assisted_v2_ANALYSIS.json" if "ai" in cohort else "original_frozen_ANALYSIS.json")),
            safe_wording="Do not replace larger three-model full-post RQ1 estimates with this subset.",
            notes="Matched four-model RQ1 only.",
        )

    # --- Correction 2: distinct RQ2 VADER IDs ---
    c_audit = conv["rq2_ai_first_author_minus_post"]
    est_a, ci_a = fmt(c_audit)
    add(
        evidence_id="RQ2-AI-VADER-full-audit-bootstrap",
        rq="RQ2",
        cohort="ai_reply_eligible",
        label_provenance=AI_UNCONFIRMED,
        analysis_unit="first_author_reply_minus_original_post",
        instrument="VADER",
        score_scale="compound delta",
        n_gds=c_audit["n_a"],
        n_tps=c_audit["n_b"],
        estimand="GDS_minus_TPS_of_paired_delta",
        estimate=est_a,
        ci95=ci_a,
        adjustment_or_sensitivity="full text; thread-stratified bootstrap seed from audit item3",
        authority_status="ALTERNATE_BOOTSTRAP_SUPERSEDED_AS_CANONICAL",
        source_artifact=str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
        safe_wording="Same estimand/mean as canonical; earlier independent bootstrap CI. Not discarded as invalid.",
        notes="CI [-0.178083, -0.048827]; mean identical to length-sensitivity within rounding.",
    )

    c_len = length["ai"]["full_vader"]
    est_l, ci_l = fmt(c_len)
    add(
        evidence_id="RQ2-AI-VADER-full",
        rq="RQ2",
        cohort="ai_reply_eligible",
        label_provenance=AI_UNCONFIRMED,
        analysis_unit="first_author_reply_minus_original_post",
        instrument="VADER",
        score_scale="compound delta",
        n_gds=c_len["n_a"],
        n_tps=c_len["n_b"],
        estimand="GDS_minus_TPS_of_paired_delta",
        estimate=est_l,
        ci95=ci_l,
        adjustment_or_sensitivity="full text; CANONICAL manuscript comparison (length-sensitivity tree)",
        authority_status="AUTHORITATIVE_CANONICAL",
        source_artifact=str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
        safe_wording="Canonical because it is the later sealed tree that also hosts prefix/adjusted sensitivities on the same pairs.",
        notes="CI [-0.178935, -0.050006]; estimate -0.114709. Justification: see CANONICAL_VADER_RQ2.md",
    )

    c_fr = conv["rq2_original_first_author_minus_post"]
    est, ci = fmt(c_fr)
    add(
        evidence_id="RQ2-F-VADER-full",
        rq="RQ2",
        cohort="frozen_score_reply_eligible",
        label_provenance=MIXED_ORIGIN_LABEL,
        analysis_unit="first_author_reply_minus_original_post",
        instrument="VADER",
        score_scale="compound delta",
        n_gds=c_fr["n_a"],
        n_tps=c_fr["n_b"],
        estimand="GDS_minus_TPS_of_paired_delta",
        estimate=est,
        ci95=ci,
        adjustment_or_sensitivity="full saved text",
        authority_status="AUTHORITATIVE",
        source_artifact=str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
        safe_wording="Observed reply-rich threads; mixed-origin labels; ≠ issue resolution.",
        notes="",
    )

    ai_len = length["ai"]
    for inst, full_key, prefix_key, scale in (
        ("TextBlob", "full_textblob", "prefix_textblob", "polarity delta"),
        ("Cardiff", "full_cardiff", "prefix_cardiff", "sign delta"),
    ):
        c = ai_len[full_key]
        est, ci = fmt(c)
        add(
            evidence_id=f"RQ2-AI-{inst}-full",
            rq="RQ2",
            cohort="ai_reply_eligible",
            label_provenance=AI_UNCONFIRMED,
            analysis_unit="first_author_reply_minus_post",
            instrument=inst,
            score_scale=scale,
            n_gds=c["n_a"],
            n_tps=c["n_b"],
            estimand="GDS_minus_TPS_of_paired_delta",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="full text",
            authority_status="AUTHORITATIVE",
            source_artifact=str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
            safe_wording="Instrument-specific; do not pool magnitudes across scales.",
            notes="",
        )
        c2 = ai_len[prefix_key]
        est, ci = fmt(c2)
        add(
            evidence_id=f"RQ2-AI-{inst}-prefix",
            rq="RQ2",
            cohort="ai_reply_eligible",
            label_provenance=AI_UNCONFIRMED,
            analysis_unit="first_author_reply_minus_post",
            instrument=inst,
            score_scale=scale,
            n_gds=c2["n_a"],
            n_tps=c2["n_b"],
            estimand="GDS_minus_TPS_of_paired_delta",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="equal-length char prefixes (different text portion)",
            authority_status="AUTHORITATIVE_SENSITIVITY",
            source_artifact=str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
            safe_wording="Prefix ≠ full-text estimand.",
            notes="",
        )

    # VADER prefix + adjusted as sensitivities alongside canonical full
    c2 = ai_len["prefix_vader"]
    est, ci = fmt(c2)
    add(
        evidence_id="RQ2-AI-VADER-prefix",
        rq="RQ2",
        cohort="ai_reply_eligible",
        label_provenance=AI_UNCONFIRMED,
        analysis_unit="first_author_reply_minus_post",
        instrument="VADER",
        score_scale="compound delta",
        n_gds=c2["n_a"],
        n_tps=c2["n_b"],
        estimand="GDS_minus_TPS_of_paired_delta",
        estimate=est,
        ci95=ci,
        adjustment_or_sensitivity="equal-length char prefixes (different text portion)",
        authority_status="AUTHORITATIVE_SENSITIVITY",
        source_artifact=str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
        safe_wording="Prefix ≠ full-text estimand.",
        notes="",
    )
    va_adj = ai_len["adjusted_vader"]
    add(
        evidence_id="RQ2-AI-VADER-adj",
        rq="RQ2",
        cohort="ai_reply_eligible",
        label_provenance=AI_UNCONFIRMED,
        analysis_unit="first_author_reply_minus_post",
        instrument="VADER",
        score_scale="compound delta",
        n_gds=va_adj["n_gds"],
        n_tps=va_adj["n_tps"],
        estimand="GDS_coef_length_source_date_adjusted",
        estimate=f"{va_adj['gds_coef_adjusted']:.6f}",
        ci95=f"[{va_adj['gds_coef_ci95'][0]:.6f}, {va_adj['gds_coef_ci95'][1]:.6f}]",
        adjustment_or_sensitivity="OLS delta ~ GDS + log post/reply len + source + date",
        authority_status="AUTHORITATIVE_SENSITIVITY",
        source_artifact=str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
        safe_wording="Sensitivity; not a causal trajectory proof.",
        notes="",
    )

    c = s4_ai["senti4sd_first_author_minus_post_gds_minus_tps"]
    est, ci = fmt(c)
    add(
        evidence_id="RQ2-AI-Senti4SD",
        rq="RQ2",
        cohort="ai_reply_eligible",
        label_provenance=AI_UNCONFIRMED,
        analysis_unit="first_author_reply_minus_post",
        instrument="Senti4SD",
        score_scale="sign delta",
        n_gds=c["n_a"],
        n_tps=c["n_b"],
        estimand="GDS_minus_TPS_of_paired_sign_delta",
        estimate=est,
        ci95=ci,
        adjustment_or_sensitivity="full text class signs; newlines collapsed for CSV",
        authority_status="AUTHORITATIVE",
        source_artifact=str(SENTI4SD_OUT / "ai_assisted_v2_ANALYSIS.json"),
        safe_wording="Direction/class-sign only; not magnitude vs VADER.",
        notes="",
    )

    c = s4_fr["senti4sd_first_author_minus_post_gds_minus_tps"]
    est, ci = fmt(c)
    add(
        evidence_id="RQ2-F-Senti4SD",
        rq="RQ2",
        cohort="frozen_score_reply_eligible",
        label_provenance=MIXED_ORIGIN_LABEL,
        analysis_unit="first_author_reply_minus_post",
        instrument="Senti4SD",
        score_scale="sign delta",
        n_gds=c["n_a"],
        n_tps=c["n_b"],
        estimand="GDS_minus_TPS_of_paired_sign_delta",
        estimate=est,
        ci95=ci,
        adjustment_or_sensitivity="BOOTSTRAP-SENSITIVE; do not treat as robust nonzero",
        authority_status="AUTHORITATIVE_WITH_UNCERTAINTY_CAVEAT",
        source_artifact=str(SENTI4SD_OUT / "original_frozen_ANALYSIS.json"),
        safe_wording="Small frozen GDS n=98; CI borderline/bootstrap-sensitive across seeds. Not a robust conclusion.",
        notes="See SENTI4SD_FROZEN_RQ2_UNCERTAINTY.md; sealed CI here vs reconciliation resample discrepancy.",
    )

    for eid, key, lab in (
        ("RQ2-AI-second_minus_first", "rq2_ai_second_minus_first", AI_UNCONFIRMED),
        ("RQ2-AI-ge4", "rq2_ai_author_ge4_shift", AI_UNCONFIRMED),
    ):
        c = conv[key]
        est, ci = fmt(c)
        add(
            evidence_id=eid,
            rq="RQ2",
            cohort="ai_deeper_reply_eligible",
            label_provenance=lab,
            analysis_unit=key.replace("rq2_ai_", ""),
            instrument="VADER",
            score_scale="compound delta",
            n_gds=c["n_a"],
            n_tps=c["n_b"],
            estimand="GDS_minus_TPS",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="different eligibility than first-reply",
            authority_status="AUTHORITATIVE_SECONDARY",
            source_artifact=str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
            safe_wording="Do not collapse with first-reply estimand.",
            notes="",
        )

    for eid, key, cohort, lab in (
        ("RQ3-AI", "rq3_ai_author_minus_community", "ai_both_roles", AI_UNCONFIRMED),
        ("RQ3-F", "rq3_original_author_minus_community", "frozen_score_both_roles", MIXED_ORIGIN_LABEL),
    ):
        c = conv[key]
        est, ci = fmt(c)
        add(
            evidence_id=eid,
            rq="RQ3",
            cohort=cohort,
            label_provenance=lab,
            analysis_unit="author_mean_minus_community_mean_per_thread",
            instrument="VADER",
            score_scale="compound gap",
            n_gds=c["n_a"],
            n_tps=c["n_b"],
            estimand="GDS_minus_TPS_of_role_gap",
            estimate=est,
            ci95=ci,
            adjustment_or_sensitivity="thread-level gap then purpose contrast",
            authority_status="AUTHORITATIVE",
            source_artifact=str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
            safe_wording="Bounded descriptive; intervals include 0. Own eligibility (not RQ2 N).",
            notes=f"AI both-role n=5879; frozen both-role n=1032",
        )

    models = rq4["models"]
    for mid, label in (
        ("text_balanced", "text_only_balanced"),
        ("text_plus_vader_tb_balanced", "text+VADER+TextBlob_balanced"),
        ("text_plus_vader_tb_cardiff_balanced", "text+VADER+TB+Cardiff_balanced"),
    ):
        m = models[mid]
        add(
            evidence_id=f"RQ4-{mid}",
            rq="RQ4",
            cohort="fixed_grouped_split_v2",
            label_provenance="mixed_origin_saved_labels",
            analysis_unit="test_rows",
            instrument=label,
            score_scale="macro_F1",
            n_gds="64 GDS test",
            n_tps="754 TPS test",
            estimand="balanced_macro_F1",
            estimate=f"{m['macro_f1']:.6f}",
            ci95="",
            adjustment_or_sensitivity=f"train/val/test={rq4['n_train']}/{rq4['n_val']}/{rq4['n_test']}",
            authority_status="AUTHORITATIVE",
            source_artifact=str(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json"),
            safe_wording="Agreement with saved labels only — not human field accuracy.",
            notes="",
        )
    boot = rq4["component_bootstrap_paired_macro_f1"]
    for eid, key in (
        ("RQ4-boot-2feat_minus_text", "two_feature_minus_text_balanced"),
        ("RQ4-boot-cardiff_minus_text", "cardiff_inclusive_minus_text_balanced"),
    ):
        b = boot[key]
        add(
            evidence_id=eid,
            rq="RQ4",
            cohort="fixed_grouped_split_v2",
            label_provenance="mixed_origin_saved_labels",
            analysis_unit="near_dup_components",
            instrument=key,
            score_scale="macro_F1_diff",
            n_gds="",
            n_tps="",
            estimand="paired_component_bootstrap_mean_diff",
            estimate=f"{b['mean']:.6f}",
            ci95=f"[{b['ci95'][0]:.6f}, {b['ci95'][1]:.6f}]",
            adjustment_or_sensitivity=f"n_components={b['n_test_components']}; prior row-bootstrap SUPERSEDED",
            authority_status="AUTHORITATIVE",
            source_artifact=str(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json"),
            safe_wording="Component bootstrap replaces superseded row bootstrap.",
            notes="",
        )

    for eid, note in (
        ("SUPERSEDED-human_frozen_shorthand", "Prior ledger 'human/prior_frozen_labels' / 'frozen_labels' shorthand superseded by mixed-origin wording."),
        ("SUPERSEDED-RQ2-AI-VADER-full-duplicate-id", "Duplicate ID RQ2-AI-VADER-full with audit CI superseded as canonical; retained as RQ2-AI-VADER-full-audit-bootstrap."),
        ("SUPERSEDED-union-18543", "Pre-removal-correction union drafts (~18,543) superseded by 18,478 v2."),
        ("SUPERSEDED-Cliff-sign", "Pre-audit Cliff's δ sign convention reversed; means/CIs retained, δ sign corrected."),
        ("SUPERSEDED-Cardiff-trunc-mixed", "Mixed frozen+AI truncated vs frozen-only nontruncated comparison superseded."),
        ("SUPERSEDED-RQ4-row-bootstrap", "RQ4 row-level bootstrap exploratory only; use component bootstrap."),
        ("SUPERSEDED-Senti4SD-as-full-RQ1", "Any reading of Senti4SD post contrast as full 15,312-post RQ1 is incorrect."),
    ):
        add(
            evidence_id=eid,
            rq="META",
            cohort="n/a",
            label_provenance="n/a",
            analysis_unit="n/a",
            instrument="n/a",
            score_scale="n/a",
            n_gds="",
            n_tps="",
            estimand="supersession",
            estimate="SUPERSEDED",
            ci95="",
            adjustment_or_sensitivity="",
            authority_status="SUPERSEDED",
            source_artifact="see notes",
            safe_wording=note,
            notes="Do not cite as current paper number.",
        )

    return rows


def json_dumps_compact(obj: Any) -> str:
    import json as _json

    return _json.dumps(obj, sort_keys=True, separators=(",", ":"))


def write_ledger(rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    csv_path = OUT_ROOT / "tables/AUTHORITATIVE_EVIDENCE_LEDGER_CORRECTED.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Authoritative evidence ledger (CORRECTED) — 2026-11-03",
        "",
        "Codex CONDITIONAL PASS corrections applied. Sealed prior reconciliation tree preserved untouched.",
        "",
        f"**Label rule:** frozen-score cohorts use `{MIXED_ORIGIN_LABEL}` (not 'human/prior frozen labels').",
        "",
        "| ID | RQ | Cohort | Labels | Unit | Instrument | N GDS/TPS | Estimate | CI95 | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if str(r["evidence_id"]).startswith("SUPERSEDED"):
            continue
        lines.append(
            "| {evidence_id} | {rq} | {cohort} | {label_provenance} | {analysis_unit} | {instrument} | {n_gds}/{n_tps} | {estimate} | {ci95} | {authority_status} |".format(
                **{k: str(r[k]).replace("|", "/")[:80] for k in r}
            )
        )
    lines += ["", "## Superseded / provenance", ""]
    for r in rows:
        if str(r["evidence_id"]).startswith("SUPERSEDED") or "SUPERSEDED" in str(r["authority_status"]) or "ALTERNATE" in str(r["authority_status"]):
            lines.append(f"- **{r['evidence_id']}** [{r['authority_status']}]: {r['safe_wording']} {r.get('notes','')}")
    (OUT_ROOT / "tables/AUTHORITATIVE_EVIDENCE_LEDGER_CORRECTED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT_ROOT / "tables/AUTHORITATIVE_EVIDENCE_LEDGER_CORRECTED.json", {"utc": utcnow(), "n_rows": len(rows), "rows": rows})


def build_flow() -> dict[str, Any]:
    # Verify protected count
    payload = load_json(PROTECTED_IDS)
    n_prot = len(payload.get("union_ids") or [])
    # Verify original split
    n_hist = n_capt = 0
    for r in load_jsonl(DATASET_JSONL):
        if r.get("sentiment_definition") == HIST_DEF:
            n_hist += 1
        elif r.get("sentiment_definition") == FROZEN_DEF:
            n_capt += 1
    assert n_hist == HISTORICAL_IN_ORIGINAL
    assert n_capt == CAPTURE_LINKED_ORIGINAL
    assert n_hist + n_capt == ORIGINAL_SELECTED
    assert n_prot == PROTECTED_IDS_N
    assert BROAD_CAPTURE == SCREENED_EXCLUDED + CAPTURE_LINKED_ORIGINAL + PROTECTED_IDS_N
    assert ANALYTICAL_UNION == ORIGINAL_SELECTED + AI_NEW_CANDIDATES

    flow = {
        "utc": utcnow(),
        "arithmetic_checks": {
            "broad_equals_screened_plus_capture_linked_plus_protected": BROAD_CAPTURE
            == SCREENED_EXCLUDED + CAPTURE_LINKED_ORIGINAL + PROTECTED_IDS_N,
            "original_equals_capture_linked_plus_historical": ORIGINAL_SELECTED
            == CAPTURE_LINKED_ORIGINAL + HISTORICAL_IN_ORIGINAL,
            "union_equals_original_plus_ai_new": ANALYTICAL_UNION == ORIGINAL_SELECTED + AI_NEW_CANDIDATES,
            "protected_ids_file_n": n_prot,
            "dataset_historical_def_n": n_hist,
            "dataset_frozen_broad_def_n": n_capt,
        },
        "nodes": [
            {"id": "broad", "label": "Broad capture frame", "n": BROAD_CAPTURE, "role": "universe"},
            {
                "id": "protected",
                "label": "Protected IDs (evaluation isolation)",
                "n": PROTECTED_IDS_N,
                "role": "excluded_from_train_screen",
                "note": "Not used for training or screening; ID-only outcomes unread",
            },
            {
                "id": "screened",
                "label": "Excluded posts screened",
                "n": SCREENED_EXCLUDED,
                "role": "screen_cohort",
            },
            {
                "id": "capture_linked",
                "label": "Capture-linked original selected",
                "n": CAPTURE_LINKED_ORIGINAL,
                "role": "original_subset",
            },
            {
                "id": "historical",
                "label": "Historical records (outside broad capture partition)",
                "n": HISTORICAL_IN_ORIGINAL,
                "role": "external_input_to_original",
                "note": "Not inside the 37,596 broad capture partition",
            },
            {
                "id": "original",
                "label": "Original selected dataset",
                "n": ORIGINAL_SELECTED,
                "role": "original_total",
                "formula": "5,905 capture-linked + 520 historical",
            },
            {
                "id": "ai_new",
                "label": "New AI-predicted TPS/GDS candidates",
                "n": AI_NEW_CANDIDATES,
                "role": "from_screen",
                "note": "After 65 unassessable removals; not human-confirmed",
            },
            {
                "id": "union",
                "label": "Analytical union v2",
                "n": ANALYTICAL_UNION,
                "role": "analysis",
                "formula": "6,425 + 12,053",
                "note": "Not all human-reviewed",
            },
            {"id": "rq1_f", "label": "RQ1 frozen-score comparable", "n": 3259, "n_gds": 231, "n_tps": 3028},
            {"id": "rq1_ai", "label": "RQ1 AI excl exact train dups", "n": 12050, "n_gds": 999, "n_tps": 11051},
            {
                "id": "rq2",
                "label": "RQ2 first-author pairs (4-model)",
                "ai_n": 5942,
                "frozen_n": 1054,
                "total": 6996,
            },
            {"id": "rq3", "label": "RQ3 both-role threads", "ai_n": 5879, "frozen_n": 1032},
            {"id": "rq4", "label": "RQ4 split", "train": 1948, "val": 493, "test": 818},
        ],
        "edges": [
            {"from": "broad", "to": "protected", "relation": "partition_exclude"},
            {"from": "broad", "to": "screened", "relation": "partition_screen"},
            {"from": "broad", "to": "capture_linked", "relation": "partition_original_capture"},
            {"from": "historical", "to": "original", "relation": "external_add (not from broad partition)"},
            {"from": "capture_linked", "to": "original", "relation": "compose"},
            {"from": "screened", "to": "ai_new", "relation": "AI_purpose_screen"},
            {"from": "original", "to": "union", "relation": "compose"},
            {"from": "ai_new", "to": "union", "relation": "compose"},
        ],
    }
    write_json(OUT_ROOT / "tables/DATASET_FLOW_CORRECTED.json", flow)

    md = [
        "# Dataset / eligibility flow (CORRECTED)",
        "",
        "## Partition of the broad capture (37,596)",
        "",
        "| Component | N | Role |",
        "|---|---:|---|",
        f"| Screened excluded posts | {SCREENED_EXCLUDED} | AI purpose screen cohort |",
        f"| Capture-linked original selected | {CAPTURE_LINKED_ORIGINAL} | Inside broad capture |",
        f"| Protected IDs | {PROTECTED_IDS_N} | Isolated; **not** used for training/screening |",
        f"| **Sum** | **{SCREENED_EXCLUDED + CAPTURE_LINKED_ORIGINAL + PROTECTED_IDS_N}** | Equals broad capture |",
        "",
        "## Original selected dataset (6,425)",
        "",
        "| Component | N |",
        "|---|---:|",
        f"| Capture-linked (frozen_broad scores) | {CAPTURE_LINKED_ORIGINAL} |",
        f"| Historical (harmonized package; **outside** broad partition) | {HISTORICAL_IN_ORIGINAL} |",
        f"| **Total original** | **{ORIGINAL_SELECTED}** |",
        "",
        "## Analytical union v2 (18,478)",
        "",
        f"| 6,425 original + {AI_NEW_CANDIDATES} new AI TPS/GDS = **{ANALYTICAL_UNION}**. Not all human-reviewed.",
        "",
        "## RQ eligibility counts",
        "",
        "| RQ | Count cell |",
        "|---|---|",
        "| RQ1 frozen-score comparable | 3,259 (231 GDS / 3,028 TPS) |",
        "| RQ1 AI excl exact train dups | 12,050 (999 / 11,051) |",
        "| RQ2 first-author pairs | AI 5,942; frozen-score 1,054; total 6,996 |",
        "| RQ3 both-role | AI **5,879**; frozen-score **1,032** |",
        "| RQ4 train/val/test | **1,948 / 493 / 818** |",
        "",
        f"Arithmetic checks OK: `{flow['arithmetic_checks']}`.",
    ]
    (OUT_ROOT / "tables/DATASET_FLOW_CORRECTED.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return flow


def draw_flow_figure(flow: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Dataset flow (corrected): broad partition, historical add-on, analytical union",
        fontsize=11,
        pad=12,
    )

    def box(x, y, w, h, text, fc="#E8EEF2"):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=1.2, edgecolor="#333", facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, wrap=True)

    # Row 1: broad
    box(3.5, 8.6, 5, 0.9, f"Broad capture frame\nN = {BROAD_CAPTURE:,}", fc="#D9E2EC")
    # Row 2: three-way partition
    box(0.3, 6.8, 3.5, 1.1, f"Screened excluded\nN = {SCREENED_EXCLUDED:,}\n(AI purpose screen)", fc="#F7D6C8")
    box(4.25, 6.8, 3.5, 1.1, f"Capture-linked original\nN = {CAPTURE_LINKED_ORIGINAL:,}\n(inside broad)", fc="#C8E0D0")
    box(8.2, 6.8, 3.5, 1.1, f"Protected IDs\nN = {PROTECTED_IDS_N}\nNOT train/screen", fc="#F0E0F0")
    # arrows from broad
    for x in (2.0, 6.0, 10.0):
        ax.annotate("", xy=(x, 7.9), xytext=(6.0, 8.6), arrowprops=dict(arrowstyle="->", color="#444", lw=1))
    ax.text(6.0, 8.25, "partition 30,998 + 5,905 + 693", ha="center", fontsize=7, color="#444")

    # Historical external
    box(0.3, 5.0, 3.5, 1.0, f"Historical records\nN = {HISTORICAL_IN_ORIGINAL}\nOUTSIDE broad partition", fc="#FFE6A7")
    ax.annotate("", xy=(5.0, 4.55), xytext=(2.0, 5.0), arrowprops=dict(arrowstyle="->", color="#886600", lw=1.2))
    ax.text(2.8, 4.7, "external add", fontsize=7, color="#886600")

    # Original total
    box(4.0, 3.9, 4.0, 1.1, f"Original selected dataset\nN = {ORIGINAL_SELECTED:,}\n= {CAPTURE_LINKED_ORIGINAL:,} + {HISTORICAL_IN_ORIGINAL}", fc="#C8E0D0")
    ax.annotate("", xy=(6.0, 5.0), xytext=(6.0, 6.8), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    # AI new from screened
    box(0.3, 2.2, 3.5, 1.1, f"New AI TPS/GDS candidates\nN = {AI_NEW_CANDIDATES:,}\n(unconfirmed purpose labels)", fc="#F7D6C8")
    ax.annotate("", xy=(2.0, 3.3), xytext=(2.0, 6.8), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    # Union
    box(4.0, 1.0, 4.0, 1.2, f"Analytical union v2\nN = {ANALYTICAL_UNION:,}\n= {ORIGINAL_SELECTED:,} + {AI_NEW_CANDIDATES:,}\n(not all human-reviewed)", fc="#BFD7EA")
    ax.annotate("", xy=(5.2, 2.2), xytext=(6.0, 3.9), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))
    ax.annotate("", xy=(4.5, 2.2), xytext=(2.5, 2.7), arrowprops=dict(arrowstyle="->", color="#444", lw=1.2))

    # RQ boxes
    box(8.3, 3.6, 3.4, 2.4, "RQ eligibility\nRQ1 frozen-score 3,259\nRQ1 AI 12,050\nRQ2 pairs AI 5,942 / F 1,054\nRQ3 both-role AI 5,879 / F 1,032\nRQ4 1,948/493/818", fc="#FFFFFF")

    fig.tight_layout()
    path = OUT_ROOT / "figures/fig_dataset_flow_CORRECTED.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    (OUT_ROOT / "figures/fig_dataset_flow_CORRECTED_CAPTION.md").write_text(
        "Figure (corrected). Broad capture (37,596) partitions into screened excluded posts (30,998), "
        "capture-linked original selected records (5,905), and protected IDs (693) held out of training/screening. "
        "The original selected dataset (6,425) additionally includes 520 historical records that are **not** part of the "
        "broad-capture partition. The analytical union v2 (18,478) joins the 6,425 original records with 12,053 newly "
        "AI-predicted TPS/GDS candidates from the screen. Purpose labels in the union are mixed-origin / AI-unconfirmed; "
        "the figure does not claim human review of the full union.\n",
        encoding="utf-8",
    )
    return path


def write_canonical_vader_rq2() -> None:
    audit = load_json(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json")[
        "rq2_ai_first_author_minus_post"
    ]
    length = load_json(LENGTH_OUT / "length_sensitivity/SUMMARY.json")["ai"]["full_vader"]
    text = f"""# Canonical VADER RQ2 (AI first-author − post)

## Same estimand, two bootstrap intervals

| ID | Tree | Mean GDS−TPS | 95% CI | N GDS/TPS | Seed / notes |
|---|---|---:|---|---|---|
| `RQ2-AI-VADER-full-audit-bootstrap` | audit_corrections item3 | {audit['mean_diff']:.12f} | {audit['mean_diff_ci95']} | {audit['n_a']}/{audit['n_b']} | seed {audit['seed']}; earlier sealed bootstrap |
| **`RQ2-AI-VADER-full` (CANONICAL)** | sentiment_length_sensitivity SUMMARY | {length['mean_diff']:.12f} | {length['mean_diff_ci95']} | {length['n_a']}/{length['n_b']} | seed {length['seed']}; later sealed tree |

Means agree to floating-point identity for the reported contrast (−0.11470882…). Intervals differ slightly because they are **independent thread-stratified bootstrap resamples with different seeds** in different sealed runs — not different cohorts or different formulas.

## Why the length-sensitivity interval is canonical

1. **Same pairs, later packaging:** The length-sensitivity tree is the sealed home of the full/prefix/adjusted RQ2 suite on these exact 5,942 pairs; manuscript comparisons that cite prefix or adjusted VADER should share that tree’s full-text baseline.
2. **Not chosen for favorability:** The canonical CI [−0.178935, −0.050006] is not narrower in a decision-relevant sense than the audit CI [−0.178083, −0.048827]; both exclude zero. Selection is organizational (one baseline for the sensitivity family), not “pick the nicer interval.”
3. **Audit interval retained:** Kept as `RQ2-AI-VADER-full-audit-bootstrap` with status `ALTERNATE_BOOTSTRAP_SUPERSEDED_AS_CANONICAL` — scientifically valid alternate resample, not an error to delete.

## What this is not

- Not a different estimand (both are GDS−TPS of reply−post VADER on AI reply-eligible pairs).
- Not evidence of data drift between trees (means match).
- Not a license to average or cherry-pick intervals.
"""
    (OUT_ROOT / "CANONICAL_VADER_RQ2.md").write_text(text, encoding="utf-8")
    write_json(
        OUT_ROOT / "CANONICAL_VADER_RQ2.json",
        {
            "canonical_id": "RQ2-AI-VADER-full",
            "canonical": length,
            "alternate_id": "RQ2-AI-VADER-full-audit-bootstrap",
            "alternate": audit,
            "justification": "Later length-sensitivity sealed tree is the shared baseline for full/prefix/adjusted RQ2; audit CI retained as alternate bootstrap.",
        },
    )


def write_senti4sd_uncertainty() -> None:
    s4 = load_json(SENTI4SD_OUT / "original_frozen_ANALYSIS.json")["senti4sd_first_author_minus_post_gds_minus_tps"]
    # reconciliation four-model may have slightly different CI
    four_path = PRIOR_RECON / "tables/FOUR_MODEL_MATCHED_RQ1_RQ2.json"
    recon_ci = None
    if four_path.exists():
        four = load_json(four_path)
        recon_ci = four["cohorts"]["original_frozen"]["rq2_delta_gds_minus_tps"]["senti4sd_sign"]
    text = f"""# Frozen-score Senti4SD RQ2 uncertainty note

## Result in view

Frozen-score reply-eligible pairs: **98 GDS / 956 TPS**.

Sealed Senti4SD analysis (`original_frozen_ANALYSIS.json`):

- mean GDS−TPS of (reply−post) sign delta = **{s4['mean_diff']:.6f}**
- CI95 = **{s4['mean_diff_ci95']}** (includes values on both sides of a razor-thin bound depending on reporting)

Independent / reconciliation resamples (Codex audit): one table listed approximately [−0.5380, −0.0006]; the sealed result lists [−0.5395, +0.0011]; other valid resamples include zero.

{f"Reconciliation matched resample CI: {recon_ci.get('mean_diff_ci95') if recon_ci else 'n/a'}"}

## Required wording

- Describe this as **bootstrap-sensitive / unstable** at the zero boundary.
- **Do not** present a robust nonzero frozen Senti4SD RQ2 conclusion.
- Prefer AI-cohort Senti4SD RQ2 (610/5,332) for any multi-model trajectory discussion, with frozen-score shown only with this caveat.
- Also disclose Senti4SD CSV newline→space normalization (whitespace only; hash match is on pre-normalization saved text).

## Ledger status

`RQ2-F-Senti4SD` → `AUTHORITATIVE_WITH_UNCERTAINTY_CAVEAT`.
"""
    (OUT_ROOT / "SENTI4SD_FROZEN_RQ2_UNCERTAINTY.md").write_text(text, encoding="utf-8")


def write_label_provenance_note(prov: dict[str, Any]) -> None:
    (OUT_ROOT / "LABEL_PROVENANCE_WORDING.md").write_text(
        f"""# Label provenance wording (CORRECTED)

## Forbidden shorthand (superseded)

- `human/prior_frozen_labels`
- `frozen_labels` (when used to imply human verification of TPS/GDS)
- “human-reviewed frozen cohort” as a blanket phrase for the 3,259

“Frozen” refers to the **sentiment score package** (`frozen_broad_release_train_package_scores`), not to human confirmation of purpose labels.

## Required wording

Use: **`{MIXED_ORIGIN_LABEL}`** for frozen-score comparable TPS/GDS rows.

### Verified breakdown (frozen comparable N=3,259)

| Purpose | F review (human) | Terra AI | Manu AI-assisted | Codex AI pilot | Total |
|---|---:|---:|---:|---:|---:|
| TPS | 1,628 | 1,346 | 31 | 23 | 3,028 |
| GDS | 57 | 155 | 6 | 13 | 231 |

Match to `decision_source` on `full_dataset.jsonl`: **{prov['match_expected']}**.

Human F-review subset = **{prov['totals']['human_f_review_total']} / 3,259**; remaining purpose labels are AI/model-assisted origins.

AI-assisted v2 screen labels remain **`{AI_UNCONFIRMED}`** with no completed independent human validation packet.
""",
        encoding="utf-8",
    )


def write_correction_summary(prov: dict[str, Any], flow: dict[str, Any]) -> None:
    (OUT_ROOT / "CORRECTION_SUMMARY.md").write_text(
        f"""# Correction summary — Codex CONDITIONAL PASS fixes

UTC: {utcnow()}

| # | Codex condition | Action | Status |
|---|---|---|---|
| 1 | Mixed-origin frozen labels | Replaced human/frozen shorthand; verified 3,259 breakdown | DONE (match={prov['match_expected']}) |
| 2 | Duplicate RQ2-AI-VADER-full CIs | Canonical = length-sensitivity CI; audit CI alternate ID | DONE |
| 3 | Dataset-flow 520/693/… | Corrected table + figure; arithmetic checks pass | DONE ({flow['arithmetic_checks']}) |
| 4 | Frozen Senti4SD uncertainty | Dedicated note + ledger caveat status | DONE |

Sealed prior trees **not** modified: audit, length-sensitivity, senti4sd, correction_rq, and prior reconciliation package remain sealed.

No sentiment model reruns. No human annotation. No manuscript draft.
""",
        encoding="utf-8",
    )


def write_codex_packet(prior_hashes: dict[str, str]) -> None:
    (OUT_ROOT / "CODEX_FINAL_VERIFICATION_PACKET.md").write_text(
        f"""# Codex final verification packet — evidence corrections 2026-11-03

**Status:** `AWAITING_FINAL_CODEX_CHECK`  
**UTC:** {utcnow()}

## Exact files to verify

1. `tables/AUTHORITATIVE_EVIDENCE_LEDGER_CORRECTED.{{md,csv,json}}` — no `human/prior_frozen_labels`; frozen rows say mixed-origin; unique RQ2 VADER IDs.
2. `tables/FROZEN_COMPARABLE_LABEL_PROVENANCE.json` — breakdown match true.
3. `CANONICAL_VADER_RQ2.md` — canonical CI [−0.178935, −0.050006]; alternate audit CI retained.
4. `tables/DATASET_FLOW_CORRECTED.{{md,json}}` + `figures/fig_dataset_flow_CORRECTED.png` — shows 520 historical, 693 protected, 5,905 capture-linked, 30,998 screened, 12,053 AI new, 18,478 union; protected not train/screen.
5. `SENTI4SD_FROZEN_RQ2_UNCERTAINTY.md` + ledger row `RQ2-F-Senti4SD` caveat.
6. `LABEL_PROVENANCE_WORDING.md`
7. `SEAL_PRESERVATION_CHECK.json` — prior sealed hashes unchanged.
8. `CORRECTION_SUMMARY.md`

## Before → after

| Item | Before (reconciliation) | After (this tree) |
|---|---|---|
| Frozen label field | `human/prior_frozen_labels` / `frozen_labels` | `mixed-origin saved purpose labels` + breakdown |
| RQ2-AI-VADER-full | Duplicate ID, two CIs | Canonical length-sens CI; audit CI as `…-audit-bootstrap` |
| Flow figure | Ambiguous 37,596→6,425; clipped label | Explicit 520/693/5,905 partition + historical external |
| Frozen Senti4SD RQ2 | Point CI without instability note | Uncertainty note + caveat status |

## Unchanged sealed hashes (must still match)

```
{json_dumps_compact(prior_hashes)}
```

Stop after verification; do not draft the manuscript in this step.
""",
        encoding="utf-8",
    )


def run_all() -> dict[str, Any]:
    assert sha256_file(DATASET_JSONL) == EXPECTED_JSONL
    prior_hashes = {
        "reconciliation": sha256_file(PRIOR_RECON / "OUTPUT_HASHES_FINAL.json"),
        "audit": sha256_file(AUDIT_OUT / "OUTPUT_HASHES_FINAL.json"),
        "length": sha256_file(LENGTH_OUT / "OUTPUT_HASHES_FINAL.json"),
        "senti4sd": sha256_file(SENTI4SD_OUT / "OUTPUT_HASHES_FINAL.json"),
        "correction_rq": sha256_file(CORR_OUT / "OUTPUT_HASHES_FINAL.json"),
        "dataset_jsonl": EXPECTED_JSONL,
    }
    write_json(OUT_ROOT / "SEALED_PRIOR_HASHES.json", prior_hashes)

    prov = verify_frozen_provenance()
    assert prov["match_expected"], prov
    rows = build_corrected_ledger(prov)
    write_ledger(rows)
    flow = build_flow()
    fig = draw_flow_figure(flow)
    write_canonical_vader_rq2()
    write_senti4sd_uncertainty()
    write_label_provenance_note(prov)
    write_correction_summary(prov, flow)
    write_codex_packet(prior_hashes)

    # Amend matched-table notes file (copy forward with corrections)
    (OUT_ROOT / "FOUR_MODEL_MATCHED_NOTES_CORRECTED.md").write_text(
        """# Four-model matched notes (CORRECTED)

- Scope remains **reply-eligible only** (AI 5,942 + frozen-score 1,054 = 6,996). Does **not** replace full three-model RQ1 (AI 12,050 / frozen-score 3,259).
- Senti4SD CSV normalizes newlines to spaces (whitespace); text-hash matching is on saved text **before** that normalization.
- Frozen-score Senti4SD RQ2 CI is **bootstrap-sensitive**; see `SENTI4SD_FROZEN_RQ2_UNCERTAINTY.md`.
- Purpose labels: AI rows unconfirmed; frozen-score rows are mixed-origin (not all human-reviewed).
""",
        encoding="utf-8",
    )

    after = {
        "reconciliation": sha256_file(PRIOR_RECON / "OUTPUT_HASHES_FINAL.json"),
        "audit": sha256_file(AUDIT_OUT / "OUTPUT_HASHES_FINAL.json"),
        "length": sha256_file(LENGTH_OUT / "OUTPUT_HASHES_FINAL.json"),
        "senti4sd": sha256_file(SENTI4SD_OUT / "OUTPUT_HASHES_FINAL.json"),
        "correction_rq": sha256_file(CORR_OUT / "OUTPUT_HASHES_FINAL.json"),
    }
    seals_ok = after == {k: prior_hashes[k] for k in after}
    write_json(OUT_ROOT / "SEAL_PRESERVATION_CHECK.json", {"ok": seals_ok, "before": prior_hashes, "after": after})

    return {
        "prov_match": prov["match_expected"],
        "n_ledger_rows": len(rows),
        "flow_ok": all(
            v is True
            for k, v in flow["arithmetic_checks"].items()
            if k.startswith("broad_") or k.startswith("original_") or k.startswith("union_")
        ),
        "figure": str(fig),
        "seals_ok": seals_ok,
    }
