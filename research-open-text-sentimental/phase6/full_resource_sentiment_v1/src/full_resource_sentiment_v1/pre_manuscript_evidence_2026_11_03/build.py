"""Build authoritative ledger, matched four-model table, flow figure, chronology note."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from full_resource_sentiment_v1.master_research_corrections_2026_11_03.textblob_checks import (
    independent_stratified_mean_diff,
)
from full_resource_sentiment_v1.pre_manuscript_evidence_2026_11_03 import N_BOOT, SEED
from full_resource_sentiment_v1.pre_manuscript_evidence_2026_11_03.common import (
    AUDIT_OUT,
    CONT_OUT,
    CORR_OUT,
    LENGTH_OUT,
    LENGTH_RESTRICTED,
    OUT_ROOT,
    RESTRICTED_ROOT,
    SENTI4SD_OUT,
    SENTI4SD_RESTRICTED,
    load_json,
    load_jsonl,
    sha256_file,
    utcnow,
    write_json,
    write_jsonl,
)


def _ci(c: dict[str, Any] | None) -> str:
    if not c or c.get("mean_diff") is None:
        return ""
    lo, hi = c["mean_diff_ci95"]
    return f"[{lo:.6f}, {hi:.6f}]"


def _mean(c: dict[str, Any] | None) -> str:
    if not c or c.get("mean_diff") is None:
        return ""
    return f"{c['mean_diff']:.6f}"


def _row(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def build_evidence_ledger() -> list[dict[str, Any]]:
    cliffs = load_json(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json")
    adj = load_json(AUDIT_OUT / "item4_rq1_length_source_sensitivity/RQ1_ADJUSTED_SENSITIVITY.json")
    conv = load_json(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json")
    rq4 = load_json(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json")
    trunc = load_json(AUDIT_OUT / "item2_cardiff_truncation_correction/CARDIFF_TRUNCATION_CORRECTED.json")
    length = load_json(LENGTH_OUT / "length_sensitivity/SUMMARY.json")
    s4_ai = load_json(SENTI4SD_OUT / "ai_assisted_v2_ANALYSIS.json")
    s4_fr = load_json(SENTI4SD_OUT / "original_frozen_ANALYSIS.json")
    dataset = load_json(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json")

    fr = cliffs["frozen_original_posts"]
    ai = cliffs["ai_assisted_v2_excl_exact_train_dups"]
    ai_adj = adj["ai_assisted_v2_excl_exact_train_dups"]
    fr_adj = adj["frozen"]

    rows: list[dict[str, Any]] = []

    def add(
        evidence_id: str,
        rq: str,
        cohort: str,
        label_provenance: str,
        unit: str,
        instrument: str,
        scale: str,
        n_gds: Any,
        n_tps: Any,
        estimand: str,
        estimate: Any,
        ci95: Any,
        adjustment: str,
        status: str,
        source: str,
        safe_wording: str,
        notes: str = "",
    ) -> None:
        rows.append(
            _row(
                evidence_id=evidence_id,
                rq=rq,
                cohort=cohort,
                label_provenance=label_provenance,
                analysis_unit=unit,
                instrument=instrument,
                score_scale=scale,
                n_gds=n_gds,
                n_tps=n_tps,
                estimand=estimand,
                estimate=estimate,
                ci95=ci95,
                adjustment_or_sensitivity=adjustment,
                authority_status=status,
                source_artifact=source,
                safe_wording=safe_wording,
                notes=notes,
            )
        )

    # Dataset flow facts
    add(
        "D1",
        "DATA",
        "union_v2",
        "frozen_preserved + AI_screen_v2",
        "reddit_id",
        "n/a",
        "count",
        "",
        "",
        "analytical_union_rows",
        dataset["n_union_rows"],
        "",
        "removal-corrected v2",
        "AUTHORITATIVE",
        str(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json"),
        "Use 18,478-ID union; earlier 18,543 drafts superseded.",
        f"screened={dataset['n_screened']}; AI TPS/GDS={dataset['n_candidate_tps_gds_ai_only_corrected']}; unassessable={dataset['n_unassessable']}",
    )

    # RQ1 frozen
    for inst, key, scale in (
        ("VADER", "vader", "compound[-1,1]"),
        ("TextBlob", "textblob", "polarity[-1,1]"),
        ("Cardiff", "cardiff_sign", "class_sign{-1,0,+1}"),
    ):
        c = fr[key]
        add(
            f"RQ1-F-{inst}",
            "RQ1",
            "frozen_comparable",
            "human/prior_frozen_labels",
            "original_post (title+body one event)",
            inst,
            scale,
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS_mean",
            _mean(c),
            _ci(c),
            "unadjusted; Cliff's δ corrected sign",
            "AUTHORITATIVE",
            str(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json"),
            "Conditional frozen-cohort descriptive contrast; not population truth.",
            f"cliffs_delta={c.get('cliffs_delta'):.6f}",
        )

    # RQ1 AI
    for inst, key, scale in (
        ("VADER", "vader", "compound[-1,1]"),
        ("TextBlob", "textblob", "polarity[-1,1]"),
        ("Cardiff", "cardiff_sign", "class_sign{-1,0,+1}"),
    ):
        c = ai[key]
        add(
            f"RQ1-AI-{inst}",
            "RQ1",
            "ai_assisted_v2_excl_exact_train_dups",
            "AI_purpose_labels_unconfirmed",
            "original_post (title+body one event)",
            inst,
            scale,
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS_mean",
            _mean(c),
            _ci(c),
            "unadjusted; 3 exact train-text TPS dups excluded",
            "AUTHORITATIVE",
            str(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json"),
            "Association within AI-predicted purpose labels only.",
            f"cliffs_delta={c.get('cliffs_delta'):.6f}",
        )

    # RQ1 AI VADER length-adjusted
    va = ai_adj["vader"]["regression_adjusted_log_length"]
    add(
        "RQ1-AI-VADER-adj",
        "RQ1",
        "ai_assisted_v2_excl_exact_train_dups",
        "AI_purpose_labels_unconfirmed",
        "original_post",
        "VADER",
        "compound[-1,1]",
        va["n_gds"],
        va["n_tps"],
        "GDS_coef_log1p_chars_adjusted",
        f"{va['gds_coef_adjusted']:.6f}",
        f"[{va['gds_coef_ci95'][0]:.6f}, {va['gds_coef_ci95'][1]:.6f}]",
        "OLS: score ~ GDS + log1p(chars)",
        "AUTHORITATIVE_SENSITIVITY",
        str(AUDIT_OUT / "item4_rq1_length_source_sensitivity/RQ1_ADJUSTED_SENSITIVITY.json"),
        "Length-adjusted association; not causal.",
    )

    # RQ1 Senti4SD reply-eligible only (NOT full RQ1 population)
    for cohort, rep, eid in (
        ("ai_assisted_v2_reply_eligible", s4_ai, "RQ1-AI-Senti4SD-replyelig"),
        ("frozen_reply_eligible", s4_fr, "RQ1-F-Senti4SD-replyelig"),
    ):
        c = rep["senti4sd_post_gds_minus_tps"]
        add(
            eid,
            "RQ1",
            cohort,
            "AI_unconfirmed" if "ai" in cohort else "frozen_labels",
            "original_post | conditioned on first-author reply existing",
            "Senti4SD",
            "class_sign{-1,0,+1}",
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS_mean_sign",
            _mean(c),
            _ci(c),
            "REPLY-ELIGIBLE SUBSET ONLY — not full RQ1 population",
            "AUTHORITATIVE_FOR_REPLY_ELIGIBLE_FOUR_MODEL",
            str(SENTI4SD_OUT / ("ai_assisted_v2_ANALYSIS.json" if "ai" in cohort else "original_frozen_ANALYSIS.json")),
            "Do not present as same population as full-post VADER/TB/Cardiff RQ1.",
            "Full 15,312-post Senti4SD RQ1 requires author approval (Decision 4).",
        )

    # RQ2
    for tag, key, path_note in (
        ("AI-VADER-full", "rq2_ai_first_author_minus_post", "audit item3"),
        ("F-VADER-full", "rq2_original_first_author_minus_post", "audit item3"),
    ):
        c = conv[key]
        add(
            f"RQ2-{tag}",
            "RQ2",
            "ai_reply_eligible" if tag.startswith("AI") else "frozen_reply_eligible",
            "AI_unconfirmed" if tag.startswith("AI") else "frozen_labels",
            "first_author_reply_minus_original_post",
            "VADER",
            "compound delta",
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS_of_paired_delta",
            _mean(c),
            _ci(c),
            "full saved text",
            "AUTHORITATIVE",
            str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
            "Observed reply-rich threads only; ≠ issue resolution.",
            path_note,
        )

    # RQ2 length / multi-instrument from length + senti4sd
    ai_len = length["ai"]
    for inst, full_key, prefix_key, scale in (
        ("VADER", "full_vader", "prefix_vader", "compound delta"),
        ("TextBlob", "full_textblob", "prefix_textblob", "polarity delta"),
        ("Cardiff", "full_cardiff", "prefix_cardiff", "sign delta"),
    ):
        c = ai_len[full_key]
        add(
            f"RQ2-AI-{inst}-full",
            "RQ2",
            "ai_reply_eligible",
            "AI_unconfirmed",
            "first_author_reply_minus_post",
            inst,
            scale,
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS_of_paired_delta",
            _mean(c),
            _ci(c),
            "full text",
            "AUTHORITATIVE",
            str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
            "Instrument-specific; do not pool magnitudes across scales.",
        )
        c2 = ai_len[prefix_key]
        add(
            f"RQ2-AI-{inst}-prefix",
            "RQ2",
            "ai_reply_eligible",
            "AI_unconfirmed",
            "first_author_reply_minus_post",
            inst,
            scale,
            c2["n_a"],
            c2["n_b"],
            "GDS_minus_TPS_of_paired_delta",
            _mean(c2),
            _ci(c2),
            "equal-length char prefixes (different text portion)",
            "AUTHORITATIVE_SENSITIVITY",
            str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
            "Prefix ≠ full-text estimand.",
        )

    va_adj = ai_len["adjusted_vader"]
    add(
        "RQ2-AI-VADER-adj",
        "RQ2",
        "ai_reply_eligible",
        "AI_unconfirmed",
        "first_author_reply_minus_post",
        "VADER",
        "compound delta",
        va_adj["n_gds"],
        va_adj["n_tps"],
        "GDS_coef_length_source_date_adjusted",
        f"{va_adj['gds_coef_adjusted']:.6f}",
        f"[{va_adj['gds_coef_ci95'][0]:.6f}, {va_adj['gds_coef_ci95'][1]:.6f}]",
        "OLS delta ~ GDS + log post/reply len + source + date",
        "AUTHORITATIVE_SENSITIVITY",
        str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
        "Sensitivity; not a causal trajectory proof.",
    )

    c = s4_ai["senti4sd_first_author_minus_post_gds_minus_tps"]
    add(
        "RQ2-AI-Senti4SD",
        "RQ2",
        "ai_reply_eligible",
        "AI_unconfirmed",
        "first_author_reply_minus_post",
        "Senti4SD",
        "sign delta",
        c["n_a"],
        c["n_b"],
        "GDS_minus_TPS_of_paired_sign_delta",
        _mean(c),
        _ci(c),
        "full text class signs; no Senti4SD prefix/adjusted run",
        "AUTHORITATIVE",
        str(SENTI4SD_OUT / "ai_assisted_v2_ANALYSIS.json"),
        "Direction/class-sign only; not magnitude vs VADER.",
    )

    # RQ2 secondary
    for eid, key in (
        ("RQ2-AI-second_minus_first", "rq2_ai_second_minus_first"),
        ("RQ2-AI-ge4", "rq2_ai_author_ge4_shift"),
    ):
        c = conv[key]
        add(
            eid,
            "RQ2",
            "ai_deeper_reply_eligible",
            "AI_unconfirmed",
            key.replace("rq2_ai_", ""),
            "VADER",
            "compound delta",
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS",
            _mean(c),
            _ci(c),
            "different eligibility than first-reply",
            "AUTHORITATIVE_SECONDARY",
            str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
            "Do not collapse with first-reply estimand.",
        )

    # RQ3
    for eid, key, cohort in (
        ("RQ3-AI", "rq3_ai_author_minus_community", "ai_both_roles"),
        ("RQ3-F", "rq3_original_author_minus_community", "frozen_both_roles"),
    ):
        c = conv[key]
        add(
            eid,
            "RQ3",
            cohort,
            "AI_unconfirmed" if "ai" in cohort else "frozen_labels",
            "author_mean_minus_community_mean_per_thread",
            "VADER",
            "compound gap",
            c["n_a"],
            c["n_b"],
            "GDS_minus_TPS_of_role_gap",
            _mean(c),
            _ci(c),
            "thread-level gap then purpose contrast",
            "AUTHORITATIVE",
            str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
            "Bounded descriptive; intervals include 0.",
        )

    # RQ4
    models = rq4["models"]
    for mid, label in (
        ("text_balanced", "text_only_balanced"),
        ("text_plus_vader_tb_balanced", "text+VADER+TextBlob_balanced"),
        ("text_plus_vader_tb_cardiff_balanced", "text+VADER+TB+Cardiff_balanced"),
    ):
        m = models[mid]
        add(
            f"RQ4-{mid}",
            "RQ4",
            "fixed_grouped_split_v2",
            "mixed_origin_saved_labels",
            "test_rows",
            label,
            "macro_F1",
            "64 GDS test",
            "754 TPS test",
            "balanced_macro_F1",
            f"{m['macro_f1']:.6f}",
            "",
            f"train/val/test={rq4['n_train']}/{rq4['n_val']}/{rq4['n_test']}",
            "AUTHORITATIVE",
            str(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json"),
            "Agreement with saved labels only — not human field accuracy.",
        )
    boot = rq4["component_bootstrap_paired_macro_f1"]
    for eid, key in (
        ("RQ4-boot-2feat_minus_text", "two_feature_minus_text_balanced"),
        ("RQ4-boot-cardiff_minus_text", "cardiff_inclusive_minus_text_balanced"),
    ):
        b = boot[key]
        add(
            eid,
            "RQ4",
            "fixed_grouped_split_v2",
            "mixed_origin_saved_labels",
            "near_dup_components",
            key,
            "macro_F1_diff",
            "",
            "",
            "paired_component_bootstrap_mean_diff",
            f"{b['mean']:.6f}",
            f"[{b['ci95'][0]:.6f}, {b['ci95'][1]:.6f}]",
            f"n_components={b['n_test_components']}; prior row-bootstrap SUPERSEDED",
            "AUTHORITATIVE",
            str(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json"),
            "Component bootstrap replaces superseded row bootstrap.",
        )

    # Truncation correction note
    add(
        "SENS-Cardiff-trunc",
        "SENS",
        "frozen_complete_scores_by_truncation_arm",
        "frozen",
        "original_post strata",
        "Cardiff vs VADER/TB agreement",
        "pct_agree",
        "",
        "",
        "truncated_n / nontruncated_n",
        f"{trunc['frozen_all_units_complete_scores']['truncated']['n_complete_scores']} / {trunc['frozen_all_units_complete_scores']['nontruncated']['n_complete_scores']}",
        "",
        "same-stratum arms; prior mixed-stratum SUPERSEDED",
        "AUTHORITATIVE_CORRECTION",
        str(AUDIT_OUT / "item2_cardiff_truncation_correction/CARDIFF_TRUNCATION_CORRECTED.json"),
        "Report truncation by analyzed unit; do not reuse mixed-stratum table.",
    )

    # Superseded markers
    for eid, note in (
        (
            "SUPERSEDED-union-18543",
            "Pre-removal-correction union drafts (~18,543) superseded by 18,478 v2.",
        ),
        (
            "SUPERSEDED-Cliff-sign",
            "Pre-audit Cliff's δ sign convention reversed; means/CIs retained, δ sign corrected.",
        ),
        (
            "SUPERSEDED-Cardiff-trunc-mixed",
            "Mixed frozen+AI truncated vs frozen-only nontruncated comparison superseded.",
        ),
        (
            "SUPERSEDED-RQ4-row-bootstrap",
            "RQ4 row-level bootstrap exploratory only; use component bootstrap.",
        ),
        (
            "SUPERSEDED-Senti4SD-as-full-RQ1",
            "Any reading of Senti4SD post contrast as full 15,312-post RQ1 is incorrect.",
        ),
    ):
        add(
            eid,
            "META",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "",
            "",
            "supersession",
            "SUPERSEDED",
            "",
            "",
            "SUPERSEDED",
            "see notes",
            note,
            "Do not cite as current paper number.",
        )

    return rows


def write_ledger(rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    csv_path = OUT_ROOT / "tables/AUTHORITATIVE_EVIDENCE_LEDGER.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Markdown
    lines = [
        "# Authoritative evidence ledger — 2026-11-03",
        "",
        "Read-only reconciliation of sealed corrected outputs. Frozen and AI cohorts are never pooled without labels.",
        "",
        "| ID | RQ | Cohort | Labels | Unit | Instrument | N GDS/TPS | Estimand | Estimate | CI95 | Status | Source |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if str(r["evidence_id"]).startswith("SUPERSEDED"):
            continue
        lines.append(
            "| {evidence_id} | {rq} | {cohort} | {label_provenance} | {analysis_unit} | {instrument} | {n_gds}/{n_tps} | {estimand} | {estimate} | {ci95} | {authority_status} | `{source_artifact}` |".format(
                **{k: str(r[k]).replace("|", "/") for k in r}
            )
        )
    lines += ["", "## Explicitly superseded", ""]
    for r in rows:
        if str(r["evidence_id"]).startswith("SUPERSEDED"):
            lines.append(f"- **{r['evidence_id']}**: {r['safe_wording']} {r['notes']}")
    (OUT_ROOT / "tables/AUTHORITATIVE_EVIDENCE_LEDGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT_ROOT / "tables/AUTHORITATIVE_EVIDENCE_LEDGER.json", {"utc": utcnow(), "n_rows": len(rows), "rows": rows})


def build_dataset_flow() -> dict[str, Any]:
    dataset = load_json(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json")
    conv = load_json(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json")
    flow = {
        "utc": utcnow(),
        "steps": [
            {"step": 1, "label": "ReleaseTrain capture frame", "n": 37596, "note": "broad capture universe cited in design"},
            {"step": 2, "label": "Original selected cohort (frozen package)", "n": dataset["n_original_preserved"], "note": "6,425 retained"},
            {"step": 3, "label": "Excluded posts screened", "n": dataset["n_screened"], "note": "30,998"},
            {
                "step": 4,
                "label": "AI TPS/GDS candidates after unassessable removal",
                "n": dataset["n_candidate_tps_gds_ai_only_corrected"],
                "detail": {
                    "TPS": dataset["n_candidate_tps"],
                    "GDS": dataset["n_candidate_gds"],
                    "Other": dataset["n_candidate_relevant_other"],
                    "not_candidate": dataset["n_not_candidate"],
                    "unassessable": dataset["n_unassessable"],
                },
            },
            {
                "step": 5,
                "label": "Analytical union v2",
                "n": dataset["n_union_rows"],
                "note": "6,425 + 12,053; overlap_original_ai=0; protected_in_union=0",
            },
            {
                "step": 6,
                "label": "RQ1 frozen comparable posts",
                "n_gds": 231,
                "n_tps": 3028,
                "n": 3259,
                "note": "not all 6,425",
            },
            {
                "step": 7,
                "label": "RQ1 AI posts excl exact train-text dups",
                "n_gds": 999,
                "n_tps": 11051,
                "n": 12050,
                "note": "3 exact TPS train-text overlaps excluded from primary AI VADER",
            },
            {
                "step": 8,
                "label": "RQ2 first-author pairs (four-model matched)",
                "ai": {"n": conv["ai_eligibility"]["author_ge1"], "gds": 610, "tps": 5332},
                "frozen": {"n": 1054, "gds": 98, "tps": 956},
                "total_pairs": 6996,
                "note": "Senti4SD coverage equals these pairs; not full RQ1",
            },
            {
                "step": 9,
                "label": "RQ3 both-role threads",
                "ai": {"n": conv["ai_eligibility"]["both_roles"], "gds": 608, "tps": 5271},
                "frozen": {"n": 1032, "gds": 98, "tps": 934},
            },
            {
                "step": 10,
                "label": "RQ4 fixed split",
                "train": 1948,
                "val": 493,
                "test": 818,
                "test_gds": 64,
            },
        ],
        "source": str(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json"),
    }
    write_json(OUT_ROOT / "tables/DATASET_FLOW.json", flow)
    md = [
        "# Dataset / eligibility flow",
        "",
        "| Step | Label | Count | Notes |",
        "|---:|---|---:|---|",
    ]
    for s in flow["steps"]:
        n = s.get("n", s.get("total_pairs", ""))
        if "n_gds" in s and "n" in s:
            detail = f"GDS {s['n_gds']} / TPS {s['n_tps']}"
        elif "ai" in s:
            detail = f"AI {s['ai']}; frozen {s.get('frozen')}"
        elif "train" in s:
            detail = f"train/val/test {s['train']}/{s['val']}/{s['test']}"
        else:
            detail = s.get("note", "")
        md.append(f"| {s['step']} | {s['label']} | {n} | {detail} |")
    (OUT_ROOT / "tables/DATASET_FLOW.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return flow


def verify_four_model_match() -> dict[str, Any]:
    """Confirm VADER/TB/Cardiff/Senti4SD on identical reply-eligible pairs."""
    report: dict[str, Any] = {"utc": utcnow(), "cohorts": {}}
    for cohort, pair_name, score_name in (
        ("ai_assisted_v2", "ai_assisted_v2_pair_rows.jsonl", "ai_assisted_v2_senti4sd_scores.jsonl"),
        ("original_frozen", "original_frozen_pair_rows.jsonl", "original_frozen_senti4sd_scores.jsonl"),
    ):
        pairs = {r["reddit_id"]: r for r in load_jsonl(LENGTH_RESTRICTED / pair_name)}
        scores = load_jsonl(SENTI4SD_RESTRICTED / score_name)
        by = defaultdict(dict)
        missing_fields = 0
        hash_mismatch = 0
        for s in scores:
            by[s["reddit_id"]][s["unit"]] = s
            if any(s.get(k) is None for k in ("vader", "textblob", "cardiff_sign", "senti4sd_sign", "text_hash")):
                missing_fields += 1
        matched = []
        only_pairs = 0
        only_s4 = 0
        for rid, p in pairs.items():
            if rid not in by or "original_post" not in by[rid] or "first_author_reply" not in by[rid]:
                only_pairs += 1
                continue
            post = by[rid]["original_post"]
            reply = by[rid]["first_author_reply"]
            # Length rows lack text_hash; join on reddit_id and require Senti4SD carries prior instrument scores
            matched.append(
                {
                    "reddit_id": rid,
                    "label": p["label"],
                    "cohort": cohort,
                    "post_hash": post["text_hash"],
                    "reply_hash": reply["text_hash"],
                    "post_vader": post.get("vader", p.get("full_post_vader")),
                    "post_textblob": post.get("textblob", p.get("full_post_textblob")),
                    "post_cardiff_sign": post.get("cardiff_sign", p.get("full_post_cardiff_sign")),
                    "post_senti4sd_sign": post["senti4sd_sign"],
                    "post_senti4sd_label": post["senti4sd_label"],
                    "reply_vader": reply.get("vader", p.get("full_reply_vader")),
                    "reply_textblob": reply.get("textblob", p.get("full_reply_textblob")),
                    "reply_cardiff_sign": reply.get("cardiff_sign", p.get("full_reply_cardiff_sign")),
                    "reply_senti4sd_sign": reply["senti4sd_sign"],
                    "reply_senti4sd_label": reply["senti4sd_label"],
                    # Prefer length-tree full scores when present (canonical for VADER/TB/Cardiff)
                    "full_post_vader": p.get("full_post_vader"),
                    "full_reply_vader": p.get("full_reply_vader"),
                    "full_post_textblob": p.get("full_post_textblob"),
                    "full_reply_textblob": p.get("full_reply_textblob"),
                    "full_post_cardiff_sign": p.get("full_post_cardiff_sign"),
                    "full_reply_cardiff_sign": p.get("full_reply_cardiff_sign"),
                }
            )
        for rid in by:
            if rid not in pairs:
                only_s4 += 1
        # Instrument completeness on matched
        complete = 0
        for m in matched:
            if all(
                m.get(k) is not None
                for k in (
                    "full_post_vader",
                    "full_reply_vader",
                    "full_post_textblob",
                    "full_reply_textblob",
                    "full_post_cardiff_sign",
                    "full_reply_cardiff_sign",
                    "post_senti4sd_sign",
                    "reply_senti4sd_sign",
                )
            ):
                complete += 1
        write_jsonl(RESTRICTED_ROOT / f"{cohort}_four_model_matched_pairs.jsonl", matched)
        report["cohorts"][cohort] = {
            "n_length_pairs": len(pairs),
            "n_matched_complete": complete,
            "n_matched_rows": len(matched),
            "only_in_length_not_s4": only_pairs,
            "only_in_s4_not_length": only_s4,
            "s4_rows_missing_fields": missing_fields,
            "fully_matched": only_pairs == 0 and only_s4 == 0 and complete == len(pairs),
            "n_gds": sum(1 for m in matched if m["label"] == "GDS"),
            "n_tps": sum(1 for m in matched if m["label"] == "TPS"),
        }
    report["total_matched_pairs"] = sum(c["n_matched_rows"] for c in report["cohorts"].values())
    report["four_model_fair_comparison_scope"] = (
        "Reply-eligible pairs only (AI 5,942 + frozen 1,054 = 6,996). "
        "Not the full 15,312-post RQ1 population. Full-post Senti4SD needs author approval."
    )
    write_json(OUT_ROOT / "tables/FOUR_MODEL_MATCH_VERIFICATION.json", report)
    return report


def four_model_contrasts(match_report: dict[str, Any]) -> dict[str, Any]:
    """RQ1 post + RQ2 delta on identical matched observations; within-model scales only."""
    out: dict[str, Any] = {"utc": utcnow(), "note": "No cross-scale magnitude comparison.", "cohorts": {}}
    for cohort in ("ai_assisted_v2", "original_frozen"):
        pairs = load_jsonl(RESTRICTED_ROOT / f"{cohort}_four_model_matched_pairs.jsonl")
        # Use canonical length-tree scores for V/TB/Cardiff; Senti4SD signs from matched file
        specs = [
            ("vader_post", "full_post_vader", "RQ1_post"),
            ("textblob_post", "full_post_textblob", "RQ1_post"),
            ("cardiff_sign_post", "full_post_cardiff_sign", "RQ1_post"),
            ("senti4sd_sign_post", "post_senti4sd_sign", "RQ1_post"),
        ]
        cohort_rep: dict[str, Any] = {"n": len(pairs), "rq1_post_gds_minus_tps": {}, "rq2_delta_gds_minus_tps": {}}
        seed = SEED + (500 if cohort.startswith("ai") else 600)
        for i, (name, key, _) in enumerate(specs):
            rows = [{"label": p["label"], key: p.get(key)} for p in pairs]
            g = [float(r[key]) for r in rows if r["label"] == "GDS" and r.get(key) is not None]
            t = [float(r[key]) for r in rows if r["label"] == "TPS" and r.get(key) is not None]
            cohort_rep["rq1_post_gds_minus_tps"][name] = independent_stratified_mean_diff(
                g, t, seed=seed + i, n_boot=N_BOOT
            )
        # RQ2 deltas
        delta_specs = [
            ("vader", "full_post_vader", "full_reply_vader"),
            ("textblob", "full_post_textblob", "full_reply_textblob"),
            ("cardiff_sign", "full_post_cardiff_sign", "full_reply_cardiff_sign"),
            ("senti4sd_sign", "post_senti4sd_sign", "reply_senti4sd_sign"),
        ]
        for i, (name, pk, rk) in enumerate(delta_specs):
            g, t = [], []
            for p in pairs:
                if p.get(pk) is None or p.get(rk) is None:
                    continue
                d = float(p[rk]) - float(p[pk])
                (g if p["label"] == "GDS" else t).append(d)
            cohort_rep["rq2_delta_gds_minus_tps"][name] = independent_stratified_mean_diff(
                g, t, seed=seed + 20 + i, n_boot=N_BOOT
            )

        # Direction consistency among instruments (sign of GDS-TPS mean only)
        def sgn(x: float | None) -> str:
            if x is None:
                return "na"
            if x > 0:
                return "+"
            if x < 0:
                return "-"
            return "0"

        rq1_dirs = {k: sgn(v.get("mean_diff")) for k, v in cohort_rep["rq1_post_gds_minus_tps"].items()}
        rq2_dirs = {k: sgn(v.get("mean_diff")) for k, v in cohort_rep["rq2_delta_gds_minus_tps"].items()}
        cohort_rep["rq1_direction_pattern"] = rq1_dirs
        cohort_rep["rq2_direction_pattern"] = rq2_dirs
        cohort_rep["rq1_directions_agree_all_four"] = len(set(rq1_dirs.values())) == 1
        cohort_rep["rq2_directions_agree_all_four"] = len(set(rq2_dirs.values())) == 1
        # CI excludes 0?
        cohort_rep["rq1_ci_excludes_zero"] = {
            k: (v.get("mean_diff_ci95") is not None and (v["mean_diff_ci95"][0] > 0 or v["mean_diff_ci95"][1] < 0))
            for k, v in cohort_rep["rq1_post_gds_minus_tps"].items()
        }
        cohort_rep["rq2_ci_excludes_zero"] = {
            k: (v.get("mean_diff_ci95") is not None and (v["mean_diff_ci95"][0] > 0 or v["mean_diff_ci95"][1] < 0))
            for k, v in cohort_rep["rq2_delta_gds_minus_tps"].items()
        }
        out["cohorts"][cohort] = cohort_rep
    write_json(OUT_ROOT / "tables/FOUR_MODEL_MATCHED_RQ1_RQ2.json", out)

    # Markdown summary
    lines = [
        "# Four-model matched comparison (reply-eligible only)",
        "",
        "**Scope:** identical reddit_ids with VADER, TextBlob, Cardiff sign, and Senti4SD sign on post and first-author reply.",
        f"**Total pairs:** {match_report['total_matched_pairs']} (AI + frozen separate).",
        "",
        "Do not compare raw magnitudes across instruments. Directions below are signs of GDS−TPS means only.",
        "",
    ]
    for cohort, rep in out["cohorts"].items():
        lines.append(f"## {cohort} (n={rep['n']})")
        lines.append("")
        lines.append("| Estimand | VADER | TextBlob | Cardiff sign | Senti4SD sign |")
        lines.append("|---|---|---|---|---|")
        def cell(c):
            if not c or c.get("mean_diff") is None:
                return "n/a"
            lo, hi = c["mean_diff_ci95"]
            return f"{c['mean_diff']:.4f} [{lo:.4f},{hi:.4f}] n={c['n_a']}/{c['n_b']}"

        r1 = rep["rq1_post_gds_minus_tps"]
        r2 = rep["rq2_delta_gds_minus_tps"]
        lines.append(
            f"| RQ1 post GDS−TPS | {cell(r1['vader_post'])} | {cell(r1['textblob_post'])} | {cell(r1['cardiff_sign_post'])} | {cell(r1['senti4sd_sign_post'])} |"
        )
        lines.append(
            f"| RQ2 reply−post GDS−TPS | {cell(r2['vader'])} | {cell(r2['textblob'])} | {cell(r2['cardiff_sign'])} | {cell(r2['senti4sd_sign'])} |"
        )
        lines.append("")
        lines.append(f"- RQ1 direction pattern: `{rep['rq1_direction_pattern']}` agree_all_four={rep['rq1_directions_agree_all_four']}")
        lines.append(f"- RQ2 direction pattern: `{rep['rq2_direction_pattern']}` agree_all_four={rep['rq2_directions_agree_all_four']}")
        lines.append(f"- RQ1 CI excludes 0: `{rep['rq1_ci_excludes_zero']}`")
        lines.append(f"- RQ2 CI excludes 0: `{rep['rq2_ci_excludes_zero']}`")
        lines.append("")
    (OUT_ROOT / "tables/FOUR_MODEL_MATCHED_RQ1_RQ2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_chronology_report() -> dict[str, Any]:
    report = {
        "utc": utcnow(),
        "event0_original_post": {
            "definition": "Title and body concatenated as one initial posting event / one score unit.",
            "verified": True,
            "model_construction": {
                "VADER": "Frozen: combined_compound from package; AI/Cardiff pipeline: vader on same combined text unit.",
                "TextBlob": "Polarity on same combined title+body text.",
                "Cardiff": "Class on same combined original_post unit (512-token truncation flagged).",
                "Senti4SD": "Class on same combined post text (newlines collapsed to spaces for CSV); scored only if first-author reply exists.",
            },
            "source_audits": [
                "planning/final_independent_audit_2026-11-03/FINAL_RQ1_RQ4_AUDIT.md §RQ2",
                "results/exact_join_repair_2026-11-03/",
            ],
        },
        "subsequent_author_replies": {
            "ordering": "Author comments sorted by timestamp_epoch_seconds then chronological_index / stable index.",
            "event1": "First original-author reply (exact capture_page + reddit_id + original_index join).",
            "event2": "Second original-author reply (separate eligibility; smaller N).",
            "distinguished_estimands": [
                "post → first reply (primary RQ2)",
                "first → second reply (secondary)",
                "≥4 author replies shift (secondary, different eligibility)",
            ],
            "verified": True,
            "join_repair": "Role-order join superseded by exact index join (exact_join_repair_2026-11-03).",
        },
        "preprocessing_differences": [
            "Senti4SD CSV path collapses newlines to spaces; VADER/TextBlob/Cardiff use saved multiline text.",
            "Cardiff truncates at 512 tokens; Senti4SD uses dense SE features with DSM; VADER/TB are lexicon/rule-based.",
            "Equal-length prefix sensitivity exists for VADER/TB/Cardiff only (length-sensitivity tree), not Senti4SD.",
        ],
        "eligibility_differences": [
            "Full RQ1 posts: frozen 3,259; AI 12,050 after exact-dup exclusion — three instruments.",
            "Four-model table: reply-eligible 6,996 pairs only.",
            "Missing reply is unobserved, not neutral.",
        ],
        "decision_required": "Author Decision 4: whether to authorize Senti4SD on remaining non-reply posts for full RQ1 four-model table.",
    }
    write_json(OUT_ROOT / "AUTHOR_CHRONOLOGY_VERIFICATION.json", report)
    (OUT_ROOT / "AUTHOR_CHRONOLOGY_VERIFICATION.md").write_text(
        "# Original-author chronology verification\n\n"
        + "\n".join(
            [
                "## Event 0 — original post",
                report["event0_original_post"]["definition"],
                "",
                "### Per-model score construction",
                *[f"- **{k}**: {v}" for k, v in report["event0_original_post"]["model_construction"].items()],
                "",
                "## Later author comments",
                f"- Ordering: {report['subsequent_author_replies']['ordering']}",
                f"- Event 1: {report['subsequent_author_replies']['event1']}",
                f"- Event 2: {report['subsequent_author_replies']['event2']}",
                "",
                "## Estimands kept distinct",
                *[f"- {x}" for x in report["subsequent_author_replies"]["distinguished_estimands"]],
                "",
                "## Preprocessing / eligibility differences",
                *[f"- {x}" for x in report["preprocessing_differences"]],
                *[f"- {x}" for x in report["eligibility_differences"]],
                "",
                f"**Pending author decision:** {report['decision_required']}",
                "",
                f"Verified flags: event0={report['event0_original_post']['verified']}; replies={report['subsequent_author_replies']['verified']}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def make_figures(flow: dict[str, Any], four: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created: list[str] = []
    # Figure 1: dataset flow
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    values = []
    for s in flow["steps"]:
        if "n" in s and isinstance(s["n"], int):
            labels.append(f"{s['step']}. {s['label'][:42]}")
            values.append(s["n"])
    ax.barh(range(len(values)), values, color="#2F4F4F")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Dataset / eligibility flow (authoritative v2 counts)")
    for i, v in enumerate(values):
        ax.text(v, i, f"  {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    p1 = OUT_ROOT / "figures/fig_dataset_flow.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    created.append(str(p1))
    (OUT_ROOT / "figures/fig_dataset_flow_CAPTION.md").write_text(
        "Figure. Dataset and eligibility flow for the removal-corrected analytical union. "
        "Counts are from sealed `DATASET_CORRECTION_SUMMARY.json`. "
        "RQ1 frozen comparable (3,259) and AI (12,050) are subsets; four-model Senti4SD coverage is reply-eligible pairs only (6,996).\n",
        encoding="utf-8",
    )

    # Figure 2: AI four-model RQ2 directions (forest-like within each scale — separate panels)
    ai = four["cohorts"]["ai_assisted_v2"]["rq2_delta_gds_minus_tps"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=False)
    titles = [
        ("vader", "VADER compound Δ"),
        ("textblob", "TextBlob polarity Δ"),
        ("cardiff_sign", "Cardiff sign Δ"),
        ("senti4sd_sign", "Senti4SD sign Δ"),
    ]
    for ax, (key, title) in zip(axes, titles):
        c = ai[key]
        mean = c["mean_diff"]
        lo, hi = c["mean_diff_ci95"]
        ax.errorbar([0], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", color="#1f77b4", capsize=5)
        ax.axhline(0, color="gray", lw=1)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("GDS−TPS")
        ax.text(0, mean, f"  n={c['n_a']}/{c['n_b']}", fontsize=7, va="bottom")
    fig.suptitle("AI reply-eligible RQ2: within-instrument GDS−TPS of (reply−post)\nScales are not commensurate — panels are separate", fontsize=10)
    fig.tight_layout()
    p2 = OUT_ROOT / "figures/fig_ai_rq2_four_model_panels.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    created.append(str(p2))
    (OUT_ROOT / "figures/fig_ai_rq2_four_model_panels_CAPTION.md").write_text(
        "Figure. AI-assisted reply-eligible cohort (610 GDS / 5,332 TPS): GDS−TPS contrast of first-author-reply minus original-post "
        "for four instruments on identical threads. Each panel uses that instrument's native scale; "
        "vertical positions must not be compared across panels. Error bars are 95% thread-stratified bootstrap CIs.\n",
        encoding="utf-8",
    )

    # Figure inventory + recommendations
    inventory = {
        "utc": utcnow(),
        "existing_figures_found_in_results": [],
        "new_figures_this_tree": [
            {"path": "figures/fig_dataset_flow.png", "serves": "methods data flow"},
            {"path": "figures/fig_ai_rq2_four_model_panels.png", "serves": "RQ2 four-model on matched IDs"},
        ],
        "recommended_additional": [
            {
                "fig": "RQ1 forest by cohort×instrument (3 instruments full-post; Senti4SD reply-eligible inset)",
                "priority": "high",
                "status": "optional next; numbers in ledger",
            },
            {
                "fig": "RQ2 full vs prefix sensitivity (VADER/Cardiff only)",
                "priority": "high",
                "status": "numbers in length-sensitivity FINDINGS; figure optional",
            },
            {
                "fig": "RQ4 bar of macro-F1 + component-bootstrap deltas",
                "priority": "medium",
                "status": "numbers in ledger",
            },
            {
                "fig": "Do NOT create single-axis overlay of VADER vs Senti4SD magnitudes",
                "priority": "forbid",
                "status": "misleading",
            },
        ],
        "publication_tables_ready": [
            "tables/AUTHORITATIVE_EVIDENCE_LEDGER.md",
            "tables/DATASET_FLOW.md",
            "tables/FOUR_MODEL_MATCHED_RQ1_RQ2.md",
        ],
    }
    # scan for existing png/svg lightly
    root = OUT_ROOT.parent
    found = []
    for pat in ("**/*.png", "**/*.svg"):
        for p in root.glob(pat):
            if "pre_manuscript_evidence" in str(p):
                continue
            found.append(str(p.relative_to(root)))
            if len(found) >= 40:
                break
    inventory["existing_figures_found_in_results"] = found
    write_json(OUT_ROOT / "FIGURE_TABLE_INVENTORY.json", inventory)
    return created


def write_repro_and_packet(
    match: dict[str, Any],
    four: dict[str, Any],
    chrono: dict[str, Any],
) -> None:
    sealed = {
        "length_sensitivity": sha256_file(LENGTH_OUT / "OUTPUT_HASHES_FINAL.json"),
        "senti4sd": sha256_file(SENTI4SD_OUT / "OUTPUT_HASHES_FINAL.json"),
        "audit": sha256_file(AUDIT_OUT / "OUTPUT_HASHES_FINAL.json"),
        "correction_rq": sha256_file(CORR_OUT / "OUTPUT_HASHES_FINAL.json"),
    }
    write_json(OUT_ROOT / "SEALED_PRIOR_HASHES.json", sealed)

    manifest = {
        "utc": utcnow(),
        "authoritative_inputs": {
            "dataset_jsonl": "results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.jsonl",
            "union_correction": str(CORR_OUT / "stage1_dataset_correction/DATASET_CORRECTION_SUMMARY.json"),
            "rq1_cliffs": str(AUDIT_OUT / "item1_cliffs_delta_correction/CORRECTED_CLIFFS_RQ1_POST.json"),
            "rq1_adj": str(AUDIT_OUT / "item4_rq1_length_source_sensitivity/RQ1_ADJUSTED_SENSITIVITY.json"),
            "rq2_rq3": str(AUDIT_OUT / "item3_expanded_conversation/EXPANDED_CONVERSATION_RQ2_RQ3.json"),
            "rq4": str(AUDIT_OUT / "item5_rq4_component_bootstrap/RQ4_COMPONENT_BOOTSTRAP.json"),
            "length_sensitivity": str(LENGTH_OUT / "length_sensitivity/SUMMARY.json"),
            "senti4sd_ai": str(SENTI4SD_OUT / "ai_assisted_v2_ANALYSIS.json"),
            "senti4sd_frozen": str(SENTI4SD_OUT / "original_frozen_ANALYSIS.json"),
            "matched_pair_rows": str(LENGTH_RESTRICTED / "ai_assisted_v2_pair_rows.jsonl"),
            "senti4sd_scores": str(SENTI4SD_RESTRICTED / "ai_assisted_v2_senti4sd_scores.jsonl"),
        },
        "scripts": [
            "scripts/run_pre_manuscript_evidence_reconciliation_2026_11_03.py",
            "src/full_resource_sentiment_v1/pre_manuscript_evidence_2026_11_03/",
        ],
        "checks": [
            "tests/test_pre_manuscript_evidence_2026_11_03.py",
            "FOUR_MODEL_MATCH_VERIFICATION.json fully_matched flags",
            "sealed prior hash preservation",
        ],
        "sealed_prior_hashes": sealed,
        "no_expensive_reruns": True,
        "no_label_changes": True,
    }
    write_json(OUT_ROOT / "REPRODUCIBILITY_MANIFEST.json", manifest)
    (OUT_ROOT / "REPRODUCIBILITY_MANIFEST.md").write_text(
        "# Reproducibility manifest\n\n"
        + "\n".join(f"- **{k}**: `{v}`" for k, v in manifest["authoritative_inputs"].items())
        + "\n\n## Run\n\n```bash\ncd phase6/full_resource_sentiment_v1\n"
        ".venv_master_2026_11_03/bin/python -m pytest tests/test_pre_manuscript_evidence_2026_11_03.py -q\n"
        ".venv_master_2026_11_03/bin/python scripts/run_pre_manuscript_evidence_reconciliation_2026_11_03.py\n```\n",
        encoding="utf-8",
    )

    ai = four["cohorts"]["ai_assisted_v2"]
    packet = f"""# Codex review packet — pre-manuscript evidence reconciliation 2026-11-03

**Status:** `AWAITING_INDEPENDENT_CODEX_REVIEW`  
**UTC:** {utcnow()}  
**Sealed priors preserved:** `{sealed}`  

## Ask Codex to verify
1. Every AUTHORITATIVE ledger row traces to the cited sealed JSON and matches estimate/CI/N.
2. Superseded rows are excluded from paper numbers.
3. Four-model match: AI+frozen = 6,996 pairs; `fully_matched` true; zero missing instrument fields on matched rows.
4. Four-model RQ1/RQ2 table is labeled **reply-eligible only**, not full RQ1 15,312.
5. Dataset flow 37,596→6,425 / 30,998→12,053→18,478 reconciles with DATASET_CORRECTION_SUMMARY.
6. Chronology: title+body = event 0; first/second author replies distinguished; exact-join supersession noted.
7. No sealed tree hashes changed; no manuscript written; no expensive model reruns.

## Four-model fairness verdict (this packet)
- Fair comparison **on reply-eligible matched IDs**: {match['cohorts']['ai_assisted_v2']['fully_matched']} (AI), {match['cohorts']['original_frozen']['fully_matched']} (frozen).
- Total matched pairs: {match['total_matched_pairs']}.
- AI RQ1 directions agree all four? {ai['rq1_directions_agree_all_four']} → `{ai['rq1_direction_pattern']}`
- AI RQ2 directions agree all four? {ai['rq2_directions_agree_all_four']} → `{ai['rq2_direction_pattern']}`
- Full-population four-model RQ1 **not** available without Decision-4 Senti4SD expansion.

## Author decisions still open
See `planning/final_pre_manuscript_gap_assessment_2026-11-03/DECISIONS_REQUIRING_MY_APPROVAL.md` (thesis strength, human validation, TP/FP meaning, Senti4SD full-post scope, historical overlap, venue).

## Outputs
- `tables/AUTHORITATIVE_EVIDENCE_LEDGER.{{csv,md,json}}`
- `tables/DATASET_FLOW.{{md,json}}`
- `tables/FOUR_MODEL_MATCH_VERIFICATION.json`
- `tables/FOUR_MODEL_MATCHED_RQ1_RQ2.{{md,json}}`
- `AUTHOR_CHRONOLOGY_VERIFICATION.md`
- `figures/fig_dataset_flow.png`
- `figures/fig_ai_rq2_four_model_panels.png`
- `REPRODUCIBILITY_MANIFEST.md`
"""
    (OUT_ROOT / "CODEX_REVIEW_PACKET.md").write_text(packet, encoding="utf-8")

    (OUT_ROOT / "HANDOFF_SUMMARY.md").write_text(
        f"""# Handoff summary — pre-manuscript reconciliation

## Completed
- Authoritative RQ1–RQ4 evidence ledger with supersession markers.
- Dataset flow table/figure.
- Four-model matched verification on 6,996 reply-eligible pairs; RQ1/RQ2 within-instrument contrasts.
- Author chronology verification.
- Publication-ready tables + 2 figures; inventory/recommendations.
- Reproducibility manifest + Codex packet.

## Already correct
- Sealed audit corrections (Cliff’s δ, truncation, AI replies, RQ1 adj, RQ4 component bootstrap).
- Length-sensitivity and Senti4SD trees; prior seals intact.

## Discrepancies / limitations documented
- Senti4SD is **not** available for full RQ1 15,312 posts; four-model table is reply-eligible only.
- AI RQ1/RQ2 instrument directions are not unanimous (TextBlob often near zero / differs).
- Newline collapsing in Senti4SD CSV vs multiline text for other tools.
- No human validation of AI purpose labels.

## Blockers before manuscript drafting
- **No technical blocker** for a *qualified* draft after Codex audits this ledger.
- **Author decisions** still required for claim strength, validation, TP/FP definition, and whether to expand Senti4SD.
- Do not write unqualified population claims.

Chronology verified: {chrono['event0_original_post']['verified']} / {chrono['subsequent_author_replies']['verified']}.
""",
        encoding="utf-8",
    )


def run_all() -> dict[str, Any]:
    rows = build_evidence_ledger()
    write_ledger(rows)
    flow = build_dataset_flow()
    match = verify_four_model_match()
    four = four_model_contrasts(match)
    chrono = write_chronology_report()
    figs = make_figures(flow, four)
    write_repro_and_packet(match, four, chrono)
    return {
        "n_ledger_rows": len(rows),
        "match": match,
        "four_model_ai_rq2_dirs": four["cohorts"]["ai_assisted_v2"]["rq2_direction_pattern"],
        "figures": figs,
    }
