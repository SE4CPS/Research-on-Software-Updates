"""
Statistical comparison of trajectory/divergence metrics: TPS vs GDS (C1′).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from evaluation.config import TIER_A_SUBREDDITS


@dataclass
class GroupComparison:
    metric: str
    n_tps: int
    n_gds: int
    mean_tps: float
    mean_gds: float
    mean_diff_tps_minus_gds: float
    cohens_d: float
    welch_t_stat: float
    welch_p: float
    mannwhitney_u: float
    mannwhitney_p: float
    bootstrap_mean_diff_ci_lower: float
    bootstrap_mean_diff_ci_upper: float
    bootstrap_n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def filter_tier_a(df: pd.DataFrame) -> pd.DataFrame:
    sub = df["subreddit"].astype(str).str.lower().str.strip()
    return df[sub.isin(TIER_A_SUBREDDITS)].copy()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_mean_diff_ci(
    tps: np.ndarray,
    gds: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    n_t, n_g = len(tps), len(gds)
    if n_t == 0 or n_g == 0:
        return float("nan"), float("nan")
    diffs = []
    for _ in range(n_boot):
        bt = rng.choice(tps, size=n_t, replace=True)
        bg = rng.choice(gds, size=n_g, replace=True)
        diffs.append(float(np.mean(bt) - np.mean(bg)))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1.0 - alpha))


def compare_groups(
    tps: np.ndarray,
    gds: np.ndarray,
    metric_name: str,
    n_boot: int = 2000,
    random_state: int = 42,
) -> GroupComparison:
    tps = np.asarray(tps, dtype=float)
    gds = np.asarray(gds, dtype=float)
    tps = tps[~np.isnan(tps)]
    gds = gds[~np.isnan(gds)]

    mean_tps = float(np.mean(tps)) if len(tps) else float("nan")
    mean_gds = float(np.mean(gds)) if len(gds) else float("nan")
    diff = mean_tps - mean_gds

    if len(tps) >= 2 and len(gds) >= 2:
        welch = stats.ttest_ind(tps, gds, equal_var=False)
        mw = stats.mannwhitneyu(tps, gds, alternative="two-sided")
        welch_t, welch_p = float(welch.statistic), float(welch.pvalue)
        mw_u, mw_p = float(mw.statistic), float(mw.pvalue)
        d = cohens_d(tps, gds)
    else:
        welch_t = welch_p = mw_u = mw_p = d = float("nan")

    lo, hi = bootstrap_mean_diff_ci(tps, gds, n_boot=n_boot, random_state=random_state)

    return GroupComparison(
        metric=metric_name,
        n_tps=int(len(tps)),
        n_gds=int(len(gds)),
        mean_tps=round(mean_tps, 4),
        mean_gds=round(mean_gds, 4),
        mean_diff_tps_minus_gds=round(diff, 4),
        cohens_d=round(d, 4) if not np.isnan(d) else d,
        welch_t_stat=round(welch_t, 4) if not np.isnan(welch_t) else welch_t,
        welch_p=round(welch_p, 6) if not np.isnan(welch_p) else welch_p,
        mannwhitney_u=round(mw_u, 4) if not np.isnan(mw_u) else mw_u,
        mannwhitney_p=round(mw_p, 6) if not np.isnan(mw_p) else mw_p,
        bootstrap_mean_diff_ci_lower=round(lo, 4),
        bootstrap_mean_diff_ci_upper=round(hi, 4),
        bootstrap_n=n_boot,
    )


DIVERGENCE_METRICS = [
    "sentiment_divergence",
    "author_mean_sentiment",
    "community_mean_sentiment",
    "author_trend",
    "community_trend",
    "author_volatility",
    "community_volatility",
    "author_early_late_shift",
    "community_early_late_shift",
]


def author_more_negative_rate(df: pd.DataFrame, label_col: str = "corrected_label") -> dict[str, Any]:
    """Proportion where author_mean < community_mean, by class."""
    out: dict[str, Any] = {}
    for label, name in [(1, "TPS"), (0, "GDS")]:
        sub = df[df[label_col] == label]
        if len(sub) == 0:
            out[name] = {"n": 0, "rate": float("nan")}
            continue
        rate = float(sub["author_more_negative"].mean())
        out[name] = {"n": int(len(sub)), "rate": round(rate, 4)}
    tps_mask = df[label_col] == 1
    gds_mask = df[label_col] == 0
    tps_r = df.loc[tps_mask, "author_more_negative"].astype(int).to_numpy()
    gds_r = df.loc[gds_mask, "author_more_negative"].astype(int).to_numpy()
    if len(tps_r) and len(gds_r):
        # Chi-square on 2x2 could work; use bootstrap on rate difference
        diff = float(np.mean(tps_r) - np.mean(gds_r))
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(2000):
            bt = rng.choice(tps_r, size=len(tps_r), replace=True)
            bg = rng.choice(gds_r, size=len(gds_r), replace=True)
            boots.append(float(np.mean(bt) - np.mean(bg)))
        lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        _, p = stats.mannwhitneyu(tps_r, gds_r, alternative="two-sided")
    else:
        diff, lo, hi, p = float("nan"), float("nan"), float("nan"), float("nan")
    out["rate_diff_tps_minus_gds"] = round(diff, 4)
    out["rate_diff_bootstrap_ci"] = [round(lo, 4), round(hi, 4)]
    out["mannwhitney_p"] = round(float(p), 6) if not np.isnan(p) else p
    return out


def run_all_comparisons(
    df: pd.DataFrame,
    n_boot: int = 2000,
    random_state: int = 42,
) -> list[GroupComparison]:
    tps_df = df[df["corrected_label"] == 1]
    gds_df = df[df["corrected_label"] == 0]
    results = []
    for col in DIVERGENCE_METRICS:
        if col not in df.columns:
            continue
        results.append(
            compare_groups(
                tps_df[col].to_numpy(),
                gds_df[col].to_numpy(),
                metric_name=col,
                n_boot=n_boot,
                random_state=random_state,
            )
        )
    return results


def comparisons_to_dataframe(comparisons: list[GroupComparison]) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in comparisons])


def format_markdown_report(
    cohort_name: str,
    df: pd.DataFrame,
    comparisons: list[GroupComparison],
    author_neg: dict[str, Any],
) -> str:
    n_tps = int((df["corrected_label"] == 1).sum())
    n_gds = int((df["corrected_label"] == 0).sum())
    lines = [
        f"# C1′ Divergence Analysis — {cohort_name}",
        "",
        f"**Posts:** {len(df)} (TPS={n_tps}, GDS={n_gds})",
        "",
        "## Author more negative than community",
        "",
        f"| Class | n | Rate |",
        f"|-------|---|------|",
        f"| TPS | {author_neg['TPS']['n']} | {author_neg['TPS']['rate']} |",
        f"| GDS | {author_neg['GDS']['n']} | {author_neg['GDS']['rate']} |",
        "",
        f"- Rate difference (TPS − GDS): **{author_neg.get('rate_diff_tps_minus_gds')}**",
        f"- Bootstrap 95% CI: {author_neg.get('rate_diff_bootstrap_ci')}",
        f"- Mann–Whitney p: {author_neg.get('mannwhitney_p')}",
        "",
        "## Continuous metrics (TPS vs GDS)",
        "",
        "Positive `mean_diff` → higher in TPS. "
        "Welch *t*-test and Mann–Whitney U; Cohen's *d*; bootstrap 95% CI on mean difference (TPS − GDS).",
        "",
        "| Metric | Mean TPS | Mean GDS | Δ mean | Cohen's d | Welch p | MW p | Bootstrap CI (Δ) |",
        "|--------|----------|----------|--------|-----------|---------|------|------------------|",
    ]
    for c in comparisons:
        ci = f"[{c.bootstrap_mean_diff_ci_lower}, {c.bootstrap_mean_diff_ci_upper}]"
        lines.append(
            f"| {c.metric} | {c.mean_tps} | {c.mean_gds} | {c.mean_diff_tps_minus_gds} | "
            f"{c.cohens_d} | {c.welch_p} | {c.mannwhitney_p} | {ci} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- **sentiment_divergence** = |author_mean − community_mean| (magnitude, not direction).",
            "- Author track: body → OP comments → title fallback (see DATA_CARD.md).",
            "- Non-significant *p* does not imply equivalence; report effect sizes and CIs.",
            "",
        ]
    )
    return "\n".join(lines)
