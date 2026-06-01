"""
Load C1 (542) merged labels + text for evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from evaluation.config import REPO_ROOT, TIER_A_SUBREDDITS

_TPS_SCRIPTS = REPO_ROOT / "tps_gds_classification" / "scripts"
if str(_TPS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_TPS_SCRIPTS))

from tps_gds_data import load_dataset, load_verified_frame, undersample_gds  # noqa: E402


def load_c1_frame(
    labels_path: Path,
    data_json_path: Path,
    tier_a_only: bool = False,
) -> pd.DataFrame:
    """Return merged frame with label, text, text_raw, subreddit, reddit_id."""
    df = load_verified_frame(labels_path, data_json_path)
    if tier_a_only:
        sub = df["subreddit"].astype(str).str.lower().str.strip()
        df = df[sub.isin(TIER_A_SUBREDDITS)].copy()
    return df


def prepare_modeling_frame(
    df: pd.DataFrame,
    undersample_gds: bool,
    gds_sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Optional GDS undersampling (legacy); default uses full 542."""
    if undersample_gds:
        return undersample_gds(
            df,
            n_gds=gds_sample_size,
            label_col="label",
            random_state=random_state,
        )
    return df.reset_index(drop=True)


def load_c1_extended_frame(
    labels_path: Path,
    data_json_path: Path,
    c1_prime_csv: Path | None = None,
    tier_a_only: bool = False,
) -> pd.DataFrame:
    """
    C1 frame plus title_raw, body_raw, and ablation text fields.
    Optionally merges C1′ trajectory columns from c1_prime_metrics.csv.
    """
    df = load_c1_frame(labels_path, data_json_path, tier_a_only=tier_a_only)

    full = load_dataset(data_json_path)
    full["reddit_id"] = full["reddit_id"].astype(str).str.strip()
    extra = full[["reddit_id", "title_raw", "body_raw"]].drop_duplicates("reddit_id")
    df = df.merge(extra, on="reddit_id", how="left")
    df["title_raw"] = df["title_raw"].fillna("").astype(str)
    df["body_raw"] = df["body_raw"].fillna("").astype(str)

    df["text_title_only"] = df["title_raw"].str.strip()
    df["text_title_body"] = (
        df["title_raw"].str.strip() + "\n" + df["body_raw"].str.strip()
    ).str.strip()

    if c1_prime_csv and Path(c1_prime_csv).exists():
        traj = pd.read_csv(c1_prime_csv)
        traj["reddit_id"] = traj["reddit_id"].astype(str).str.strip()
        traj_cols = [
            "reddit_id",
            "sentiment_divergence",
            "author_mean_sentiment",
            "community_mean_sentiment",
            "author_more_negative",
            "author_trend",
            "community_trend",
            "author_volatility",
            "community_volatility",
            "author_early_late_shift",
            "community_early_late_shift",
        ]
        traj_cols = [c for c in traj_cols if c in traj.columns]
        df = df.merge(traj[traj_cols], on="reddit_id", how="left")

    return df.reset_index(drop=True)


TRAJECTORY_FEATURE_COLS = [
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
