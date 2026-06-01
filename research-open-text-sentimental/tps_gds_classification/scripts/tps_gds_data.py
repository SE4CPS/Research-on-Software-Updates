"""
Load TPS/GDS dataset, undersample GDS, stratified train/val/test split.
Verified manual labels merged with JSON posts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def write_unique_labels_csv(src: Path, dst: Path) -> int:
    """Deduplicate manual label CSV by reddit_id; return row count."""
    df = pd.read_csv(src).drop_duplicates(subset=["reddit_id"], keep="first")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return len(df)


def load_verified_frame(labels_path: Path, json_path: Path) -> pd.DataFrame:
    """Merge deduped manual labels with deduped JSON records. Label = corrected_label."""
    lab = pd.read_csv(labels_path).drop_duplicates(subset=["reddit_id"], keep="first")
    lab["reddit_id"] = lab["reddit_id"].astype(str).str.strip()
    lab = lab[["reddit_id", "corrected_label"]]

    full = load_dataset(json_path)
    full["reddit_id"] = full["reddit_id"].astype(str).str.strip()
    full = full.drop_duplicates(subset=["reddit_id"], keep="first")

    cols = [
        "reddit_id",
        "text",
        "text_raw",
        "num_comments_in_post",
        "_id",
        "subreddit",
        "url",
    ]
    missing = [c for c in cols if c not in full.columns]
    if missing:
        raise ValueError(f"tps_gds_dataset.json missing columns: {missing}")

    m = lab.merge(full[cols], on="reddit_id", how="inner")
    m["label"] = m["corrected_label"].astype(int)
    return m


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)
    records = payload.get("records") or []
    return pd.DataFrame(records)


def undersample_gds(
    df: pd.DataFrame,
    n_gds: int,
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Keep all TPS (label==1); randomly sample `n_gds` rows from GDS (label==0).
    Shuffles the combined set for reproducibility.
    """
    tps = df[df[label_col] == 1].copy()
    gds = df[df[label_col] == 0].copy()
    if len(gds) < n_gds:
        raise ValueError(
            f"Not enough GDS rows: have {len(gds)}, need {n_gds}. "
            "Lower --gds-sample-size or refresh data."
        )
    gds_sample = gds.sample(n=n_gds, random_state=random_state)
    out = pd.concat([tps, gds_sample], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def stratified_train_val_test(
    df: pd.DataFrame,
    y_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Stratified split: train 70%, validation 15%, test 15% of full data.
    """
    if abs(test_size + val_size - 0.30) > 1e-6:
        raise ValueError("test_size + val_size must equal 0.30 for 70/15/15 split.")
    X = df.drop(columns=[y_col])
    y = df[y_col]
    # First: hold out test_size (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    # Remaining 85% -> train 70/85 and val 15/85
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
