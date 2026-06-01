"""
Build C1′: trajectory + divergence for all labeled posts from raw API JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from trajectory.metrics import TrajectoryMetrics, early_late_shift, linear_slope, mean_abs_step, mean_or_zero
from trajectory.sentiment import compound_score

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = REPO_ROOT / "tps_gds_classification" / "data" / "updated_labeled_dataset_unique.csv"
DEFAULT_RAW_DIR = REPO_ROOT / "tps_gds_classification" / "data" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "c1_prime"


def load_raw_posts_by_id(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Merge tps_response.json and gds_response.json posts by redditId."""
    by_id: dict[str, dict[str, Any]] = {}
    for fname in ("tps_response.json", "gds_response.json"):
        path = raw_dir / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        posts = payload.get("data") or []
        for post in posts:
            rid = str(post.get("redditId") or post.get("reddit_id") or "").strip()
            if rid:
                by_id[rid] = post
    return by_id


def _sort_key(comment: dict[str, Any]) -> str:
    return str(comment.get("created_utc") or "")


def analyze_post_trajectories(
    post: dict[str, Any],
    analyzer: SentimentIntensityAnalyzer,
) -> tuple[list[float], list[float], float]:
    """
    Returns (author_compounds, community_compounds, title_compound)
    in chronological comment order within each role track.
    """
    title = post.get("title") or ""
    title_compound = compound_score(analyzer, title)

    author_name = (post.get("author") or "").strip()
    comments = sorted(post.get("comments") or [], key=_sort_key)

    author_traj: list[float] = []
    community_traj: list[float] = []

    # OP voice in post body (selftext) precedes reply comments — aligns with C2 author track intent
    body = (post.get("author_description") or "").strip()
    if body:
        author_traj.append(compound_score(analyzer, body))

    for c in comments:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        comp = compound_score(analyzer, body)
        is_author = bool(c.get("is_submitter")) or (
            author_name and (c.get("author") or "").strip() == author_name
        )
        if is_author:
            author_traj.append(comp)
        else:
            community_traj.append(comp)

    if not author_traj:
        author_traj.append(title_compound)

    return author_traj, community_traj, title_compound


def compute_metrics_row(
    reddit_id: str,
    subreddit: str,
    corrected_label: int,
    url: str,
    author_traj: list[float],
    community_traj: list[float],
    title_compound: float,
    min_author: int = 1,
    min_community: int = 1,
) -> TrajectoryMetrics:
    author_mean = mean_or_zero(author_traj)
    community_mean = mean_or_zero(community_traj)
    divergence = abs(author_mean - community_mean)

    notes: list[str] = []
    eligible = True
    if len(author_traj) < min_author:
        notes.append(f"author_replies<{min_author}")
        eligible = False
    if len(community_traj) < min_community:
        notes.append(f"community_comments<{min_community}")
        eligible = False

    return TrajectoryMetrics(
        reddit_id=reddit_id,
        subreddit=subreddit,
        corrected_label=int(corrected_label),
        url=url,
        title_compound=round(title_compound, 4),
        n_author_replies=len(author_traj),
        n_community_comments=len(community_traj),
        n_comments_total=len(author_traj) + len(community_traj),
        author_mean_sentiment=round(author_mean, 4),
        community_mean_sentiment=round(community_mean, 4),
        sentiment_divergence=round(divergence, 4),
        author_more_negative=author_mean < community_mean,
        author_trend=round(linear_slope(author_traj), 4),
        community_trend=round(linear_slope(community_traj), 4),
        author_volatility=round(mean_abs_step(author_traj), 4),
        community_volatility=round(mean_abs_step(community_traj), 4),
        author_early_late_shift=round(early_late_shift(author_traj), 4),
        community_early_late_shift=round(early_late_shift(community_traj), 4),
        trajectory_eligible=eligible,
        trajectory_notes="; ".join(notes) if notes else "ok",
    )


def build_c1_prime(
    labels_path: Path = DEFAULT_LABELS,
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    include_trajectory_arrays: bool = True,
    min_author: int = 1,
    min_community: int = 1,
) -> Path:
    """
    Build C1′ artifacts for all labeled reddit_ids.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(labels_path)
    labels["reddit_id"] = labels["reddit_id"].astype(str).str.strip()

    posts_by_id = load_raw_posts_by_id(raw_dir)
    analyzer = SentimentIntensityAnalyzer()

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for _, lab in labels.iterrows():
        rid = lab["reddit_id"]
        post = posts_by_id.get(rid)
        if not post:
            missing_ids.append(rid)
            continue

        author_traj, community_traj, title_c = analyze_post_trajectories(post, analyzer)
        m = compute_metrics_row(
            reddit_id=rid,
            subreddit=str(lab.get("subreddit") or post.get("subreddit") or ""),
            corrected_label=int(lab["corrected_label"]),
            url=str(lab.get("url") or post.get("url") or ""),
            author_traj=author_traj,
            community_traj=community_traj,
            title_compound=title_c,
            min_author=min_author,
            min_community=min_community,
        )
        row = m.to_dict()
        rows.append(row)

        rec: dict[str, Any] = {**row}
        if include_trajectory_arrays:
            rec["author_trajectory"] = [round(v, 4) for v in author_traj]
            rec["community_trajectory"] = [round(v, 4) for v in community_traj]
        records.append(rec)

    metrics_df = pd.DataFrame(rows)
    csv_path = out_dir / "c1_prime_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    dataset = {
        "meta": {
            "name": "ReleaseTrain-TPSGDS-542-Trajectories",
            "version": "1.0",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_labeled": int(len(labels)),
            "n_built": int(len(rows)),
            "n_missing_raw": int(len(missing_ids)),
            "sentiment": "VADER compound",
            "role_separation": "is_submitter or author match; author_description as first author point",
            "comment_order": "created_utc ascending",
            "labels_path": str(labels_path),
            "raw_dir": str(raw_dir),
        },
        "records": records,
    }
    json_path = out_dir / "c1_prime_dataset.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    build_meta = {
        **dataset["meta"],
        "missing_reddit_ids": missing_ids[:50],
        "n_missing_list_truncated": len(missing_ids) > 50,
        "tier_note": "Use trajectory_eligible flag for strict cohort; full 542 for population stats.",
    }
    with (out_dir / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(build_meta, f, indent=2)

    print(f"C1′ built: {len(rows)}/{len(labels)} posts → {out_dir}")
    if missing_ids:
        print(f"  Warning: {len(missing_ids)} labeled IDs missing from raw JSON", file=__import__("sys").stderr)
    return out_dir
