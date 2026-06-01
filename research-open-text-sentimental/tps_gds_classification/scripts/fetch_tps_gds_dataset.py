#!/usr/bin/env python3
"""
Fetch candidate TPS / GDS posts from ReleaseTrain API, preprocess text, save dataset.

Endpoints provide CANDIDATE labels only — manual verification is recommended.

TPS (expanded, high-quality candidates): minComments=3, minScore=0.3
GDS candidates: minComments=3, maxScore=0.5

Run from repo root:
  python tps_gds_classification/scripts/fetch_tps_gds_dataset.py

Outputs:
  data/raw/tps_response.json   — full last TPS API response
  data/raw/gds_response.json   — full last GDS API response
  data/tps_gds_dataset.json    — records + meta
  data/tps_gds_dataset.csv
  data/tps_gds_text_label.json — minimal {text, label} for quick ML loads
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import requests

from text_preprocessing import clean_text, combine_text, top_comments

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_BASE = "https://releasetrain.io/api/reddit/query/filter"

# Pagination: offset/skip/page were tested against this API (2026-03) and return
# the same first page; use a sufficiently large `limit` per request instead.
DEFAULT_PAGE_LIMIT = 2000


def build_filter_url(min_comments: int, **score_filters: float | None) -> str:
    params: dict[str, Any] = {"minComments": min_comments}
    if "minScore" in score_filters and score_filters["minScore"] is not None:
        params["minScore"] = score_filters["minScore"]
    if "maxScore" in score_filters and score_filters["maxScore"] is not None:
        params["maxScore"] = score_filters["maxScore"]
    params["limit"] = score_filters.get("limit", DEFAULT_PAGE_LIMIT)
    return f"{API_BASE}?{urlencode(params)}"


def fetch_json(url: str, timeout: int = 120) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def save_raw(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_record(
    post: dict[str, Any],
    label: int,
    max_comments: int,
) -> dict[str, Any]:
    title = post.get("title") or ""
    body = post.get("author_description") or ""
    comments = post.get("comments") or []
    comment_bodies = top_comments(comments, max_comments)
    text_raw = combine_text(title, body, comment_bodies)
    text_clean = clean_text(text_raw)

    return {
        "_id": str(post.get("_id", "")),
        "reddit_id": post.get("redditId") or "",
        "url": post.get("url") or "",
        "subreddit": post.get("subreddit") or "",
        "label": label,
        "label_name": "TPS" if label == 1 else "GDS",
        "label_source": "api_candidate",
        "title_raw": title,
        "body_raw": body,
        "num_comments_in_post": len(comments),
        "comments_included": len(comment_bodies),
        "post_score": post.get("score"),
        "text_raw": text_raw,
        "text": text_clean,
        "char_len_raw": len(text_raw),
        "char_len_clean": len(text_clean),
    }


def write_dataset_json(path: Path, meta: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"meta": meta, "records": records}
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def write_text_label_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    minimal = [{"text": r["text"], "label": r["label"]} for r in records]
    with path.open("w", encoding="utf-8") as f:
        json.dump({"records": minimal}, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "_id",
                "reddit_id",
                "label",
                "text",
                "subreddit",
                "post_score",
                "num_comments_in_post",
                "label_source",
            ],
        )
        w.writeheader()
        for r in records:
            w.writerow(
                {
                    "_id": r["_id"],
                    "reddit_id": r["reddit_id"],
                    "label": r["label"],
                    "text": r["text"],
                    "subreddit": r["subreddit"],
                    "post_score": r["post_score"],
                    "num_comments_in_post": r["num_comments_in_post"],
                    "label_source": r["label_source"],
                }
            )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fetch TPS/GDS candidate dataset from ReleaseTrain API (with preprocessing).",
    )
    p.add_argument(
        "--max-comments",
        type=int,
        default=10,
        help="Max comments per post (by score, descending). Use 5–10 to limit noise.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PAGE_LIMIT,
        help="API `limit` per request (pagination params are ineffective; raise this to fetch more).",
    )
    p.add_argument(
        "--min-comments",
        type=int,
        default=3,
        help="API minComments filter (default 3).",
    )
    p.add_argument(
        "--skip-empty-text",
        action="store_true",
        help="Drop records where cleaned text is empty.",
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "raw",
        help="Directory for full API JSON responses.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_PROJECT_ROOT / "data" / "tps_gds_dataset.json",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=_PROJECT_ROOT / "data" / "tps_gds_dataset.csv",
    )
    p.add_argument(
        "--out-minimal-json",
        type=Path,
        default=_PROJECT_ROOT / "data" / "tps_gds_text_label.json",
    )
    args = p.parse_args()

    tps_url = build_filter_url(
        args.min_comments,
        minScore=0.3,
        maxScore=None,
        limit=args.limit,
    )
    gds_url = build_filter_url(
        args.min_comments,
        minScore=None,
        maxScore=0.5,
        limit=args.limit,
    )

    try:
        tps_payload = fetch_json(tps_url)
        gds_payload = fetch_json(gds_url)
    except requests.RequestException as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return 1

    ts = datetime.now(timezone.utc).isoformat()
    save_raw(args.raw_dir / "tps_response.json", {"fetched_at_utc": ts, **tps_payload})
    save_raw(args.raw_dir / "gds_response.json", {"fetched_at_utc": ts, **gds_payload})

    tps_posts = tps_payload.get("data") or []
    gds_posts = gds_payload.get("data") or []

    tps_records = [build_record(post, label=1, max_comments=args.max_comments) for post in tps_posts]
    gds_records = [build_record(post, label=0, max_comments=args.max_comments) for post in gds_posts]

    records = tps_records + gds_records
    if args.skip_empty_text:
        records = [r for r in records if r["text"]]

    meta = {
        "fetched_at_utc": ts,
        "n_records": len(records),
        "label_1_tps": sum(1 for r in records if r["label"] == 1),
        "label_0_gds": sum(1 for r in records if r["label"] == 0),
        "label_policy": (
            "Candidate labels: TPS from minScore=0.3 filter; GDS from maxScore=0.5 filter. "
            "Not gold labels — verify manually before publication."
        ),
        "pagination_note": (
            "ReleaseTrain filter API ignores offset/skip/page in tests; "
            "fetch uses a single request per endpoint with `limit`. "
            "Increase --limit if the backend supports more rows per call."
        ),
        "endpoints": {
            "tps_candidates": tps_url,
            "gds_candidates": gds_url,
        },
        "imbalance_handling_note": (
            "Prefer class_weight / weighted loss in training rather than dropping rows; "
            "optionally verify and relabel before comparing models."
        ),
    }

    write_dataset_json(args.out_json, meta, records)
    write_csv(args.out_csv, records)
    write_text_label_json(args.out_minimal_json, records)

    summary = {
        "fetched_tps_posts": len(tps_posts),
        "fetched_gds_posts": len(gds_posts),
        "saved_records": len(records),
        "label_tps": meta["label_1_tps"],
        "label_gds": meta["label_0_gds"],
        "out_json": str(args.out_json),
        "out_csv": str(args.out_csv),
        "out_minimal_json": str(args.out_minimal_json),
        "raw_tps": str(args.raw_dir / "tps_response.json"),
        "raw_gds": str(args.raw_dir / "gds_response.json"),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
