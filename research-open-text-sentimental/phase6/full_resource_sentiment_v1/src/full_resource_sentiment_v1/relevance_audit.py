"""Phase 6 human relevance-audit sampler. Does not create labels or read protected outcomes."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .adapter import extract_post_body
from .broad_sentiment import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    git_revision,
    utcnow,
    verify_capture_pages,
)
from .hashing import sha256_file, sha256_text
from .io_utils import iter_jsonl, read_json
from .paths import (
    BROAD_CAPTURE_DIR,
    BROAD_COMPACT_INDEX,
    BROAD_SENTIMENT_RUN_DIR,
    RESULTS_DIR,
    WORKSPACE,
)
from .privacy import scan_paths
from .scoring import is_reddit_url

SAMPLE_SEED = 20261028
N_PER_STRATUM = 120
CAPTURE_ID = "2026-10-22-main"
OCTOBER_MONTH = "2026-10"
RESTRICTED_DIR = WORKSPACE / "restricted" / "relevance_audit_sample_2026-10-28"
PUBLIC_DIR = RESULTS_DIR / "relevance_audit_sample_2026-10-28"
I_ARCHIVE_DIR = WORKSPACE / "restricted" / "accepted_workstream_i_2026-10-28"
I_LIVE_DIR = RESULTS_DIR / "workstream_i_2026-10-22-main"
PROTECTED_MEMBERSHIPS = frozenset({"reserved_test", "fresh_500"})
PACKET_FIELDS = ("case_code", "title", "body", "relevance", "evidence_phrase", "uncertainty_note")
PACKET_FORBIDDEN_KEYS = {
    "reddit_id",
    "redditId",
    "author",
    "username",
    "url",
    "permalink",
    "comments",
    "created_month",
    "stratum",
    "period",
    "count_gap",
    "reserved_membership",
    "fresh_membership",
    "s2_target",
    "s2_in_scope",
    "compound",
    "title_compound",
    "body_compound",
    "source",
    "api_route",
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]*)?(?:\(?\d{3}\)?[\s.-]*)\d{3}[\s.-]\d{4}(?!\d)")
HANDLE_RE = re.compile(r"(?<!\w)u/[A-Za-z0-9_-]+")

EXPECTED_FRAME = {
    "n_primary": 37596,
    "n_october_before_exclusion": 20412,
    "n_older_before_exclusion": 17184,
    "n_protected_union": 693,
    "n_reserved": 197,
    "n_fresh": 496,
    "n_eligible_october": 20337,
    "n_eligible_older": 16566,
}

LABEL_GUIDE = """# Relevance audit — reviewer instructions

Status: first-pass labeling only. Do not consult live Reddit, other reviewers, or any model/API output.

## What you will see

Each case has a random case code, a saved original-post title, and a saved original-post body. Usernames, comments, dates, routes, predicted labels, and sentiment scores are hidden. Some contact details, URLs, or Reddit handles may appear as placeholders such as `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_URL]`, or `[REDACTED_HANDLE]`.

## Labels (choose one)

- `relevant`: the original author explicitly discusses an actual, proposed, failed, deferred, or reversed update/upgrade/patch/release/version change to software, an operating system, an app, or firmware, or a consequence attributed to that change. Troubleshooting is not required. Multiple updates in one post remain relevant if at least one meets this rule.
- `not_relevant`: “update” means a news/status refresh; the post discusses a general bug or product without an identifiable software-version change; or the only update link would have to come from a commenter rather than the original post.
- `unclear`: the original text is genuinely too thin or conflicting to decide.
- `unassessable`: original-post title and body are absent or corrupt.

Do not infer an update merely from a subreddit name or from a guess about what a model would say.

## Also record

- A short evidence phrase copied or paraphrased from the shown title/body.
- An uncertainty note (ok to leave blank if you are sure).

Lock this first pass. Do not change labels after discussion.
"""


def _ids_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(str(i) for i in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sequence_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(str(i) for i in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stratum_of_month(created_month: str | None) -> str:
    if created_month == OCTOBER_MONTH:
        return "october_2026"
    return "older"


def is_protected_membership(memberships: Iterable[str] | None) -> bool:
    return bool(PROTECTED_MEMBERSHIPS.intersection(memberships or []))


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts = {"email": 0, "phone": 0, "url": 0, "handle": 0, "handle_flagged_not_redacted": 0}

    def sub_email(match: re.Match[str]) -> str:
        counts["email"] += 1
        return "[REDACTED_EMAIL]"

    def sub_phone(match: re.Match[str]) -> str:
        counts["phone"] += 1
        return "[REDACTED_PHONE]"

    def sub_url(match: re.Match[str]) -> str:
        counts["url"] += 1
        return "[REDACTED_URL]"

    def sub_handle(match: re.Match[str]) -> str:
        counts["handle"] += 1
        return "[REDACTED_HANDLE]"

    redacted = EMAIL_RE.sub(sub_email, text)
    redacted = URL_RE.sub(sub_url, redacted)
    redacted = PHONE_RE.sub(sub_phone, redacted)
    redacted = HANDLE_RE.sub(sub_handle, redacted)
    counts["handle_flagged_not_redacted"] = 0
    return redacted, counts


def unredacted_handles(text: str) -> list[str]:
    return HANDLE_RE.findall(text or "")


def assert_packets_have_zero_handles(rows: list[dict[str, Any]], *, source: str) -> None:
    leftover = []
    for row in rows:
        hits = unredacted_handles(row.get("title", "")) + unredacted_handles(row.get("body", ""))
        if hits:
            leftover.append({"case_code": row.get("case_code"), "n_hits": len(hits)})
    if leftover:
        raise ValueError(f"{source} still contains u/ handles: {leftover}")


def packet_is_blinded(row: dict[str, Any]) -> list[str]:
    hits = [key for key in row if key in PACKET_FORBIDDEN_KEYS]
    hits.extend(f"unexpected:{key}" for key in row if key not in PACKET_FIELDS)
    if row.get("relevance") not in ("", None):
        hits.append("preassigned_relevance")
    return hits


def verify_g_hashes(run_dir: Path = BROAD_SENTIMENT_RUN_DIR) -> dict[str, str]:
    declared = read_json(run_dir / "hashes.json")
    names = (
        "post_scores.jsonl",
        "comment_scores.jsonl",
        "thread_metrics.jsonl",
        "analysis_A.json",
        "analysis_B.json",
        "analysis_C.json",
        "analysis_D.json",
        "analysis_E.json",
        "analysis_F.json",
    )
    actual = {name: sha256_file(run_dir / name) for name in names}
    mismatches = {name: {"declared": declared[name], "actual": actual[name]} for name in names if declared[name] != actual[name]}
    if mismatches:
        raise ValueError(f"Workstream G hash mismatch: {mismatches}")
    return actual


def verify_i_archive(archive_dir: Path = I_ARCHIVE_DIR, live_dir: Path = I_LIVE_DIR) -> dict[str, Any]:
    manifest = read_json(archive_dir / "ARCHIVE_MANIFEST.json")
    rows = []
    for item in manifest["files"]:
        archived = archive_dir / item["archive"]
        actual = sha256_file(archived)
        if actual != item["sha256"] or archived.stat().st_size != item["bytes"]:
            raise ValueError(f"I archive mismatch for {item['archive']}")
        live = WORKSPACE / item["source"]
        live_hash = sha256_file(live) if live.exists() else None
        if live_hash is not None and live_hash != item["sha256"]:
            raise ValueError(f"live I/G-related file drifted from accepted archive: {item['source']}")
        rows.append({"archive": item["archive"], "sha256": actual, "live_match": live_hash == actual})
    prefix_hash = sha256_file(archive_dir / "restricted" / "workstream_i_2026-10-22-main" / "prefix_diagnostics.jsonl")
    if prefix_hash != manifest.get("restricted_prefix_sha256"):
        raise ValueError("archived prefix diagnostics hash mismatch")
    if not live_dir.exists():
        raise ValueError("live Workstream I directory missing")
    return {"n_files": len(rows), "files": rows, "restricted_prefix_sha256": prefix_hash}


def load_primary_frame(
    *,
    post_scores_path: Path = BROAD_SENTIMENT_RUN_DIR / "post_scores.jsonl",
    compact_index_path: Path = BROAD_COMPACT_INDEX,
) -> dict[str, Any]:
    posts: dict[str, dict[str, Any]] = {}
    with post_scores_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = str(row["reddit_id"])
            if rid in posts:
                raise ValueError(f"duplicate post_scores id {rid}")
            posts[rid] = {
                "reddit_id": rid,
                "created_month": row.get("created_month"),
                "stratum": stratum_of_month(row.get("created_month")),
                "reserved_membership": bool(row.get("reserved_membership")),
                "fresh_membership": bool(row.get("fresh_membership")),
                "count_gap": bool(row.get("count_gap")),
                "title_nonempty": bool(row.get("title_nonempty")),
                "body_nonempty": bool(row.get("body_nonempty")),
                "capture_page": row.get("capture_page"),
            }
    index: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(compact_index_path):
        if not row.get("reddit_url"):
            continue
        rid = str(row["reddit_id"])
        if rid in index:
            raise ValueError(f"duplicate compact reddit_id {rid}")
        memberships = list(row.get("local_memberships") or [])
        index[rid] = {
            "local_memberships": memberships,
            "created_month": row.get("created_month"),
            "protected": is_protected_membership(memberships),
        }
    if set(posts) != set(index):
        raise ValueError("primary frame IDs differ between post_scores and compact index")
    mismatches = []
    reserved_ids = []
    fresh_ids = []
    for rid, post in posts.items():
        idx = index[rid]
        idx_reserved = "reserved_test" in idx["local_memberships"]
        idx_fresh = "fresh_500" in idx["local_memberships"]
        if post["reserved_membership"] != idx_reserved or post["fresh_membership"] != idx_fresh:
            mismatches.append(rid)
        if post["created_month"] != idx["created_month"]:
            mismatches.append(rid)
        if post["reserved_membership"]:
            reserved_ids.append(rid)
        if post["fresh_membership"]:
            fresh_ids.append(rid)
    if mismatches:
        raise ValueError(f"membership/month mismatch vs compact index: {mismatches[:10]}")
    union = sorted(set(reserved_ids) | set(fresh_ids))
    eligible = []
    for rid, post in posts.items():
        if post["reserved_membership"] or post["fresh_membership"]:
            continue
        eligible.append(post)
    october = [row["reddit_id"] for row in eligible if row["stratum"] == "october_2026"]
    older = [row["reddit_id"] for row in eligible if row["stratum"] == "older"]
    counts = {
        "n_primary": len(posts),
        "n_october_before_exclusion": sum(1 for row in posts.values() if row["stratum"] == "october_2026"),
        "n_older_before_exclusion": sum(1 for row in posts.values() if row["stratum"] == "older"),
        "n_reserved": len(reserved_ids),
        "n_fresh": len(fresh_ids),
        "n_protected_union": len(union),
        "n_protected_both": len(set(reserved_ids) & set(fresh_ids)),
        "n_eligible": len(eligible),
        "n_eligible_october": len(october),
        "n_eligible_older": len(older),
        "n_eligible_count_gap": sum(1 for row in eligible if row["count_gap"]),
    }
    for key, expected in EXPECTED_FRAME.items():
        if counts[key] != expected:
            raise ValueError(f"frame count {key}={counts[key]} != {expected}")
    return {
        "posts": posts,
        "eligible": {row["reddit_id"]: row for row in eligible},
        "october_ids": october,
        "older_ids": older,
        "reserved_ids": sorted(reserved_ids),
        "fresh_ids": sorted(fresh_ids),
        "protected_union_ids": union,
        "counts": counts,
        "reserved_ids_sha256": _ids_hash(reserved_ids),
        "fresh_ids_sha256": _ids_hash(fresh_ids),
        "protected_union_ids_sha256": _ids_hash(union),
    }


def draw_stratified_sample(
    october_ids: list[str],
    older_ids: list[str],
    *,
    seed: int = SAMPLE_SEED,
    n_per_stratum: int = N_PER_STRATUM,
) -> dict[str, Any]:
    if len(set(october_ids)) != len(october_ids) or len(set(older_ids)) != len(older_ids):
        raise ValueError("duplicate IDs in a stratum")
    if set(october_ids) & set(older_ids):
        raise ValueError("stratum ID overlap")
    oct_sorted = sorted(october_ids)
    older_sorted = sorted(older_ids)
    rng = random.Random(seed)
    oct_sample = rng.sample(oct_sorted, n_per_stratum)
    older_sample = rng.sample(older_sorted, n_per_stratum)
    if len(set(oct_sample)) != n_per_stratum or len(set(older_sample)) != n_per_stratum:
        raise ValueError("sample has duplicates")
    selected = [{"reddit_id": rid, "stratum": "october_2026"} for rid in oct_sample]
    selected.extend({"reddit_id": rid, "stratum": "older"} for rid in older_sample)
    selected.sort(key=lambda row: row["reddit_id"])
    used_codes: set[str] = set()
    for row in selected:
        while True:
            code = f"C-{rng.randrange(0, 1_000_000):06d}"
            if code not in used_codes:
                used_codes.add(code)
                row["case_code"] = code
                break
    codes = [row["case_code"] for row in selected]
    order_a = list(codes)
    rng.shuffle(order_a)
    order_b = list(codes)
    rng.shuffle(order_b)
    while order_b == order_a:
        rng.shuffle(order_b)
    by_code = {row["case_code"]: row for row in selected}
    return {
        "seed": seed,
        "n_per_stratum": n_per_stratum,
        "october_ids": sorted(oct_sample),
        "older_ids": sorted(older_sample),
        "all_ids": [row["reddit_id"] for row in selected],
        "rows": selected,
        "reviewer_a_order": order_a,
        "reviewer_b_order": order_b,
        "by_code": by_code,
        "rng_algorithm": "Python random.Random.sample then unique C-###### codes on lexicographic IDs, then independent shuffles",
    }


def collect_selected_text(
    selected_ids: set[str],
    *,
    capture_dir: Path = BROAD_CAPTURE_DIR,
) -> dict[str, dict[str, Any]]:
    capture = verify_capture_pages(capture_dir)
    found: dict[str, dict[str, Any]] = {}
    for page in capture["pages"]:
        payload = json.loads((capture_dir / page["file"]).read_text(encoding="utf-8"))
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"page schema failure {page['file']}")
        for raw in rows:
            if not isinstance(raw, dict) or not raw.get("redditId"):
                continue
            if not is_reddit_url(raw.get("url")):
                continue
            rid = str(raw["redditId"]).strip()
            if rid not in selected_ids:
                continue
            title = "" if raw.get("title") is None else str(raw.get("title") or "")
            body, body_field, _present = extract_post_body(raw)
            found[rid] = {
                "reddit_id": rid,
                "title": title,
                "body": body,
                "body_field": body_field,
                "title_nonempty": bool(title.strip()),
                "body_nonempty": bool(body.strip()),
            }
    missing = sorted(selected_ids - set(found))
    if missing:
        raise ValueError(f"selected IDs missing from capture text pass: {missing[:10]}")
    return found


def build_review_rows(
    sample: dict[str, Any],
    texts: dict[str, dict[str, Any]],
    eligible: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packets_by_code: dict[str, dict[str, str]] = {}
    private_rows: list[dict[str, Any]] = []
    for row in sample["rows"]:
        rid = row["reddit_id"]
        text = texts[rid]
        title_redacted, title_pii = redact_pii(text["title"])
        body_redacted, body_pii = redact_pii(text["body"])
        readable = bool(text["title"].strip() or text["body"].strip())
        packet = {
            "case_code": row["case_code"],
            "title": title_redacted,
            "body": body_redacted,
            "relevance": "",
            "evidence_phrase": "",
            "uncertainty_note": "",
        }
        hits = packet_is_blinded(packet)
        if hits:
            raise ValueError(f"packet blinding failure {row['case_code']}: {hits}")
        packets_by_code[row["case_code"]] = packet
        private_rows.append(
            {
                "case_code": row["case_code"],
                "reddit_id": rid,
                "stratum": row["stratum"],
                "count_gap": bool(eligible[rid]["count_gap"]),
                "created_month": eligible[rid]["created_month"],
                "title_sha256": sha256_text(text["title"]),
                "body_sha256": sha256_text(text["body"]),
                "packet_title_sha256": sha256_text(title_redacted),
                "packet_body_sha256": sha256_text(body_redacted),
                "readable_original_post_text": readable,
                "title_nonempty": text["title_nonempty"],
                "body_nonempty": text["body_nonempty"],
                "pii_title": title_pii,
                "pii_body": body_pii,
                "protected_membership": False,
            }
        )
    if any(row["protected_membership"] for row in private_rows):
        raise ValueError("protected ID present in private mapping")
    if len({row["reddit_id"] for row in private_rows}) != len(private_rows):
        raise ValueError("duplicate sampled reddit_id")
    return [packets_by_code[code] for code in sample["reviewer_a_order"]], private_rows


def packet_csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(PACKET_FIELDS), quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in PACKET_FIELDS})
    return buffer.getvalue()


def _code_fingerprint() -> tuple[str, list[str]]:
    files = [
        WORKSPACE / "src" / "full_resource_sentiment_v1" / "relevance_audit.py",
        WORKSPACE / "scripts" / "prepare_relevance_audit_sample.py",
        WORKSPACE / "scripts" / "correct_relevance_audit_packet_handles.py",
    ]
    digest = hashlib.sha256()
    parts = []
    for path in files:
        file_hash = sha256_file(path)
        parts.append(f"{path.name}:{file_hash}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), parts


def write_docs(
    *,
    public_dir: Path,
    restricted_dir: Path,
    manifest: dict[str, Any],
    sample: dict[str, Any],
    frame_counts: dict[str, Any],
) -> None:
    selected_oct = "\n".join(sample["october_ids"])
    selected_older = "\n".join(sample["older_ids"])
    run_md = f"""# Relevance audit sampling run

Capture: `{manifest['capture_id']}`
Status: **Awaiting Codex sampling validation**

```
{WORKSPACE}
```

```bash
.venv/bin/python -m pytest -q tests/test_relevance_audit_sample.py
.venv/bin/python scripts/prepare_relevance_audit_sample.py
```

Seed: `{manifest['seed']}`
Elapsed seconds: {manifest['elapsed_seconds']:.1f}
"""
    report = f"""# Relevance audit sampling execution report

Date: {manifest['run_finished_utc'][:10]}. Executor: Cursor. Status for Codex: **Awaiting Codex sampling validation**. No relevance labels were assigned. Packets are not released to reviewers until Codex validates sampling and blinding.

Workspace: `{WORKSPACE}`
Restricted packets: `{restricted_dir}`
Public sampling docs: `{public_dir}`

## Commands

```bash
.venv/bin/python -m pytest -q tests/test_relevance_audit_sample.py
.venv/bin/python scripts/prepare_relevance_audit_sample.py
```

Seed: `{SAMPLE_SEED}`. Python `random.Random({SAMPLE_SEED})`. 120 IDs without replacement from each eligible stratum after lexicographic sort. October sampled first, then older. Case codes assigned on lexicographic selected IDs. Reviewer A/B orders are independent shuffles of those codes.

## Eligible frame (membership flags only)

Primary Reddit-URL frame: {frame_counts['n_primary']}.
October before exclusion: {frame_counts['n_october_before_exclusion']}; older before exclusion: {frame_counts['n_older_before_exclusion']}.
Protected excluded union: {frame_counts['n_protected_union']} (reserved {frame_counts['n_reserved']}, fresh {frame_counts['n_fresh']}, both {frame_counts['n_protected_both']}).
Eligible October: {frame_counts['n_eligible_october']}; eligible older: {frame_counts['n_eligible_older']}.
Eligible count-gap retained in the frame: {frame_counts['n_eligible_count_gap']}.
Protected-union ID SHA-256: `{manifest['protected_union_ids_sha256']}`.

## Sample

N=240. October 2026: 120. Older: 120.
Selected-ID SHA-256: `{manifest['selected_ids_sha256']}`.
Count-gap IDs in sample: {manifest['n_selected_count_gap']}.
Unreadable original-post text in sample: {manifest['n_unreadable']}.
Protected IDs in sample: 0.

October IDs (lexicographic):

```
{selected_oct}
```

Older IDs (lexicographic):

```
{selected_older}
```

## Packets

Two blinded packets (JSONL + CSV) contain only `case_code`, `title`, `body`, and empty coding fields. PII/URL placeholders may appear. The private key maps case codes to Reddit IDs and is restricted. Reviewer A/B presentation orders differ.

## Integrity

- G hashes unchanged: {manifest['g_outputs_modified'] is False}
- I archive/live match: {manifest['i_archive_ok']}
- Public privacy scan hits: {manifest['privacy_scan_hits']}
- Git revision: `{manifest.get('git_revision')}`
- Code fingerprint: `{manifest['code_fingerprint']}`

Output SHA-256 values are in `hashes.json`.

## Limits

This is a bounded audit sample of nonprotected ReleaseTrain candidates. No prevalence estimate, no auto-labels, no protected outcomes, no within-author trajectory, and no manuscript.
"""
    atomic_write_text(public_dir / "RUN.md", run_md)
    atomic_write_text(public_dir / "EXECUTION_REPORT.md", report)
    atomic_write_text(restricted_dir / "RUN.md", run_md)
    atomic_write_text(restricted_dir / "EXECUTION_REPORT.md", report)
    atomic_write_text(WORKSPACE / "RELEVANCE_AUDIT_SAMPLE_RUN.md", run_md)
    atomic_write_text(WORKSPACE / "RELEVANCE_AUDIT_SAMPLE_EXECUTION_REPORT.md", report)


def prepare_relevance_audit_sample(
    *,
    capture_dir: Path = BROAD_CAPTURE_DIR,
    g_run_dir: Path = BROAD_SENTIMENT_RUN_DIR,
    compact_index_path: Path = BROAD_COMPACT_INDEX,
    restricted_dir: Path = RESTRICTED_DIR,
    public_dir: Path = PUBLIC_DIR,
    archive_dir: Path = I_ARCHIVE_DIR,
) -> dict[str, Any]:
    import time

    started = time.monotonic()
    started_utc = utcnow()
    g_hashes_before = verify_g_hashes(g_run_dir)
    i_archive = verify_i_archive(archive_dir)
    frame = load_primary_frame(post_scores_path=g_run_dir / "post_scores.jsonl", compact_index_path=compact_index_path)
    sample = draw_stratified_sample(frame["october_ids"], frame["older_ids"], seed=SAMPLE_SEED)
    selected_set = set(sample["all_ids"])
    if selected_set & set(frame["protected_union_ids"]):
        raise ValueError("sample overlaps protected IDs")
    texts = collect_selected_text(selected_set, capture_dir=capture_dir)
    reviewer_a, private_rows = build_review_rows(sample, texts, frame["eligible"])
    by_code = {row["case_code"]: row for row in reviewer_a}
    reviewer_b = [dict(by_code[code]) for code in sample["reviewer_b_order"]]
    # rebuild A in stated order (build_review_rows already did)
    n_unreadable = sum(1 for row in private_rows if not row["readable_original_post_text"])
    n_gap = sum(1 for row in private_rows if row["count_gap"])
    pii_summary = {
        "n_email": sum(row["pii_title"]["email"] + row["pii_body"]["email"] for row in private_rows),
        "n_phone": sum(row["pii_title"]["phone"] + row["pii_body"]["phone"] for row in private_rows),
        "n_url": sum(row["pii_title"]["url"] + row["pii_body"]["url"] for row in private_rows),
        "n_handle": sum(row["pii_title"].get("handle", 0) + row["pii_body"].get("handle", 0) for row in private_rows),
        "n_handle_flagged_not_redacted": sum(
            row["pii_title"]["handle_flagged_not_redacted"] + row["pii_body"]["handle_flagged_not_redacted"] for row in private_rows
        ),
        "n_cases_with_redaction": sum(
            1
            for row in private_rows
            if any(row["pii_title"].get(k, 0) or row["pii_body"].get(k, 0) for k in ("email", "phone", "url", "handle"))
        ),
    }
    restricted_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(restricted_dir, 0o700)
    atomic_write_jsonl(restricted_dir / "reviewer_A_packet.jsonl", reviewer_a)
    atomic_write_jsonl(restricted_dir / "reviewer_B_packet.jsonl", reviewer_b)
    atomic_write_text(restricted_dir / "reviewer_A_packet.csv", packet_csv_text(reviewer_a))
    atomic_write_text(restricted_dir / "reviewer_B_packet.csv", packet_csv_text(reviewer_b))
    atomic_write_text(restricted_dir / "REVIEWER_INSTRUCTIONS.md", LABEL_GUIDE)
    atomic_write_jsonl(restricted_dir / "private_id_key.jsonl", sorted(private_rows, key=lambda row: row["case_code"]))
    atomic_write_json(
        restricted_dir / "excluded_protected_ids.json",
        {
            "note": "Membership flags only; no protected outcomes.",
            "reserved_ids": frame["reserved_ids"],
            "fresh_ids": frame["fresh_ids"],
            "union_ids": frame["protected_union_ids"],
        },
    )
    atomic_write_json(restricted_dir / "pii_audit.json", pii_summary)
    selected_payload = {
        "seed": SAMPLE_SEED,
        "n_october": N_PER_STRATUM,
        "n_older": N_PER_STRATUM,
        "october_2026_ids": sample["october_ids"],
        "older_ids": sample["older_ids"],
        "all_ids_sorted": sample["all_ids"],
        "selected_ids_sha256": _ids_hash(sample["all_ids"]),
    }
    atomic_write_json(public_dir / "selected_ids.json", selected_payload)
    atomic_write_json(restricted_dir / "selected_ids.json", selected_payload)
    public_names = ["selected_ids.json"]
    restricted_packet_names = [
        "reviewer_A_packet.jsonl",
        "reviewer_B_packet.jsonl",
        "reviewer_A_packet.csv",
        "reviewer_B_packet.csv",
        "REVIEWER_INSTRUCTIONS.md",
        "private_id_key.jsonl",
        "excluded_protected_ids.json",
        "pii_audit.json",
        "selected_ids.json",
    ]
    privacy_hits = scan_paths([public_dir / "selected_ids.json"])
    if privacy_hits:
        raise ValueError(f"public sampling docs failed privacy scan: {privacy_hits[:20]}")
    for packet_name in ("reviewer_A_packet.jsonl", "reviewer_B_packet.jsonl"):
        for row in iter_jsonl(restricted_dir / packet_name):
            hits = packet_is_blinded(row)
            if hits:
                raise ValueError(f"{packet_name} blinding failure: {hits}")
            if row.get("relevance") not in ("", None):
                raise ValueError("packet contains a preassigned relevance label")
        assert_packets_have_zero_handles(list(iter_jsonl(restricted_dir / packet_name)), source=packet_name)
    g_hashes_after = verify_g_hashes(g_run_dir)
    if g_hashes_after != g_hashes_before:
        raise ValueError("G outputs changed during sampling")
    verify_i_archive(archive_dir)
    elapsed = time.monotonic() - started
    fingerprint, code_files = _code_fingerprint()
    output_hashes = {name: sha256_file(public_dir / name) for name in public_names}
    output_hashes.update({f"restricted/{name}": sha256_file(restricted_dir / name) for name in restricted_packet_names})
    capture = verify_capture_pages(capture_dir)
    manifest = {
        "status": "awaiting_codex_sampling_validation",
        "complete": True,
        "capture_id": CAPTURE_ID,
        "run_started_utc": started_utc,
        "run_finished_utc": utcnow(),
        "elapsed_seconds": elapsed,
        "seed": SAMPLE_SEED,
        "n_per_stratum": N_PER_STRATUM,
        "n_sample": N_PER_STRATUM * 2,
        "rng": "Python random.Random",
        "git_revision": git_revision(),
        "code_fingerprint": fingerprint,
        "code_files": code_files,
        "g_input_hashes": g_hashes_before,
        "g_outputs_modified": False,
        "i_archive_ok": True,
        "i_archive_prefix_sha256": i_archive["restricted_prefix_sha256"],
        "capture_raw_manifest_sha256": capture["raw_manifest_sha256"],
        "frame_counts": frame["counts"],
        "protected_union_ids_sha256": frame["protected_union_ids_sha256"],
        "selected_ids_sha256": _ids_hash(sample["all_ids"]),
        "n_selected_count_gap": n_gap,
        "n_unreadable": n_unreadable,
        "pii_summary": pii_summary,
        "privacy_scan_hits": privacy_hits,
        "protected_outcomes_read": False,
        "labels_assigned": False,
        "validated_by_cursor": False,
        "output_sha256": output_hashes,
        "reviewer_a_order_sha256": _sequence_hash(sample["reviewer_a_order"]),
        "reviewer_b_order_sha256": _sequence_hash(sample["reviewer_b_order"]),
        "notes": [
            "Sampling only. Status is Awaiting Codex sampling validation.",
            "Packets must not be distributed until Codex confirms hashes, counts, blinding, and zero protected overlap.",
        ],
    }
    atomic_write_json(public_dir / "manifest.json", manifest)
    atomic_write_json(restricted_dir / "sampling_manifest.json", manifest)
    atomic_write_json(public_dir / "hashes.json", output_hashes)
    atomic_write_json(restricted_dir / "hashes.json", output_hashes)
    output_hashes["manifest.json"] = sha256_file(public_dir / "manifest.json")
    atomic_write_json(public_dir / "hashes.json", output_hashes)
    write_docs(
        public_dir=public_dir,
        restricted_dir=restricted_dir,
        manifest=manifest,
        sample=sample,
        frame_counts=frame["counts"],
    )
    print(json.dumps({"status": manifest["status"], "restricted_dir": str(restricted_dir), "elapsed_seconds": elapsed}, indent=2))
    return manifest


HANDLE_CORRECTION_CASES = ("C-291399", "C-175973")


def _pii_summary_from_private_rows(private_rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("email", "phone", "url", "handle")
    return {
        "n_email": sum(row["pii_title"].get("email", 0) + row["pii_body"].get("email", 0) for row in private_rows),
        "n_phone": sum(row["pii_title"].get("phone", 0) + row["pii_body"].get("phone", 0) for row in private_rows),
        "n_url": sum(row["pii_title"].get("url", 0) + row["pii_body"].get("url", 0) for row in private_rows),
        "n_handle": sum(row["pii_title"].get("handle", 0) + row["pii_body"].get("handle", 0) for row in private_rows),
        "n_handle_flagged_not_redacted": sum(
            row["pii_title"].get("handle_flagged_not_redacted", 0) + row["pii_body"].get("handle_flagged_not_redacted", 0)
            for row in private_rows
        ),
        "n_cases_with_redaction": sum(
            1 for row in private_rows if any(row["pii_title"].get(k, 0) or row["pii_body"].get(k, 0) for k in keys)
        ),
    }


def correct_prepared_packet_handles(
    *,
    restricted_dir: Path = RESTRICTED_DIR,
    public_dir: Path = PUBLIC_DIR,
    archive_dir: Path = I_ARCHIVE_DIR,
    g_run_dir: Path = BROAD_SENTIMENT_RUN_DIR,
) -> dict[str, Any]:
    """Replace remaining u/ handles in prepared packets. Does not redraw the sample."""
    g_before = verify_g_hashes(g_run_dir)
    verify_i_archive(archive_dir)
    selected_before = read_json(public_dir / "selected_ids.json")

    def load_packet(name: str) -> list[dict[str, Any]]:
        return list(iter_jsonl(restricted_dir / name))

    reviewer_a = load_packet("reviewer_A_packet.jsonl")
    reviewer_b = load_packet("reviewer_B_packet.jsonl")
    if [row["case_code"] for row in reviewer_a] == [row["case_code"] for row in reviewer_b]:
        raise ValueError("reviewer packet orders unexpectedly identical")
    if {row["case_code"] for row in reviewer_a} != {row["case_code"] for row in reviewer_b}:
        raise ValueError("reviewer packets have different case sets")

    corrected_cases: set[str] = set()

    def redact_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        new_title, title_handle = HANDLE_RE.subn("[REDACTED_HANDLE]", row["title"])
        new_body, body_handle = HANDLE_RE.subn("[REDACTED_HANDLE]", row["body"])
        if title_handle or body_handle:
            corrected_cases.add(row["case_code"])
        out = dict(row)
        out["title"] = new_title
        out["body"] = new_body
        if out.get("relevance") not in ("", None):
            raise ValueError("packet contains a preassigned relevance label")
        return out, {"title_handle": title_handle, "body_handle": body_handle}

    a_out = []
    b_out = []
    handle_counts: dict[str, dict[str, int]] = {}
    for row in reviewer_a:
        updated, counts = redact_row(row)
        a_out.append(updated)
        if counts["title_handle"] or counts["body_handle"]:
            handle_counts[row["case_code"]] = counts
    for row in reviewer_b:
        updated, counts = redact_row(row)
        b_out.append(updated)
        if row["case_code"] in handle_counts and handle_counts[row["case_code"]] != counts:
            raise ValueError(f"handle redaction mismatch between packets for {row['case_code']}")
    if corrected_cases != set(HANDLE_CORRECTION_CASES):
        raise ValueError(f"handle correction cases {sorted(corrected_cases)} != {list(HANDLE_CORRECTION_CASES)}")
    assert_packets_have_zero_handles(a_out, source="reviewer_A_packet.jsonl")
    assert_packets_have_zero_handles(b_out, source="reviewer_B_packet.jsonl")
    for row in a_out + b_out:
        hits = packet_is_blinded(row)
        if hits:
            raise ValueError(f"packet blinding failure after handle correction: {hits}")

    private_rows = list(iter_jsonl(restricted_dir / "private_id_key.jsonl"))
    original_text_hashes = {
        row["case_code"]: {"title_sha256": row["title_sha256"], "body_sha256": row["body_sha256"]} for row in private_rows
    }
    packet_by_code = {row["case_code"]: row for row in a_out}
    for row in private_rows:
        code = row["case_code"]
        if row["title_sha256"] != original_text_hashes[code]["title_sha256"] or row["body_sha256"] != original_text_hashes[code]["body_sha256"]:
            raise ValueError("original-text hashes drifted before correction")
        packet = packet_by_code[code]
        row["packet_title_sha256"] = sha256_text(packet["title"])
        row["packet_body_sha256"] = sha256_text(packet["body"])
        row["pii_title"] = dict(row.get("pii_title") or {})
        row["pii_body"] = dict(row.get("pii_body") or {})
        if code in HANDLE_CORRECTION_CASES:
            n_title = handle_counts[code]["title_handle"]
            n_body = handle_counts[code]["body_handle"]
            row["pii_title"]["handle"] = n_title
            row["pii_body"]["handle"] = n_body
            row["pii_title"]["handle_flagged_not_redacted"] = 0
            row["pii_body"]["handle_flagged_not_redacted"] = 0
        else:
            row["pii_title"].setdefault("handle", 0)
            row["pii_body"].setdefault("handle", 0)
        if row["title_sha256"] != original_text_hashes[code]["title_sha256"] or row["body_sha256"] != original_text_hashes[code]["body_sha256"]:
            raise ValueError("original-text hashes were modified")
    pii_summary = _pii_summary_from_private_rows(private_rows)

    atomic_write_jsonl(restricted_dir / "reviewer_A_packet.jsonl", a_out)
    atomic_write_jsonl(restricted_dir / "reviewer_B_packet.jsonl", b_out)
    atomic_write_text(restricted_dir / "reviewer_A_packet.csv", packet_csv_text(a_out))
    atomic_write_text(restricted_dir / "reviewer_B_packet.csv", packet_csv_text(b_out))
    atomic_write_jsonl(restricted_dir / "private_id_key.jsonl", private_rows)
    atomic_write_json(restricted_dir / "pii_audit.json", pii_summary)
    atomic_write_text(restricted_dir / "REVIEWER_INSTRUCTIONS.md", LABEL_GUIDE)

    assert_packets_have_zero_handles(list(iter_jsonl(restricted_dir / "reviewer_A_packet.jsonl")), source="reviewer_A_after_write")
    assert_packets_have_zero_handles(list(iter_jsonl(restricted_dir / "reviewer_B_packet.jsonl")), source="reviewer_B_after_write")
    for name in ("reviewer_A_packet.csv", "reviewer_B_packet.csv"):
        leftover = unredacted_handles((restricted_dir / name).read_text(encoding="utf-8"))
        if leftover:
            raise ValueError(f"{name} still contains u/ handles")

    selected_after = read_json(public_dir / "selected_ids.json")
    if selected_after != selected_before:
        raise ValueError("selected IDs changed during handle correction")
    g_after = verify_g_hashes(g_run_dir)
    if g_after != g_before:
        raise ValueError("G outputs changed during handle correction")
    verify_i_archive(archive_dir)

    restricted_packet_names = [
        "reviewer_A_packet.jsonl",
        "reviewer_B_packet.jsonl",
        "reviewer_A_packet.csv",
        "reviewer_B_packet.csv",
        "REVIEWER_INSTRUCTIONS.md",
        "private_id_key.jsonl",
        "excluded_protected_ids.json",
        "pii_audit.json",
        "selected_ids.json",
    ]
    output_hashes = {"selected_ids.json": sha256_file(public_dir / "selected_ids.json")}
    output_hashes.update({f"restricted/{name}": sha256_file(restricted_dir / name) for name in restricted_packet_names})
    fingerprint, code_files = _code_fingerprint()
    manifest = read_json(public_dir / "manifest.json")
    manifest["status"] = "awaiting_codex_packet_revalidation"
    manifest["pii_summary"] = pii_summary
    manifest["output_sha256"] = output_hashes
    manifest["code_fingerprint"] = fingerprint
    manifest["code_files"] = code_files
    manifest["privacy_correction"] = {
        "date_utc": utcnow(),
        "cases": list(HANDLE_CORRECTION_CASES),
        "replacement": "[REDACTED_HANDLE]",
        "sample_redrawn": False,
        "original_text_hashes_preserved": True,
        "zero_unredacted_handles": True,
    }
    manifest["notes"] = [
        "Sampling unchanged. Privacy correction replaced two u/ handles with [REDACTED_HANDLE].",
        "Status is Awaiting Codex packet revalidation. Packets must not be distributed yet.",
    ]
    manifest["validated_by_cursor"] = False
    manifest["labels_assigned"] = False
    manifest["g_outputs_modified"] = False
    atomic_write_json(public_dir / "manifest.json", manifest)
    atomic_write_json(restricted_dir / "sampling_manifest.json", manifest)
    output_hashes_with_manifest = dict(output_hashes)
    output_hashes_with_manifest["manifest.json"] = sha256_file(public_dir / "manifest.json")
    atomic_write_json(public_dir / "hashes.json", output_hashes_with_manifest)
    atomic_write_json(restricted_dir / "hashes.json", output_hashes_with_manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "corrected_cases": list(HANDLE_CORRECTION_CASES),
                "packet_hashes": {
                    "reviewer_A_packet.jsonl": output_hashes["restricted/reviewer_A_packet.jsonl"],
                    "reviewer_B_packet.jsonl": output_hashes["restricted/reviewer_B_packet.jsonl"],
                    "reviewer_A_packet.csv": output_hashes["restricted/reviewer_A_packet.csv"],
                    "reviewer_B_packet.csv": output_hashes["restricted/reviewer_B_packet.csv"],
                },
            },
            indent=2,
        )
    )
    return manifest
