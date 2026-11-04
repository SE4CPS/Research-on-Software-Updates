#!/usr/bin/env python3
"""Prepare blinded TPS/GDS reannotation packets for the 179 F-set Other posts.

Read-only on frozen sources. Writes only to the dated results/restricted dirs.
Does not auto-label, call APIs, or read protected evaluation outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE6 = Path(__file__).resolve().parents[1]  # .../phase6/full_resource_sentiment_v1
WORKSPACE = PHASE6.parents[1]  # research-open-text-sentimental

DEVELOPMENT_V2 = WORKSPACE / "phase3/revised_development_v2/development_v2.jsonl"
DEVELOPMENT_MANIFEST = WORKSPACE / "phase3/revised_development_v2/MANIFEST.json"
POST_SCORES = PHASE6 / "results/broad_sentiment_2026-10-22-main/post_scores.jsonl"
THREAD_METRICS = PHASE6 / "results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl"
CAPTURE_DIR = PHASE6 / "restricted/broad_release_train/2026-10-22-main"
PROTECTED_IDS = PHASE6 / "restricted/relevance_audit_sample_2026-10-28/excluded_protected_ids.json"
ANNOTATION_GUIDELINES = WORKSPACE / "ANNOTATION_GUIDELINES.md"
PHASE2_REVIEWER_GUIDE = WORKSPACE / "phase2/annotation/REVIEWER_GUIDE.md"

PUBLIC_DIR = PHASE6 / "results/tps_gds_other_reannotation_2026-11-02"
RESTRICTED_DIR = PHASE6 / "restricted/tps_gds_other_reannotation_2026-11-02"

RUN_ID = "tps-gds-other-reannotation-2026-11-02"
LABELS = ("TPS", "GDS", "Other", "Unclear")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    return text.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_development() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(DEVELOPMENT_V2):
        out[str(row["reddit_id"])] = row
    return out


def load_post_scores() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(POST_SCORES):
        out[str(row["reddit_id"])] = row
    return out


def load_thread_metrics() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(THREAD_METRICS):
        out[str(row["reddit_id"])] = row
    return out


def load_protected_union() -> set[str]:
    if not PROTECTED_IDS.exists():
        return set()
    payload = json.loads(PROTECTED_IDS.read_text(encoding="utf-8"))
    return {str(x) for x in (payload.get("union_ids") or [])}


def load_capture_post(reddit_id: str, capture_page: int | None, cache: dict[int, dict[str, dict]]) -> dict[str, Any] | None:
    if capture_page is None:
        return None
    page = int(capture_page)
    if page not in cache:
        path = CAPTURE_DIR / f"page_{page:05d}.json"
        if not path.exists():
            cache[page] = {}
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        by_id = {}
        for post in data.get("data") or []:
            rid = str(post.get("redditId") or "")
            if rid:
                by_id[rid] = post
        cache[page] = by_id
    return cache[page].get(reddit_id)


def format_comment_block(comments: list[dict[str, Any]]) -> tuple[str, str, str, int, int, int]:
    """Return full context, author-only, community-only, and counts."""
    author_parts: list[str] = []
    community_parts: list[str] = []
    all_parts: list[str] = []
    n_author = 0
    n_community = 0
    n_valid_ts = 0
    for idx, comment in enumerate(comments, start=1):
        body = clean_text(comment.get("body"))
        if not body:
            continue
        is_submitter = bool(comment.get("is_submitter"))
        role = "ORIGINAL AUTHOR" if is_submitter else "COMMUNITY"
        ts = clean_text(comment.get("created_utc"))
        if ts:
            n_valid_ts += 1
        header = f"[{idx}] {role}"
        if ts:
            header += f" | {ts}"
        block = f"{header}\n{body}"
        all_parts.append(block)
        if is_submitter:
            n_author += 1
            author_parts.append(block)
        else:
            n_community += 1
            community_parts.append(block)
    sep = "\n\n----\n\n"
    return (
        sep.join(all_parts),
        sep.join(author_parts),
        sep.join(community_parts),
        n_author,
        n_community,
        n_valid_ts,
    )


def build_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    development = load_development()
    post_scores = load_post_scores()
    thread_metrics = load_thread_metrics()
    protected_union = load_protected_union()
    page_cache: dict[int, dict[str, dict]] = {}

    f_label_counts = Counter()
    f_ids: list[str] = []
    for rid, ps in post_scores.items():
        if not ps.get("f_eligible"):
            continue
        f_ids.append(rid)
        dt = development.get(rid, {}).get("discussion_type")
        f_label_counts[dt] += 1

    other_ids = [
        rid
        for rid in f_ids
        if development.get(rid, {}).get("discussion_type") == "other"
    ]
    other_ids = sorted(set(other_ids))

    records: list[dict[str, Any]] = []
    for rid in other_ids:
        dev = development[rid]
        ps = post_scores[rid]
        tm = thread_metrics.get(rid) or {}
        capture_page = ps.get("capture_page")
        capture = load_capture_post(rid, capture_page, page_cache) or {}

        title = clean_text(capture.get("title") or dev.get("title"))
        body = clean_text(capture.get("author_description") or dev.get("post_body"))
        post_ts = clean_text(capture.get("created_utc") or dev.get("created_utc"))
        comments = list(capture.get("comments") or [])
        conversation, author_replies, community_replies, n_author, n_community, n_comment_ts = format_comment_block(
            comments
        )
        # Prefer development discussion_context only as private provenance note when capture empty.
        n_nonempty_comments = sum(1 for c in comments if clean_text(c.get("body")))
        has_saved_comments = n_nonempty_comments >= 1
        has_author_replies = n_author >= 1
        has_valid_post_ts = bool(post_ts)
        # Priority: comments first (1..66), then remaining (67..179)
        records.append(
            {
                "reddit_id": rid,
                "title": title,
                "post_body": body,
                "post_created_utc": post_ts,
                "subreddit": clean_text(capture.get("subreddit") or dev.get("subreddit")),
                "conversation_context": conversation,
                "author_replies": author_replies,
                "community_replies": community_replies,
                "n_nonempty_comments": n_nonempty_comments,
                "n_author_replies": n_author,
                "n_community_replies": n_community,
                "n_comment_timestamps": n_comment_ts,
                "has_saved_comments": has_saved_comments,
                "has_author_replies": has_author_replies,
                "has_valid_post_timestamp": has_valid_post_ts,
                "capture_page": capture_page,
                "capture_found": bool(capture),
                "body_source": (
                    "capture_author_description"
                    if clean_text(capture.get("author_description"))
                    else ("development_post_body" if clean_text(dev.get("post_body")) else "empty")
                ),
                "title_source": (
                    "capture_title"
                    if clean_text(capture.get("title"))
                    else ("development_title" if clean_text(dev.get("title")) else "empty")
                ),
                # private fields
                "original_discussion_type": dev.get("discussion_type"),
                "original_s2_target": dev.get("s2_target"),
                "original_s2_in_scope": dev.get("s2_in_scope"),
                "original_about_software": dev.get("about_software"),
                "original_update_related": dev.get("update_related"),
                "study_id": dev.get("study_id"),
                "text_availability": dev.get("text_availability"),
                "development_discussion_context": clean_text(dev.get("discussion_context")),
                "f_eligible": bool(ps.get("f_eligible")),
                "reserved_membership": bool(ps.get("reserved_membership") or tm.get("reserved_membership")),
                "fresh_membership": bool(ps.get("fresh_membership") or tm.get("fresh_membership")),
                "protected_union_membership": rid in protected_union,
                "author_reply_eligible_metric": bool(tm.get("author_reply_eligible")),
                "post_timestamp_valid_metric": bool(tm.get("post_timestamp_valid")),
                "title_sha256": sha256_text(title),
                "body_sha256": sha256_text(body),
                "conversation_sha256": sha256_text(conversation),
            }
        )

    # Sort: comments first, then by reddit_id within band
    with_comments = sorted(
        [r for r in records if r["has_saved_comments"]],
        key=lambda r: r["reddit_id"],
    )
    without_comments = sorted(
        [r for r in records if not r["has_saved_comments"]],
        key=lambda r: r["reddit_id"],
    )
    ordered = with_comments + without_comments
    for i, row in enumerate(ordered, start=1):
        row["review_order"] = i
        row["review_id"] = f"O-{i:03d}"
        row["priority_band"] = "has_saved_comments" if row["has_saved_comments"] else "no_saved_comments"

    verification = {
        "utc": utcnow(),
        "run_id": RUN_ID,
        "frame_f_eligible_n": len(set(f_ids)),
        "frame_label_counts": {
            "TPS_technical_troubleshooting": f_label_counts.get("technical_troubleshooting", 0),
            "GDS_general_dissatisfaction": f_label_counts.get("general_dissatisfaction", 0),
            "Other_other": f_label_counts.get("other", 0),
            "other_keys": {k: v for k, v in f_label_counts.items() if k not in {
                "technical_troubleshooting", "general_dissatisfaction", "other"
            }},
        },
        "n_other_unique": len(ordered),
        "n_with_saved_comments": sum(1 for r in ordered if r["has_saved_comments"]),
        "n_with_author_replies": sum(1 for r in ordered if r["has_author_replies"]),
        "n_with_valid_post_timestamp": sum(1 for r in ordered if r["has_valid_post_timestamp"]),
        "n_duplicate_reddit_ids": len(other_ids) - len(set(other_ids)),
        "n_missing_capture_post": sum(1 for r in ordered if not r["capture_found"]),
        "n_empty_title": sum(1 for r in ordered if not r["title"]),
        "n_empty_body": sum(1 for r in ordered if not r["post_body"]),
        "n_protected_union_overlap": sum(1 for r in ordered if r["protected_union_membership"]),
        "n_reserved_membership": sum(1 for r in ordered if r["reserved_membership"]),
        "n_fresh_membership": sum(1 for r in ordered if r["fresh_membership"]),
        "reconsideration_candidate_list_on_disk": False,
        "reconsideration_candidate_n_reported_by_user": 32,
        "note": (
            "32 reconsideration candidates mentioned in the task brief were not found as a saved "
            "artifact on disk; all 179 Other posts are included. Priority is comments-first."
        ),
    }
    return ordered, verification


def write_blinded_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "review_order",
        "review_id",
        "priority_band",
        "reddit_id",
        "subreddit",
        "title",
        "post_body",
        "post_created_utc",
        "conversation_context",
        "author_replies",
        "community_replies",
        "n_nonempty_comments",
        "n_author_replies",
        "n_community_replies",
        "has_saved_comments",
        "has_author_replies",
        "has_valid_post_timestamp",
        "proposed_label",
        "reviewer_notes",
        "adjudication_label",
        "adjudication_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            out = {k: row.get(k, "") for k in fields}
            out["proposed_label"] = ""
            out["reviewer_notes"] = ""
            out["adjudication_label"] = ""
            out["adjudication_notes"] = ""
            writer.writerow(out)


def write_reviewer_sheet(records: list[dict[str, Any]], path: Path, reviewer: str) -> None:
    fields = [
        "review_order",
        "review_id",
        "priority_band",
        "reddit_id",
        "title",
        "has_saved_comments",
        "has_author_replies",
        "proposed_label",
        "reviewer_notes",
        "confidence",
        "reviewer_id",
        "reviewed_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "review_order": row["review_order"],
                    "review_id": row["review_id"],
                    "priority_band": row["priority_band"],
                    "reddit_id": row["reddit_id"],
                    "title": row["title"],
                    "has_saved_comments": row["has_saved_comments"],
                    "has_author_replies": row["has_author_replies"],
                    "proposed_label": "",
                    "reviewer_notes": "",
                    "confidence": "",
                    "reviewer_id": reviewer,
                    "reviewed_utc": "",
                }
            )


def write_adjudication_sheet(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "review_order",
        "review_id",
        "reddit_id",
        "title",
        "reviewer_A_label",
        "reviewer_B_label",
        "labels_agree",
        "adjudication_label",
        "adjudication_notes",
        "adjudicator_id",
        "adjudicated_utc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "review_order": row["review_order"],
                    "review_id": row["review_id"],
                    "reddit_id": row["reddit_id"],
                    "title": row["title"],
                    "reviewer_A_label": "",
                    "reviewer_B_label": "",
                    "labels_agree": "",
                    "adjudication_label": "",
                    "adjudication_notes": "",
                    "adjudicator_id": "",
                    "adjudicated_utc": "",
                }
            )


def html_escape_pre(text: str) -> str:
    return html.escape(text or "", quote=False)


def write_html(records: list[dict[str, Any]], path: Path) -> None:
    cards = []
    for row in records:
        body = html_escape_pre(row["post_body"]) or "<em>(empty post body)</em>"
        conv = html_escape_pre(row["conversation_context"]) or "<em>(no saved comments)</em>"
        author = html_escape_pre(row["author_replies"]) or "<em>(none)</em>"
        community = html_escape_pre(row["community_replies"]) or "<em>(none)</em>"
        options = "".join(f'<option value="{lab}">{lab}</option>' for lab in LABELS)
        cards.append(
            f"""
<article class="card" id="{html.escape(row['review_id'])}" data-order="{row['review_order']}" data-band="{html.escape(row['priority_band'])}">
  <header>
    <div class="meta">
      <span class="badge">{html.escape(row['review_id'])}</span>
      <span class="badge">order {row['review_order']}/179</span>
      <span class="badge">{'comments' if row['has_saved_comments'] else 'no comments'}</span>
      <span class="badge muted">r/{html.escape(row['subreddit'] or '')}</span>
    </div>
    <h2>{html.escape(row['title'] or '(no title)')}</h2>
    <p class="submeta">Reddit ID: <code>{html.escape(row['reddit_id'])}</code>
      · Posted: <code>{html.escape(row['post_created_utc'] or 'missing')}</code>
      · Comments saved: {row['n_nonempty_comments']}
      · Author replies: {row['n_author_replies']}
      · Community replies: {row['n_community_replies']}
    </p>
  </header>
  <section>
    <h3>Original post body</h3>
    <pre>{body}</pre>
  </section>
  <section>
    <h3>Conversation context</h3>
    <pre>{conv}</pre>
  </section>
  <section class="split">
    <div>
      <h3>Original-author replies</h3>
      <pre>{author}</pre>
    </div>
    <div>
      <h3>Community replies</h3>
      <pre>{community}</pre>
    </div>
  </section>
  <section class="form">
    <label>Proposed label
      <select class="label-select" data-review-id="{html.escape(row['review_id'])}">
        <option value="">— select —</option>
        {options}
      </select>
    </label>
    <label>Reviewer notes
      <textarea class="notes" data-review-id="{html.escape(row['review_id'])}" rows="3" placeholder="Why this primary purpose? Note mixed-purpose cases."></textarea>
    </label>
    <label>Adjudication (use only after both reviewers finish)
      <select class="adj-select" data-review-id="{html.escape(row['review_id'])}">
        <option value="">— select —</option>
        {options}
      </select>
    </label>
    <label>Adjudication notes
      <textarea class="adj-notes" data-review-id="{html.escape(row['review_id'])}" rows="2"></textarea>
    </label>
  </section>
</article>
"""
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TPS/GDS Other Reannotation — Blinded Review ({RUN_ID})</title>
<style>
  :root {{
    --bg: #f4f1ea;
    --ink: #1c1917;
    --muted: #57534e;
    --card: #fffdf8;
    --line: #d6d3d1;
    --accent: #0f766e;
    --badge: #ecfdf5;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    color: var(--ink);
    background:
      radial-gradient(circle at top left, rgba(15,118,110,.08), transparent 28%),
      linear-gradient(180deg, #f7f3eb, #efe8dc);
  }}
  header.top {{
    position: sticky; top: 0; z-index: 5;
    backdrop-filter: blur(8px);
    background: rgba(244,241,234,.92);
    border-bottom: 1px solid var(--line);
    padding: 1rem 1.25rem;
  }}
  header.top h1 {{ margin: 0 0 .35rem; font-size: 1.35rem; }}
  header.top p {{ margin: .15rem 0; color: var(--muted); font-size: .95rem; }}
  .controls {{ display:flex; flex-wrap:wrap; gap:.6rem; margin-top:.7rem; align-items:center; }}
  .controls input, .controls select, .controls button {{
    font: inherit; padding: .4rem .6rem; border: 1px solid var(--line); border-radius: 6px; background: white;
  }}
  .controls button {{ background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 1rem 1.25rem 4rem; }}
  .rubric {{
    background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 1rem;
  }}
  .rubric ul {{ margin: .4rem 0 .2rem 1.1rem; }}
  .card {{
    background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem 1.2rem; margin: 1rem 0 1.4rem;
  }}
  .card h2 {{ margin: .35rem 0 .2rem; font-size: 1.2rem; line-height: 1.3; }}
  .card h3 {{ margin: .9rem 0 .35rem; font-size: .95rem; letter-spacing: .02em; text-transform: uppercase; color: var(--muted); }}
  .meta {{ display:flex; flex-wrap:wrap; gap:.35rem; }}
  .badge {{ background: var(--badge); color: var(--accent); border: 1px solid #99f6e4; border-radius: 999px; padding: .15rem .55rem; font-size: .78rem; font-family: ui-sans-serif, system-ui, sans-serif; }}
  .badge.muted {{ background: #f5f5f4; color: var(--muted); border-color: var(--line); }}
  .submeta {{ color: var(--muted); font-size: .9rem; }}
  pre {{
    white-space: pre-wrap; word-break: break-word;
    background: #fafaf9; border: 1px solid var(--line); border-radius: 8px;
    padding: .75rem .85rem; font-family: ui-sans-serif, system-ui, sans-serif; font-size: .92rem; line-height: 1.45;
  }}
  .split {{ display:grid; grid-template-columns: 1fr 1fr; gap: .8rem; }}
  @media (max-width: 800px) {{ .split {{ grid-template-columns: 1fr; }} }}
  .form {{ display:grid; gap: .65rem; margin-top: .8rem; padding-top: .8rem; border-top: 1px dashed var(--line); }}
  .form label {{ display:grid; gap: .25rem; font-family: ui-sans-serif, system-ui, sans-serif; font-size: .9rem; }}
  .form select, .form textarea {{ font: inherit; padding: .45rem .55rem; border: 1px solid var(--line); border-radius: 6px; }}
  .hidden {{ display:none !important; }}
  .progress {{ font-family: ui-sans-serif, system-ui, sans-serif; font-size: .9rem; }}
</style>
</head>
<body>
<header class="top">
  <h1>Blinded TPS / GDS reannotation — 179 Other posts</h1>
  <p>Run <code>{html.escape(RUN_ID)}</code>. Original discussion labels, VADER scores, AI predictions, and screening flags are hidden.</p>
  <p>Label from <strong>discussion purpose</strong>, not sentiment scores or desired class balance. Complete Reviewer A and Reviewer B sheets independently.</p>
  <div class="controls">
    <label>Filter
      <select id="bandFilter">
        <option value="all">All 179</option>
        <option value="has_saved_comments">Priority: with comments (66)</option>
        <option value="no_saved_comments">Remaining without comments (113)</option>
      </select>
    </label>
    <label>Search <input id="searchBox" type="search" placeholder="review id, reddit id, title…"/></label>
    <span class="progress" id="progress">0 / 179 labeled in this browser</span>
    <button type="button" id="exportBtn">Download browser draft CSV</button>
  </div>
</header>
<main>
  <section class="rubric">
    <h2>Approved definitions</h2>
    <ul>
      <li><strong>TPS</strong> — Discussions primarily concerned with understanding, diagnosing, reporting with useful technical detail, or resolving software update problems.</li>
      <li><strong>GDS</strong> — Discussions primarily expressing dissatisfaction, criticism, frustration, unfavorable evaluations, or negative experiences associated with software updates, where technical troubleshooting is not the main purpose.</li>
      <li><strong>Other</strong> — Update-related announcements, neutral discussions, praise, or other posts that are neither TPS nor GDS.</li>
      <li><strong>Unclear</strong> — Posts without sufficient information to establish the primary discussion purpose.</li>
    </ul>
    <p>Official reviewer answers belong in <code>reviewer_A_sheet.csv</code> / <code>reviewer_B_sheet.csv</code>. This HTML is a reading aid; browser draft export is optional and local-only.</p>
  </section>
  {''.join(cards)}
</main>
<script>
(() => {{
  const cards = [...document.querySelectorAll('.card')];
  const bandFilter = document.getElementById('bandFilter');
  const searchBox = document.getElementById('searchBox');
  const progress = document.getElementById('progress');
  const key = 'tps_gds_other_reannotation_2026_11_02_draft';

  function load() {{
    try {{ return JSON.parse(localStorage.getItem(key) || '{{}}'); }} catch {{ return {{}}; }}
  }}
  function save(data) {{ localStorage.setItem(key, JSON.stringify(data)); }}

  function applyDraft() {{
    const data = load();
    let n = 0;
    cards.forEach(card => {{
      const id = card.id;
      const row = data[id] || {{}};
      const label = card.querySelector('.label-select');
      const notes = card.querySelector('.notes');
      const adj = card.querySelector('.adj-select');
      const adjNotes = card.querySelector('.adj-notes');
      if (label) label.value = row.proposed_label || '';
      if (notes) notes.value = row.reviewer_notes || '';
      if (adj) adj.value = row.adjudication_label || '';
      if (adjNotes) adjNotes.value = row.adjudication_notes || '';
      if (row.proposed_label) n += 1;
    }});
    progress.textContent = n + ' / 179 labeled in this browser';
  }}

  function persistFrom(el) {{
    const id = el.getAttribute('data-review-id');
    const data = load();
    const card = document.getElementById(id);
    data[id] = {{
      proposed_label: card.querySelector('.label-select').value,
      reviewer_notes: card.querySelector('.notes').value,
      adjudication_label: card.querySelector('.adj-select').value,
      adjudication_notes: card.querySelector('.adj-notes').value,
    }};
    save(data);
    applyDraft();
  }}

  function filterCards() {{
    const band = bandFilter.value;
    const q = (searchBox.value || '').trim().toLowerCase();
    cards.forEach(card => {{
      const bandOk = band === 'all' || card.dataset.band === band;
      const hay = (card.innerText || '').toLowerCase();
      const qOk = !q || hay.includes(q);
      card.classList.toggle('hidden', !(bandOk && qOk));
    }});
  }}

  document.querySelectorAll('.label-select, .notes, .adj-select, .adj-notes').forEach(el => {{
    el.addEventListener('change', () => persistFrom(el));
    el.addEventListener('input', () => persistFrom(el));
  }});
  bandFilter.addEventListener('change', filterCards);
  searchBox.addEventListener('input', filterCards);

  document.getElementById('exportBtn').addEventListener('click', () => {{
    const data = load();
    const lines = [['review_id','proposed_label','reviewer_notes','adjudication_label','adjudication_notes']];
    cards.forEach(card => {{
      const row = data[card.id] || {{}};
      lines.push([
        card.id,
        row.proposed_label || '',
        (row.reviewer_notes || '').replaceAll('"','""'),
        row.adjudication_label || '',
        (row.adjudication_notes || '').replaceAll('"','""'),
      ].map(v => `"${{v}}"`));
    }});
    const blob = new Blob([lines.map(r => r.join(',')).join('\\n')], {{type:'text/csv'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'browser_draft_labels.csv';
    a.click();
  }});

  applyDraft();
  filterCards();
}})();
</script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_private_manifest(records: list[dict[str, Any]], verification: dict[str, Any], source_hashes: dict[str, str]) -> None:
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)
    private_rows = []
    for row in records:
        private_rows.append(
            {
                "review_order": row["review_order"],
                "review_id": row["review_id"],
                "priority_band": row["priority_band"],
                "reddit_id": row["reddit_id"],
                "original_discussion_type": row["original_discussion_type"],
                "original_label_display": "Other",
                "original_s2_target": row["original_s2_target"],
                "original_s2_in_scope": row["original_s2_in_scope"],
                "original_about_software": row["original_about_software"],
                "original_update_related": row["original_update_related"],
                "study_id": row["study_id"],
                "text_availability": row["text_availability"],
                "has_saved_comments": row["has_saved_comments"],
                "has_author_replies": row["has_author_replies"],
                "has_valid_post_timestamp": row["has_valid_post_timestamp"],
                "n_nonempty_comments": row["n_nonempty_comments"],
                "n_author_replies": row["n_author_replies"],
                "capture_page": row["capture_page"],
                "body_source": row["body_source"],
                "title_source": row["title_source"],
                "title_sha256": row["title_sha256"],
                "body_sha256": row["body_sha256"],
                "conversation_sha256": row["conversation_sha256"],
                "reserved_membership": row["reserved_membership"],
                "fresh_membership": row["fresh_membership"],
                "protected_union_membership": row["protected_union_membership"],
                "f_eligible": row["f_eligible"],
                "review_priority_reason": (
                    "saved_comments_present"
                    if row["has_saved_comments"]
                    else "remaining_other_without_saved_comments"
                ),
            }
        )

    manifest_path = RESTRICTED_DIR / "private_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in private_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "utc": utcnow(),
        "run_id": RUN_ID,
        "purpose": "Private provenance for blinded TPS/GDS reannotation of F-set Other posts.",
        "do_not_share_with_reviewers": True,
        "original_labels_preserved": True,
        "auto_relabeling_performed": False,
        "protected_outcomes_accessed": False,
        "source_hashes": source_hashes,
        "verification": verification,
        "n_private_rows": len(private_rows),
        "files": {
            "private_manifest_jsonl": str(manifest_path.relative_to(PHASE6)),
            "public_dir": str(PUBLIC_DIR.relative_to(PHASE6)),
        },
    }
    (RESTRICTED_DIR / "private_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_instructions() -> None:
    text = f"""# TPS/GDS Other Reannotation — Review Instructions

**Run ID:** `{RUN_ID}`  
**Status:** Preparation only. No labels have been changed.

## Goal

Independently re-label the **179** posts currently marked `discussion_type=other` in the frozen F-set (1,864 in-scope posts) using the **approved revised rubric**.

## Approved definitions

- **TPS:** Discussions primarily concerned with understanding, diagnosing, reporting with useful technical detail, or resolving software update problems.
- **GDS:** Discussions primarily expressing dissatisfaction, criticism, frustration, unfavorable evaluations, or negative experiences associated with software updates, where technical troubleshooting is not the main purpose.
- **Other:** Update-related announcements, neutral discussions, praise, or other posts that are neither TPS nor GDS.
- **Unclear:** Posts without sufficient information to establish the primary discussion purpose.

Decide from **discussion purpose**, not VADER scores, negative keywords alone, or desired class balance.

## What to open

1. **Reading interface (recommended):**  
   `phase6/full_resource_sentiment_v1/results/tps_gds_other_reannotation_2026-11-02/blinded_review_packet.html`
2. **Official answer sheets (required):**
   - Reviewer A: `reviewer_A_sheet.csv`
   - Reviewer B: `reviewer_B_sheet.csv`
3. Optional full packet table: `blinded_review_packet.csv`
4. After both reviewers finish: `adjudication_sheet.csv`

Do **not** open `restricted/tps_gds_other_reannotation_2026-11-02/` until adjudication is complete. That folder contains original labels.

## Review order

1. Complete the **66** posts with saved comments first (`priority_band=has_saved_comments`, review_order 1–66).
2. Then complete the remaining **113** (`priority_band=no_saved_comments`, review_order 67–179).
3. All 179 must be labeled.

## How to fill a reviewer sheet

For each row:

1. Read title, post body, conversation context, author replies, and community replies in the HTML (or CSV).
2. Enter `proposed_label` as exactly one of: `TPS`, `GDS`, `Other`, `Unclear`.
3. Enter brief `reviewer_notes` for mixed-purpose or difficult cases.
4. Optionally set `confidence` to `high`, `medium`, or `low`.
5. Set `reviewed_utc` when done with that row (ISO-8601 preferred).

Work independently. Do not look at the other reviewer’s sheet until both are finished.

## Adjudication

1. Copy Reviewer A and B labels into `adjudication_sheet.csv`.
2. Set `labels_agree` to `yes` or `no`.
3. For disagreements (and any Unclear/mixed cases needing a final call), set `adjudication_label` and `adjudication_notes`.
4. Do not invent a second reviewer’s decisions. If only one reviewer is available, stop after that sheet and record the gap.

## Prohibited

- Do not modify frozen source datasets.
- Do not use VADER, AI predictions, or original `discussion_type` while labeling.
- Do not access protected evaluation outcomes.
- Do not auto-fill labels with scripts or models for this human review.

## After review

Return the completed `reviewer_A_sheet.csv`, `reviewer_B_sheet.csv`, and `adjudication_sheet.csv`.  
We will then finalize verified TPS/GDS/Other/Unclear counts before the four research questions.
"""
    (PUBLIC_DIR / "REVIEW_INSTRUCTIONS.md").write_text(text, encoding="utf-8")


def write_rubric() -> None:
    text = f"""# Approved TPS/GDS labeling rubric (reannotation)

**Run ID:** `{RUN_ID}`  
**Source of approval:** Research team decision with Professor Solomon Berhe (task brief 2026-11-02).

## Labels

### TPS (Technical Problem Solving)
Discussions primarily concerned with understanding, diagnosing, reporting with useful technical detail, or resolving software update problems.

### GDS (General Discontent)
Discussions primarily expressing dissatisfaction, criticism, frustration, unfavorable evaluations, or negative experiences associated with software updates, where technical troubleshooting is not the main purpose.

### Other
Update-related announcements, neutral discussions, praise, or other posts that are neither TPS nor GDS.

### Unclear
Posts without sufficient information to establish the primary discussion purpose.

## Decision principle
Determine the label from **discussion purpose**, not VADER scores, negative keywords alone, or desired class balance.

## Historical references (not substitutes for this rubric)
- Original binary guidelines: `ANNOTATION_GUIDELINES.md`
- Phase-2 operational guide: `phase2/annotation/REVIEWER_GUIDE.md`
"""
    (PUBLIC_DIR / "APPROVED_RUBRIC.md").write_text(text, encoding="utf-8")


def write_verification_report(verification: dict[str, Any], source_hashes: dict[str, str], output_hashes: dict[str, str]) -> None:
    expected_ok = (
        verification["frame_f_eligible_n"] == 1864
        and verification["frame_label_counts"]["TPS_technical_troubleshooting"] == 1628
        and verification["frame_label_counts"]["GDS_general_dissatisfaction"] == 57
        and verification["frame_label_counts"]["Other_other"] == 179
        and verification["n_other_unique"] == 179
        and verification["n_with_saved_comments"] == 66
        and verification["n_with_author_replies"] == 38
        and verification["n_duplicate_reddit_ids"] == 0
        and verification["n_protected_union_overlap"] == 0
    )
    text = f"""# Verification report — TPS/GDS Other reannotation packet

**Run ID:** `{RUN_ID}`  
**UTC:** {verification["utc"]}  
**Independent count check:** {"PASS" if expected_ok else "CHECK FAILED — inspect counts below"}

## 1. Unique posts prepared
**{verification["n_other_unique"]}** (expected 179)

## 2. Posts with saved comments
**{verification["n_with_saved_comments"]}** (expected 66)  
These are review_order 1–{verification["n_with_saved_comments"]}.

## 3. Posts with original-author replies
**{verification["n_with_author_replies"]}** (expected 38)

## 4. Posts with valid post timestamps
**{verification["n_with_valid_post_timestamp"]}** of 179

## 5. Missing data / duplicates
| Check | Count |
|---|---:|
| Duplicate Reddit IDs | {verification["n_duplicate_reddit_ids"]} |
| Missing capture page post | {verification["n_missing_capture_post"]} |
| Empty title | {verification["n_empty_title"]} |
| Empty body | {verification["n_empty_body"]} |

## 6. Source files and outputs

### Frozen sources (not modified)
| File | SHA-256 |
|---|---|
| `phase3/revised_development_v2/development_v2.jsonl` | `{source_hashes["development_v2.jsonl"]}` |
| `phase3/revised_development_v2/MANIFEST.json` | `{source_hashes["development_MANIFEST.json"]}` |
| `results/broad_sentiment_2026-10-22-main/post_scores.jsonl` | `{source_hashes["post_scores.jsonl"]}` |
| `results/broad_sentiment_2026-10-22-main/thread_metrics.jsonl` | `{source_hashes["thread_metrics.jsonl"]}` |
| `ANNOTATION_GUIDELINES.md` | `{source_hashes["ANNOTATION_GUIDELINES.md"]}` |
| `phase2/annotation/REVIEWER_GUIDE.md` | `{source_hashes["phase2_REVIEWER_GUIDE.md"]}` |
| Capture dir | `restricted/broad_release_train/2026-10-22-main/` (joined by `capture_page`) |

### New outputs (this run only)
| File | Role |
|---|---|
| `results/tps_gds_other_reannotation_2026-11-02/blinded_review_packet.html` | Blinded reading / local draft UI |
| `results/tps_gds_other_reannotation_2026-11-02/blinded_review_packet.csv` | Blinded full packet |
| `results/tps_gds_other_reannotation_2026-11-02/reviewer_A_sheet.csv` | Reviewer A answers |
| `results/tps_gds_other_reannotation_2026-11-02/reviewer_B_sheet.csv` | Reviewer B answers |
| `results/tps_gds_other_reannotation_2026-11-02/adjudication_sheet.csv` | Final adjudication |
| `results/tps_gds_other_reannotation_2026-11-02/REVIEW_INSTRUCTIONS.md` | Workflow |
| `results/tps_gds_other_reannotation_2026-11-02/APPROVED_RUBRIC.md` | Rubric copy |
| `restricted/tps_gds_other_reannotation_2026-11-02/private_manifest.jsonl` | Original labels + provenance (private) |

Output hashes:
```
{json.dumps(output_hashes, indent=2, sort_keys=True)}
```

## 7. Protected evaluation records
- Overlap with reserved/fresh protected union (693 IDs): **{verification["n_protected_union_overlap"]}**
- Protected evaluation outcomes were **not** read.
- Reserved membership among these 179: **{verification["n_reserved_membership"]}**
- Fresh membership among these 179: **{verification["n_fresh_membership"]}**

## 8. F-set independent verification
| Label | Count |
|---|---:|
| In-scope F (`f_eligible`) | {verification["frame_f_eligible_n"]} |
| TPS (`technical_troubleshooting`) | {verification["frame_label_counts"]["TPS_technical_troubleshooting"]} |
| GDS (`general_dissatisfaction`) | {verification["frame_label_counts"]["GDS_general_dissatisfaction"]} |
| Other (`other`) | {verification["frame_label_counts"]["Other_other"]} |

## 9. Reconsideration-candidate list
The task brief mentioned **32** possible reconsideration candidates. No saved on-disk list with that membership was found. All **179** Other posts are in the packet; priority is the 66 with comments, then the remaining 113.

## 10. Exact review instructions
Follow `REVIEW_INSTRUCTIONS.md` in the same public folder. Open the HTML interface first, then fill the reviewer CSVs.
"""
    (PUBLIC_DIR / "VERIFICATION_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    RESTRICTED_DIR.mkdir(parents=True, exist_ok=True)

    source_hashes = {
        "development_v2.jsonl": sha256_file(DEVELOPMENT_V2),
        "development_MANIFEST.json": sha256_file(DEVELOPMENT_MANIFEST),
        "post_scores.jsonl": sha256_file(POST_SCORES),
        "thread_metrics.jsonl": sha256_file(THREAD_METRICS),
        "ANNOTATION_GUIDELINES.md": sha256_file(ANNOTATION_GUIDELINES),
        "phase2_REVIEWER_GUIDE.md": sha256_file(PHASE2_REVIEWER_GUIDE),
        "excluded_protected_ids.json": sha256_file(PROTECTED_IDS),
    }

    records, verification = build_records()
    if verification["n_other_unique"] != 179:
        raise SystemExit(f"expected 179 Other posts, got {verification['n_other_unique']}")

    write_blinded_csv(records, PUBLIC_DIR / "blinded_review_packet.csv")
    write_html(records, PUBLIC_DIR / "blinded_review_packet.html")
    write_reviewer_sheet(records, PUBLIC_DIR / "reviewer_A_sheet.csv", "A")
    write_reviewer_sheet(records, PUBLIC_DIR / "reviewer_B_sheet.csv", "B")
    write_adjudication_sheet(records, PUBLIC_DIR / "adjudication_sheet.csv")
    write_private_manifest(records, verification, source_hashes)
    write_instructions()
    write_rubric()

    output_files = [
        "blinded_review_packet.csv",
        "blinded_review_packet.html",
        "reviewer_A_sheet.csv",
        "reviewer_B_sheet.csv",
        "adjudication_sheet.csv",
        "REVIEW_INSTRUCTIONS.md",
        "APPROVED_RUBRIC.md",
    ]
    output_hashes = {name: sha256_file(PUBLIC_DIR / name) for name in output_files}
    output_hashes["private_manifest.jsonl"] = sha256_file(RESTRICTED_DIR / "private_manifest.jsonl")
    output_hashes["private_manifest_summary.json"] = sha256_file(RESTRICTED_DIR / "private_manifest_summary.json")
    write_verification_report(verification, source_hashes, output_hashes)
    output_hashes["VERIFICATION_REPORT.md"] = sha256_file(PUBLIC_DIR / "VERIFICATION_REPORT.md")
    (PUBLIC_DIR / "OUTPUT_HASHES.json").write_text(
        json.dumps({"utc": utcnow(), "run_id": RUN_ID, "hashes": output_hashes}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": "ok",
        "n_other": verification["n_other_unique"],
        "n_with_comments": verification["n_with_saved_comments"],
        "n_author_replies": verification["n_with_author_replies"],
        "public_dir": str(PUBLIC_DIR),
        "html": str(PUBLIC_DIR / "blinded_review_packet.html"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
