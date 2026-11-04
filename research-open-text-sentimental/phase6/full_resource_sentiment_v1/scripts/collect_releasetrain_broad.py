"""Bounded, resumable ReleaseTrain /api/reddit capture for the Berhe lab study.

Raw pages stay local under restricted/ and are never written to public results.
One request at a time; server-provided cursor only. No authentication bypass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://releasetrain.io/api/reddit"
USER_AGENT = "BerheLabResearchCapture/1.0"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def fetch(cursor, limit, timeout, retries=4):
    params = {"limit": limit, "showCount": "true"}
    if cursor:
        params["cursor"] = cursor
    url = BASE + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return url, raw, dict(response.headers)
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise
            retry = exc.headers.get("Retry-After")
            try:
                delay = min(120, max(1, float(retry))) if retry else min(60, 2 ** attempt)
            except ValueError:
                delay = min(60, 2 ** attempt)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError):
            if attempt == retries:
                raise
            time.sleep(min(60, 2 ** attempt))


def safe_rows(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("unexpected response: expected object with data array")
    rows = payload["data"]
    if not all(isinstance(r, dict) and r.get("redditId") for r in rows):
        raise ValueError("missing exact Reddit ID in page")
    return rows


def audit_page(rows):
    counts = Counter()
    ids = []
    for row in rows:
        ids.append(str(row["redditId"]))
        counts["title_nonempty"] += bool(str(row.get("title") or "").strip())
        counts["body_nonempty"] += bool(str(row.get("author_description") or "").strip())
        counts["post_timestamp_present"] += bool(row.get("created_utc"))
        counts["post_author_present"] += bool(str(row.get("author") or "").strip())
        comments = row.get("comments")
        counts["comments_field_absent"] += "comments" not in row or comments is None
        counts["comments_empty"] += isinstance(comments, list) and len(comments) == 0
        counts["comments_nonempty"] += isinstance(comments, list) and len(comments) > 0
        counts["comments_malformed"] += comments is not None and not isinstance(comments, list)
        if isinstance(comments, list):
            counts["stored_comment_turns"] += len(comments)
            for comment in comments:
                if not isinstance(comment, dict):
                    counts["malformed_comment_turns"] += 1
                    continue
                if str(comment.get("body") or "").strip():
                    counts["nonempty_comment_turns"] += 1
                if comment.get("created_utc"):
                    counts["timestamp_present_turns"] += 1
                if str(comment.get("author") or "").strip():
                    counts["comment_author_present_turns"] += 1
                if comment.get("is_submitter") is True:
                    counts["is_submitter_true_turns"] += 1
        source = row.get("source")
        counts["source:" + str(source)] += 1
    return ids, dict(counts)


def run(args):
    capture = ROOT / "restricted" / "broad_release_train" / args.capture_id
    capture.mkdir(parents=True, exist_ok=True)
    os.chmod(capture, 0o700)
    (capture.parent.parent / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    manifest_path = capture / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["endpoint"] != BASE or manifest["limit"] != args.limit:
            raise ValueError("resume parameters do not match manifest")
    else:
        manifest = {"capture_id": args.capture_id, "endpoint": BASE, "limit": args.limit,
                    "started_utc": utcnow(), "access_basis": "Manu confirms Berhe lab ReleaseTrain research access and explicit professor instruction; public read endpoint did not request credentials",
                    "pages": [], "complete": False}
        atomic_json(manifest_path, manifest)
    if manifest["complete"]:
        print("capture already complete", capture)
        return
    seen_cursors = {page.get("request_cursor") for page in manifest["pages"]}
    cursor = manifest["pages"][-1].get("next_cursor") if manifest["pages"] else None
    if manifest["pages"] and not cursor:
        raise ValueError("incomplete manifest has no next cursor")
    started = time.monotonic()
    prior_bytes = sum(p["bytes"] for p in manifest["pages"])
    for _ in range(args.max_pages - len(manifest["pages"])):
        if time.monotonic() - started > args.max_seconds or prior_bytes > args.max_bytes:
            print("bounded stop: time/byte cap")
            break
        if cursor in seen_cursors:
            raise ValueError("repeated request cursor")
        url, raw, headers = fetch(cursor, args.limit, args.timeout)
        if prior_bytes + len(raw) > args.max_bytes:
            raise ValueError("byte cap would be exceeded before saving page")
        payload = json.loads(raw)
        rows = safe_rows(payload)
        ids, counts = audit_page(rows)
        page_no = len(manifest["pages"]) + 1
        next_cursor = headers.get("X-Next-Cursor") or headers.get("x-next-cursor")
        if next_cursor == cursor:
            raise ValueError("cursor did not advance")
        filename = f"page_{page_no:05d}.json"
        temp = capture / (filename + ".tmp")
        temp.write_bytes(raw)
        os.chmod(temp, 0o600)
        os.replace(temp, capture / filename)
        manifest["pages"].append({"page": page_no, "file": filename, "request_url": url,
                                  "request_cursor": cursor, "next_cursor": next_cursor,
                                  "captured_utc": utcnow(), "server_date": headers.get("Date"),
                                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                  "rows": len(rows), "first_id": ids[0] if ids else None,
                                  "last_id": ids[-1] if ids else None, "reported_total": payload.get("totalCount"),
                                  "counts": counts})
        seen_cursors.add(cursor)
        prior_bytes += len(raw)
        if not rows or not next_cursor:
            manifest["complete"] = True
            manifest["finished_utc"] = utcnow()
        atomic_json(manifest_path, manifest)
        print(f"page {page_no}: {len(rows)} rows; reported total {payload.get('totalCount')}; next {bool(next_cursor)}", flush=True)
        if manifest["complete"]:
            break
        cursor = next_cursor
        time.sleep(args.delay)
    print("capture", capture, "complete", manifest["complete"], "pages", len(manifest["pages"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500 or args.max_pages < 1 or args.delay < 0.5:
        parser.error("limit must be 1..500; max-pages >= 1; delay >= 0.5")
    run(args)


if __name__ == "__main__":
    main()
