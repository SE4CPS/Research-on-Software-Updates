#!/usr/bin/env python3
"""Localhost-only blinded review app for the 200-case validation sample.

Serves only opaque case codes and redacted title/body. Never loads the private ID key.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parents[1]
RESTRICTED = BASE / "restricted/validation_200_2026-11-02"
CASES_PATH = RESTRICTED / "blinded_cases.jsonl"
DB_PATH = RESTRICTED / "review_state.sqlite"
EXPORT_DIR = RESTRICTED / "exports"
HOST = "127.0.0.1"
PORT = 8765

ALLOWED_A = {"relevant", "not_relevant", "unclear", "unassessable"}
ALLOWED_B = {"TPS", "GDS", "Other", "Unclear", "not_applicable"}
ALLOWED_CONF = {"high", "medium", "low", ""}
ALLOWED_REC = {"yes", "no", "unsure", ""}

_lock = threading.Lock()
_cases: list[dict] = []
_by_code: dict[str, dict] = {}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_cases() -> None:
    global _cases, _by_code
    rows = []
    with CASES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["presentation_order"]))
    # Hard ban on analyst fields in memory.
    forbidden = {
        "reddit_id",
        "final_relevance",
        "final_discussion",
        "weight",
        "stratum",
        "inclusion_probability",
        "model_reason",
        "decision_source",
        "sentiment",
        "compound",
    }
    cleaned = []
    for row in rows:
        if set(row) & forbidden:
            raise SystemExit(f"refusing to serve case with forbidden fields: {set(row)&forbidden}")
        cleaned.append(
            {
                "case_code": row["case_code"],
                "presentation_order": int(row["presentation_order"]),
                "title": row.get("title") or "",
                "body": row.get("body") or "",
            }
        )
    _cases = cleaned
    _by_code = {row["case_code"]: row for row in _cases}
    if not _cases:
        raise SystemExit("no blinded cases loaded")
    # Production draw is exactly 200; synthetic tests may use a smaller packet.


def db() -> sqlite3.Connection:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            case_code TEXT PRIMARY KEY,
            presentation_order INTEGER NOT NULL,
            decision_a TEXT NOT NULL DEFAULT '',
            decision_b TEXT NOT NULL DEFAULT '',
            confidence TEXT NOT NULL DEFAULT '',
            evidence_phrase TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            boundary_note TEXT NOT NULL DEFAULT '',
            recognized_prior_case TEXT NOT NULL DEFAULT '',
            updated_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_code TEXT NOT NULL,
            presentation_order INTEGER NOT NULL,
            decision_a TEXT NOT NULL,
            decision_b TEXT NOT NULL,
            confidence TEXT NOT NULL,
            evidence_phrase TEXT NOT NULL,
            reason TEXT NOT NULL,
            boundary_note TEXT NOT NULL,
            recognized_prior_case TEXT NOT NULL,
            revised_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def validate_payload(payload: dict) -> dict:
    code = str(payload.get("case_code") or "")
    if code not in _by_code:
        raise ValueError("unknown case_code")
    a = str(payload.get("decision_a") or "").strip()
    b = str(payload.get("decision_b") or "").strip()
    conf = str(payload.get("confidence") or "").strip()
    rec = str(payload.get("recognized_prior_case") or "").strip()
    if a and a not in ALLOWED_A:
        raise ValueError("invalid decision_a")
    if b and b not in ALLOWED_B:
        raise ValueError("invalid decision_b")
    if conf not in ALLOWED_CONF:
        raise ValueError("invalid confidence")
    if rec not in ALLOWED_REC:
        raise ValueError("invalid recognized_prior_case")
    if a and a != "relevant":
        b = "not_applicable"
    if a == "relevant" and b == "not_applicable":
        raise ValueError("decision_b required when relevant")
    return {
        "case_code": code,
        "presentation_order": _by_code[code]["presentation_order"],
        "decision_a": a,
        "decision_b": b,
        "confidence": conf,
        "evidence_phrase": str(payload.get("evidence_phrase") or ""),
        "reason": str(payload.get("reason") or ""),
        "boundary_note": str(payload.get("boundary_note") or ""),
        "recognized_prior_case": rec,
        "updated_utc": utcnow(),
    }


def save_decision(payload: dict) -> dict:
    row = validate_payload(payload)
    with _lock:
        conn = db()
        prev = conn.execute("SELECT * FROM decisions WHERE case_code=?", (row["case_code"],)).fetchone()
        conn.execute(
            """
            INSERT INTO decisions (
                case_code, presentation_order, decision_a, decision_b, confidence,
                evidence_phrase, reason, boundary_note, recognized_prior_case, updated_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_code) DO UPDATE SET
                decision_a=excluded.decision_a,
                decision_b=excluded.decision_b,
                confidence=excluded.confidence,
                evidence_phrase=excluded.evidence_phrase,
                reason=excluded.reason,
                boundary_note=excluded.boundary_note,
                recognized_prior_case=excluded.recognized_prior_case,
                updated_utc=excluded.updated_utc
            """,
            (
                row["case_code"],
                row["presentation_order"],
                row["decision_a"],
                row["decision_b"],
                row["confidence"],
                row["evidence_phrase"],
                row["reason"],
                row["boundary_note"],
                row["recognized_prior_case"],
                row["updated_utc"],
            ),
        )
        # Append revision if first save or any field changed.
        changed = prev is None or any(
            str(prev[k]) != str(row[k])
            for k in (
                "decision_a",
                "decision_b",
                "confidence",
                "evidence_phrase",
                "reason",
                "boundary_note",
                "recognized_prior_case",
            )
        )
        if changed:
            conn.execute(
                """
                INSERT INTO revisions (
                    case_code, presentation_order, decision_a, decision_b, confidence,
                    evidence_phrase, reason, boundary_note, recognized_prior_case, revised_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["case_code"],
                    row["presentation_order"],
                    row["decision_a"],
                    row["decision_b"],
                    row["confidence"],
                    row["evidence_phrase"],
                    row["reason"],
                    row["boundary_note"],
                    row["recognized_prior_case"],
                    row["updated_utc"],
                ),
            )
        conn.commit()
        conn.close()
    return row


def export_latest() -> dict:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with _lock:
        conn = db()
        decisions = {
            row["case_code"]: dict(row)
            for row in conn.execute("SELECT * FROM decisions").fetchall()
        }
        revisions = [dict(row) for row in conn.execute("SELECT * FROM revisions ORDER BY id").fetchall()]
        conn.close()

    latest_rows = []
    for case in _cases:
        d = decisions.get(case["case_code"], {})
        latest_rows.append(
            {
                "presentation_order": case["presentation_order"],
                "case_code": case["case_code"],
                "decision_a": d.get("decision_a", ""),
                "decision_b": d.get("decision_b", ""),
                "confidence": d.get("confidence", ""),
                "evidence_phrase": d.get("evidence_phrase", ""),
                "reason": d.get("reason", ""),
                "boundary_note": d.get("boundary_note", ""),
                "recognized_prior_case": d.get("recognized_prior_case", ""),
                "updated_utc": d.get("updated_utc", ""),
            }
        )

    jsonl_path = EXPORT_DIR / f"decisions_latest_{stamp}.jsonl"
    csv_path = EXPORT_DIR / f"decisions_latest_{stamp}.csv"
    rev_path = EXPORT_DIR / f"revisions_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in latest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(latest_rows)
    with rev_path.open("w", encoding="utf-8") as handle:
        for row in revisions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n_complete = sum(1 for row in latest_rows if row["decision_a"])
    return {
        "utc": utcnow(),
        "n_cases": len(latest_rows),
        "n_with_decision_a": n_complete,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "revisions": str(rev_path),
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Validation 200 — Blinded Review</title>
<style>
  :root { --bg:#f6f3ec; --ink:#1c1917; --muted:#57534e; --card:#fffdf8; --line:#d6d3d1; --accent:#0f766e; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: Georgia, "Iowan Old Style", serif; background:linear-gradient(180deg,#f7f3eb,#efe8dc); color:var(--ink); }
  header { position:sticky; top:0; background:rgba(246,243,236,.95); border-bottom:1px solid var(--line); padding:0.85rem 1.1rem; }
  header h1 { margin:0; font-size:1.15rem; }
  header p { margin:0.2rem 0 0; color:var(--muted); font-size:0.92rem; }
  main { max-width:920px; margin:0 auto; padding:1rem 1.1rem 3rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; }
  .meta { color:var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; font-size:0.85rem; }
  h2 { margin:0.4rem 0 0.6rem; font-size:1.25rem; line-height:1.3; }
  pre { white-space:pre-wrap; word-break:break-word; background:#fafaf9; border:1px solid var(--line); border-radius:8px; padding:0.8rem; font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.95rem; }
  form { display:grid; gap:0.7rem; margin-top:1rem; }
  label { display:grid; gap:0.25rem; font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.9rem; }
  select, textarea, input, button { font:inherit; padding:0.45rem 0.55rem; border:1px solid var(--line); border-radius:6px; }
  .row { display:flex; flex-wrap:wrap; gap:0.5rem; }
  button { cursor:pointer; background:white; }
  button.primary { background:var(--accent); color:white; border-color:var(--accent); }
  .status { font-family:ui-sans-serif,system-ui,sans-serif; font-size:0.85rem; color:var(--muted); min-height:1.2em; }
  .tree { font-size:0.9rem; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Blinded validation review — 200 cases</h1>
  <p>Localhost only. No prior labels, model scores, Reddit IDs, or sampling weights are available here.</p>
  <p class="meta"><span id="progress">Loading…</span> · <button type="button" id="exportBtn">Export latest decisions</button></p>
</header>
<main>
  <section class="card">
    <div class="meta" id="caseMeta"></div>
    <h2 id="title"></h2>
    <pre id="body"></pre>
    <p class="tree">Decide A (relevance) before B (purpose). If A is not relevant, B is not_applicable.</p>
    <form id="form">
      <input type="hidden" id="case_code"/>
      <label>Decision A — software-update relevance
        <select id="decision_a">
          <option value="">— select —</option>
          <option value="relevant">relevant</option>
          <option value="not_relevant">not_relevant</option>
          <option value="unclear">unclear</option>
          <option value="unassessable">unassessable</option>
        </select>
      </label>
      <label>Decision B — primary purpose (if relevant)
        <select id="decision_b">
          <option value="">— select —</option>
          <option value="TPS">TPS</option>
          <option value="GDS">GDS</option>
          <option value="Other">Other</option>
          <option value="Unclear">Unclear</option>
          <option value="not_applicable">not_applicable</option>
        </select>
      </label>
      <label>Confidence
        <select id="confidence">
          <option value="">— select —</option>
          <option value="high">high</option>
          <option value="medium">medium</option>
          <option value="low">low</option>
        </select>
      </label>
      <label>Exact evidence phrase (from visible text)
        <textarea id="evidence_phrase" rows="2"></textarea>
      </label>
      <label>Short reason
        <textarea id="reason" rows="2"></textarea>
      </label>
      <label>Boundary note (optional)
        <textarea id="boundary_note" rows="2"></textarea>
      </label>
      <label>Recognized prior case?
        <select id="recognized_prior_case">
          <option value="">— select —</option>
          <option value="yes">yes</option>
          <option value="no">no</option>
          <option value="unsure">unsure</option>
        </select>
      </label>
      <div class="row">
        <button type="button" id="backBtn">Back</button>
        <button type="button" id="saveBtn" class="primary">Save</button>
        <button type="button" id="nextBtn" class="primary">Save &amp; Next</button>
      </div>
      <div class="status" id="status"></div>
    </form>
  </section>
</main>
<script>
(() => {
  let index = 0;
  let total = 0;
  const $ = (id) => document.getElementById(id);

  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  function syncB() {
    const a = $('decision_a').value;
    if (a && a !== 'relevant') {
      $('decision_b').value = 'not_applicable';
    }
  }

  async function loadCase(i) {
    const data = await api('/api/case?i=' + i);
    index = data.index;
    total = data.total;
    $('progress').textContent = 'Case ' + (index+1) + ' / ' + total + ' · answered ' + data.n_answered;
    $('caseMeta').textContent = data.case.case_code + ' · presentation order ' + data.case.presentation_order;
    $('title').textContent = data.case.title || '(no title)';
    $('body').textContent = data.case.body || '(empty body)';
    $('case_code').value = data.case.case_code;
    const d = data.decision || {};
    $('decision_a').value = d.decision_a || '';
    $('decision_b').value = d.decision_b || '';
    $('confidence').value = d.confidence || '';
    $('evidence_phrase').value = d.evidence_phrase || '';
    $('reason').value = d.reason || '';
    $('boundary_note').value = d.boundary_note || '';
    $('recognized_prior_case').value = d.recognized_prior_case || '';
    $('status').textContent = d.updated_utc ? ('Last saved ' + d.updated_utc) : '';
    syncB();
  }

  function payload() {
    return {
      case_code: $('case_code').value,
      decision_a: $('decision_a').value,
      decision_b: $('decision_b').value,
      confidence: $('confidence').value,
      evidence_phrase: $('evidence_phrase').value,
      reason: $('reason').value,
      boundary_note: $('boundary_note').value,
      recognized_prior_case: $('recognized_prior_case').value,
    };
  }

  async function save() {
    syncB();
    const data = await api('/api/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload()),
    });
    $('status').textContent = 'Saved ' + data.updated_utc;
    return data;
  }

  $('decision_a').addEventListener('change', syncB);
  $('saveBtn').addEventListener('click', async () => {
    try { await save(); } catch (e) { $('status').textContent = String(e); }
  });
  $('nextBtn').addEventListener('click', async () => {
    try {
      await save();
      if (index + 1 < total) await loadCase(index + 1);
      else $('status').textContent = 'Saved. End of packet.';
    } catch (e) { $('status').textContent = String(e); }
  });
  $('backBtn').addEventListener('click', async () => {
    if (index > 0) await loadCase(index - 1);
  });
  $('exportBtn').addEventListener('click', async () => {
    try {
      const data = await api('/api/export', {method:'POST'});
      $('status').textContent = 'Exported ' + data.n_with_decision_a + '/' + data.n_cases + ' to ' + data.csv;
    } catch (e) { $('status').textContent = String(e); }
  });

  // Autosave on blur of fields after edits.
  ['decision_a','decision_b','confidence','evidence_phrase','reason','boundary_note','recognized_prior_case'].forEach(id => {
    $(id).addEventListener('change', () => { save().catch(e => $('status').textContent = String(e)); });
  });

  loadCase(0).catch(e => $('status').textContent = String(e));
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "Validation200Review/1.0"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/case":
            qs = dict(x.split("=", 1) for x in (parsed.query.split("&") if parsed.query else []) if x)
            i = int(qs.get("i", "0"))
            i = max(0, min(i, len(_cases) - 1))
            case = _cases[i]
            with _lock:
                conn = db()
                row = conn.execute("SELECT * FROM decisions WHERE case_code=?", (case["case_code"],)).fetchone()
                n_answered = conn.execute(
                    "SELECT COUNT(*) AS n FROM decisions WHERE decision_a != ''"
                ).fetchone()["n"]
                conn.close()
            decision = dict(row) if row else {}
            # Strip sqlite internals if any.
            decision.pop("presentation_order", None)
            self._json(
                200,
                {
                    "index": i,
                    "total": len(_cases),
                    "n_answered": n_answered,
                    "case": case,
                    "decision": decision,
                },
            )
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if parsed.path == "/api/save":
            try:
                payload = json.loads(raw.decode("utf-8"))
                saved = save_decision(payload)
                self._json(200, saved)
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/export":
            try:
                self._json(200, export_latest())
            except Exception as exc:  # noqa: BLE001
                self._json(400, {"error": str(exc)})
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("review_app: " + (fmt % args) + "\n")


def main() -> int:
    if not CASES_PATH.exists():
        print("missing blinded cases; run scripts/run_validation_200_draw_2026-11-02.py first", file=sys.stderr)
        return 2
    # Refuse if private key somehow placed next to cases with same basename patterns in browser root — we simply never open it.
    private = RESTRICTED / "private_id_key.jsonl"
    if not private.exists():
        print("warning: private key missing; review can still proceed for labeling", file=sys.stderr)
    load_cases()
    db().close()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Blinded review app at http://{HOST}:{PORT}/")
    print(f"Cases: {CASES_PATH}")
    print(f"SQLite: {DB_PATH}")
    print("Private ID key is NOT served.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
