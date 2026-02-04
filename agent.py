    # agent.py
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb
from filelock import FileLock

from lake_silver_layer import (
    infer_intent_from_query,
    build_silver,
    clean_vendor_set,
)
from lake_gold import build_gold_facts, build_gold_latest_version

# -----------------------------
# Config
# -----------------------------
VENDOR_STOPWORDS: Set[str] = {
    "version", "release", "latest", "patch", "hotfix", "fix", "update",
    "cve", "build", "changelog", "driver", "security",
}

DEFAULT_TTL_SEC = 6 * 3600  # 6 hours (tune this)

# -----------------------------
# Vendor detection (query)
# -----------------------------
def detect_vendor_from_query(query: str, allowed_vendors: Set[str]) -> Optional[str]:
    q = (query or "").lower().strip()
    if not q or not allowed_vendors:
        return None

    q_pad = " " + re.sub(r"[^a-z0-9]+", " ", q).strip() + " "

    best: Optional[str] = None
    best_len = 0

    for v in allowed_vendors:
        v = (v or "").strip().lower()
        if not v or v in VENDOR_STOPWORDS:
            continue
        if len(v) < 3:
            continue
        v_pad = " " + re.sub(r"[^a-z0-9]+", " ", v).strip() + " "
        if v_pad in q_pad and len(v) > best_len:
            best = v
            best_len = len(v)

    return best

# -----------------------------
# Lake state (TTL rebuild)
# -----------------------------
def ensure_state_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS lake_state (
        vendor TEXT PRIMARY KEY,
        last_built_at_utc TEXT
    );
    """)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _age_sec(iso: str) -> Optional[int]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return int((datetime.now(timezone.utc) - dt).total_seconds())
    except Exception:
        return None

def should_rebuild_vendor(con: duckdb.DuckDBPyConnection, vendor: str, ttl_sec: int) -> bool:
    ensure_state_table(con)
    row = con.execute("SELECT last_built_at_utc FROM lake_state WHERE vendor = ? LIMIT 1", [vendor]).fetchone()
    if not row:
        return True
    age = _age_sec(row[0] or "")
    if age is None:
        return True
    return age > ttl_sec

def mark_built_vendor(con: duckdb.DuckDBPyConnection, vendor: str) -> None:
    ensure_state_table(con)
    con.execute("""
      INSERT INTO lake_state (vendor, last_built_at_utc)
      VALUES (?, ?)
      ON CONFLICT(vendor) DO UPDATE SET last_built_at_utc=excluded.last_built_at_utc
    """, [vendor, _now_iso()])

# -----------------------------
# Core: agent_answer
# -----------------------------
def agent_answer(
    q: str,
    allowed_vendors: Set[str],
    get_json,
    normalize_results,
    os_api: str,        # not used once vendor-scoped (kept for compatibility)
    reddit_api: str,    # not used once vendor-scoped (kept for compatibility)
    limit: int = 12,
    db_path: str = "releasetrain.duckdb",
    ttl_sec: int = DEFAULT_TTL_SEC,
) -> Dict[str, Any]:
    q = (q or "").strip()
    intent = infer_intent_from_query(q)

    allowed_vendors_clean = clean_vendor_set(set(allowed_vendors or set()))
    vendor = detect_vendor_from_query(q, allowed_vendors_clean)

    if not vendor:
        return {
            "abstained": True,
            "confidence": 0.30,
            "intent": intent,
            "resolved_vendors": [],
            "short_answer": "I don’t know. Vendor not found in vendor index.",
            "meta": f"Intent: {intent} · VendorMatch: no · Confidence: 30%",
            "citations": [],
            "evidence": [],
        }

    # ✅ Vendor-scoped APIs
    os_url = f"https://releasetrain.io/api/component?q={vendor}"
    rd_url = f"https://releasetrain.io/api/reddit?q={vendor}"

    lock_path = str(Path(db_path).with_suffix(".lock"))
    with FileLock(lock_path):
        conw = duckdb.connect(db_path)
        rebuild = should_rebuild_vendor(conw, vendor, ttl_sec)

        if rebuild:
            # fetch live items (scoped)
            items: List[dict] = []
            for src, url in [("os", os_url), ("reddit", rd_url)]:
                raw = get_json(url, src) or []
                for it in normalize_results(raw):
                    if isinstance(it, dict):
                        it2 = dict(it)
                        it2["_source"] = src
                        items.append(it2)

            os_items = [it for it in items if it.get("_source") == "os"]
            rd_items = [it for it in items if it.get("_source") == "reddit"]

            # build silver
            build_silver(db_path, os_items, "os", allowed_vendors_clean)
            build_silver(db_path, rd_items, "reddit", allowed_vendors_clean)

            # build gold
            build_gold_facts(db_path, allowed_vendors_clean)
            build_gold_latest_version(db_path)

            mark_built_vendor(conw, vendor)

        conw.close()

    # ✅ Answer phase (read-only connection)
    if intent == "VERSION":
        return answer_version(db_path, vendor, intent)

    if intent in ("CVE", "PATCH"):
        return answer_cve_patch(db_path, vendor, intent, limit=limit)

    return answer_vendor_only(db_path, vendor, intent, limit=limit)

# -----------------------------
# Answer helpers
# -----------------------------
def answer_version(db_path: str, vendor: str, intent: str) -> Dict[str, Any]:
    con = duckdb.connect(db_path, read_only=True)
    row = con.execute("""
        SELECT latest_version, fact_date, source, url, snippet
        FROM gold_latest_version
        WHERE vendor = ?
        LIMIT 1
    """, [vendor]).fetchone()
    con.close()

    if row:
        latest_version, fact_date, source, url, snippet = row
        conf = 0.85
        evidence = [{
            "source": source or "",
            "title": f"{vendor} {latest_version}",
            "date": fact_date or "",
            "url": url or "",
            "snippet": snippet or "",
        }]
        return {
            "abstained": False,
            "confidence": conf,
            "intent": intent,
            "resolved_vendors": [vendor],
            "short_answer": f"Latest version of {vendor} is **{latest_version}**.",
            "meta": f"Intent: {intent} · Vendor: {vendor} · VersionFound: yes · Confidence: {conf*100:.0f}%",
            "citations": [],
            "evidence": evidence,
        }

    conf = 0.45
    return {
        "abstained": True,
        "confidence": conf,
        "intent": intent,
        "resolved_vendors": [vendor],
        "short_answer": f"I don’t know the latest version of {vendor} from the current evidence.",
        "meta": f"Intent: {intent} · Vendor: {vendor} · VersionFound: no · Confidence: {conf*100:.0f}%",
        "citations": [],
        "evidence": [],
    }

def answer_cve_patch(db_path: str, vendor: str, intent: str, limit: int = 12) -> Dict[str, Any]:
    con = duckdb.connect(db_path, read_only=True)
    intent_filter = "has_cve_kw = 1" if intent == "CVE" else "has_patch_kw = 1"

    rows = con.execute(f"""
        SELECT source, url, published_at, sentence
        FROM silver_sentences
        WHERE vendors_json LIKE '%' || ? || '%'
          AND {intent_filter}
        ORDER BY published_at DESC
        LIMIT ?
    """, [vendor, limit]).fetchall()
    con.close()

    if not rows:
        conf = 0.49
        return {
            "abstained": True,
            "confidence": conf,
            "intent": intent,
            "resolved_vendors": [vendor],
            "short_answer": f"I don’t know about {vendor} for {intent} from the current evidence.",
            "meta": f"Intent: {intent} · Vendor: {vendor} · SameSentence: no · Evidence: 0 · Confidence: {conf*100:.0f}%",
            "citations": [],
            "evidence": [],
        }

    evidence = [{
        "source": source or "",
        "title": f"{vendor} {intent} evidence",
        "date": published_at or "",
        "url": url or "",
        "snippet": (sentence or "")[:240],
    } for (source, url, published_at, sentence) in rows[:limit]]

    conf = 0.70
    return {
        "abstained": False,
        "confidence": conf,
        "intent": intent,
        "resolved_vendors": [vendor],
        "short_answer": f"Here’s what I found for **{vendor}** related to **{intent}**.",
        "meta": f"Intent: {intent} · Vendor: {vendor} · SameSentence: yes · Evidence: {len(evidence)} · Confidence: {conf*100:.0f}%",
        "citations": [],
        "evidence": evidence,
    }

def answer_vendor_only(db_path: str, vendor: str, intent: str, limit: int = 12) -> Dict[str, Any]:
    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("""
        SELECT source, url, published_at, sentence
        FROM silver_sentences
        WHERE vendors_json LIKE '%' || ? || '%'
        ORDER BY published_at DESC
        LIMIT ?
    """, [vendor, limit]).fetchall()
    con.close()

    if not rows:
        conf = 0.45
        return {
            "abstained": True,
            "confidence": conf,
            "intent": intent,
            "resolved_vendors": [vendor],
            "short_answer": f"I don’t know about {vendor} from the current evidence.",
            "meta": f"Intent: {intent} · Vendor: {vendor} · Evidence: 0 · Confidence: {conf*100:.0f}%",
            "citations": [],
            "evidence": [],
        }

    evidence = [{
        "source": source or "",
        "title": f"{vendor} evidence",
        "date": published_at or "",
        "url": url or "",
        "snippet": (sentence or "")[:240],
    } for (source, url, published_at, sentence) in rows[:limit]]

    conf = 0.50
    return {
        "abstained": False,
        "confidence": conf,
        "intent": intent,
        "resolved_vendors": [vendor],
        "short_answer": f"Here’s what I found mentioning **{vendor}**.",
        "meta": f"Intent: {intent} · Vendor: {vendor} · Evidence: {len(evidence)} · Confidence: {conf*100:.0f}%",
        "citations": [],
        "evidence": evidence,
    }