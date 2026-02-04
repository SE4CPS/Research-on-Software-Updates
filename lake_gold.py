# lake_gold.py
from __future__ import annotations
import re
import duckdb
from typing import Optional, Tuple, List

# basic semver compare: split numbers, ignore suffix
def semver_key(v: str) -> Tuple[int, ...]:
    # keep leading numeric segments only
    parts = re.split(r"[^\d]+", (v or "").strip())
    nums = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
    return tuple(nums) if nums else (0,)

def ensure_gold_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS gold_release_facts (
        fact_id TEXT PRIMARY KEY,
        vendor TEXT,
        fact_type TEXT,
        fact_value TEXT,
        fact_date TEXT,
        source TEXT,
        sent_id TEXT,
        url TEXT,
        snippet TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS gold_latest_version (
        vendor TEXT PRIMARY KEY,
        latest_version TEXT,
        fact_date TEXT,
        source TEXT,
        url TEXT,
        snippet TEXT
    );
    """)

def build_gold_facts(db_path: str, allowed_vendors: set[str]) -> int:
    """
    Create version facts from silver_sentences for known vendors.
    Stores one fact per (vendor, sentence version).
    """
    con = duckdb.connect(db_path)
    ensure_gold_tables(con)

    # Pull all sentences that have version mentions.
    rows = con.execute("""
        SELECT sent_id, source, url, published_at, sentence, versions_json
        FROM silver_sentences
        WHERE has_version_kw = 1
  AND versions_json IS NOT NULL
  AND versions_json != '[]'
    """).fetchall()

    inserted = 0
    for sent_id, source, url, published_at, sentence, versions_json in rows:
        sent_lc = (sentence or "").lower()

        # vendor detection against allowed list: choose longest match
        vendor = best_vendor_match(sent_lc, allowed_vendors)
        if not vendor:
            continue

        versions = parse_versions_json(versions_json)
        for v in versions:
            fact_id = sha1_like(f"{vendor}|{sent_id}|{v}")
            con.execute("""
                INSERT INTO gold_release_facts
                (fact_id, vendor, fact_type, fact_value, fact_date, source, sent_id, url, snippet)
                VALUES (?, ?, 'LATEST_VERSION_CANDIDATE', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO NOTHING
            """, [fact_id, vendor, v, (published_at or ""), source, sent_id, url, sentence[:260]])
            inserted += 1

    con.close()
    return inserted

def build_gold_latest_version(db_path: str) -> int:
    """
    Recompute gold_latest_version by ranking:
    - newest date first (ISO string works)
    - then highest semver
    - prefer os over reddit
    """
    con = duckdb.connect(db_path)
    ensure_gold_tables(con)

    facts = con.execute("""
        SELECT vendor, fact_value, fact_date, source, url, snippet
        FROM gold_release_facts
        WHERE fact_type='LATEST_VERSION_CANDIDATE'
    """).fetchall()

    # IMPORTANT: use a name that won't get shadowed accidentally
    best_by_vendor: dict[str, tuple] = {}

    for vendor, value, fact_date, source, url, snippet in facts:
        key = (
            fact_date or "",            # newest first
            semver_key(value),          # highest semver
            1 if source == "os" else 0  # prefer os
        )
        prev = best_by_vendor.get(vendor)
        if (prev is None) or (key > prev[0]):
            best_by_vendor[vendor] = (key, value, fact_date, source, url, snippet)

    # overwrite table
    con.execute("DELETE FROM gold_latest_version")

    for vendor, pack in best_by_vendor.items():
        _, value, fact_date, source, url, snippet = pack
        con.execute("""
            INSERT INTO gold_latest_version (vendor, latest_version, fact_date, source, url, snippet)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [vendor, value, fact_date or "", source or "", url or "", snippet or ""])

    n = len(best_by_vendor)
    con.close()
    return n

def best_vendor_match(text_lc: str, allowed_vendors: set[str]) -> Optional[str]:
    if not text_lc or not allowed_vendors:
        return None
    # boundary-ish match
    text_pad = " " + re.sub(r"[^a-z0-9]+", " ", text_lc).strip() + " "
    best = None
    best_len = 0
    for v in allowed_vendors:
        v = (v or "").strip().lower()
        if len(v) < 3:
            continue
        v_pad = " " + re.sub(r"[^a-z0-9]+", " ", v).strip() + " "
        if v_pad in text_pad and len(v) > best_len:
            best = v
            best_len = len(v)
    return best

def parse_versions_json(s: str) -> List[str]:
    try:
        import json
        out = json.loads(s)
        return [str(x) for x in out] if isinstance(out, list) else []
    except Exception:
        return []

def sha1_like(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()