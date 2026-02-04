# lake_silver_layer.py
from __future__ import annotations

import re
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple, Set, Optional

import duckdb

# -----------------------------
# Vendor stopwords + cleaning
# -----------------------------
VENDOR_STOPWORDS: Set[str] = {
    "version", "release", "latest", "patch", "hotfix", "fix", "update",
    "cve", "build", "changelog", "driver", "security",
    # extra junk tokens you observed
    "https", "http", "www", "the", "and", "or", "for", "with", "from",
    "page", "read", "bug", "can", "will", "may", "user", "process"
}

def clean_vendor_set(allowed_vendors: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for v in (allowed_vendors or set()):
        v = (v or "").strip().lower()
        if len(v) < 3:
            continue
        if v in VENDOR_STOPWORDS:
            continue
        out.add(v)
    return out

# -----------------------------
# Intent inference (query)
# -----------------------------
PATCH_KWS = ("patch", "hotfix", "fixed", "security fix", "security patch", "resolved", "addressed")
VERSION_KWS = ("version", "release", "build", "latest", "changelog")

def infer_intent_from_query(q: str) -> str:
    t = (q or "").lower()
    if "cve-" in t:
        return "CVE"
    if any(k in t for k in PATCH_KWS):
        return "PATCH"
    return "VERSION"

# -----------------------------
# Helpers
# -----------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def json_dumps_safe(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return "[]"

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\[!\w+\]", " ", s)     # [!NOTE], [!IMPORTANT]
    s = re.sub(r"[#>*`]", " ", s)       # markdown-ish
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_dt_to_iso(it: Dict[str, Any]) -> str:
    for k in ("updatedAt", "createdAt", "date", "published", "created_utc"):
        v = it.get(k)
        if v is None:
            continue
        if k == "created_utc":
            try:
                return datetime.utcfromtimestamp(int(v)).isoformat()
            except Exception:
                pass
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).isoformat()
        except Exception:
            pass
    return ""

def pick_url(it: Dict[str, Any]) -> str:
    for k in ("url", "link", "permalink", "sourceUrl", "source_url"):
        v = it.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    return ""

def pick_title(it: Dict[str, Any]) -> str:
    return (
        it.get("title")
        or it.get("name")
        or it.get("versionProductName")
        or it.get("product")
        or "Untitled"
    )

def pick_body(it: Dict[str, Any]) -> str:
    return (
        it.get("versionReleaseNotes")
        or it.get("summary")
        or it.get("description")
        or it.get("content")
        or it.get("selftext")
        or ""
    )

# -----------------------------
# Sentence split
# -----------------------------
_SENT_SPLIT = re.compile(r"(?<=[\.\?\!])\s+|[\n\r]+")

def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 12:
            continue
        out.append(p)
    return out

# -----------------------------
# Regex extraction
# -----------------------------
VERSION_RE = re.compile(r"\b\d+\.\d+(?:\.\d+)?(?:[-._]?[a-z0-9]+)?\b", re.I)
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)

def sentence_intent_flags(sentence: str) -> Dict[str, int]:
    s = (sentence or "").lower()
    has_cve = 1 if ("cve-" in s or CVE_RE.search(sentence or "")) else 0
    has_patch = 1 if any(k in s for k in PATCH_KWS) else 0
    has_version = 1 if any(k in s for k in VERSION_KWS) else 0
    return {"has_cve_kw": has_cve, "has_patch_kw": has_patch, "has_version_kw": has_version}

def extract_versions(sentence: str) -> List[str]:
    return [m.group(0) for m in VERSION_RE.finditer(sentence or "")]

def extract_cves(sentence: str) -> List[str]:
    return [m.group(0).upper() for m in CVE_RE.finditer(sentence or "")]

# -----------------------------
# Vendor detection (sentence-level)
# -----------------------------
def vendors_in_sentence(sentence: str, allowed_vendors: Set[str], max_hits: int = 3) -> List[str]:
    if not sentence or not allowed_vendors:
        return []
    s = " " + re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip() + " "
    hits: List[str] = []
    for v in sorted(allowed_vendors, key=len, reverse=True):
        if v in VENDOR_STOPWORDS:
            continue
        v_pad = " " + re.sub(r"[^a-z0-9]+", " ", v.lower()).strip() + " "
        if v_pad in s:
            hits.append(v)
            if len(hits) >= max_hits:
                break
    return hits

# -----------------------------
# Silver filter (latency win)
# -----------------------------
_URLY = re.compile(r"https?://", re.I)

def should_keep_sentence(sent: str, v_hits: List[str], flags: Dict[str, int]) -> bool:
    """
    Keep sentence if it’s likely useful.
    - Must have (vendor hit) OR (intent flag)
    - Drop very URL-heavy lines
    - Drop very long kernel/log blobs
    """
    s = (sent or "").strip()
    if not s:
        return False

    # drop if it's basically a URL line
    if len(s) < 120 and _URLY.search(s) and s.count(" ") < 6:
        return False

    # drop giant log dumps
    if len(s) > 600:
        return False

    if v_hits:
        return True

    if flags.get("has_cve_kw") == 1 or flags.get("has_patch_kw") == 1 or flags.get("has_version_kw") == 1:
        return True

    return False

# -----------------------------
# DuckDB schema + build
# -----------------------------
def ensure_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS silver_documents (
        doc_id TEXT PRIMARY KEY,
        source TEXT,
        title TEXT,
        body_text TEXT,
        url TEXT,
        published_at TEXT,
        raw_json TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS silver_sentences (
        sent_id TEXT PRIMARY KEY,
        doc_id TEXT,
        source TEXT,
        url TEXT,
        published_at TEXT,
        sentence TEXT,
        sentence_lc TEXT,
        has_cve_kw INTEGER,
        has_patch_kw INTEGER,
        has_version_kw INTEGER,
        versions_json TEXT,
        cves_json TEXT,
        vendors_json TEXT,
        vendor_count INTEGER
    );
    """)

def upsert_document(con: duckdb.DuckDBPyConnection, row: Dict[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO silver_documents (doc_id, source, title, body_text, url, published_at, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
          source=excluded.source,
          title=excluded.title,
          body_text=excluded.body_text,
          url=excluded.url,
          published_at=excluded.published_at,
          raw_json=excluded.raw_json
        """,
        [row["doc_id"], row["source"], row["title"], row["body_text"],
         row["url"], row["published_at"], row["raw_json"]],
    )

def upsert_sentence(con: duckdb.DuckDBPyConnection, row: Dict[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO silver_sentences
        (sent_id, doc_id, source, url, published_at, sentence, sentence_lc,
         has_cve_kw, has_patch_kw, has_version_kw,
         versions_json, cves_json, vendors_json, vendor_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sent_id) DO UPDATE SET
          sentence=excluded.sentence,
          sentence_lc=excluded.sentence_lc,
          has_cve_kw=excluded.has_cve_kw,
          has_patch_kw=excluded.has_patch_kw,
          has_version_kw=excluded.has_version_kw,
          versions_json=excluded.versions_json,
          cves_json=excluded.cves_json,
          vendors_json=excluded.vendors_json,
          vendor_count=excluded.vendor_count
        """,
        [row["sent_id"], row["doc_id"], row["source"], row["url"], row["published_at"],
         row["sentence"], row["sentence_lc"],
         row["has_cve_kw"], row["has_patch_kw"], row["has_version_kw"],
         row["versions_json"], row["cves_json"], row["vendors_json"], row["vendor_count"]],
    )

def build_silver(
    db_path: str,
    items: List[Dict[str, Any]],
    source: str,
    allowed_vendors: Set[str],
) -> Tuple[int, int]:
    """
    Writes:
    - silver_documents
    - silver_sentences (filtered)
    """
    allowed_vendors = clean_vendor_set(set(allowed_vendors or set()))

    con = duckdb.connect(db_path)
    ensure_tables(con)

    docs = 0
    sents = 0

    for it in items:
        if not isinstance(it, dict):
            continue

        title = pick_title(it)
        body = pick_body(it)
        url = pick_url(it)
        published_at = parse_dt_to_iso(it)

        full_text = clean_text(f"{title}. {body}".strip())
        stable_key = f"{source}|{url or title}"
        doc_id = sha256_hex(stable_key)

        upsert_document(con, {
            "doc_id": doc_id,
            "source": source,
            "title": title,
            "body_text": body,
            "url": url,
            "published_at": published_at,
            "raw_json": json_dumps_safe(it),
        })
        docs += 1

        for idx, sent in enumerate(split_sentences(full_text), start=1):
            flags = sentence_intent_flags(sent)
            versions = extract_versions(sent)
            cves = extract_cves(sent)
            v_hits = vendors_in_sentence(sent, allowed_vendors, max_hits=3)

            # ✅ FILTER: don't insert junk
            if not should_keep_sentence(sent, v_hits, flags):
                continue

            sent_id = sha256_hex(f"{doc_id}|{idx}|{sent[:80]}")

            upsert_sentence(con, {
                "sent_id": sent_id,
                "doc_id": doc_id,
                "source": source,
                "url": url,
                "published_at": published_at,
                "sentence": sent,
                "sentence_lc": sent.lower(),
                "has_cve_kw": flags["has_cve_kw"],
                "has_patch_kw": flags["has_patch_kw"],
                "has_version_kw": flags["has_version_kw"],
                "versions_json": json_dumps_safe(versions),
                "cves_json": json_dumps_safe(cves),
                "vendors_json": json_dumps_safe(v_hits),
                "vendor_count": len(v_hits),
            })
            sents += 1

    con.close()
    return docs, sents