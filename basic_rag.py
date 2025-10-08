# basic_rag.py
# Purpose: Build/refresh the vector store (RAG) used by the app/server.
# It pulls CSV + OS API + Reddit API -> embeds -> saves HF dataset + FAISS.

import os
import re
import json
import shutil
import argparse
from pathlib import Path

import requests
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------- Config (keep in sync with app/server) ----------
CSV_PATH   = "SoftwareUpdateSurvey.csv"
OS_API     = "https://releasetrain.io/api/component?q=os"
REDDIT_API = "https://releasetrain.io/api/reddit"

EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"
DATA_DIR   = "release_notes_store"
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")

# Optional lightweight cache to be resilient during rebuilds
CACHE_DIR = Path(".live_cache"); CACHE_DIR.mkdir(exist_ok=True)

# ----------------------------- Utils -----------------------------------------
def log(msg: str):
    print(f"[basic_rag] {msg}")

def _normalize_results(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return []

def _get_json(url: str, name: str):
    """
    Resilient GET: retries -> cache (no seeds).
    """
    safe = re.sub(r"[^a-z0-9]+", "_", f"{name}_{url}".lower()).strip("_")
    cache_path = CACHE_DIR / f"{safe}.json"

    sess = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=0.6,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    sess.mount("https://", HTTPAdapter(max_retries=retry))

    try:
        r = sess.get(url, timeout=15, headers={"User-Agent": "ReleaseNotesRec-RAG/Builder"})
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            data = r.json()
            try:
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            except Exception:
                pass
            log(f"{name}: live ✓")
            return data
        raise RuntimeError(f"HTTP {r.status_code} ({r.headers.get('content-type','')})")
    except Exception as e:
        if cache_path.exists():
            try:
                log(f"{name}: cache ✓ (live failed: {e})")
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        log(f"{name}: fetch failed and no cache available ({e})")
        return None

# ----------------------------- Ingestion --------------------------------------
def load_csv(path: str):
    if not os.path.exists(path):
        log(f"CSV not found at {path}; continuing without it.")
        return []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log(f"CSV load failed: {e}; continuing without CSV.")
        return []
    docs = []
    for _, row in df.iterrows():
        lines = [f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c])]
        if lines:
            docs.append({"text": "\n".join(lines)})
    log(f"CSV docs: {len(docs)}")
    return docs

def fetch_api(url: str, max_items: int, mapping: dict, name: str):
    raw = _get_json(url, name=name)
    if raw is None:
        return []
    data = _normalize_results(raw) or (raw if isinstance(raw, list) else [])
    out = []
    for item in data[:max_items]:
        lines = []
        for k, v in mapping.items():
            lines.append(f"{k}: {item.get(v, '')}")
        out.append({"text": "\n".join(lines)})
    log(f"{name} docs: {len(out)} (max {max_items})")
    return out

# ----------------------------- Build / Save -----------------------------------
def build_store(max_os: int, max_reddit: int):
    log("Starting rebuild of vector store…")

    docs  = load_csv(CSV_PATH)
    docs += fetch_api(
        OS_API,
        max_os,
        {"OS_ID": "_id", "OS_Name": "versionProductName", "OS_ReleaseNotes": "versionReleaseNotes"},
        name="os",
    )
    docs += fetch_api(
        REDDIT_API,
        max_reddit,
        {"REDDIT_ID": "_id", "Subreddit": "subreddit", "Title": "title", "URL": "url"},
        name="reddit",
    )

    if not docs:
        raise RuntimeError("No documents gathered; aborting.")

    log(f"Total documents to embed: {len(docs)}")
    texts = [d["text"] for d in docs]

    log(f"Loading embedding model: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL)

    def _embed(batch):
        return {
            "embeddings": model.encode(
                batch["text"],
                show_progress_bar=False,
                batch_size=16,
            )
        }

    log("Building HF dataset…")
    ds = DatasetDict({"train": Dataset.from_dict({"text": texts})})

    log("Embedding (this may take a bit)…")
    ds = ds.map(_embed, batched=True, batch_size=16)

    # Clean & save
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    log(f"Saving dataset to {DATA_DIR} …")
    ds.save_to_disk(DATA_DIR)

    log("Loading back & creating FAISS index…")
    ds2 = load_from_disk(DATA_DIR)
    ds2["train"].add_faiss_index("embeddings")
    ds2["train"].save_faiss_index("embeddings", FAISS_PATH)

    log(f"Done. Vector store saved to '{DATA_DIR}' with FAISS index at '{FAISS_PATH}'.")

def info():
    if not os.path.exists(DATA_DIR):
        print(json.dumps({"status": "absent", "path": DATA_DIR}, indent=2))
        return
    try:
        ds = load_from_disk(DATA_DIR)
        n = len(ds["train"])
        has_idx = os.path.exists(FAISS_PATH)
        print(json.dumps(
            {"status": "ok", "path": DATA_DIR, "rows": n, "faiss_index": has_idx, "faiss_path": FAISS_PATH},
            indent=2
        ))
    except Exception as e:
        print(json.dumps({"status": "corrupt", "path": DATA_DIR, "error": str(e)}, indent=2))

# ----------------------------- CLI -------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build/refresh the vector store for Release Notes RAG.")
    parser.add_argument("--max-os", type=int, default=50, help="Max OS API items to ingest (default: 50)")
    parser.add_argument("--max-reddit", type=int, default=50, help="Max Reddit API items to ingest (default: 50)")
    parser.add_argument("--info", action="store_true", help="Show vector store status and exit")
    args = parser.parse_args()

    if args.info:
        info()
        return

    build_store(max_os=args.max_os, max_reddit=args.max_reddit)

if __name__ == "__main__":
    main()