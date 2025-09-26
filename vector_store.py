"""
Builds the FAISS vector store once and saves it to disk.
Run this ONLY when you refresh the corpus.
"""

import os, shutil, requests, pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer

###############################################################################
# paths & parameters
###############################################################################
OUT_DIR     = "release_notes_store"                 # folder for dataset + index
FAISS_PATH  = os.path.join(OUT_DIR, "faiss.index")
EMB_MODEL   = "sentence-transformers/all-mpnet-base-v2"

CSV_PATH    = "SoftwareUpdateSurvey.csv"
OS_API      = "https://releasetrain.io/api/component?q=os"
REDDIT_API  = "https://releasetrain.io/api/reddit"
MAX_OS, MAX_REDDIT = 50, 50

###############################################################################
# helpers
###############################################################################
def load_csv(path):
    df = pd.read_csv(path)
    return [
        {"text": "\n".join(f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c]))}
        for _, row in df.iterrows()
    ]

def fetch_api(url, max_items, field_map):
    try:
        data = requests.get(url, timeout=15).json()
    except Exception:
        return []
    docs = []
    for item in data[:max_items]:
        lines = [f"{k}: {item.get(v, '')}" for k, v in field_map.items()]
        docs.append({"text": "\n".join(lines)})
    return docs

###############################################################################
# ingest
###############################################################################
docs = []
docs += load_csv(CSV_PATH)
docs += fetch_api(OS_API, MAX_OS, {
    "OS_ID": "_id",
    "OS_Name": "versionProductName",
    "OS_ReleaseNotes": "versionReleaseNotes",
})
docs += fetch_api(REDDIT_API, MAX_REDDIT, {
    "REDDIT_ID": "_id",
    "Subreddit": "subreddit",
    "Title": "title",
    "URL": "url",
})

print(f"Ingested {len(docs)} total documents")

###############################################################################
# embed
###############################################################################
model = SentenceTransformer(EMB_MODEL)
ds = Dataset.from_dict({"text": [d["text"] for d in docs]})
ds = DatasetDict({"train": ds})

def embed(batch):
    return {"embeddings": model.encode(batch["text"],
                                       show_progress_bar=False,
                                       batch_size=16)}
ds = ds.map(embed, batched=True, batch_size=16)

###############################################################################
# save arrow dataset FIRST
###############################################################################
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

ds.save_to_disk(OUT_DIR)            # ← no indexes attached yet

###############################################################################
# build & save FAISS separately
###############################################################################
ds["train"].add_faiss_index("embeddings")
ds["train"].save_faiss_index("embeddings", FAISS_PATH)

print("Vector store built →", OUT_DIR)