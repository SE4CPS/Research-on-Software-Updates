import os, json, re, requests, pandas as pd, streamlit as st
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
import calendar
from dateutil.relativedelta import relativedelta

OS_API     = "https://releasetrain.io/api/component?q=os"
REDDIT_API = "https://releasetrain.io/api/reddit"
EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"
DATA_DIR   = "release_notes_store_basic"
FAISS_PATH = os.path.join(DATA_DIR, "faiss.index")

CACHE_DIR = Path(".live_cache"); CACHE_DIR.mkdir(exist_ok=True)
SEED_DIR  = Path(".seeds");      SEED_DIR.mkdir(exist_ok=True)

def _normalize_results(payload):
    if isinstance(payload, list): return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list): return payload["results"]
    return []

def _load_seed(name: str):
    p = SEED_DIR / f"{name}.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: return None
    return None

def _get_json(url: str, name: str):
    safe = re.sub(r"[^a-z0-9]+", "_", f"{name}_{url}".lower()).strip("_")
    cache_path = CACHE_DIR / f"{safe}.json"
    sess = requests.Session()
    retry = Retry(total=2, backoff_factor=0.7, status_forcelist=(502,503,504), allowed_methods=frozenset(["GET"]))
    sess.mount("http://", HTTPAdapter(max_retries=retry))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    try:
        r = sess.get(url, timeout=12, headers={"User-Agent":"ReleaseNotesRec/1.0"})
        if r.status_code == 200 and "json" in r.headers.get("content-type",""):
            data = r.json()
            try: cache_path.write_text(json.dumps(data), encoding="utf-8")
            except Exception: pass
            st.caption(f"✓ {name}: live")
            return data
        raise RuntimeError(f"HTTP {r.status_code}")
    except Exception:
        if cache_path.exists():
            try:
                st.caption(f"ℹ {name}: cache")
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        seed = _load_seed(name.lower())
        if seed is not None:
            st.caption(f"ℹ {name}: seed")
            return seed
        st.caption(f"• {name}: none")
        return None

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
WEEK_REX = re.compile(r"\bweek\s+(\d{1,2})\s+of\s+(19|20)\d{2}\b", re.I)

def _as_utc(dt):
    if not dt.tzinfo: return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parse_isoish(s: str | None):
    if not s: return None
    try:
        s = s.replace("Z","+00:00").split(".")[0]
        return _as_utc(datetime.fromisoformat(s))
    except Exception: return None

def parse_time_window(q: str, now=None):
    q = (q or "").strip()
    now = _as_utc(now or datetime.now(timezone.utc))
    s = q.lower()

    if "yesterday" in s:
        d = now.date() - timedelta(days=1)
        return (datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(d, datetime.max.time(), tzinfo=timezone.utc))
    if "last week" in s:
        monday = (now - timedelta(days=now.weekday()+7)).date()
        sunday = monday + timedelta(days=6)
        return (datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc))
    if "this week" in s:
        monday = (now - timedelta(days=now.weekday())).date()
        sunday = monday + timedelta(days=6)
        return (datetime.combine(monday, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(sunday, datetime.max.time(), tzinfo=timezone.utc))
    if "last month" in s:
        first = (now.replace(day=1) - relativedelta(months=1)).date()
        last  = (now.replace(day=1) - timedelta(days=1)).date()
        return (datetime.combine(first, datetime.min.time(), tzinfo=timezone.utc),
                datetime.combine(last, datetime.max.time(), tzinfo=timezone.utc))

    m = WEEK_REX.search(q)  # ← fixed
    if m:
        wk = int(m.group(1)); yr = int(q[m.start(2):m.start(2)+4])
        monday = datetime.fromisocalendar(yr, wk, 1).replace(tzinfo=timezone.utc)
        sunday = datetime.fromisocalendar(yr, wk, 7).replace(tzinfo=timezone.utc, hour=23, minute=59, second=59)
        return monday, sunday

    for name, idx in _MONTHS.items():
        m2 = re.search(rf"\b{name}\b\s+(\d{{4}})", q, re.I)
        if m2:
            yr = int(m2.group(1))
            first = datetime(yr, idx, 1, tzinfo=timezone.utc)
            last_day = calendar.monthrange(yr, idx)[1]
            last = datetime(yr, idx, last_day, 23,59,59, tzinfo=timezone.utc)
            return first, last

    y = re.search(r"\bin\s+(20\d{2}|19\d{2})\b", q)
    if y:
        yr = int(y.group(1))
        return (datetime(yr,1,1,tzinfo=timezone.utc), datetime(yr,12,31,23,59,59,tzinfo=timezone.utc))

    return None

_STOP = set("the a an and or of for to with on in at by about into from is are was were be been being".split())
VENDOR_SYNONYMS = {
    "microsoft": {"windows", "msft"}, "windows": {"microsoft"},
    "nvidia": {"geforce", "cuda"}, "intel": {"arc", "xe"}, "amd": {"radeon"},
    "ubuntu": {"canonical"}, "redhat": {"rhel", "red hat"}, "debian": set(),
    "fedora": set(), "arch": set(), "android": {"aosp"}, "kernel": {"linux"}
}

def extract_keywords(q: str):
    ql = (q or "").lower()
    vendors = []
    for v, alts in VENDOR_SYNONYMS.items():
        if v in ql or any(a in ql for a in alts):
            vendors.append(v)
    if vendors: return vendors, True
    toks = [re.sub(r"[^a-z0-9.+#-]", "", t) for t in ql.split()]
    toks = [t for t in toks if t and t not in _STOP and len(t) >= 3]
    return toks[:6], False

def _item_datetime(it):
    return _parse_isoish(it.get("updatedAt") or it.get("createdAt") or it.get("date") or it.get("published") or it.get("created_utc"))

def filter_items(items, window, terms, vendor_strict=False):
    if not items: return []
    s, e = (window if window else (None, None))
    out = []
    for it in items:
        if window:
            dt = _item_datetime(it)
            if not dt or not (s <= dt <= e): continue
        hay = " ".join(str(it.get(k,"")) for k in ("title","name","versionProductName","versionReleaseNotes","summary","description","subreddit")).lower()
        if terms:
            if vendor_strict:
                if not all(t in hay for t in terms): continue
            else:
                if not any(t in hay for t in terms): continue
        out.append(it)
    return out

def summarize_items(title, items, limit=8):
    if not items:
        return f"**{title}**\n\n_No matching items._"
    lines = [f"**{title}**"]
    for it in items[:limit]:
        t = it.get("title") or it.get("name") or it.get("versionProductName") or "Untitled"
        url = it.get("url") or it.get("link") or ""
        dt = _item_datetime(it); ds = dt.date().isoformat() if dt else ""
        notes = (it.get("versionReleaseNotes") or it.get("summary") or it.get("description") or it.get("content") or "")
        blurb = (notes[:220] + "…") if notes and len(notes) > 220 else notes
        lines.append(f"- **{t}** — {blurb} _(date: {ds})_{' • [source]('+url+')' if url else ''}")
    return "\n\n".join(lines)

def build_store():
    docs = []
    os_raw = _get_json(OS_API, "os") or []
    rd_raw = _get_json(REDDIT_API, "reddit") or []
    os_items = _normalize_results(os_raw) if isinstance(os_raw,(list,dict)) else []
    rd_items = _normalize_results(rd_raw) if isinstance(rd_raw,(list,dict)) else []
    for it in os_items[:200]:
        docs.append({"text": " | ".join(str(it.get(k,"")) for k in ("versionProductName","versionReleaseNotes","updatedAt","createdAt"))})
    for it in rd_items[:200]:
        docs.append({"text": " | ".join(str(it.get(k,"")) for k in ("title","subreddit","url","updatedAt","created_utc"))})
    model = SentenceTransformer(EMB_MODEL)
    ds = DatasetDict({"train": Dataset.from_dict({"text":[d["text"] for d in docs]})})
    ds = ds.map(lambda b: {"embeddings": model.encode(b["text"], batch_size=16, show_progress_bar=False)}, batched=True, batch_size=16)
    os.makedirs(DATA_DIR, exist_ok=True)
    ds.save_to_disk(DATA_DIR)
    ds["train"].add_faiss_index("embeddings")
    ds["train"].save_faiss_index("embeddings", FAISS_PATH)

def load_store():
    ds = load_from_disk(DATA_DIR)
    ds["train"].load_faiss_index("embeddings", FAISS_PATH)
    return SentenceTransformer(EMB_MODEL), ds

@st.cache_resource(show_spinner="Loading vector store…")
def get_store():
    if not os.path.exists(DATA_DIR):
        build_store()
    return load_store()

embedder, datastore = get_store()

def retrieve(query, k):
    emb = embedder.encode(query, show_progress_bar=False)
    _, ex = datastore["train"].get_nearest_examples("embeddings", emb, k=k)
    return ex["text"]

SYSTEM_PROMPT = "Answer only what the user asked. Prefer live OS/Reddit; use RAG snippets only to refine."

_gem = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def call_llm(user_q, live_os_md, live_rd_md, rag_ctx_md):
    prompt = (
        f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_q}\n\n"
        f"LIVE OS:\n{live_os_md}\n\nL IVE REDDIT:\n{live_rd_md}\n\nRAG:\n{rag_ctx_md}\n\nASSISTANT:"
    )
    resp = _gem.invoke(prompt)
    return getattr(resp, "content", str(resp))

st.title("Minimal RAG + Live APIs (strict)")
top_k = st.slider("Top-K vector snippets to assist", 1, 12, 4)

if st.button("Rebuild vector store"):
    build_store()
    st.cache_resource.clear()
    st.success("Rebuilt.")

if "hist" not in st.session_state: st.session_state.hist = []
for role, msg in st.session_state.hist: st.chat_message(role).write(msg)

st.markdown(
    """
    <script>
      window.addEventListener('load', ()=>{
        const focus = () => {
          const xs = document.querySelectorAll('textarea[data-testid="stChatInput"]');
          if (xs.length) xs[xs.length-1].focus(); else setTimeout(focus, 200);
        };
        focus();
      });
    </script>
    """,
    unsafe_allow_html=True,
)

user_q = st.chat_input("Ask anything (vendor/product/time)…")

if user_q:
    st.chat_message("user").write(user_q); st.session_state.hist.append(("user", user_q))

    os_raw = _get_json(OS_API, "os") or []
    rd_raw = _get_json(REDDIT_API, "reddit") or []
    os_items = _normalize_results(os_raw) if isinstance(os_raw,(list,dict)) else []
    rd_items = _normalize_results(rd_raw) if isinstance(rd_raw,(list,dict)) else []

    window = parse_time_window(user_q)
    terms, vendor_strict = extract_keywords(user_q)

    os_f = filter_items(os_items, window, terms, vendor_strict=vendor_strict)
    rd_f = filter_items(rd_items, window, terms, vendor_strict=vendor_strict)

    live_os_md = summarize_items("OS Updates", os_f)
    live_rd_md = summarize_items("Reddit Discussions", rd_f)

    rag_snips = retrieve(user_q, top_k)
    rag_ctx_md = "\n\n".join(f"- {t[:500]}" for t in rag_snips) if rag_snips else "_(empty)_"

    answer = call_llm(user_q, live_os_md, live_rd_md, rag_ctx_md)
    st.chat_message("assistant").write(answer)
    st.session_state.hist.append(("assistant", answer))