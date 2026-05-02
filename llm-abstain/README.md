# Release Master — LLM Abstain System

> **Software Update Q&A System** powered by a 5-gate abstain pipeline, NVIDIA NeMo stack, and real-time hallucination detection.

The system answers questions about software releases, CVEs, patches, and breaking changes by querying **releasetrain.io**. Instead of guessing, it says **"I don't know"** whenever it cannot find verified data — a deliberate design decision called *controlled abstention*.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture Overview](#architecture-overview)
3. [The 5-Gate Abstain Pipeline](#the-5-gate-abstain-pipeline)
4. [NVIDIA NeMo Components](#nvidia-nemo-components)
5. [Hallucination Detection — BERTScore](#hallucination-detection--bertscore)
6. [Composite Confidence Score](#composite-confidence-score)
7. [KPI Dashboard](#kpi-dashboard)
8. [LRU Cache Strategy](#lru-cache-strategy)
9. [Example Queries — Confident Answers](#example-queries--confident-answers)
10. [Example Queries — I Don't Know](#example-queries--i-dont-know)
11. [Setup & Run](#setup--run)
12. [Environment Variables](#environment-variables)
13. [Project Structure](#project-structure)

---

## What It Does

| User asks | System does |
|---|---|
| "What CVEs affect Firefox?" | Queries releasetrain.io, runs 5-gate validation, returns verified CVE data |
| "Latest Node.js patches?" | Finds patch-channel releases, scores confidence, returns structured card |
| "How do I cook pasta?" | Gate 1 detects off-topic → returns "I don't know" immediately |
| "Latest updates for FigJam?" | Gate 3 finds no data in releasetrain.io → returns "I don't know" |

The core principle: **never hallucinate**. Every answer is traceable back to a releasetrain.io source URL.

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  NLP Processor                                          │
│  • Extract entity  (Firefox, Android, Node.js…)         │
│  • Detect intent   (CVE / patch / version / breaking)   │
│  • Parse date filter (yesterday, last 7 days…)          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  5-Gate Abstain Pipeline                                │
│                                                         │
│  Gate 1 → Topic Relevance   (NeMo Guardrails LLM)       │
│  Gate 2 → NLP Confidence    (entity confidence ≥ 0.70)  │
│  Gate 3 → Software Lookup   (releasetrain.io API)       │
│  Gate 4 → Software Validity (entries returned > 0)      │
│  Gate 5 → Response Quality  (data completeness ≥ 0.30)  │
│                                                         │
│  ──── NeMo Reranker (between Gate 3 and Gate 4) ──────  │
│  ──── Composite Score Gate (weighted avg ≥ 0.60) ─────  │
└────────────────────────┬────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        CONFIDENT                 ABSTAIN
    Structured response        "I don't know"
    + BERTScore computed       reason returned
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  Frontend — React / Vite                                │
│  • KPI Bar  (BERTScore ring + Composite progress bar)   │
│  • Gate Timeline  (collapsible pass/fail per gate)      │
│  • Software Card  (CVE / Patch / Version / Breaking)    │
└─────────────────────────────────────────────────────────┘
```

---

## The 5-Gate Abstain Pipeline

Each gate must **pass** before the next runs. If any gate fails, the pipeline short-circuits and returns "I don't know" with the reason.

### Gate 1 — Topic Relevance

**Purpose:** Is this query about software/OS updates, CVEs, patches, or releases?

**Method:** LLM zero-shot classification via NeMo Guardrails (`meta/llama-3.1-8b-instruct`). Falls back to regex keyword scoring when no API key is set.

**Threshold:** Score must be ≥ **0.60**

**Fail example:**
```
Query: "How do I deploy to AWS?"
→ LLM classifies as off-topic, score = 0.05
→ ABSTAIN: "Off-topic (score 0.050 < 0.60)"
```

**Pass example:**
```
Query: "What CVEs affect Firefox?"
→ LLM classifies as relevant, confidence = 0.95
→ PASS: score = 0.95
```

---

### Gate 2 — NLP Extraction Confidence

**Purpose:** Was a specific software or OS entity extracted with sufficient confidence?

**Confidence levels by match type:**

| Match type | Confidence | Example |
|---|---|---|
| OS regex pattern | 0.95 | "windows 11", "ubuntu 22" |
| Software regex | 0.92 | "firefox", "docker", "node.js" |
| Device pattern | 0.88 | "ipad", "pixel 7" |
| Component list match | 0.95 | any of ~6,400 known names |
| NLP noun fallback | 0.55 | generic nouns (always fails gate) |

**Threshold:** Confidence must be ≥ **0.70**

**Fail example:**
```
Query: "Any new updates today?"
→ No named entity found → confidence = 0
→ ABSTAIN: "No software/OS entity could be extracted"
```

---

### Gate 3 — Software Lookup

**Purpose:** Does releasetrain.io have any data for this software?

**Method:** `GET https://releasetrain.io/api/v/?q={entity}`

If 0 results are returned, the system automatically tries a **canonical name resolution**:
- `"node.js"` → retries as `"Node"`
- `"vs code"` → retries as `"vscode"`
- `"k8s"` → retries as `"kubernetes"`

**Fail example:**
```
Query: "Latest updates for Obsidian?"
→ "obsidian" → 0 results
→ Canonical resolver finds no alias
→ ABSTAIN: '"obsidian" was not found in releasetrain.io'
```

---

### Gate 4 — Software Validity

**Purpose:** Confirm the entries array from Gate 3 is non-empty after processing.

This gate is a safety check — it always passes when Gate 3 passes unless an edge case clears the entries between gates.

---

### Gate 5 — Response Quality

**Purpose:** Filter entries by the detected query type and score data completeness.

**Query types and filters:**

| Query type | Filter applied |
|---|---|
| `cve` | `isCve === true` OR nvd.nist.gov URL OR SECURITY classification |
| `patch` | `versionReleaseChannel === "patch"` OR `"hotfix"` |
| `version` | Top entry must have `versionNumber` present |
| `critical` | `classification.breakingType` includes `"Critical Failure"` |
| `breaking` | `classification.breakingType` includes detected sub-type |
| `general` | Any entry with `versionNumber` |

**Completeness scoring:**
```
quality = populated_fields / total_fields
```
Threshold: quality ≥ **0.30**

**Fail example:**
```
Query: "Critical failures in Notepad?"
→ Notepad found in releasetrain.io (Gate 3 passes)
→ Zero entries with breakingType = "Critical Failure"
→ ABSTAIN: 'no "Critical Failure" releases for "Notepad"'
```

---

### Composite Score Gate

After all 5 gates pass, the weighted average must exceed **0.60** to return a confident answer.

**Formula:**
```
Composite = (Gate1_score × 0.25)
          + (Gate2_score × 0.25)
          + (Gate3_score × 0.25)
          + (Gate5_score × 0.25)
```

**Example calculation — "GitHub breaking update":**
```
Gate 1 (topic relevance)    = 0.85  × 0.25 = 0.2125
Gate 2 (NLP confidence)     = 0.92  × 0.25 = 0.2300
Gate 3 (software lookup)    = 1.00  × 0.25 = 0.2500
Gate 5 (response quality)   = 0.70  × 0.25 = 0.1750
                                      ─────────────
Composite                             = 0.8675  → 86.8%
Decision: CONFIDENT  (0.8675 ≥ 0.60 ✓)
```

**Example — barely passes individually but fails composite:**
```
Gate 1 = 0.61 × 0.25 = 0.1525
Gate 2 = 0.71 × 0.25 = 0.1775
Gate 3 = 1.00 × 0.25 = 0.2500
Gate 5 = 0.31 × 0.25 = 0.0775
                ──────────────
Composite      = 0.5575  → 55.8%
Decision: ABSTAIN  (0.5575 < 0.60 ✗)
"Overall confidence 55.8% is below the minimum 60%"
```

---

## NVIDIA NeMo Components

### 1. NeMo Guardrails — Gate 1 LLM Classifier

**File:** `server/services/nemoGuardrails.js`

**Model:** `meta/llama-3.1-8b-instruct` (via NVIDIA NIM)

**How it works:** Sends the user query to the LLM with a NeMo Guardrails-style system prompt that defines the allowed topic space in natural language. The LLM returns a JSON object:

```json
{ "relevant": true, "confidence": 0.95, "reason": "asks about software CVEs" }
```

**Fallback chain:**
```
NVIDIA_API_KEY set?
  Yes → call LLM classifier
    LLM call succeeds? → use LLM score
    LLM call fails?    → fall back to regex (positiveScore / isUpdateRelated)
  No  → use regex scores directly
```

**LRU Cache:** Results cached for **30 minutes** (200-entry capacity). Same query never calls the LLM twice within that window.

---

### 2. Triton Client — Local Embedding Server

**File:** `server/services/tritonClient.js`

**Purpose:** Provides embeddings for the reranker and BERTScore. Tries a local GPU server before falling back to NVIDIA's cloud.

**Priority chain:**
```
1. Local Triton server (TRITON_URL, default: http://localhost:8000)
   → GET /v2/health/ready  (health check, result cached for session)
   → POST /v2/models/{TRITON_MODEL}/infer
     inputs:  text_input (BYTES), input_type (BYTES)
     outputs: embedding (FP32 [N, dim])

2. NVIDIA NIM cloud  (requires NVIDIA_API_KEY)
   → POST https://integrate.api.nvidia.com/v1/embeddings
     model: nvidia/nv-embedqa-e5-v5
```

Once Triton is found unreachable, it is skipped for the rest of the server session (cached `false`).

---

### 3. NeMo Reranker — Cross-Encoder Ranking

**File:** `server/services/nemoRetriever.js`

**Model:** `nvidia/nv-rerankqa-mistral-4b-v3`

**What it replaces:** Cosine similarity between embeddings (bi-encoder). A cross-encoder reads the query and each passage *together*, giving a much more accurate relevance score.

**How it works:**
```
POST /v1/ranking
{
  "model": "nvidia/nv-rerankqa-mistral-4b-v3",
  "query":    { "text": "Firefox CVEs" },
  "passages": [{ "text": "Firefox v128 — CVE security vulnerability — ..." }, ...],
  "truncate": "END"
}

Response:
{ "rankings": [{ "index": 2, "logit": 1.42 }, { "index": 0, "logit": 0.87 }, ...] }
```

Entries are re-sorted by `logit` descending → most relevant first → Gate 5 sees the best entries at the top.

**Fallback chain:**
```
NVIDIA_API_KEY set?
  Yes → try cross-encoder reranker
    Succeeds? → return re-ranked entries
    Fails?    → fall back to Triton/NIM embeddings + cosine similarity
  No  → try Triton embeddings + cosine similarity
    Triton unavailable + no API key? → return original order
```

---

## Hallucination Detection — BERTScore

**File:** `server/services/hallucination.js`

**What it measures:** How well the structured response is *grounded* in the raw releasetrain.io source data. Since this system extracts data directly from an API (no LLM generation), hallucination risk is low — but the score gives a quantitative grounding guarantee.

### The Three Numbers

| Metric | Name in UI | Meaning |
|---|---|---|
| Precision (P) | **Grounded** | % of response words found in the source data |
| Recall (R) | **Coverage** | % of source data words that appear in the response |
| F1 | **BERTScore** | Harmonic mean of P and R |

### Why Coverage Is Always Low

The source pool contains **all entries** for a software (potentially hundreds). The response shows only the top 1–5 results. A focused answer that shows "Firefox v128" will only touch a tiny fraction of all Firefox source text.

```
Coverage = 23% is EXPECTED and NORMAL for a summary answer.
It does NOT indicate hallucination.
```

### Why Risk Uses Grounded (P), Not F1

F1 is dragged down by low Coverage, making a well-grounded answer look risky. Risk is based solely on Grounded:

```
Grounded ≥ 78%  →  Low Risk    (green)
Grounded ≥ 55%  →  Medium Risk (amber)
Grounded  < 55%  →  High Risk   (red)
```

### Calculation Example

**Query:** "What is the latest version of Firefox?"
**Response:** version 128.0, released 2024-07-09, stable channel
**Source pool:** 79 Firefox entries from releasetrain.io

```
Hypothesis phrases (from response):
  "software Firefox"
  "version 128.0"
  "released 2024-07-09"
  "stable release channel"

Reference phrases (from top source entries):
  "Firefox v128.0 stable Released 2024-07-09 ..."
  "Firefox v127.0 stable ..."
  ... (10 entries)

Grounded (P) = tokens in response ∩ tokens in source / tokens in response
             = 0.786  →  78.6%

Coverage (R) = tokens in source ∩ tokens in response / tokens in source
             = 0.324  →  32.4%

BERTScore F1 = 2 × 0.786 × 0.324 / (0.786 + 0.324)
             = 0.509 / 1.110
             = 0.459  →  45.9%

Risk: Grounded 78.6% ≥ 78% threshold → LOW RISK ✓
```

### Embedding vs Lexical

| Mode | When active | Method |
|---|---|---|
| `embedding` | Triton or NVIDIA NIM available | Pairwise cosine similarity between BERT vectors |
| `lexical` | No embeddings available | Token overlap (ROUGE-1 style) |

Both produce the same P / R / F1 output shape. Embedding mode is more semantically aware (catches synonyms, paraphrases).

---

## Composite Confidence Score

Shown as a **progress bar** in the KPI dashboard.

### Thresholds

```
≥ 80%   High confidence   (teal / green)
60–79%  Acceptable        (teal / green)
< 60%   Too low → ABSTAIN (bar still shown, decision = Abstain)
```

### What Each Gate Score Represents

| Gate | Score source |
|---|---|
| Gate 1 | LLM confidence (0–1) or regex positiveScore |
| Gate 2 | Entity extraction confidence (0.88–0.95 for regex, 0.55 for NLP fallback) |
| Gate 3 | Always 1.0 when passing (binary found/not-found) |
| Gate 5 | `populated_fields / total_fields` for the query type |

---

## KPI Dashboard

The center-top strip that appears after the first query result.

```
┌──────────────────────────────┐  ┌──────────────────────┐
│  HALLUCINATION DETECTION     │  │  COMPOSITE CONFIDENCE │
│                              │  │                       │
│   ╭────╮  BERTScore 45.9%   │  │   86%                 │
│   │ 46 │  Grounded  78%     │  │   ████████░░          │
│   ╰────╯  Coverage  32%     │  │   60%  80%            │
│           ● Low Risk        │  │   ● Confident         │
└──────────────────────────────┘  └──────────────────────┘
```

| Element | Description |
|---|---|
| Ring gauge | Animated SVG circle fill = BERTScore F1 |
| Risk badge | Color-coded: green (Low), amber (Medium), red (High) |
| Progress bar | Width = composite score %, animated on update |
| Threshold markers | Vertical lines at 60% and 80% |
| Decision pill | Confident / Suggestion / Abstained |

On **Abstain**: BERTScore shows 0%, risk = High (no response to evaluate).
Updates after **every** query — reflects the most recent result.

---

## LRU Cache Strategy

**File:** `server/utils/lruCache.js`

Standard O(1) LRU using JavaScript `Map` insertion-order:
- `get(key)` → deletes + re-inserts at tail (promotes to MRU), checks TTL
- `set(key, val)` → inserts at tail; if at capacity, deletes head (LRU)
- `_map.keys().next().value` → always the least-recently-used key

### Cache Instances

| Cache | Capacity | TTL | Purpose |
|---|---|---|---|
| `apiClient` searchCache | 100 entries | 5 min | releasetrain.io API responses |
| `nemoGuardrails` classifyCache | 200 entries | 30 min | LLM topic classification results |

### Why LRU Over Plain Map

The original implementation used a plain `Map` that grew indefinitely. With LRU:
- Memory is bounded — old software lookups are evicted automatically
- Frequently queried software (Firefox, Android, Chrome) stays warm
- Hit-rate is logged: `[API Client] LRU hit "firefox" (79 entries) | 50.0% hit-rate`

### LRU Eviction Example

```
Cache capacity = 3, queries in order:
  firefox   → MISS, insert  [firefox]
  android   → MISS, insert  [firefox, android]
  chrome    → MISS, insert  [firefox, android, chrome]
  node      → MISS, capacity full → evict "firefox" (LRU head)
              insert  [android, chrome, node]
  android   → HIT,  promote to tail
              [chrome, node, android]
  firefox   → MISS, evict "chrome" → [node, android, firefox]
```

---

## Example Queries — Confident Answers

These queries pass all 5 gates and return structured data.

```bash
# Latest version of a software
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the latest version of Firefox?"}'

# CVE security vulnerabilities
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What CVEs affect Android?"}'

# Recent patches / hotfixes
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Latest patches for Chrome?"}'

# Breaking changes
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"GitHub breaking update"}'

# Critical failures
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Critical failures in Docker?"}'

# Date-filtered query
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Firefox CVEs last 30 days"}'

# Network issues
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Network issues in Kubernetes?"}'

# Specific OS version
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Latest iOS security updates"}'
```

---

## Example Queries — I Don't Know

Each fails at a different gate — the response includes `reason` and `gates[]` showing exactly where it stopped.

### Gate 1 Fails — Off-topic query

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I deploy to AWS?"}'
# reason: "Off-topic (score 0.050 < 0.60)"

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What programming language should I learn?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain what a REST API is"}'
```

### Gate 2 Fails — No software entity found

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the latest security patches this week?"}'
# reason: "No software/OS entity could be extracted"

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Any breaking changes released yesterday?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me recent CVEs"}'
```

### Gate 3 Fails — Software not in releasetrain.io

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Latest updates for Obsidian?"}'
# reason: '"obsidian" was not found in releasetrain.io'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Any CVEs for Cursor IDE?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Patches for FigJam?"}'
```

### Gate 5 Fails — Software found but no matching data for query type

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Critical failures in Notepad?"}'
# reason: 'no "Critical Failure" releases for "Notepad"'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the CVEs for Winamp?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Breaking changes in Paint?"}'
```

### Composite Score Too Low

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me something about git"}'
# reason: "Overall confidence X% is below the minimum 60%"
```

### Quick decision-only output (no full JSON)

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Latest updates for Obsidian?"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['decision'].upper(), '—', d.get('reason',''))"
```

---

## Setup & Run

### Prerequisites
- Node.js ≥ 18
- (Optional) NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) — enables NeMo Guardrails, Reranker, and NIM embeddings
- (Optional) Local Triton Inference Server for GPU embedding inference

### Install & Start

```bash
# Clone the repo
git clone https://github.com/SE4CPS/Research-on-Software-Updates.git
cd Research-on-Software-Updates/llm-abstain

# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env — add NVIDIA_API_KEY if available

# Start both servers (Express + Vite)
npm run dev
```

**Backend** runs at `http://localhost:3001`  
**Frontend** runs at `http://localhost:5173`

### Run servers separately

```bash
npm run dev:server   # Express API only
npm run dev:client   # Vite frontend only
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3001` | Express server port |
| `API_BASE_URL` | `https://releasetrain.io/api` | releasetrain.io API base |
| `NVIDIA_API_KEY` | *(blank)* | Enables NeMo Guardrails, Reranker, NIM embeddings |
| `GUARDRAILS_MODEL` | `meta/llama-3.1-8b-instruct` | LLM used for Gate 1 topic classification |
| `TRITON_URL` | `http://localhost:8000` | Local Triton Inference Server base URL |
| `TRITON_MODEL` | `nv-embedqa-e5-v5` | Model name deployed in Triton |

All NVIDIA features degrade gracefully when `NVIDIA_API_KEY` is not set — the app runs fully in offline/regex mode.

---

## Project Structure

```
llm-abstain/
├── client/                          # React / Vite frontend
│   ├── src/
│   │   ├── App.jsx                  # Main chat layout + KPI state
│   │   ├── components/
│   │   │   ├── ChatMessage.jsx      # Message renderer + pipeline result
│   │   │   ├── GateTimeline.jsx     # Collapsible gate pass/fail list
│   │   │   ├── KpiBar.jsx           # BERTScore ring + composite bar
│   │   │   └── SoftwareCard.jsx     # CVE / Patch / Version / Breaking cards
│   │   └── styles/app.css           # Dark theme design system
│   └── index.html
│
├── server/                          # Express backend
│   ├── index.js                     # Server entry point
│   ├── routes/
│   │   └── query.js                 # POST /api/query
│   ├── services/
│   │   ├── abstainPipeline.js       # 5-gate pipeline orchestrator
│   │   ├── nlpProcessor.js          # Entity extraction + intent detection
│   │   ├── nemoGuardrails.js        # Gate 1 LLM topic classifier
│   │   ├── nemoRetriever.js         # NeMo Reranker (cross-encoder)
│   │   ├── tritonClient.js          # Triton → NIM embedding fallback
│   │   ├── hallucination.js         # BERTScore grounding detector
│   │   ├── apiClient.js             # releasetrain.io HTTP client + LRU cache
│   │   ├── componentNames.js        # 6,400+ component name index
│   │   └── similarityMatcher.js     # Jaro-Winkler + Dice fuzzy matching
│   └── utils/
│       ├── lruCache.js              # O(1) LRU cache implementation
│       └── similarity.js            # Jaro-Winkler + Levenshtein algorithms
│
├── .env.example                     # Environment variable template
├── package.json
└── render.yaml                      # Render.com deployment config
```

---

*Data source: [releasetrain.io](https://releasetrain.io) — software release tracking API*  
*NVIDIA NeMo stack: Guardrails · NV-EmbedQA · NV-RerankQA-Mistral · Triton Inference Server*
