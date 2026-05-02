# Release Master — LLM Abstain System

> **Software Update Q&A System** powered by a 5-gate abstain pipeline, NVIDIA NeMo stack, RAG response generation, and real-time hallucination detection.

The system answers questions about software releases, CVEs, patches, and breaking changes by querying **releasetrain.io**. Instead of guessing, it says **"I don't know"** whenever it cannot find verified data — a deliberate design decision called *controlled abstention*.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture Overview](#architecture-overview)
3. [The 5-Gate Abstain Pipeline](#the-5-gate-abstain-pipeline)
4. [NVIDIA NeMo Components](#nvidia-nemo-components)
5. [RAG — LLM Response Generation](#rag--llm-response-generation)
6. [Hallucination Detection — BERTScore](#hallucination-detection--bertscore)
7. [Composite Confidence Score](#composite-confidence-score)
8. [KPI Dashboard](#kpi-dashboard)
9. [LRU Cache Strategy](#lru-cache-strategy)
10. [Example Queries — Confident Answers](#example-queries--confident-answers)
11. [Example Queries — I Don't Know](#example-queries--i-dont-know)
12. [Screenshots](#screenshots)
13. [Setup & Run](#setup--run)
14. [Environment Variables](#environment-variables)
15. [Project Structure](#project-structure)

---

## What It Does

| User asks | System does |
|---|---|
| "What CVEs affect Firefox?" | Queries releasetrain.io → 5-gate validation → LLM narrates → structured card |
| "Latest Node.js patches?" | Finds patch releases → LLM summarises → returns verified card |
| "How do I cook pasta?" | Gate 1 detects off-topic → returns "I don't know" immediately |
| "Latest updates for FigJam?" | Gate 3 finds no data → returns "I don't know" |

The core principle: **never hallucinate**. The LLM only narrates data that was already retrieved and verified. Every answer is traceable back to a releasetrain.io source URL.

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
                    All gates pass
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  RAG — LLM Response Generation                          │
│  meta/llama-3.1-70b-instruct via NVIDIA NIM             │
│  Context: verified structured data + top source entries │
│  Output: 2–4 sentence natural language summary          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Hallucination Detection                                │
│  BERTScore — LLM text vs raw source entries             │
│  Grounded (P) · Coverage (R) · F1 · Risk level         │
└────────────────────────┬────────────────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        CONFIDENT                 ABSTAIN
    LLM summary text          "I don't know"
    + Structured card         reason returned
    + BERTScore KPI
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

If 0 results are returned, the system automatically tries **canonical name resolution**:
- `"node.js"` → retries as `"Node"`
- `"vs code"` → retries as `"vscode"`
- `"k8s"` → retries as `"kubernetes"`

**Fail example:**
```
Query: "Latest updates for Obsidian?"
→ "obsidian" → 0 results, no canonical alias found
→ ABSTAIN: '"obsidian" was not found in releasetrain.io'
```

---

### Gate 4 — Software Validity

Confirms entries array from Gate 3 is non-empty after processing. Always passes when Gate 3 passes.

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

---

### Composite Score Gate

After all 5 gates pass, the weighted average must exceed **0.60**.

**Formula:**
```
Composite = (Gate1_score × 0.25)
          + (Gate2_score × 0.25)
          + (Gate3_score × 0.25)
          + (Gate5_score × 0.25)
```

**Example — "Any GitHub breaking updates April 2026?"  →  87%**
```
Gate 1 (topic relevance)    = 0.95 × 0.25 = 0.2375
Gate 2 (NLP confidence)     = 0.92 × 0.25 = 0.2300
Gate 3 (software lookup)    = 1.00 × 0.25 = 0.2500
Gate 5 (response quality)   = 0.75 × 0.25 = 0.1875
                                     ──────────────
Composite                           = 0.8750  →  87%
Decision: CONFIDENT  (0.8750 ≥ 0.60 ✓)
```

Date filter resolved: `"April 2026"` → `20260401` to `20260430`. Only GitHub entries with breaking types within that date range were shown.

**Example — composite fails even when all individual gates scrape past:**
```
Gate 1 = 0.61 × 0.25 = 0.1525
Gate 2 = 0.71 × 0.25 = 0.1775
Gate 3 = 1.00 × 0.25 = 0.2500
Gate 5 = 0.31 × 0.25 = 0.0775
                ──────────────
Composite      = 0.5575  →  55.8%
Decision: ABSTAIN  (0.5575 < 0.60 ✗)
```

---

## NVIDIA NeMo Components

### 1. NeMo Guardrails — Gate 1 LLM Classifier

**File:** `server/services/nemoGuardrails.js`  
**Model:** `meta/llama-3.1-8b-instruct` (via NVIDIA NIM)

Sends the user query to the LLM with a NeMo Guardrails-style system prompt that defines the allowed topic space in natural language. Returns:

```json
{ "relevant": true, "confidence": 0.95, "reason": "asks about software CVEs" }
```

**Fallback:** regex `positiveScore` / `isUpdateRelated` when no API key or LLM failure.  
**LRU Cache:** 200 entries, 30-min TTL.

---

### 2. Triton Client — Local Embedding Server

**File:** `server/services/tritonClient.js`

| Priority | Backend | Endpoint |
|---|---|---|
| 1 | Local Triton server | `POST {TRITON_URL}/v2/models/{model}/infer` |
| 2 | NVIDIA NIM cloud | `POST https://integrate.api.nvidia.com/v1/embeddings` |

Health check is cached per session. Falls back to NIM automatically on Triton failure.

---

### 3. NeMo Reranker — Cross-Encoder Ranking

**File:** `server/services/nemoRetriever.js`  
**Model:** `nvidia/nv-rerankqa-mistral-4b-v3`

Cross-encoder reads query + each passage together → logit score → re-sorted best-first before Gate 5.

**Fallback chain:**
```
Reranker API → Triton embeddings + cosine → NIM embeddings + cosine → original order
```

---

## RAG — LLM Response Generation

**File:** `server/services/llmResponse.js`  
**Model:** `meta/llama-3.1-70b-instruct` (configurable via `LLM_RESPONSE_MODEL`)

### How it works

After all 5 gates pass and structured data is extracted, the LLM receives a grounded context and generates a 2–4 sentence natural language summary.

```
User query  +  Structured data  +  Top 3 raw source entries
                        ↓
          meta/llama-3.1-70b-instruct
                        ↓
        "Firefox is affected by 50 CVEs, with the most
         recent being CVE-2026-2447 (SECURITY severity)..."
```

### System prompt rules given to the LLM

- Use **only** the verified data provided — never add external knowledge
- State version numbers, dates, and CVE IDs exactly as given
- 2–4 sentences maximum
- No filler phrases ("As an AI…", "Based on the data…")

### Context passed to LLM

```
=== Verified data from releasetrain.io ===
Software: Firefox
Top CVE ID: CVE-2026-2447
Severity: SECURITY
Total CVEs found: 50
...

=== Raw source entries (top 3) ===
1. v128.0 | stable | 20260410 | CVE | Security fix for...
2. v127.0 | stable | 20260301 | ...
3. v126.0 | patch  | 20260215 | ...
```

### Fallback behaviour

If the LLM call fails (timeout, API error), the system still returns the structured card — the LLM summary is additive, not a dependency.

**LRU Cache:** 100 entries, 30-min TTL.

---

## Hallucination Detection — BERTScore

**File:** `server/services/hallucination.js`

### What changed in RAG mode

In RAG mode the **LLM-generated text** is the hypothesis (not structured fields). This is the real hallucination check — did the LLM stick to what the source data actually says?

```
Before RAG:  hypothesis = extracted structured fields
After RAG:   hypothesis = LLM response sentences  ← more meaningful
```

### The Three Numbers

| Metric | Name in UI | Meaning |
|---|---|---|
| Precision (P) | **Grounded** | % of LLM response words found in source data |
| Recall (R) | **Coverage** | % of source data words that appear in the LLM response |
| F1 | **BERTScore** | Harmonic mean of P and R |

### Real example — "Any GitHub breaking updates April 2026?"

```
LLM text: "GitHub had 3 Breaking Update releases in April 2026,
           including v3.12.1 on 2026-04-10..."

Grounded (P) = LLM words found in source / total LLM words = 77%
Coverage (R) = source words found in LLM / total source words = 56%

BERTScore F1 = 2 × 0.77 × 0.56 / (0.77 + 0.56)
             = 0.862 / 1.33
             = 65.1%

Risk: Grounded 77% → Medium Risk (just below 78% Low threshold)
```

### Risk thresholds (based on Grounded / Precision)

```
Grounded ≥ 78%  →  Low Risk     (green)
Grounded ≥ 55%  →  Medium Risk  (amber)
Grounded  < 55%  →  High Risk    (red)
```

Risk is based on **Grounded (P)**, not F1. F1 is dragged down by Coverage which is always low for summaries.

### Embedding vs Lexical

| Mode | When | Method |
|---|---|---|
| `embedding` | NVIDIA NIM or Triton available | Pairwise cosine similarity between BERT vectors |
| `lexical` | No embeddings available | Token overlap (ROUGE-1 style) |

---

## Composite Confidence Score

**Progress bar** in the KPI dashboard — width = composite %, animated on each update.

### Thresholds

```
≥ 80%   High confidence   (teal)
60–79%  Acceptable        (teal)
< 60%   Too low → ABSTAIN (coral — bar still shown)
```

### Threshold markers on the bar

Two vertical markers at **60%** and **80%** show where the acceptable zone starts and where high confidence begins.

---

## KPI Dashboard

Center-top card strip — appears after the first query, updates on every subsequent result.

```
┌──────────────────────────────┐  ┌──────────────────────────┐
│  HALLUCINATION DETECTION     │  │  COMPOSITE CONFIDENCE    │
│                              │  │                          │
│   ╭────╮  BERTScore 65.1%   │  │   87%                    │
│   │ 65 │  Grounded  77%     │  │   ████████░░             │
│   ╰────╯  Coverage  56%     │  │   60%  80%               │
│           ● Medium Risk     │  │   ● Confident            │
└──────────────────────────────┘  └──────────────────────────┘
```

| Element | Description |
|---|---|
| Ring gauge | Animated SVG — fills to BERTScore F1 % |
| Grounded | % of LLM response backed by source data |
| Coverage | % of source represented in response |
| Risk badge | Green / Amber / Red based on Grounded score |
| Progress bar | Width = composite score, animated transition |
| Threshold markers | Lines at 60% and 80% |
| Decision pill | Confident / Suggestion / Abstained |

On **Abstain**: BERTScore shows 0%, risk = High.

---

## LRU Cache Strategy

**File:** `server/utils/lruCache.js`

O(1) LRU using JavaScript `Map` insertion-order. `get()` promotes to tail (MRU); head is always evicted first.

### Cache instances

| Cache | Capacity | TTL | Covers |
|---|---|---|---|
| `apiClient` searchCache | 100 | 5 min | releasetrain.io API responses |
| `nemoGuardrails` classifyCache | 200 | 30 min | LLM topic classification results |
| `llmResponse` responseCache | 100 | 30 min | LLM-generated response text |

### Eviction example

```
Capacity = 3, queries: firefox → android → chrome → node
  [firefox, android, chrome]  ← full
  node arrives → evict firefox (LRU head)
  [android, chrome, node]
  android accessed → promote to tail
  [chrome, node, android]
```

---

## Example Queries — Confident Answers

```bash
# Latest version
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the latest version of Firefox?"}'

# CVE security vulnerabilities
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What CVEs affect Android?"}'

# Recent patches
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

# Month-specific breaking changes
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Any GitHub breaking updates April 2026?"}'

# OS security updates
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Latest iOS security updates"}'
```

---

## Example Queries — I Don't Know

### Gate 1 fails — off-topic

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I deploy to AWS?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What programming language should I learn?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain what a REST API is"}'
```

### Gate 2 fails — no entity extracted

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the latest security patches this week?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Any breaking changes released yesterday?"}'
```

### Gate 3 fails — not in releasetrain.io

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Latest updates for Obsidian?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Any CVEs for Cursor IDE?"}'
```

### Gate 5 fails — data type mismatch

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Critical failures in Notepad?"}'

curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"What are the CVEs for Winamp?"}'
```

### Composite score too low

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me something about git"}'
```

### Quick decision + reason only

```bash
curl -s -X POST http://localhost:3001/api/query \
  -H "Content-Type: application/json" \
  -d '{"message":"Any CVEs for Cursor IDE?"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['decision'].upper(), '—', d.get('reason',''))"
```

---

## Application Screenshots

### Dashboard — Confident Answer with LLM Summary

#### 1) LLM Summary

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/91bc92d7-876b-47dd-a957-fee205a37b67" />


#### 2) Source Evicence

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/85830592-d898-4158-b7e0-ab14e2d4b54b" />


#### 3) Source Verify Against Releasetrain — For GitHub Failure today

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/18101d35-43cf-40f9-ae3c-41e9427e3394" />


---

### KPI Bar — BERTScore + Composite Confidence

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/285c4b39-924b-4e2b-81f9-ba13a2cc2707" />


---

### Gate Timeline — All 5 Gates Passed

<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/9c5dc12f-f0a4-4103-8642-e491c19c3b50" />


---

### I Don't Know — Abstain Response


#### 1) I Don't Know Response

<img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/5bdfa46a-c26e-409f-b3f6-e650f2b78909" />

#### 2) BERTScore + Composite Confidence

<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/8ac33b43-134d-481f-9a62-ec670f5c7a42" />


---

### CVE Card



---

### Breaking Update Card with Date Filter



---

## Setup & Run

### Prerequisites

- Node.js ≥ 18
- NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) — enables NeMo Guardrails, Reranker, LLM responses, and NIM embeddings
- (Optional) Local Triton Inference Server for GPU embedding inference

### Install & Start

```bash
git clone https://github.com/SE4CPS/Research-on-Software-Updates.git
cd Research-on-Software-Updates/llm-abstain

npm install
cp .env.example .env
# Edit .env — add NVIDIA_API_KEY

npm run dev
```

**Backend** → `http://localhost:3001`  
**Frontend** → `http://localhost:5173`

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
| `NVIDIA_API_KEY` | *(blank)* | Enables all NVIDIA features |
| `GUARDRAILS_MODEL` | `meta/llama-3.1-8b-instruct` | LLM for Gate 1 topic classification |
| `LLM_RESPONSE_MODEL` | `meta/llama-3.1-70b-instruct` | LLM for RAG response generation |
| `TRITON_URL` | `http://localhost:8000` | Local Triton Inference Server |
| `TRITON_MODEL` | `nv-embedqa-e5-v5` | Model deployed in Triton |

All NVIDIA features degrade gracefully — app runs in offline/regex mode without `NVIDIA_API_KEY`.

---

## Project Structure

```
llm-abstain/
├── client/                          # React / Vite frontend
│   ├── src/
│   │   ├── App.jsx                  # Chat layout, KPI state, LLM result handling
│   │   ├── components/
│   │   │   ├── ChatMessage.jsx      # LLM summary block + pipeline result renderer
│   │   │   ├── GateTimeline.jsx     # Collapsible gate pass/fail list
│   │   │   ├── KpiBar.jsx           # BERTScore ring gauge + composite progress bar
│   │   │   └── SoftwareCard.jsx     # CVE / Patch / Version / Breaking data cards
│   │   └── styles/app.css           # Dark theme design system + LLM summary styles
│   └── index.html
│
├── server/                          # Express backend
│   ├── index.js                     # Server entry point
│   ├── routes/
│   │   └── query.js                 # POST /api/query
│   ├── services/
│   │   ├── abstainPipeline.js       # 5-gate orchestrator → LLM → BERTScore
│   │   ├── nlpProcessor.js          # Entity extraction + intent + date filter
│   │   ├── nemoGuardrails.js        # Gate 1 LLM topic classifier + LRU cache
│   │   ├── nemoRetriever.js         # NeMo Reranker (cross-encoder)
│   │   ├── tritonClient.js          # Triton → NIM embedding fallback
│   │   ├── llmResponse.js           # RAG LLM response generator + LRU cache
│   │   ├── hallucination.js         # BERTScore on LLM text vs source entries
│   │   ├── apiClient.js             # releasetrain.io HTTP client + LRU cache
│   │   ├── componentNames.js        # 6,400+ component name index
│   │   └── similarityMatcher.js     # Jaro-Winkler + Dice fuzzy matching
│   └── utils/
│       ├── lruCache.js              # O(1) LRU cache (Map insertion-order)
│       └── similarity.js            # Jaro-Winkler + Levenshtein algorithms
│
├── .env.example                     # Environment variable template
├── package.json
└── render.yaml                      # Render.com deployment config
```

---

*Data source: [releasetrain.io](https://releasetrain.io) — software release tracking API*  
*NVIDIA NeMo stack: Guardrails · NV-EmbedQA · NV-RerankQA-Mistral · Triton Inference Server · Llama-3.1-70B*
