import re
from typing import Any, Dict, List

RELEASETRAIN_BASE = "https://releasetrain.io/api/component/"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
REDDIT_QUESTIONS_API = "https://releasetrain.io/api/reddit/query/questions"

# Groq has per-request token limits; cap data volume for the prioritizer prompt.
MAX_RELEASETRAIN_ITEMS_TOTAL = 80
MAX_CVES_FOR_LLM = 35
MAX_RELEASE_NOTES_FOR_LLM = 35
MAX_NEWS_FOR_LLM = 18
MAX_FIELD_CHARS = 500
MAX_REDDIT_QUESTIONS_FETCH = 20
MAX_REDDIT_QUESTIONS_ANSWER = 5

CVE_ID_PATTERN = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

# Shipped with API for UI: maps LangGraph nodes to agentic roles (plan → tools → reason).
AGENT_PIPELINE_META: List[Dict[str, Any]] = [
    {
        "id": "orchestrate",
        "role": "llm_agent",
        "node": "orchestrator",
        "label": "Orchestrator",
        "detail": "Reads the user goal and plans which data sources and search terms to use.",
    },
    {
        "id": "gather",
        "role": "parallel_tools",
        "node": "release_train_fetch + google_news_fetch",
        "label": "Tool calls",
        "detail": "Fetches ReleaseTrain (CVEs + releases) and Google News RSS in parallel—no LLM.",
    },
    {
        "id": "merge",
        "role": "state_merge",
        "node": "merge",
        "label": "Context assembly",
        "detail": "Combines tool outputs into one structured context for reasoning.",
    },
    {
        "id": "prioritize",
        "role": "llm_agent",
        "node": "prioritize + evidence_repair + deterministic_match",
        "label": "Prioritizer",
        "detail": "Ranks updates with evidence ids; missing refs get a batched repair LLM then rule-based matching.",
    },
    {
        "id": "reddit_signal",
        "role": "parallel_tools",
        "node": "reddit_questions_fetch",
        "label": "Reddit signals",
        "detail": "Fetches current Reddit questions as community context — fed into the prioritizer to inform its answer.",
    },
    {
        "id": "format",
        "role": "presentation",
        "node": "format",
        "label": "Response formatter",
        "detail": "Shapes results for the chat UI, including evidence and scores.",
    },
]
