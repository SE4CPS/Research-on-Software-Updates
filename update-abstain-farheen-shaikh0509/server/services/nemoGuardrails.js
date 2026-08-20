// ============================================
// FILE: server/services/nemoGuardrails.js
// NeMo Guardrails — LLM-based Gate 1 topic classifier
// Replaces regex positiveScore with a zero-shot LLM classifier
// that decides if the query is about software/OS updates/security.
//
// Falls back to regex-derived scores when:
//   • NVIDIA_API_KEY is not set
//   • The LLM call times out or errors
// ============================================

const axios    = require("axios");
const LRUCache = require("../utils/lruCache");

// LRU cache for topic classifications — avoids calling the LLM twice for the
// same query. 200 entries, 30-min TTL (topic relevance doesn't change quickly).
const classifyCache = new LRUCache(200, 30 * 60 * 1000);

const NIM_BASE     = "https://integrate.api.nvidia.com/v1";
const GUARD_MODEL  = process.env.GUARDRAILS_MODEL || "meta/llama-3.1-8b-instruct";
const ENABLED      = !!process.env.NVIDIA_API_KEY;
const TIMEOUT_MS   = 8000;

// NeMo Guardrails-style system prompt: defines allowed topic space in natural language.
// The LLM acts as a binary classifier — relevant (software release / security domain)
// vs. off-topic (general knowledge, small-talk, unrelated tech questions).
const SYSTEM_PROMPT = `You are a topic guardrail for a software-release tracking assistant.

Classify the user query as RELEVANT or NOT RELEVANT to the following domain:
  - Software, OS, library, or package updates / releases / changelogs
  - Security advisories, CVEs, patches, hotfixes
  - Breaking changes, compatibility issues, dependency failures
  - Version information for any software or operating system

Respond ONLY with a JSON object on one line:
{"relevant": <true|false>, "confidence": <0.0-1.0>, "reason": "<short phrase>"}

RELEVANT examples (ALL of these must be marked relevant=true):
  "What CVEs affect Firefox 128?"
  "Latest Node.js LTS patches"
  "Is Android 15 stable?"
  "Breaking changes in React 19"
  "What is the latest version of GitHub?"
  "What is the newest release of Linux?"
  "GitHub patch releases this month"
  "Critical failure of GitHub this week"
  "Critical failure of GitHub this month"
  "Breaking Update of GitHub this month"
  "Network issues in Docker last month"
  "NETWORK ISSUE GitHub last month"
  "Configuration errors in Kubernetes today"
  "GitHub Configuration Errors this month"
  "Logging and Monitoring Failures GitHub this week"
  "GitHub Logging & Monitoring Failures last month"
  "Data integrity issue GitHub today"
  "LIMITED FUNCTIONALITY GitHub this month"
  "Dependency failures in Node this week"
  "Resource exhaustion in Redis yesterday"

NOT RELEVANT examples:
  "How do I make pasta?"
  "What is the capital of France?"
  "Tell me a joke"
  "Who won the game last night?"

Any query mentioning a software name (GitHub, Docker, Linux, Node, etc.) alongside a time period (today, this month, last week, etc.) and any technical term (failure, error, issue, update, patch, CVE, network, logging, configuration, data, functionality) is RELEVANT.`;

/**
 * LLM-based topic classification for Gate 1.
 *
 * @param {string} userQuery       - Raw user message
 * @param {object} fallbackScores  - { positiveScore, isUpdateRelated } from NLP regex
 * @returns {Promise<{
 *   score: number,
 *   isRelevant: boolean,
 *   method: "llm"|"regex",
 *   model?: string,
 *   reason: string
 * }>}
 */
async function classifyTopic(userQuery, fallbackScores = {}) {
  if (!ENABLED) {
    return buildFallback(fallbackScores, "No NVIDIA_API_KEY set");
  }

  // LRU cache check — skip LLM call if we've already classified this query
  const cacheKey = userQuery.toLowerCase().trim();
  const cached   = classifyCache.get(cacheKey);
  if (cached) {
    console.log(`[Guardrails] LRU hit "${cacheKey.slice(0, 40)}" | ${classifyCache.stats().hitRate} hit-rate`);
    return cached;
  }

  try {
    const response = await axios.post(
      `${NIM_BASE}/chat/completions`,
      {
        model: GUARD_MODEL,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user",   content: userQuery },
        ],
        temperature: 0.0,
        max_tokens: 120,
        response_format: { type: "json_object" },
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
        },
        timeout: TIMEOUT_MS,
      }
    );

    const raw    = response.data.choices?.[0]?.message?.content || "{}";
    const parsed = JSON.parse(raw);

    const relevant   = parsed.relevant === true;
    const confidence = typeof parsed.confidence === "number"
      ? Math.min(1, Math.max(0, parsed.confidence))
      : (relevant ? 0.85 : 0.1);

    console.log(
      `[Guardrails] LLM: relevant=${relevant} conf=${confidence.toFixed(3)} reason="${parsed.reason}" model=${GUARD_MODEL}`
    );

    // Override: if NLP already flagged this as clearly update-related and
    // positiveScore is high, don't let a low-confidence LLM rejection win.
    const nlpIsConfident =
      fallbackScores.isUpdateRelated && (fallbackScores.positiveScore || 0) >= 0.7;
    const llmLowConfidence = !relevant && confidence < 0.85;
    if (nlpIsConfident && llmLowConfidence) {
      console.log(`[Guardrails] NLP override — LLM rejected with low confidence (${confidence.toFixed(3)}) but NLP signals update-related query`);
      const override = buildFallback(fallbackScores, `llm-override: conf=${confidence.toFixed(3)}`);
      classifyCache.set(cacheKey, override);
      return override;
    }

    const result = {
      score:      confidence,
      isRelevant: relevant,
      method:     "llm",
      model:      GUARD_MODEL,
      reason:     parsed.reason || (relevant ? "on-topic" : "off-topic"),
    };
    classifyCache.set(cacheKey, result);
    return result;
  } catch (err) {
    console.warn(`[Guardrails] LLM call failed (${err.message}) — using regex fallback`);
    return buildFallback(fallbackScores, `LLM error: ${err.message}`);
  }
}

// ── Regex fallback ───────────────────────────────

function buildFallback({ positiveScore = 0, isUpdateRelated = false } = {}, note = "") {
  const score      = Math.max(positiveScore, isUpdateRelated ? 0.7 : 0);
  const isRelevant = isUpdateRelated || positiveScore >= 0.6;
  return {
    score,
    isRelevant,
    method: "regex",
    reason: `positiveScore=${positiveScore.toFixed ? positiveScore.toFixed(3) : positiveScore}, isUpdateRelated=${isUpdateRelated}${note ? ` (${note})` : ""}`,
  };
}

module.exports = { classifyTopic, isEnabled: () => ENABLED, GUARD_MODEL };
