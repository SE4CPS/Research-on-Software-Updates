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

const axios = require("axios");

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

RELEVANT examples:
  "What CVEs affect Firefox 128?"
  "Latest Node.js LTS patches"
  "Is Android 15 stable?"
  "Breaking changes in React 19"
  "Docker critical failure updates last week"

NOT RELEVANT examples:
  "How do I make pasta?"
  "What is the capital of France?"
  "Tell me a joke"
  "Who won the game last night?"

Be strict — only mark relevant=true when the query clearly asks about software or OS release/security/update information.`;

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

    return {
      score:      confidence,
      isRelevant: relevant,
      method:     "llm",
      model:      GUARD_MODEL,
      reason:     parsed.reason || (relevant ? "on-topic" : "off-topic"),
    };
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
