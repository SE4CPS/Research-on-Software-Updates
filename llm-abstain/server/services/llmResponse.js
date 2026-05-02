// ============================================
// FILE: server/services/llmResponse.js
// RAG response generator — LLM narrates verified releasetrain.io data.
//
// The LLM receives ONLY the structured data + top source entries as
// context. It cannot add facts not present in those entries.
// Falls back gracefully (returns null) when API key is absent or call fails.
// ============================================

const axios    = require("axios");
const LRUCache = require("../utils/lruCache");

const NIM_BASE       = "https://integrate.api.nvidia.com/v1";
const RESPONSE_MODEL = process.env.LLM_RESPONSE_MODEL || "meta/llama-3.1-70b-instruct";
const ENABLED        = !!process.env.NVIDIA_API_KEY;

// LRU: 100 queries, 30-min TTL — same question won't hit the LLM twice
const responseCache = new LRUCache(100, 30 * 60 * 1000);

// ── System prompt ────────────────────────────────
const SYSTEM_PROMPT = `You are Release Master, a software release and security expert.
Answer the user's question using ONLY the verified data provided below from releasetrain.io.

Rules:
- 2–4 sentences maximum. Be concise and factual.
- State version numbers, dates, and CVE IDs exactly as given — never invent them.
- Do NOT add any information not present in the provided data.
- CVE queries: mention total count and the most recent CVE ID and severity.
- Patch queries: state the latest patch version, release date, and channel.
- Breaking change queries: name the breaking type and number of affected releases.
- Version queries: state the exact latest version and release date.
- Never say "I" or "As an AI". Speak in plain present-tense facts.`;

// ── Public API ──────────────────────────────────

/**
 * Generate a natural-language summary of verified software release data.
 *
 * @param {string}   userQuery      - Original user question
 * @param {object}   structuredData - Gate 5 extracted fields
 * @param {object[]} sourceEntries  - Raw releasetrain.io entries (Gate 3 pool)
 * @param {string}   queryType      - cve | patch | version | breaking | critical | general
 * @returns {Promise<{ text: string, model: string } | null>}
 */
async function generateLLMResponse(userQuery, structuredData, sourceEntries, queryType) {
  if (!ENABLED || !structuredData) return null;

  const cacheKey = userQuery.toLowerCase().trim();
  const cached   = responseCache.get(cacheKey);
  if (cached) {
    console.log(`[LLM] Cache hit for "${cacheKey.slice(0, 40)}" | ${responseCache.stats().hitRate} hit-rate`);
    return cached;
  }

  try {
    const context  = buildContext(structuredData, sourceEntries, queryType);
    const response = await axios.post(
      `${NIM_BASE}/chat/completions`,
      {
        model: RESPONSE_MODEL,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user",   content: `Question: ${userQuery}\n\n${context}` },
        ],
        temperature: 0.1,
        max_tokens:  220,
      },
      {
        headers: {
          Authorization:  `Bearer ${process.env.NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
        },
        timeout: 15000,
      }
    );

    const text = response.data.choices?.[0]?.message?.content?.trim();
    if (!text) return null;

    const result = { text, model: RESPONSE_MODEL };
    responseCache.set(cacheKey, result);
    console.log(`[LLM] Response (${text.length} chars) for "${userQuery.slice(0, 40)}" via ${RESPONSE_MODEL}`);
    return result;
  } catch (err) {
    console.warn(`[LLM] Response generation failed (${err.message}) — showing structured card only`);
    return null;
  }
}

// ── Context builder ──────────────────────────────
// Converts structured data + raw entries into a grounded prompt context.

function buildContext(d, sourceEntries, queryType) {
  const lines = ["=== Verified data from releasetrain.io ==="];

  if (d.software)       lines.push(`Software: ${d.software}`);
  if (d.version)        lines.push(`Latest version: ${d.version}`);
  if (d.releaseDate)    lines.push(`Release date: ${d.releaseDate}`);
  if (d.releaseChannel) lines.push(`Channel: ${d.releaseChannel}`);
  if (d.cveId)          lines.push(`Top CVE ID: ${d.cveId}`);
  if (d.severity)       lines.push(`Severity: ${d.severity}`);
  if (d.totalCves)      lines.push(`Total CVEs found: ${d.totalCves}`);
  if (d.totalCritical)  lines.push(`Total matching entries: ${d.totalCritical}`);
  if (d.breakingLabel)  lines.push(`Breaking type: ${d.breakingLabel}`);
  if (d.description)    lines.push(`Release notes: ${d.description.slice(0, 200)}`);
  if (d.dateFilter)     lines.push(`Date filter applied: ${d.dateLabel} (${d.dateFilter})`);

  const items = d.recentCves || d.recentPatches || d.recentVersions || d.entries || [];
  if (items.length > 0) {
    lines.push("\nRecent entries:");
    items.slice(0, 4).forEach((item, i) => {
      const id    = item.cveId  || item.version || "";
      const dt    = item.date   || "";
      const notes = (item.notes || "").slice(0, 120);
      lines.push(`  ${i + 1}. ${[id, dt, notes].filter(Boolean).join(" | ")}`);
    });
  }

  if (sourceEntries && sourceEntries.length > 0) {
    lines.push("\n=== Raw source entries (top 3) ===");
    sourceEntries.slice(0, 3).forEach((e, i) => {
      const parts = [
        e.versionNumber      ? `v${e.versionNumber}`             : "",
        e.versionReleaseChannel || "",
        e.versionReleaseDate || "",
        e.isCve              ? "CVE"                             : "",
        (e.versionReleaseNotes || "").slice(0, 120),
      ].filter(Boolean);
      lines.push(`  ${i + 1}. ${parts.join(" | ")}`);
    });
  }

  return lines.join("\n");
}

module.exports = { generateLLMResponse, isEnabled: () => ENABLED, RESPONSE_MODEL };
