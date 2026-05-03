// ============================================
// FILE: server/services/agent.js
// Tool-calling agent — LLM decides which search
// tools to invoke before Gate 3 runs.
// ============================================

const axios   = require("axios");
const LRUCache = require("../utils/lruCache");
const { fetchVersionSearch } = require("./apiClient");

const ENABLED  = !!process.env.NVIDIA_API_KEY;
const NIM_BASE = "https://integrate.api.nvidia.com/v1";
const MODEL    = process.env.LLM_RESPONSE_MODEL || "meta/llama-3.1-70b-instruct";

const agentCache = new LRUCache(100, 15 * 60 * 1000);

// ── Tool Definitions ──────────────────────────────────────────────────────────

const TOOLS = [
  {
    type: "function",
    function: {
      name: "search_releases",
      description:
        "Search releasetrain.io for general software release versions, changelogs, and update history. Use this for questions about latest versions, release dates, or update timelines.",
      parameters: {
        type: "object",
        properties: {
          entity: { type: "string", description: "Software or OS name (e.g. 'Firefox', 'Node.js', 'Android')" },
          reason: { type: "string", description: "Brief reason why you chose this tool" },
        },
        required: ["entity"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_cves",
      description:
        "Search for CVEs, security advisories, and vulnerability disclosures for a software. Use this when the query involves security issues, exploits, CVE IDs, or vulnerability patches.",
      parameters: {
        type: "object",
        properties: {
          entity: { type: "string", description: "Software or OS name to search CVEs for" },
          reason: { type: "string", description: "Brief reason why you chose this tool" },
        },
        required: ["entity"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_breaking_changes",
      description:
        "Search for breaking updates, critical failures, compatibility issues, dependency failures, and incidents for a software. Use this for queries about failures, breakages, regressions, or critical issues.",
      parameters: {
        type: "object",
        properties: {
          entity: { type: "string", description: "Software or OS name to search breaking changes for" },
          reason: { type: "string", description: "Brief reason why you chose this tool" },
        },
        required: ["entity"],
      },
    },
  },
];

// ── Tool Executor ─────────────────────────────────────────────────────────────
// All three tools call the same releasetrain.io endpoint.
// Dedup prevents multiple identical HTTP requests for the same entity.

async function executeTool(toolName, args) {
  const entity = (args.entity || "").trim();
  const result = await fetchVersionSearch(entity);
  return { toolName, entity, reason: args.reason || null, result };
}

// Merge entries, deduplicating by version+date composite key
function mergeEntries(existing, incoming) {
  const seen = new Set(
    existing.map((e) => `${e.versionNumber}::${e.versionReleaseDate}`)
  );
  return [
    ...existing,
    ...incoming.filter((e) => !seen.has(`${e.versionNumber}::${e.versionReleaseDate}`)),
  ];
}

// ── Main Agent Entry Point ────────────────────────────────────────────────────

async function runAgent(entityName, userQuery, queryType) {
  if (!ENABLED) {
    console.log("[Agent] Skipped — no NVIDIA_API_KEY");
    return null;
  }

  const cacheKey = `${entityName.toLowerCase()}::${queryType}`;
  const cached = agentCache.get(cacheKey);
  if (cached) {
    console.log(`[Agent] Cache hit for "${cacheKey}"`);
    return { ...cached, agentDecision: { ...cached.agentDecision, cached: true } };
  }

  try {
    const res = await axios.post(
      `${NIM_BASE}/chat/completions`,
      {
        model: MODEL,
        messages: [
          {
            role: "system",
            content:
              "You are a software release intelligence agent. Given a user query, decide which search tool(s) to call to answer it. Call one or more tools. Do not respond with text — only make tool calls.",
          },
          {
            role: "user",
            content: `Query: "${userQuery}"\nSoftware entity: "${entityName}"\nIntent type: ${queryType}\n\nSelect the appropriate tool(s).`,
          },
        ],
        tools: TOOLS,
        tool_choice: "auto",
        temperature: 0,
        max_tokens: 300,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.NVIDIA_API_KEY}`,
          "Content-Type": "application/json",
        },
        timeout: 15000,
      }
    );

    const message = res.data?.choices?.[0]?.message;
    const toolCalls = message?.tool_calls || [];

    if (toolCalls.length === 0) {
      console.log("[Agent] No tool calls returned — Gate 3 will run normally");
      return null;
    }

    const parsed = toolCalls.map((tc) => ({
      toolName: tc.function.name,
      args: JSON.parse(tc.function.arguments || "{}"),
    }));

    console.log(`[Agent] Tools chosen: ${parsed.map((t) => t.toolName).join(", ")}`);

    // Deduplicate by entity — one HTTP call per unique entity string
    const seen = new Set();
    let mergedEntries = [];
    const executedTools = [];

    for (const tc of parsed) {
      const key = (tc.args.entity || entityName).trim().toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        const out = await executeTool(tc.toolName, {
          ...tc.args,
          entity: tc.args.entity || entityName,
        });
        if (out.result.success && out.result.data?.length > 0) {
          mergedEntries = mergeEntries(mergedEntries, out.result.data);
        }
        executedTools.push({
          tool:   tc.toolName,
          entity: tc.args.entity || entityName,
          reason: tc.args.reason || null,
        });
      } else {
        // Same entity, different tool label — log without extra fetch
        executedTools.push({
          tool:    tc.toolName,
          entity:  tc.args.entity || entityName,
          reason:  tc.args.reason || null,
          deduped: true,
        });
      }
    }

    const agentDecision = {
      toolsSelected: executedTools,
      cached: false,
    };

    const result = { entries: mergedEntries, agentDecision };
    if (mergedEntries.length > 0) agentCache.set(cacheKey, result);
    return result;

  } catch (err) {
    console.error(`[Agent] Error: ${err.message} — falling back to Gate 3`);
    return null;
  }
}

module.exports = { runAgent, isEnabled: () => ENABLED };
