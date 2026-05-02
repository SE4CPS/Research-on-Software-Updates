// ============================================
// FILE: server/services/abstainPipeline.js
// 5-Gate Abstain Pipeline - core validation engine
// Decides: "confident" | "suggest" | "abstain"
// ============================================

const { fetchVersionSearch } = require("./apiClient");
const { rankEntriesByRelevance, isEnabled: nemoEnabled } = require("./nemoRetriever");
const { resolveCanonicalSearchName } = require("./componentNames");
const { classifyTopic } = require("./nemoGuardrails");
const { computeBERTScore } = require("./hallucination");
const { generateLLMResponse } = require("./llmResponse");

// ── Configurable Thresholds ──
const THRESHOLDS = {
  topicRelevance: 0.6,
  nlpConfidence: 0.7,
  similarityHigh: 0.9,
  similarityPartial: 0.6,
  responseQuality: 0.3,
  compositeMinimum: 0.6,   // final gate: composite score must exceed this to give an answer
};

// ── Composite Score Weights ──
// Gate 3 is now a binary found/not-found (score always 1.0 when passing),
// so more weight goes to NLP confidence and data quality.
const WEIGHTS = {
  topicRelevance: 0.25,
  nlpConfidence:  0.25,
  similarity:     0.25,   // Gate 3 — software lookup
  responseQuality: 0.25,  // Gate 5 — data completeness
};

// ── Main Pipeline ──

async function runPipeline(processedPrompt) {
  const gates = [];
  const queryType    = processedPrompt.metadata?.queryType    || "general";
  const dateFilter   = processedPrompt.metadata?.dateFilter   || null;
  const breakingSubType = processedPrompt.metadata?.breakingSubType || null;

  console.log(`[DEBUG][Pipeline] Query       : "${processedPrompt.originalTitle}"`);
  console.log(`[DEBUG][Pipeline] QueryType   : ${queryType}${breakingSubType ? ` → "${breakingSubType}"` : ""}`);
  console.log(`[DEBUG][Pipeline] DateFilter  : ${dateFilter ? `${dateFilter.label} (${dateFilter.displayDate})` : "none"}`);
  console.log(`[DEBUG][Pipeline] Entity      : "${processedPrompt.extraction?.primaryEntity?.name}" (conf: ${processedPrompt.extraction?.primaryEntity?.confidence}`);

  // Gate 1: Topic Relevance — LLM-based classifier (NeMo Guardrails), regex fallback
  const gate1 = await checkTopicRelevance(processedPrompt);
  gates.push(gate1);
  if (gate1.result === "fail") {
    return buildResult("abstain", gates, gate1.reason, null, null, queryType);
  }

  // Gate 2: NLP Extraction Confidence
  const gate2 = checkNLPConfidence(processedPrompt);
  gates.push(gate2);
  if (gate2.result === "fail") {
    return buildResult("abstain", gates, gate2.reason, null, null, queryType);
  }

  // Gate 3: Software Lookup — searches releasetrain.io directly (same endpoint as the website)
  const gate3 = await checkSoftwareExists(processedPrompt);
  gates.push(gate3);
  if (gate3.result === "fail") {
    return buildResult("abstain", gates, gate3.reason, null, null, queryType);
  }

  // NeMo Retriever: semantically re-rank Gate 3 entries before Gate 5 filters them.
  // If NVIDIA_API_KEY is not set this is a no-op and original order is preserved.
  const rankedEntries = await rankEntriesByRelevance(
    processedPrompt.originalTitle || processedPrompt.prompt,
    gate3.entries || []
  );
  if (gate3.entries) gate3.entries = rankedEntries;

  // Gate 4: Software Validity — confirms releasetrain.io returned entries for this software
  const gate4 = {
    gate: 4,
    name: "Software validity",
    result: gate3.entries?.length > 0 ? "pass" : "fail",
    score: gate3.entries?.length > 0 ? 1.0 : 0,
    validatedName: gate3.validatedName,
    reason: gate3.entries?.length > 0 ? undefined : "No entries returned from releasetrain.io",
  };
  gates.push(gate4);
  if (gate4.result === "fail") {
    return buildResult("abstain", gates, gate4.reason, null, null, queryType);
  }

  // Gate 5: Response Quality — filters Gate 3's entries by queryType + dateFilter (no extra API call)
  const gate5 = await checkResponseQuality(gate3.validatedName, queryType, dateFilter, gate3.entries, breakingSubType);
  gates.push(gate5);
  if (gate5.result === "fail") {
    return buildResult("abstain", gates, gate5.reason, null, null, queryType);
  }

  console.log(`[DEBUG][Pipeline] Gate5 score : ${gate5.score?.toFixed(3)} | result: ${gate5.result}`);

  // Cross-validation: intent (queryType) must align with data actually found for the matched software.
  const alignment = checkIntentDataAlignment(queryType, gate5.structuredData, gate3.validatedName, breakingSubType);
  if (!alignment.valid) {
    return buildResult("abstain", gates, alignment.reason, null, null, queryType);
  }

  // Composite score gate: weighted average across all gates must exceed 60% to answer.
  // This prevents answering when several gates barely scraped past their individual thresholds.
  const compositeScore = computeComposite(gates);
  if (compositeScore < THRESHOLDS.compositeMinimum) {
    return buildResult(
      "abstain",
      gates,
      `Overall confidence ${(compositeScore * 100).toFixed(1)}% is below the minimum ${THRESHOLDS.compositeMinimum * 100}% required to give a reliable answer`,
      null,
      null,
      queryType
    );
  }

  console.log(`[DEBUG][Pipeline] Composite   : ${(compositeScore * 100).toFixed(1)}% | decision: confident`);

  // LLM: generate a natural-language summary grounded in the verified source data (RAG)
  const llmResult = await generateLLMResponse(
    processedPrompt.originalTitle || processedPrompt.prompt,
    gate5.structuredData,
    gate3.entries || [],
    queryType
  );

  // BERTScore: when LLM text exists, check IT against source (real hallucination check)
  const bertscore = await computeBERTScore(
    gate5.structuredData,
    gate3.entries || [],
    llmResult?.text || null
  );

  const result = buildResult("confident", gates, null, null, gate5.structuredData, queryType);
  if (llmResult) result.llmResponse = llmResult;
  if (bertscore)  result.bertscore  = bertscore;
  return result;
}

// ── Gate Implementations ──

async function checkTopicRelevance(prompt) {
  const { positiveScore, isUpdateRelated } = prompt.metadata;

  // NeMo Guardrails LLM classifier — falls back to regex scores automatically
  const classification = await classifyTopic(
    prompt.originalTitle || prompt.prompt,
    { positiveScore, isUpdateRelated }
  );

  const score = classification.score;

  if (!classification.isRelevant && score < THRESHOLDS.topicRelevance) {
    return {
      gate: 1,
      name: "Topic relevance",
      result: "fail",
      score,
      reason: `Off-topic (score ${score.toFixed(3)} < ${THRESHOLDS.topicRelevance}): ${classification.reason}`,
      details: { positiveScore, isUpdateRelated, method: classification.method, classifierReason: classification.reason },
    };
  }

  return {
    gate: 1,
    name: "Topic relevance",
    result: "pass",
    score,
    details: { positiveScore, isUpdateRelated, method: classification.method, classifierReason: classification.reason },
  };
}

function checkNLPConfidence(prompt) {
  const { extraction } = prompt;

  if (!extraction.primaryEntity) {
    return {
      gate: 2,
      name: "NLP extraction confidence",
      result: "fail",
      score: 0,
      reason: "No software/OS entity could be extracted from the question",
    };
  }

  if (extraction.primaryEntity.confidence < THRESHOLDS.nlpConfidence) {
    return {
      gate: 2,
      name: "NLP extraction confidence",
      result: "fail",
      score: extraction.primaryEntity.confidence,
      reason: `"${extraction.primaryEntity.name}" confidence too low (${extraction.primaryEntity.confidence.toFixed(3)})`,
    };
  }

  return {
    gate: 2,
    name: "NLP extraction confidence",
    result: "pass",
    score: extraction.primaryEntity.confidence,
    details: { entity: extraction.primaryEntity },
  };
}

// Gate 3: searches https://releasetrain.io/api/v/?q={entity} — the same endpoint
// the releasetrain.io website uses. Returns entries for Gate 5 to avoid a second call.
// If the primary search returns 0 entries, resolves a canonical name from the
// component list and retries once (e.g. "node.js" → "Node").
async function checkSoftwareExists(prompt) {
  const entityName = prompt.extraction.primaryEntity.name;
  const queryText  = prompt.originalTitle || entityName;

  let result      = await fetchVersionSearch(entityName);
  let usedName    = entityName;

  // Retry with canonical name when primary search returns no entries
  if (result.success && (result.data || []).length === 0) {
    const canonical = resolveCanonicalSearchName(entityName, queryText);
    if (canonical && canonical.toLowerCase() !== entityName.toLowerCase()) {
      console.log(`[DEBUG][Gate3] "${entityName}" → 0 results, retrying with "${canonical}"`);
      const retry = await fetchVersionSearch(canonical);
      if (retry.success && (retry.data || []).length > 0) {
        result   = retry;
        usedName = canonical;
      }
    }
  }

  if (!result.success) {
    return {
      gate: 3, name: "Software lookup", result: "fail", score: 0,
      reason: `Could not reach releasetrain.io to look up "${entityName}"`,
    };
  }

  const entries = result.data || [];

  if (entries.length === 0) {
    return {
      gate: 3, name: "Software lookup", result: "fail", score: 0,
      reason: `I don't know — "${entityName}" was not found in releasetrain.io`,
    };
  }

  const validatedName =
    entries[0]?.versionProductBrand ||
    entries[0]?.versionProductName ||
    usedName;

  console.log(`[DEBUG][Gate3] Resolved "${entityName}" → "${usedName}" → "${validatedName}" (${entries.length} entries)`);

  return {
    gate: 3,
    name: "Software lookup",
    result: "pass",
    score: 1.0,
    validatedName,
    entries,
    details: { query: usedName, totalEntries: entries.length, matchedBrand: validatedName },
  };
}

// Gate 5 receives entries already fetched by Gate 3 — no second API call.
async function checkResponseQuality(softwareName, queryType, dateFilter, entries, breakingSubType = null) {
  if (!entries || entries.length === 0) {
    return {
      gate: 5,
      name: "Response quality",
      result: "fail",
      score: 0,
      reason: `I don't know — releasetrain.io returned no data for "${softwareName}"`,
    };
  }

  // Extract type-specific structured data from the pre-fetched entries
  const extracted = extractStructuredData(entries, queryType, softwareName, dateFilter, breakingSubType);

  if (!extracted.found) {
    return {
      gate: 5,
      name: "Response quality",
      result: "fail",
      score: 0,
      reason: extracted.reason,
    };
  }

  // Score based on how many structured fields are populated
  const structured = extracted.structured;
  const valuableFields = Object.entries(structured).filter(
    ([key, val]) => key !== "queryType" && val !== null && val !== undefined && val !== ""
  );
  const totalFields = Object.keys(structured).length - 1; // exclude queryType
  const quality = totalFields > 0 ? valuableFields.length / totalFields : 0;

  if (quality < THRESHOLDS.responseQuality) {
    return {
      gate: 5,
      name: "Response quality",
      result: "fail",
      score: quality,
      reason: `I don't know — releasetrain.io has too little ${queryType} data for "${softwareName}" (${(quality * 100).toFixed(0)}% completeness)`,
    };
  }

  return {
    gate: 5,
    name: "Response quality",
    result: "pass",
    score: quality,
    structuredData: structured,
    details: { queryType, fieldsPresent: valuableFields.length, totalFields },
  };
}

// ── Date Filter Helper ──
// Narrows entries to those matching a resolved date or date range.

function filterByDate(entries, dateFilter) {
  if (!dateFilter) return entries;
  return entries.filter((e) => {
    const d = String(e.versionReleaseDate || "");
    if (dateFilter.type === "exact") return d === dateFilter.date;
    if (dateFilter.type === "range") return d >= dateFilter.from && d <= dateFilter.to;
    return true;
  });
}

// ── Structured Data Extractor ──
// Handles releasetrain.io's actual response schema:
//   { "softwareName": [{ versionNumber, versionReleaseChannel, isCve, ... }] }
// Returns { found: true, structured } or { found: false, reason }.

// entries is a flat array from GET /api/v/?q={software}
function extractStructuredData(entries, queryType, softwareName, dateFilter, breakingSubType = null) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return {
      found: false,
      reason: `I don't know — releasetrain.io has no data for "${softwareName}"`,
    };
  }

  const displayName =
    entries[0]?.versionProductBrand ||
    entries[0]?.versionProductName ||
    softwareName;

  // Narrow to the requested date / date range before any further filtering
  const allEntries = entries;
  entries = filterByDate(entries, dateFilter);
  const dateLabel = dateFilter ? ` for ${dateFilter.label} (${dateFilter.displayDate})` : "";

  if (dateFilter && entries.length === 0) {
    return {
      found: false,
      reason: `I don't know — releasetrain.io has no data for "${displayName}"${dateLabel}`,
    };
  }

  // Format "20260410" → "2026-04-10"
  const fmtDate = (d) => {
    const s = String(d || "");
    return s.length === 8 ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : s || null;
  };

  // Extract CVE-XXXX-XXXXX from an NVD URL
  const parseCveId = (url) => {
    const m = String(url || "").match(/CVE-\d{4}-\d+/i);
    return m ? m[0].toUpperCase() : null;
  };

  if (queryType === "cve") {
    const cveEntries = entries.filter(
      (e) =>
        e.isCve === true ||
        (e.versionUrl && e.versionUrl.includes("nvd.nist.gov")) ||
        (e.classification?.securityType || []).includes("SECURITY")
    );

    if (cveEntries.length === 0) {
      return {
        found: false,
        reason: `I don't know — releasetrain.io has no CVE data for "${displayName}"${dateLabel}`,
      };
    }

    const top = cveEntries[0];
    return {
      found: true,
      structured: {
        queryType: "cve",
        software: displayName,
        dateFilter: dateFilter ? dateFilter.displayDate : null,
        dateLabel: dateFilter ? dateFilter.label : null,
        cveId: parseCveId(top.versionUrl) || top.versionId,
        severity: (top.classification?.securityType || []).join(", ") || null,
        version: top.versionNumber,
        releaseDate: fmtDate(top.versionReleaseDate),
        description: top.versionReleaseNotes,
        sourceUrl: top.versionUrl,
        totalCves: cveEntries.length,
        recentCves: cveEntries.slice(0, 5).map((e) => ({
          cveId: parseCveId(e.versionUrl) || e.versionId,
          version: e.versionNumber,
          date: fmtDate(e.versionReleaseDate),
          notes: (e.versionReleaseNotes || "").slice(0, 180),
          url: e.versionUrl,
        })),
      },
    };
  }

  if (queryType === "patch") {
    const patchEntries = entries
      .filter(
        (e) =>
          e.versionReleaseChannel === "patch" ||
          e.versionReleaseChannel === "hotfix"
      )
      .slice(0, 10);

    if (patchEntries.length === 0) {
      return {
        found: false,
        reason: `I don't know — releasetrain.io has no patch releases for "${displayName}"${dateLabel}`,
      };
    }

    const top = patchEntries[0];
    return {
      found: true,
      structured: {
        queryType: "patch",
        software: displayName,
        dateFilter: dateFilter ? dateFilter.displayDate : null,
        dateLabel: dateFilter ? dateFilter.label : null,
        version: top.versionNumber,
        releaseDate: fmtDate(top.versionReleaseDate),
        releaseChannel: top.versionReleaseChannel,
        description: top.versionReleaseNotes,
        sourceUrl: top.versionUrl,
        isCve: top.isCve,
        recentPatches: patchEntries.map((e) => ({
          version: e.versionNumber,
          date: fmtDate(e.versionReleaseDate),
          isCve: e.isCve,
          cveId: e.isCve ? parseCveId(e.versionUrl) : null,
          notes: (e.versionReleaseNotes || "").slice(0, 180),
          url: e.versionUrl,
        })),
      },
    };
  }

  if (queryType === "version") {
    const top = entries[0];
    if (!top?.versionNumber) {
      return {
        found: false,
        reason: `I don't know — releasetrain.io has no version information for "${displayName}"${dateLabel}`,
      };
    }

    return {
      found: true,
      structured: {
        queryType: "version",
        software: displayName,
        dateFilter: dateFilter ? dateFilter.displayDate : null,
        dateLabel: dateFilter ? dateFilter.label : null,
        version: top.versionNumber,
        releaseDate: fmtDate(top.versionReleaseDate),
        releaseChannel: top.versionReleaseChannel,
        description: top.versionReleaseNotes,
        sourceUrl: top.versionUrl,
        totalEntries: entries.length,
        recentVersions: entries.slice(0, 5).map((e) => ({
          version: e.versionNumber,
          date: fmtDate(e.versionReleaseDate),
          channel: e.versionReleaseChannel,
          isCve: e.isCve,
        })),
      },
    };
  }

  if (queryType === "critical") {
    const criticalEntries = entries.filter(
      (e) => (e.classification?.breakingType || []).includes("Critical Failure")
    );
    console.log(`[DEBUG][Extract] critical filter → ${criticalEntries.length} / ${entries.length} entries`);

    if (criticalEntries.length === 0) {
      return {
        found: false,
        reason: `I don't know — releasetrain.io has no "Critical Failure" releases for "${displayName}"${dateLabel}`,
      };
    }

    return {
      found: true,
      structured: {
        queryType: "critical",
        software: displayName,
        dateFilter: dateFilter ? dateFilter.displayDate : null,
        dateLabel: dateFilter ? dateFilter.label : null,
        totalCritical: criticalEntries.length,
        breakingLabel: "Critical Failure",
        entries: criticalEntries.slice(0, 20).map((e) => ({
          version: e.versionNumber,
          date: fmtDate(e.versionReleaseDate),
          channel: e.versionReleaseChannel,
          breakingTypes: e.classification?.breakingType || [],
          securityTypes: e.classification?.securityType || [],
          isCve: e.isCve,
          notes: (e.versionReleaseNotes || "").slice(0, 200),
          url: e.versionUrl,
        })),
      },
    };
  }

  if (queryType === "breaking") {
    const targetType = breakingSubType || "Breaking Update";
    const matched = entries.filter(
      (e) => (e.classification?.breakingType || []).includes(targetType)
    );
    console.log(`[DEBUG][Extract] breaking "${targetType}" → ${matched.length} / ${entries.length} entries`);

    if (matched.length === 0) {
      return {
        found: false,
        reason: `I don't know — releasetrain.io has no "${targetType}" releases for "${displayName}"${dateLabel}`,
      };
    }

    return {
      found: true,
      structured: {
        queryType: "breaking",
        software: displayName,
        dateFilter: dateFilter ? dateFilter.displayDate : null,
        dateLabel: dateFilter ? dateFilter.label : null,
        totalCritical: matched.length,
        breakingLabel: targetType,
        entries: matched.slice(0, 20).map((e) => ({
          version: e.versionNumber,
          date: fmtDate(e.versionReleaseDate),
          channel: e.versionReleaseChannel,
          breakingTypes: e.classification?.breakingType || [],
          securityTypes: e.classification?.securityType || [],
          isCve: e.isCve,
          notes: (e.versionReleaseNotes || "").slice(0, 200),
          url: e.versionUrl,
        })),
      },
    };
  }

  // general
  const top = entries[0];
  if (!top) {
    return {
      found: false,
      reason: `I don't know — releasetrain.io has no usable information for "${displayName}"${dateLabel}`,
    };
  }

  return {
    found: true,
    structured: {
      queryType: "general",
      software: displayName,
      dateFilter: dateFilter ? dateFilter.displayDate : null,
      dateLabel: dateFilter ? dateFilter.label : null,
      version: top.versionNumber,
      releaseDate: fmtDate(top.versionReleaseDate),
      releaseChannel: top.versionReleaseChannel,
      description: top.versionReleaseNotes,
      sourceUrl: top.versionUrl,
      totalEntries: entries.length,
    },
  };
}

// ── Intent-Data Alignment Check ──
// Verifies the query's intent (queryType) matches what releasetrain.io actually returned.
// For "CVEs for Android": queryType='cve' AND data must contain real CVE entries for Android.

function checkIntentDataAlignment(queryType, structuredData, softwareName, breakingSubType = null) {
  if (!structuredData) {
    return { valid: false, reason: "I don't know — no structured data returned from releasetrain.io" };
  }

  if (queryType === "cve") {
    if (!structuredData.totalCves || structuredData.totalCves === 0) {
      return {
        valid: false,
        reason: `I don't know — your query asks about CVEs for "${softwareName}" but releasetrain.io has no CVE entries for this software`,
      };
    }
    if (!structuredData.recentCves || structuredData.recentCves.length === 0) {
      return {
        valid: false,
        reason: `I don't know — CVE intent confirmed but no CVE records found in releasetrain.io for "${softwareName}"`,
      };
    }
  }

  if (queryType === "patch") {
    if (!structuredData.recentPatches || structuredData.recentPatches.length === 0) {
      return {
        valid: false,
        reason: `I don't know — your query asks about patches for "${softwareName}" but releasetrain.io has no patch entries for this software`,
      };
    }
  }

  if (queryType === "version") {
    if (!structuredData.version) {
      return {
        valid: false,
        reason: `I don't know — your query asks about the version of "${softwareName}" but releasetrain.io has no version data for this software`,
      };
    }
  }

  if (queryType === "critical") {
    if (!structuredData.entries || structuredData.entries.length === 0) {
      return {
        valid: false,
        reason: `I don't know — your query asks about critical failures for "${softwareName}" but releasetrain.io has no Critical Failure releases for this software`,
      };
    }
  }

  if (queryType === "breaking") {
    const label = breakingSubType || "Breaking Update";
    if (!structuredData.entries || structuredData.entries.length === 0) {
      return {
        valid: false,
        reason: `I don't know — your query asks about "${label}" for "${softwareName}" but releasetrain.io has no matching releases`,
      };
    }
  }

  return { valid: true };
}

// ── Composite Score Calculator ──
// Extracted here so runPipeline can use it as a gate condition, not just for display.

function computeComposite(gates) {
  const scores = {
    topicRelevance: gates.find((g) => g.gate === 1)?.score || 0,
    nlpConfidence:  gates.find((g) => g.gate === 2)?.score || 0,
    similarity:     gates.find((g) => g.gate === 3)?.score || 0,
    responseQuality: gates.find((g) => g.gate === 5)?.score || 0,
  };
  return (
    WEIGHTS.topicRelevance * scores.topicRelevance +
    WEIGHTS.nlpConfidence  * scores.nlpConfidence +
    WEIGHTS.similarity     * scores.similarity +
    WEIGHTS.responseQuality * scores.responseQuality
  );
}

// ── Result Builder ──

function buildResult(
  decision,
  gates,
  reason = null,
  suggestions = null,
  data = null,
  queryType = "general"
) {
  const composite = computeComposite(gates);

  return {
    decision,
    compositeScore: Math.round(composite * 1000) / 1000,
    reason,
    suggestions,
    gates,
    data,
    queryType,
    timestamp: new Date().toISOString(),
  };
}

module.exports = { runPipeline, THRESHOLDS };