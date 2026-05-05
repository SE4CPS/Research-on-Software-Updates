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
const { runAgent } = require("./agent");
const { BROAD_SEARCH_SOFTWARE } = require("./nlpProcessor");

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

// ── Broad Pipeline — no entity, search top software, return up to 10 results ──

async function runBroadPipeline(processedPrompt) {
  const { queryType, dateFilter, breakingSubType } = processedPrompt.metadata;

  // Gate 1: keyword-based pass — NeMo LLM rejects no-entity queries so use
  // the fact that NLP already confirmed breaking keyword + date filter.
  const gate1 = {
    gate: 1, name: "Topic relevance", result: "pass", score: 0.85,
    details: { method: "keyword", classifierReason: "broad failure query with date filter" },
  };

  const fmtDate = (d) => {
    const s = String(d || "");
    return s.length === 8 ? `${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}` : s || null;
  };

  // Search popular software in parallel, filter by date + breaking type
  const results = await Promise.allSettled(
    BROAD_SEARCH_SOFTWARE.map((sw) => fetchVersionSearch(sw))
  );

  const collected = [];
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    if (r.status !== "fulfilled" || !r.value.success) continue;
    let entries = r.value.data || [];

    // Date filter
    if (dateFilter) {
      entries = entries.filter((e) => {
        const d = String(e.versionReleaseDate || "");
        if (dateFilter.type === "exact") return d === dateFilter.date;
        if (dateFilter.type === "range") return d >= dateFilter.from && d <= dateFilter.to;
        return true;
      });
    }

    // Breaking type filter
    if (breakingSubType) {
      entries = entries.filter((e) =>
        (e.classification?.breakingType || []).includes(breakingSubType)
      );
    } else {
      entries = entries.filter((e) =>
        (e.classification?.breakingType || []).length > 0
      );
    }

    for (const e of entries) {
      collected.push({
        software: e.versionProductBrand || e.versionProductName || BROAD_SEARCH_SOFTWARE[i],
        version:       e.versionNumber,
        date:          fmtDate(e.versionReleaseDate),
        channel:       e.versionReleaseChannel,
        breakingTypes: e.classification?.breakingType || [],
        isCve:         e.isCve,
        notes:         (e.versionReleaseNotes || "").slice(0, 200),
        versionUrl:    e.versionUrl,
      });
    }
  }

  if (collected.length === 0) {
    return buildResult(
      "abstain",
      [gate1],
      `I don't know — no ${breakingSubType || "failures"} found across popular software for ${dateFilter?.displayDate || "that date"}.`,
      null, null, queryType
    );
  }

  // Sort by date descending, cap at 10
  collected.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
  const top10 = collected.slice(0, 10);

  const structuredData = {
    queryType: "breaking",
    software: "Multiple Software",
    dateFilter: dateFilter ? dateFilter.displayDate : null,
    dateLabel:  dateFilter ? dateFilter.label : null,
    totalCritical: collected.length,
    breakingLabel: breakingSubType || "All Failures",
    isBroadQuery: true,
    entries: top10,
  };

  const compositeScore = gate1.score * 0.5 + 0.5; // simplified composite for broad
  return {
    decision: "confident",
    compositeScore: Math.round(compositeScore * 1000) / 1000,
    reason: null,
    suggestions: null,
    gates: [gate1],
    data: structuredData,
    queryType,
    timestamp: new Date().toISOString(),
  };
}

// ── Main Pipeline ──

async function runPipeline(processedPrompt) {
  // Broad query: no entity — search across popular software
  if (processedPrompt.metadata?.isBroadQuery) {
    return runBroadPipeline(processedPrompt);
  }

  const gates = [];
  const queryType    = processedPrompt.metadata?.queryType    || "general";
  const dateFilter   = processedPrompt.metadata?.dateFilter   || null;
  const breakingSubType = processedPrompt.metadata?.breakingSubType || null;
  const wantAll      = processedPrompt.metadata?.wantAll      || false;

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

  // Agent Layer: LLM decides which search tool(s) to call (search_releases / search_cves /
  // search_breaking_changes). Runs after entity is confirmed (Gate 2) so the agent has a
  // clean entity name. Returns null and falls back to Gate 3's own call when unavailable.
  const agentResult = await runAgent(
    processedPrompt.extraction.primaryEntity.name,
    processedPrompt.originalTitle || processedPrompt.prompt,
    queryType
  );
  console.log(`[DEBUG][Pipeline] Agent        : ${
    agentResult
      ? `tools=[${agentResult.agentDecision.toolsSelected.map((t) => t.tool).join(", ")}] entries=${agentResult.entries.length}`
      : "skipped (Gate 3 fallback)"
  }`);

  // Gate 3: Software Lookup — uses agent-fetched entries when available
  const gate3 = await checkSoftwareExists(processedPrompt, agentResult);
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
  const gate5 = await checkResponseQuality(gate3.validatedName, queryType, dateFilter, gate3.entries, breakingSubType, wantAll);
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
  if (llmResult)              result.llmResponse   = llmResult;
  if (bertscore)              result.bertscore     = bertscore;
  if (gate3.agentDecision)    result.agentDecision = gate3.agentDecision;
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

// Gate 3: searches https://releasetrain.io/api/component/?q={entity}.
// Returns entries for Gate 5 to avoid a second call.
// If the primary search returns 0 entries, resolves a canonical name from the
// component list and retries once (e.g. "node.js" → "Node").
async function checkSoftwareExists(prompt, agentResult = null) {
  const entityName = prompt.extraction.primaryEntity.name;
  const queryText  = prompt.originalTitle || entityName;

  // Agent pre-fetched entries — skip the API call
  if (agentResult?.entries?.length > 0) {
    const entries = agentResult.entries;
    const validatedName =
      entries[0]?.versionProductBrand ||
      entries[0]?.versionProductName  ||
      entityName;
    console.log(`[DEBUG][Gate3] Agent supplied ${entries.length} entries for "${validatedName}"`);
    return {
      gate: 3, name: "Software lookup", result: "pass", score: 1.0,
      validatedName,
      entries,
      agentDecision: agentResult.agentDecision,
      details: { query: entityName, totalEntries: entries.length, matchedBrand: validatedName, source: "agent" },
    };
  }

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
async function checkResponseQuality(softwareName, queryType, dateFilter, entries, breakingSubType = null, wantAll = false) {
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

// entries is a flat array from GET /api/component/?q={software}
function extractStructuredData(entries, queryType, softwareName, dateFilter, breakingSubType = null) {
  // ── Date keyword → date equivalent trace ──────────────────────────────
  const apiUrl = `https://releasetrain.io/api/component/?q=${encodeURIComponent((softwareName || "").toLowerCase())}`;
  console.log(`\n[DEBUG][DateFlow] ══════════════════════════════════════`);
  console.log(`[DEBUG][DateFlow] step 1 — URL formed    : ${apiUrl}`);
  if (dateFilter) {
    const keywordLabel = dateFilter.label;
    let converted = "";
    if (dateFilter.type === "latest")  converted = "→ no date boundary — return most recent entry";
    if (dateFilter.type === "exact")   converted = `→ exact date : ${dateFilter.date} (${dateFilter.displayDate})`;
    if (dateFilter.type === "range")   converted = `→ range      : ${dateFilter.from} → ${dateFilter.to} (${dateFilter.displayDate})`;
    console.log(`[DEBUG][DateFlow] step 2 — keyword      : "${keywordLabel}"`);
    console.log(`[DEBUG][DateFlow] step 3 — converted    : ${converted}`);
  } else {
    console.log(`[DEBUG][DateFlow] step 2 — keyword      : none (no temporal filter)`);
    console.log(`[DEBUG][DateFlow] step 3 — converted    : no date boundary — all entries returned`);
  }
  console.log(`[DEBUG][DateFlow] step 4 — raw entries  : ${Array.isArray(entries) ? entries.length : 0} fetched from API`);
  console.log(`[DEBUG][DateFlow] ══════════════════════════════════════\n`);

  console.log(`\n[DEBUG][Extract] ══════════════════════════════════════`);
  console.log(`[DEBUG][Extract] softwareName : "${softwareName}"`);
  console.log(`[DEBUG][Extract] queryType    : ${queryType}`);
  console.log(`[DEBUG][Extract] dateFilter   : ${dateFilter ? `${dateFilter.label} (${dateFilter.displayDate})` : "none"}`);
  console.log(`[DEBUG][Extract] raw entries  : ${Array.isArray(entries) ? entries.length : 0}`);

  if (!Array.isArray(entries) || entries.length === 0) {
    console.log(`[DEBUG][Extract] → ABORT: no entries`);
    return {
      found: false,
      reason: `I don't know — no data for "${softwareName}"`,
    };
  }

  const displayName =
    entries[0]?.versionProductBrand ||
    entries[0]?.versionProductName ||
    softwareName;

  // Canonical vendor (component) source — always used as primary
  const vendorSourceUrl =
    `https://releasetrain.io/api/component/?q=${encodeURIComponent(displayName.toLowerCase())}`;

  console.log(`[DEBUG][Extract] displayName  : "${displayName}"`);
  console.log(`[DEBUG][Extract] sourceUrl    : ${vendorSourceUrl}`);

  // Sample of raw entry fields to show what the API returned
  const sample = entries[0];
  console.log(`[DEBUG][Extract] sample entry fields: ${Object.keys(sample).join(", ")}`);
  console.log(`[DEBUG][Extract] sample[0]    : versionId=${sample.versionId} | channel=${sample.versionReleaseChannel} | isCve=${sample.isCve} | date=${sample.versionReleaseDate}`);

  const fmtDate = (d) => {
    const s = String(d || "");
    return s.length === 8 ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}` : s || null;
  };

  const parseCveId = (url) => {
    const m = String(url || "").match(/CVE-\d{4}-\d+/i);
    return m ? m[0].toUpperCase() : null;
  };

  // Sort entries by versionReleaseDate descending — guarantees entries[0] is always the newest
  const sortedEntries = [...entries].sort((a, b) =>
    String(b.versionReleaseDate || "").localeCompare(String(a.versionReleaseDate || ""))
  );

  // Date filter — converts keyword type to entry predicate
  const filterByDate = (entries, dateFilter) => {
    if (!dateFilter) return entries;
    if (dateFilter.type === "latest") {
      // Take all entries with the same date as the top (newest) entry
      const newestDate = String(entries[0]?.versionReleaseDate || "");
      const matched = entries.filter((e) => String(e.versionReleaseDate || "") === newestDate);
      console.log(`[DEBUG][DateFlow] "latest" resolved → newest date in data: ${newestDate} (${matched.length} entries on that date)`);
      return matched;
    }
    return entries.filter((e) => {
      const d = String(e.versionReleaseDate || "");
      if (dateFilter.type === "exact") return d === dateFilter.date;
      if (dateFilter.type === "range") return d >= dateFilter.from && d <= dateFilter.to;
      return true;
    });
  };

  const beforeDate = sortedEntries.length;
  entries = filterByDate(sortedEntries, dateFilter);
  console.log(`[DEBUG][Extract] step 5 — after date filter: ${entries.length} / ${beforeDate} entries kept | top date: ${entries[0]?.versionReleaseDate || "n/a"}`);

  if (entries.length === 0) {
    console.log(`[DEBUG][Extract] → ABORT: all entries filtered out by date`);
    return {
      found: false,
      reason: `I don't know — no data for "${displayName}" in requested time range`,
    };
  }

  // Standard 3-field entry format returned for every query type
  const fmt = (e) => ({
    versionNumber:      e.versionNumber,
    versionProductName: e.versionProductBrand || e.versionProductName || displayName,
    releaseDate:        fmtDate(e.versionReleaseDate),
    versionUrl:         e.versionUrl || null,
  });

  // Case-insensitive match against classification.breakingType AND versionSearchTags
  const matchesBreaking = (e, subType) => {
    if (!subType) return (e.classification?.breakingType || []).length > 0;
    const bt = (e.classification?.breakingType || []).map((t) => t.toLowerCase());
    const st = (e.versionSearchTags   || []).map((t) => t.toLowerCase());
    return bt.includes(subType.toLowerCase()) || st.includes(subType.toLowerCase());
  };

  // ── BREAKING (Critical failure / Breaking Update / Network Issue / etc.) ──
  if (queryType === "breaking") {
    const label   = breakingSubType || "All Breaking";
    const matched = entries.filter((e) => matchesBreaking(e, breakingSubType));

    console.log(`[DEBUG][Extract] breaking filter "${label}": ${matched.length} / ${entries.length} matched`);
    matched.slice(0, 3).forEach((e, i) =>
      console.log(`[DEBUG][Extract]   [${i}] ver=${e.versionNumber} | date=${e.versionReleaseDate} | breakingType=${JSON.stringify(e.classification?.breakingType)}`)
    );

    if (matched.length === 0) {
      console.log(`[DEBUG][Extract] → ABORT: no entries match breaking type "${label}"`);
      return { found: false, reason: `I don't know — no "${label}" entries for "${displayName}"` };
    }

    console.log(`[DEBUG][Extract] → RESULT: breaking | label="${label}" | found=${matched.length}`);
    console.log(`[DEBUG][Extract] ══════════════════════════════════════\n`);
    return {
      found: true,
      structured: {
        queryType:    "breaking",
        software:     displayName,
        sourceUrl:    vendorSourceUrl,
        breakingLabel: label,
        totalFound:   matched.length,
        entries:      matched.slice(0, 20).map(fmt),
      },
    };
  }

  // ── PATCH ──
  if (queryType === "patch") {
    const matched = entries.filter(
      (e) => e.versionReleaseChannel === "patch" || e.versionReleaseChannel === "hotfix"
    );

    console.log(`[DEBUG][Extract] patch filter: ${matched.length} / ${entries.length} matched`);
    if (matched.length === 0) {
      console.log(`[DEBUG][Extract] → ABORT: no patch entries`);
      return { found: false, reason: `I don't know — no patch releases for "${displayName}"` };
    }

    console.log(`[DEBUG][Extract] → RESULT: patch | found=${matched.length}`);
    console.log(`[DEBUG][Extract] ══════════════════════════════════════\n`);
    return {
      found: true,
      structured: {
        queryType:  "patch",
        software:   displayName,
        sourceUrl:  vendorSourceUrl,
        totalFound: matched.length,
        entries:    matched.slice(0, 20).map(fmt),
      },
    };
  }

  // ── CVE ──
  if (queryType === "cve") {
    const cveEntries = entries.filter(
      (e) =>
        e.isCve === true ||
        (e.versionUrl && e.versionUrl.includes("nvd.nist.gov")) ||
        (e.classification?.securityType || []).includes("SECURITY") ||
        (e.versionReleaseTags || []).some((t) => /security|vulnerability|cve/i.test(t)) ||
        (e.versionSearchTags  || []).some((t) => /cve-?\d{4}/i.test(t))
    );

    console.log(`[DEBUG][Extract] CVE filter: ${cveEntries.length} / ${entries.length} matched`);
    cveEntries.slice(0, 3).forEach((e, i) =>
      console.log(`[DEBUG][Extract]   cve[${i}] id=${parseCveId(e.versionUrl) || e.versionId} | ver=${e.versionNumber} | date=${e.versionReleaseDate}`)
    );

    if (cveEntries.length === 0) {
      console.log(`[DEBUG][Extract] → ABORT: no CVE entries`);
      return { found: false, reason: `I don't know — no CVE data for "${displayName}"` };
    }

    console.log(`[DEBUG][Extract] → RESULT: cve | found=${cveEntries.length}`);
    console.log(`[DEBUG][Extract] ══════════════════════════════════════\n`);
    return {
      found: true,
      structured: {
        queryType:  "cve",
        software:   displayName,
        sourceUrl:  vendorSourceUrl,
        totalFound: cveEntries.length,
        entries:    cveEntries.slice(0, 20).map((e) => ({
          ...fmt(e),
          cveId: parseCveId(e.versionUrl) || e.versionId,
        })),
      },
    };
  }

  // ── VERSION / GENERAL / LATEST ──
  const top = entries[0];
  console.log(`[DEBUG][Extract] → RESULT: ${queryType} | ver=${top.versionNumber} | date=${top.versionReleaseDate} | total=${entries.length}`);
  console.log(`[DEBUG][Extract] ══════════════════════════════════════\n`);
  return {
    found: true,
    structured: {
      queryType:  queryType === "version" ? "version" : "general",
      software:   displayName,
      sourceUrl:  vendorSourceUrl,
      totalFound: entries.length,
      entries:    entries.slice(0, 20).map(fmt),
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

  // All query types now return entries[] — just verify at least one entry was found
  if (!structuredData.entries || structuredData.entries.length === 0) {
    const label = breakingSubType || queryType;
    return {
      valid: false,
      reason: `I don't know — no "${label}" data found for "${softwareName}" in releasetrain.io`,
    };
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

module.exports = { runPipeline, runBroadPipeline, THRESHOLDS };