// ============================================
// FILE: server/services/nlpProcessor.js
// NLP entity extraction from user text
// ============================================

const nlp = require("compromise");
const { findComponentInText } = require("./componentNames");

const OS_PATTERNS = [
  /\b(windows(?:\s*\d+(?:\.\d+)*)?|win\s*\d+(?:\.\d+)*)\b/gi,             // "windows", "windows 11", "win 10"
  /\b(macos(?:\s*\d+(?:\.\d+)*)?|mac\s*os(?:\s*\d+(?:\.\d+)*)?|os\s*x(?:\s*\d+(?:\.\d+)*)?)\b/gi,
  /\b(ios(?:\s*\d+(?:\.\d+)*)?)\b/gi,
  /\b(ipados(?:\s*\d+(?:\.\d+)*)?)\b/gi,
  /\b(android(?:\s*\d+(?:\.\d+)*)?)\b/gi,
  /\b(ubuntu\s*[\d.]*|debian\s*[\d.]*|fedora\s*[\d.]*|centos\s*[\d.]*)\b/gi,
  /\b(chrome\s*os|chromeos)\b/gi,
  /\b(linux\s*mint\s*[\d.]*)\b/gi,
];

const SOFTWARE_PATTERNS = [
  /\b(firefox(?:\s*[\d.]+)?)\b/gi,
  /\b(chrome(?:\s*[\d.]+)?|chromium(?:\s*[\d.]+)?)\b/gi,
  /\b(safari(?:\s*[\d.]+)?)\b/gi,
  /\b(edge(?:\s*[\d.]+)?)\b/gi,
  /\b(visual\s*studio\s*code|vs\s*code|vscode)\b/gi,
  /\b(microsoft\s*office|office\s*365|excel|word|outlook|powerpoint|teams)\b/gi,
  /\b(slack|discord|zoom|skype)\b/gi,
  /\b(docker(?:\s*[\d.]+)?|kubernetes|k8s)\b/gi,
  /\b(node\.?js(?:\s*[\d.]+)?|python(?:\s*[\d.]+)?|java(?:\s*[\d.]+)?)\b/gi,
  /\b(photoshop|illustrator|premiere|lightroom)\b/gi,
  /\b(spotify|netflix|whatsapp|telegram|signal)\b/gi,
  /\b(openssl(?:\s*[\d.]+)?|openssh(?:\s*[\d.]+)?|nginx(?:\s*[\d.]+)?|apache(?:\s*[\d.]+)?)\b/gi,
  /\b(linux\s*kernel|kernel(?:\s*[\d.]+)?)\b/gi,
  /\b(git(?:\s*[\d.]+)?|curl(?:\s*[\d.]+)?|bash(?:\s*[\d.]+)?)\b/gi,
  /\b(github(?:\s*[\d.]+)?|gitlab(?:\s*[\d.]+)?|bitbucket(?:\s*[\d.]+)?)\b/gi,
  /\b(npm(?:\s*[\d.]+)?|yarn(?:\s*[\d.]+)?|pip(?:\s*[\d.]+)?|gradle(?:\s*[\d.]+)?|maven(?:\s*[\d.]+)?)\b/gi,
  /\b(terraform(?:\s*[\d.]+)?|ansible(?:\s*[\d.]+)?|jenkins(?:\s*[\d.]+)?|prometheus(?:\s*[\d.]+)?|grafana(?:\s*[\d.]+)?)\b/gi,
  /\b(react(?:\s*[\d.]+)?|vue(?:\s*[\d.]+)?|angular(?:\s*[\d.]+)?|next\.?js(?:\s*[\d.]+)?|express(?:\s*[\d.]+)?)\b/gi,
  /\b(postgres(?:ql)?(?:\s*[\d.]+)?|mysql(?:\s*[\d.]+)?|mongodb(?:\s*[\d.]+)?|redis(?:\s*[\d.]+)?|sqlite(?:\s*[\d.]+)?)\b/gi,
];

const DEVICE_PATTERNS = [
  /\b(ipad|iphone|macbook|imac)\b/gi,
  /\b(pixel\s*\d*|galaxy\s*[\w]*|surface\s*[\w]*)\b/gi,
];

// Requires "v"/"version" prefix OR at least one dot — prevents matching bare
// numbers in date expressions like "last 7 days" or "this month".
const VERSION_PATTERN =
  /\b(?:(?:version|ver|v)\s*(\d+(?:\.\d+)*(?:\s*(?:beta|alpha|rc|dev|preview)\s*\d*)?)|(\d+\.\d+(?:\.\d+)*))\b/gi;

function extractEntities(text) {
  const combined = text.toLowerCase();
  const entities = { os: [], software: [], device: [], versions: [] };
  const confidences = {};

  for (const pattern of OS_PATTERNS) {
    const matches = combined.matchAll(new RegExp(pattern));
    for (const match of matches) {
      const name = match[1].trim();
      if (!entities.os.includes(name)) {
        entities.os.push(name);
        confidences[name] = 0.95;
      }
    }
  }

  for (const pattern of SOFTWARE_PATTERNS) {
    const matches = combined.matchAll(new RegExp(pattern));
    for (const match of matches) {
      const name = match[1].trim();
      if (!entities.software.includes(name)) {
        entities.software.push(name);
        confidences[name] = 0.92;
      }
    }
  }

  for (const pattern of DEVICE_PATTERNS) {
    const matches = combined.matchAll(new RegExp(pattern));
    for (const match of matches) {
      const name = match[1].trim();
      if (!entities.device.includes(name)) {
        entities.device.push(name);
        confidences[name] = 0.88;
      }
    }
  }

  // ── Component names lookup (always runs — primary vendor source) ──
  // Searches the full releasetrain.io/api/c/names list (~6000+ vendors).
  // Sorted longest-first so the most specific match always wins.
  // Confidence 0.95 beats regex software patterns (0.92) so canonical names take priority.

  // Words that appear in the c/names list but are structural query words, not vendor names.
  const VENDOR_STOPLIST = new Set([
    "version", "patch", "update", "issue", "issues", "failure", "failures",
    "error", "errors", "release", "releases", "latest", "newest", "current",
    "critical", "breaking", "network", "logging", "monitoring", "data",
    "configuration", "security", "dependency", "resource", "compatibility",
    "today", "yesterday", "month", "week",
  ]);

  const componentMatch = findComponentInText(text, VENDOR_STOPLIST);
  if (componentMatch) {
    const cName      = componentMatch.name.toLowerCase();
    const apiUrl     = `https://releasetrain.io/api/component/?q=${encodeURIComponent(cName)}`;
    const alreadyIn  = entities.software.includes(cName) ||
                       entities.os.includes(cName) ||
                       entities.device.includes(cName);

    if (!alreadyIn) {
      entities.software.push(cName);
      confidences[cName] = componentMatch.confidence;
      console.log(`[DEBUG][NLP] c/names → NEW  : "${componentMatch.name}" (conf=${componentMatch.confidence}) | URL: ${apiUrl}`);
    } else {
      // Upgrade confidence so canonical c/names entry wins over a generic regex hit
      const prev = confidences[cName] || 0;
      confidences[cName] = Math.max(prev, componentMatch.confidence);
      console.log(`[DEBUG][NLP] c/names → UPGRADE: "${componentMatch.name}" (conf ${prev.toFixed(2)}→${confidences[cName].toFixed(2)}) | URL: ${apiUrl}`);
    }
  } else {
    console.log(`[DEBUG][NLP] c/names → no match in text`);
  }

  // ── NLP noun fallback (last resort, confidence 0.55 — below Gate 2 threshold) ──
  const doc = nlp(text);
  const nouns = doc.nouns().out("array");
  for (const noun of nouns) {
    const lower = noun.toLowerCase().trim();
    const alreadyFound =
      entities.os.includes(lower) ||
      entities.software.includes(lower) ||
      entities.device.includes(lower);
    if (!alreadyFound && lower.length > 2 && /^[a-z]/.test(lower)) {
      entities.software.push(lower);
      confidences[lower] = 0.55;
    }
  }

  const versionMatches = combined.matchAll(VERSION_PATTERN);
  for (const match of versionMatches) {
    const ver = (match[1] || match[2] || "").trim();
    if (ver) entities.versions.push(ver);
  }

  const allEntities = [...entities.os, ...entities.software, ...entities.device];
  const avgConfidence =
    allEntities.length > 0
      ? allEntities.reduce((sum, e) => sum + (confidences[e] || 0.5), 0) / allEntities.length
      : 0;

  return {
    entities,
    confidences,
    overallConfidence: Math.round(avgConfidence * 100) / 100,
    primaryEntity: selectPrimaryEntity(entities, confidences),
  };
}

function selectPrimaryEntity(entities, confidences) {
  if (entities.os.length > 0) {
    const best = entities.os.reduce((a, b) =>
      (confidences[a] || 0) >= (confidences[b] || 0) ? a : b
    );
    return { name: cleanName(best), type: "os", confidence: confidences[best] || 0.5 };
  }

  if (entities.software.length > 0) {
    const best = entities.software.reduce((a, b) =>
      (confidences[a] || 0) >= (confidences[b] || 0) ? a : b
    );
    return { name: cleanName(best), type: "software", confidence: confidences[best] || 0.5 };
  }

  const deviceToOS = {
    ipad: "ipados", iphone: "ios", macbook: "macos",
    imac: "macos", pixel: "android", galaxy: "android", surface: "windows",
  };

  if (entities.device.length > 0) {
    const device = entities.device[0];
    for (const [key, os] of Object.entries(deviceToOS)) {
      if (device.includes(key)) {
        return { name: os, type: "os", confidence: 0.75, mappedFrom: device };
      }
    }
  }

  return null;
}

function cleanName(name) {
  return name.replace(/\s*\d+(\.\d+)*\s*$/, "").replace(/\s+/g, " ").trim().toLowerCase();
}

// ── For Reddit question objects (kept for compatibility) ──
function generatePrompt(redditQuestion) {
  const text = `${redditQuestion.title || ""} ${redditQuestion.author_description || ""}`;
  const extraction = extractEntities(text);
  if (!extraction.primaryEntity) return null;

  const entity = extraction.primaryEntity;
  const versions = extraction.entities.versions;
  let promptText = "";
  if (versions.length > 0) {
    promptText = `What's new in ${entity.name} ${versions[0]}?`;
  } else if (redditQuestion.isAboutCve) {
    promptText = `What are the latest security updates for ${entity.name}?`;
  } else if (redditQuestion.isAboutLatestUpdate) {
    promptText = `What's the latest update for ${entity.name}?`;
  } else {
    promptText = `Tell me about ${entity.name} releases`;
  }

  return {
    id: redditQuestion._id || redditQuestion.redditId,
    prompt: promptText,
    originalTitle: redditQuestion.title,
    originalDescription: redditQuestion.author_description,
    subreddit: redditQuestion.subreddit,
    extraction,
    metadata: {
      positiveScore: redditQuestion.metadata?.predicted?.positiveScore || 0,
      isUpdateRelated: redditQuestion.metadata?.predicted?.isUpdateRelated || false,
      isAboutCve: redditQuestion.isAboutCve || false,
      isAboutLatestUpdate: redditQuestion.isAboutLatestUpdate || false,
    },
    createdAt: redditQuestion.created_utc,
  };
}

// ── Breaking type keyword → API classification.breakingType value ──
// apiType values must exactly match strings in the API's breakingType array.
// Order matters: most specific phrases first.
// apiType values match the exact strings returned by the API's classification.breakingType field.
const BREAKING_TYPE_MAP = [
  { pattern: /\b(critical\s+fail(?:ure)?s?|critical\s+break)\b/i,                                   apiType: "Critical Failure",              queryType: "breaking" },
  { pattern: /\b(limited\s+function(?:ality)?)\b/i,                                                  apiType: "Limited Functionality",         queryType: "breaking" },
  { pattern: /\b(breaking\s+updates?|breaking\s+changes?|breaking\s+releases?)\b/i,                  apiType: "Breaking Update",               queryType: "breaking" },
  { pattern: /\b(network\s+issues?|network\s+failures?|network\s+problems?)\b/i,                     apiType: "Network Issues",                queryType: "breaking" },
  { pattern: /\b(logging.{0,15}monitoring\s*failures?|log\s+failures?|monitoring\s+fails?)\b/i,      apiType: "Logging & Monitoring Failures", queryType: "breaking" },
  { pattern: /\b(config(?:uration)?\s+errors?|config(?:uration)?\s+issues?)\b/i,                     apiType: "Configuration Errors",          queryType: "breaking" },
  { pattern: /\b(data\s+integrity|data\s+corruption|data\s+loss)\b/i,                                apiType: "Data Integrity Issues",         queryType: "breaking" },
  { pattern: /\b(compatibility\s+issues?|compatibility\s+problems?|incompatib)\b/i,                  apiType: "Compatibility Issues",          queryType: "breaking" },
  { pattern: /\b(resource\s+exhaustion|resource\s+limit|out\s+of\s+memory|oom)\b/i,                  apiType: "Resource Exhaustion",           queryType: "breaking" },
  { pattern: /\b(dependency\s+fail(?:ure)?s?|dependency\s+issues?|dependency\s+errors?)\b/i,        apiType: "Dependency Failures",           queryType: "breaking" },
  // Broad — matches any breaking type (apiType null = no subtype filter)
  { pattern: /\b(failures?|breakings?|broke|broken|incidents?|outages?)\b/i,                         apiType: null,                            queryType: "breaking" },
];

// ── NEW: For raw chatbot user text ──
// Popular software searched for broad queries (no entity specified)
const BROAD_SEARCH_SOFTWARE = [
  "GitHub", "Docker", "Kubernetes", "Node", "Chrome", "Firefox",
  "Android", "Windows", "Ubuntu", "PostgreSQL", "MongoDB", "Redis",
  "Nginx", "AWS", "Azure",
];

function generatePromptFromText(userMessage) {
  const extraction = extractEntities(userMessage);

  // Broad query: no entity (or only a low-confidence noun-fallback entity)
  // + has breaking/failure keyword + date filter
  // e.g. "Any critical failures today?" / "Failures this month?"
  const noRealEntity = !extraction.primaryEntity || extraction.primaryEntity.confidence <= 0.55;
  if (noRealEntity) {
    const lower = userMessage.toLowerCase();
    const dateFilter = extractDateFilter(userMessage);
    let breakingMatch = null;
    for (const entry of BREAKING_TYPE_MAP) {
      if (entry.pattern.test(lower)) { breakingMatch = entry; break; }
    }
    const hasCritical = /\b(critical|failure|failures|breaking|incident|outage)\b/.test(lower);
    if ((breakingMatch || hasCritical) && dateFilter) {
      const queryType = breakingMatch?.queryType || "breaking";
      const breakingSubType = breakingMatch?.apiType || null;
      const typeLabel = breakingSubType || "failures";
      return {
        id: Date.now().toString(),
        prompt: `List all "${typeLabel}" across popular software on ${dateFilter.displayDate} (${dateFilter.label})`,
        originalTitle: userMessage,
        originalDescription: userMessage,
        extraction: { primaryEntity: null, entities: { os: [], software: [], device: [], versions: [] }, overallConfidence: 1.0 },
        metadata: {
          positiveScore: 0.85,
          isUpdateRelated: true,
          isBreakingQuery: true,
          breakingSubType,
          queryType,
          dateFilter,
          wantAll: false,
          isBroadQuery: true,  // no entity — search across popular software
        },
        createdAt: new Date().toISOString(),
      };
    }
    // No entity and not a broad failure query — can't process
    if (!extraction.primaryEntity) return null;
  }

  const entity = extraction.primaryEntity;
  const versions = extraction.entities.versions;

  // Detect intent from user message
  const lower = userMessage.toLowerCase();
  const isAboutUpdate = /\b(update|upgrade|latest|new version|changelog)\b/.test(lower);
  const isAboutCve    = /\b(cves?[\s-]?\d*|vulnerabilit(y|ies)|exploit|zero.?day|advisor(y|ies))\b/.test(lower);
  const isAboutPatch  = /\b(patch(es|ed)?|hotfix(es)?|hot.?fix(es)?|bug.?fix(es)?|security.?fix(es)?|security.?patch(es)?|security.?update)\b/.test(lower);
  const isAboutVersion = versions.length > 0 || /\bversion\b/.test(lower);
  // User explicitly wants all/every result, not just the top one
  const wantAll = /\b(all|every|each|list|show\s+all|show\s+every|give\s+(me\s+)?all|any)\b/.test(lower);

  // Match any known breaking type
  let breakingMatch = null;
  for (const entry of BREAKING_TYPE_MAP) {
    if (entry.pattern.test(lower)) { breakingMatch = entry; break; }
  }

  // Derive a single query type — drives structured response and "I don't know" logic
  let queryType = "general";
  if      (isAboutCve)    queryType = "cve";
  else if (isAboutPatch)  queryType = "patch";
  else if (breakingMatch) queryType = breakingMatch.queryType;
  else if (isAboutVersion || isAboutUpdate) queryType = "version";

  const breakingSubType = breakingMatch?.apiType || null;
  const isBreakingQuery = !!breakingMatch;

  // Resolve any relative date expression in the message
  const dateFilter = extractDateFilter(userMessage);
  const dateSuffix = dateFilter ? ` on ${dateFilter.displayDate} (${dateFilter.label})` : "";

  let promptText = "";
  if (isAboutVersion && versions.length > 0) {
    promptText = `What's new in ${entity.name} ${versions[0]}?`;
  } else if (isAboutCve) {
    promptText = `What are the CVEs or security vulnerabilities for ${entity.name}${dateSuffix}?`;
  } else if (isAboutPatch) {
    promptText = `What patches are available for ${entity.name}${dateSuffix}?`;
  } else if (breakingMatch) {
    const typeLabel = breakingMatch.apiType || "failures";
    promptText = wantAll
      ? `List all "${typeLabel}" releases for ${entity.name}${dateSuffix}`
      : `What are the "${typeLabel}" releases for ${entity.name}${dateSuffix}?`;
  } else if (isAboutUpdate) {
    promptText = `What's the latest version of ${entity.name}${dateSuffix}?`;
  } else {
    promptText = `Tell me about ${entity.name}${dateSuffix}`;
  }

  return {
    id: Date.now().toString(),
    prompt: promptText,
    originalTitle: userMessage,
    originalDescription: userMessage,
    extraction,
    metadata: {
      positiveScore: isAboutUpdate || isAboutCve || isAboutPatch || isAboutVersion || isBreakingQuery ? 0.85 : 0.6,
      isUpdateRelated: isAboutUpdate || isAboutVersion || isAboutPatch || isBreakingQuery,
      isAboutCve,
      isAboutPatch,
      isBreakingQuery,
      breakingSubType,   // e.g. "Network Issues", "Configuration Errors", etc.
      isAboutLatestUpdate: isAboutUpdate,
      queryType,
      dateFilter,
      wantAll,
    },
    createdAt: new Date().toISOString(),
  };
}

// ── Relative Date Resolver ──
// Converts temporal keywords into { type, date/from/to, displayDate, label }.
// Supported: latest/newest/most recent/current, today, yesterday, tomorrow,
//            this week, last week, this month, last month, last N days.

function extractDateFilter(text) {
  const lower = text.toLowerCase();
  const now = new Date();

  // Both use UTC so date labels match releasetrain.io's UTC-based dates
  const toYMD = (d) =>
    `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, "0")}${String(d.getUTCDate()).padStart(2, "0")}`;

  const toISO = (d) => d.toISOString().slice(0, 10);

  // "latest" / "newest" / "most recent" / "current" → no date boundary,
  // just return the top entry sorted by date (entries come newest-first from API)
  if (/\b(latest|newest|most\s+recent|current)\b/.test(lower)) {
    return { type: "latest", displayDate: "most recent", label: "latest" };
  }

  if (/\byesterday\b/.test(lower)) {
    const d = new Date(now);
    d.setDate(d.getDate() - 1);
    return { type: "exact", date: toYMD(d), displayDate: toISO(d), label: "yesterday" };
  }

  if (/\btoday\b/.test(lower)) {
    return { type: "exact", date: toYMD(now), displayDate: toISO(now), label: "today" };
  }

  if (/\btomorrow\b/.test(lower)) {
    const d = new Date(now);
    d.setDate(d.getDate() + 1);
    return { type: "exact", date: toYMD(d), displayDate: toISO(d), label: "tomorrow" };
  }

  if (/\bthis\s+week\b/.test(lower)) {
    const start = new Date(now);
    start.setDate(now.getDate() - now.getDay());
    const end = new Date(start);
    end.setDate(start.getDate() + 6);
    return { type: "range", from: toYMD(start), to: toYMD(end), displayDate: `${toISO(start)} to ${toISO(end)}`, label: "this week" };
  }

  if (/\blast\s+week\b/.test(lower)) {
    const end = new Date(now);
    end.setDate(now.getDate() - now.getDay() - 1);
    const start = new Date(end);
    start.setDate(end.getDate() - 6);
    return { type: "range", from: toYMD(start), to: toYMD(end), displayDate: `${toISO(start)} to ${toISO(end)}`, label: "last week" };
  }

  if (/\bthis\s+month\b/.test(lower)) {
    const start = new Date(now.getFullYear(), now.getMonth(), 1);
    const end = new Date(now.getFullYear(), now.getMonth() + 1, 0);
    return { type: "range", from: toYMD(start), to: toYMD(end), displayDate: `${toISO(start)} to ${toISO(end)}`, label: "this month" };
  }

  if (/\blast\s+month\b/.test(lower)) {
    const start = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    const end = new Date(now.getFullYear(), now.getMonth(), 0);
    return { type: "range", from: toYMD(start), to: toYMD(end), displayDate: `${toISO(start)} to ${toISO(end)}`, label: "last month" };
  }

  const lastNDays = lower.match(/\blast\s+(\d+)\s+days?\b/);
  if (lastNDays) {
    const n = parseInt(lastNDays[1], 10);
    const start = new Date(now);
    start.setDate(now.getDate() - n);
    return { type: "range", from: toYMD(start), to: toYMD(now), displayDate: `${toISO(start)} to ${toISO(now)}`, label: `last ${n} days` };
  }

  return null;
}

module.exports = { extractEntities, generatePrompt, generatePromptFromText, extractDateFilter, cleanName, BROAD_SEARCH_SOFTWARE };