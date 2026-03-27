import express from 'express';
import { createClient } from 'redis';
import { GoogleGenerativeAI } from '@google/generative-ai';
import {
  fetchOSComponents,
  findOSByProductAndDate,
  fetchPatchSearch,
  fetchPatchSearchAll,
  findLinuxPatchByDate,
  formatLinuxPatchResponse,
  docHasVendor,
} from './releasetrain.js';
import { formatOSResponse } from './schema-os.js';

const app = express();
app.use(express.json());

/** Health check for load balancers / PaaS (Render, Fly, etc.) */
app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'releasehub-api' });
});

// Allow frontend on another origin (e.g. Vite dev server) to call the API
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const VENDORS_NAMES_URL = 'https://releasetrain.io/api/c/names';
const OS_COMPONENT_SOURCE_URL = 'https://releasetrain.io/api/component?q=os';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY || '';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-1.5-flash';
const GEMINI_TIMEOUT_MS = Math.max(1000, parseInt(String(process.env.GEMINI_TIMEOUT_MS || '12000'), 10) || 12000);
const REDIS_KEY_LAST_PROMPTS = 'releasehub:last_prompts';
const REDIS_KEY_ANALYTICS_EVENTS = 'releasehub:analytics_events';
const MAX_LAST_PROMPTS = 3;
const MAX_ANALYTICS_EVENTS = 5000;

let cachedVendorNames = null;
let cachedVendorNamesFetchedAt = 0;
const VENDORS_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const analyticsEvents = [];
let geminiClient = null;

function getGeminiClient() {
  if (!GEMINI_API_KEY) return null;
  if (!geminiClient) {
    geminiClient = new GoogleGenerativeAI(GEMINI_API_KEY);
  }
  return geminiClient;
}

function runWithTimeout(promise, timeoutMs) {
  return Promise.race([
    promise,
    new Promise((_, reject) => {
      setTimeout(() => reject(new Error('timeout')), timeoutMs);
    }),
  ]);
}

function hasAllRequiredLiterals(text, requiredLiterals) {
  const out = String(text || '');
  for (const lit of requiredLiterals) {
    if (!lit) continue;
    if (!out.includes(lit)) return false;
  }
  return true;
}

async function phraseAnswerWithGemini({ rawQuestion, deterministicAnswer, status, version, sourceUrl, notes, license }) {
  if (status !== 'answer') return { answer: deterministicAnswer, llm: { used: false, reason: 'status_not_answer' } };
  const client = getGeminiClient();
  if (!client) return { answer: deterministicAnswer, llm: { used: false, reason: 'gemini_not_configured' } };

  const base = String(deterministicAnswer || '').trim();
  if (!base) return { answer: deterministicAnswer, llm: { used: false, reason: 'empty_base_answer' } };

  const requiredLiterals = [
    String(version || '').trim() || null,
    String(sourceUrl || '').trim() || null,
  ].filter(Boolean);

  const system = [
    'You are a strict response editor for software release data.',
    'Rewrite only for readability. Do not invent facts.',
    'Keep the same meaning and keep exact factual literals unchanged when present.',
    'If any URL or version string exists in input, keep those exact strings verbatim.',
    'Return plain text only, concise, no markdown.',
  ].join(' ');

  const userPrompt = [
    `User question: ${rawQuestion || ''}`,
    `Deterministic answer: ${base}`,
    `Version: ${version || 'N/A'}`,
    `Source URL: ${sourceUrl || 'N/A'}`,
    `Release notes excerpt: ${notes || 'N/A'}`,
    `License: ${license || 'N/A'}`,
    'Rewrite the deterministic answer in 1-3 short sentences. Keep all factual literals exactly unchanged.',
  ].join('\n');

  try {
    const model = client.getGenerativeModel({ model: GEMINI_MODEL });
    const generation = model.generateContent({
      generationConfig: {
        temperature: 0,
        maxOutputTokens: 220,
      },
      contents: [
        {
          role: 'user',
          parts: [{ text: `${system}\n\n${userPrompt}` }],
        },
      ],
    });
    const result = await runWithTimeout(generation, GEMINI_TIMEOUT_MS);
    const text = result?.response?.text?.()?.trim?.() || '';
    if (!text) return { answer: deterministicAnswer, llm: { used: false, reason: 'empty_llm_output' } };
    if (!hasAllRequiredLiterals(text, requiredLiterals)) {
      return { answer: deterministicAnswer, llm: { used: false, reason: 'llm_output_failed_literal_check' } };
    }
    return { answer: text, llm: { used: true, model: GEMINI_MODEL } };
  } catch (e) {
    return { answer: deterministicAnswer, llm: { used: false, reason: `llm_error:${e?.message || 'unknown'}` } };
  }
}

function getDayKey(ts = Date.now()) {
  const d = new Date(ts);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function percentile(values, p) {
  if (!Array.isArray(values) || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function recordAnalyticsEvent({ status, flow, vendor, latencyMs, reason, isVendorFound }) {
  const event = {
    ts: Date.now(),
    status: status || 'unknown',
    flow: flow || 'unknown',
    vendor: (vendor || 'unknown').toString(),
    latencyMs: Number.isFinite(latencyMs) ? latencyMs : null,
    reason: reason || null,
    isVendorFound: isVendorFound === 1 ? 1 : isVendorFound === 0 ? 0 : null,
  };
  analyticsEvents.push(event);
  if (analyticsEvents.length > MAX_ANALYTICS_EVENTS) {
    analyticsEvents.splice(0, analyticsEvents.length - MAX_ANALYTICS_EVENTS);
  }
  if (redisClient) {
    // Write-through cache for "Recent events" so analytics survives process restarts.
    redisClient
      .lPush(REDIS_KEY_ANALYTICS_EVENTS, JSON.stringify(event))
      .then(() => redisClient?.lTrim(REDIS_KEY_ANALYTICS_EVENTS, 0, MAX_ANALYTICS_EVENTS - 1))
      .catch((e) => console.warn('Redis analytics event cache write:', e.message));
  }
}

/** Last N calendar days (system local date keys), oldest first. */
function getDayRangeKeys(rangeDays) {
  const n = Math.min(90, Math.max(1, Number(rangeDays) || 1));
  const keys = [];
  for (let i = n - 1; i >= 0; i--) {
    keys.push(getDayKey(Date.now() - i * 86400000));
  }
  return keys;
}

function filterAnalyticsEvents({ rangeDays = 1, flow = 'all', status = 'all', vendor = '' }) {
  const dayKeys = getDayRangeKeys(rangeDays);
  const startDay = dayKeys[0];
  const endDay = dayKeys[dayKeys.length - 1];
  let list = analyticsEvents.filter((e) => {
    const d = getDayKey(e.ts);
    return d >= startDay && d <= endDay;
  });
  if (flow && flow !== 'all') list = list.filter((e) => e.flow === flow);
  if (status && status !== 'all') list = list.filter((e) => e.status === status);
  if (vendor && String(vendor).trim()) {
    const v = String(vendor).trim().toLowerCase();
    list = list.filter((e) => (e.vendor || '').toLowerCase() === v || (e.vendor || '').toLowerCase().includes(v));
  }
  return { list, dayKeys };
}

function buildAnalyticsSummaryFromEvents(list, dayKeys) {
  const totalQuestions = list.length;
  const abstainCount = list.filter((e) => e.status === 'abstain').length;
  const answeredCount = list.filter((e) => e.status === 'answer').length;
  const errorCount = list.filter((e) => e.status === 'error').length;
  const latencies = list.map((e) => e.latencyMs).filter((v) => Number.isFinite(v));
  const vendorCounts = new Map();
  for (const e of list) {
    const key = (e.vendor || 'unknown').toString().trim() || 'unknown';
    vendorCounts.set(key, (vendorCounts.get(key) || 0) + 1);
  }
  const topVendors = [...vendorCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([vendor, count]) => ({ vendor, count }));

  const reasonCounts = new Map();
  for (const e of list) {
    if (e.status === 'abstain' || e.status === 'error') {
      const r = e.reason || (e.status === 'error' ? 'error' : 'unknown');
      reasonCounts.set(r, (reasonCounts.get(r) || 0) + 1);
    }
  }
  const topReasons = [...reasonCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([reason, count]) => ({ reason, count }));

  const perDay = Object.fromEntries(dayKeys.map((d) => [d, 0]));
  for (const e of list) {
    const d = getDayKey(e.ts);
    if (d in perDay) perDay[d] += 1;
  }
  const timeBuckets = dayKeys.map((day) => ({ day, count: perDay[day] }));

  const maxVendorCount = topVendors.length ? Math.max(...topVendors.map((v) => v.count), 1) : 1;
  const maxDayCount = timeBuckets.length ? Math.max(...timeBuckets.map((b) => b.count), 1) : 1;

  return {
    totalQuestions,
    answeredCount,
    abstainCount,
    errorCount,
    abstainRate: totalQuestions ? Number(((abstainCount / totalQuestions) * 100).toFixed(2)) : 0,
    answerRate: totalQuestions ? Number(((answeredCount / totalQuestions) * 100).toFixed(2)) : 0,
    latencyMs: {
      p50: percentile(latencies, 50),
      p95: percentile(latencies, 95),
      average: latencies.length ? Number((latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(2)) : null,
    },
    topVendors,
    topReasons,
    timeBuckets,
    maxVendorCount,
    maxDayCount,
  };
}

/** KPI breakdown for the same filtered event list as charts (range + vendor + flow + status). */
function buildKpiBreakdown(list) {
  const queriesAnswered = list.filter((e) => e.status === 'answer').length;
  const queriesAbstain = list.filter((e) => e.status === 'abstain').length;
  const vendorNotFoundCount = list.filter((e) => e.isVendorFound === 0).length;
  const osVersionQueryCount = list.filter((e) => e.flow === 'os').length;
  const patchQueryCount = list.filter((e) => e.flow === 'patch').length;
  return {
    queriesAnswered,
    queriesAbstain,
    /** @deprecated use queriesAnswered */
    respondedCount: queriesAnswered,
    /** @deprecated use queriesAbstain */
    abstainCountKpi: queriesAbstain,
    vendorNotFoundCount,
    osQueryCount: osVersionQueryCount,
    patchQueryCount,
    osVersionQueryCount,
  };
}

function getAnalyticsSummaryQuery(opts) {
  const rangeDays = Math.min(90, Math.max(1, Number(opts.rangeDays) || 1));
  const flowRaw = String(opts.flow ?? 'all').toLowerCase();
  const flow = ['all', 'os', 'patch'].includes(flowRaw) ? flowRaw : 'all';
  const statusRaw = String(opts.status ?? 'all').toLowerCase();
  const status = ['all', 'answer', 'abstain', 'error'].includes(statusRaw) ? statusRaw : 'all';
  const vendor = String(opts.vendor || '').trim();
  const { list, dayKeys } = filterAnalyticsEvents({ rangeDays, flow, status, vendor });
  const summary = buildAnalyticsSummaryFromEvents(list, dayKeys);
  const kpiBreakdown = buildKpiBreakdown(list);
  const recent = [...list]
    .sort((a, b) => b.ts - a.ts)
    .slice(0, 12)
    .map((e) => ({
      ts: e.ts,
      status: e.status,
      flow: e.flow,
      vendor: e.vendor,
      latencyMs: e.latencyMs,
      reason: e.reason,
      isVendorFound: e.isVendorFound,
    }));
  return {
    rangeDays,
    dayFrom: dayKeys[0],
    dayTo: dayKeys[dayKeys.length - 1],
    filters: { flow, status, vendor: vendor || null },
    ...summary,
    ...kpiBreakdown,
    recent,
  };
}

let redisClient = null;
if (process.env.REDIS_URL) {
  try {
    redisClient = createClient({ url: process.env.REDIS_URL });
    redisClient.on('error', (err) => console.warn('Redis:', err.message));
    redisClient.connect()
      .then(() => hydrateAnalyticsEventsFromCache())
      .catch(() => { redisClient = null; });
  } catch {
    redisClient = null;
  }
}

async function hydrateAnalyticsEventsFromCache() {
  if (!redisClient) return;
  try {
    const cached = await redisClient.lRange(REDIS_KEY_ANALYTICS_EVENTS, 0, MAX_ANALYTICS_EVENTS - 1);
    if (!Array.isArray(cached) || cached.length === 0) return;
    const parsed = [];
    // lPush stores newest first; reverse to keep oldest->newest in memory list.
    for (const raw of [...cached].reverse()) {
      try {
        const ev = JSON.parse(raw);
        if (ev && typeof ev === 'object') parsed.push(ev);
      } catch {
        // ignore malformed cache entries
      }
    }
    if (parsed.length > 0) {
      analyticsEvents.splice(0, analyticsEvents.length, ...parsed.slice(-MAX_ANALYTICS_EVENTS));
    }
  } catch (e) {
    console.warn('Redis analytics event cache hydrate:', e.message);
  }
}

async function pushLastPrompt(question) {
  const q = (question || '').trim();
  if (!q || !redisClient) return;
  try {
    await redisClient.lPush(REDIS_KEY_LAST_PROMPTS, q);
    await redisClient.lTrim(REDIS_KEY_LAST_PROMPTS, 0, MAX_LAST_PROMPTS - 1);
  } catch (e) {
    console.warn('Redis pushLastPrompt:', e.message);
  }
}

async function getVendorNames() {
  const now = Date.now();
  if (cachedVendorNames && now - cachedVendorNamesFetchedAt < VENDORS_CACHE_TTL_MS) {
    return cachedVendorNames;
  }
  try {
    if (!r.ok) throw new Error(`Vendors API error: ${r.status}`);
    const data = await r.json();
    const list = Array.isArray(data) ? data : (data?.names ?? data?.data ?? []);
    cachedVendorNames = Array.isArray(list) ? list : [];
    cachedVendorNamesFetchedAt = now;
  } catch (e) {
    console.warn('getVendorNames failed:', e.message);
    cachedVendorNames = cachedVendorNames || [];
  }
  return cachedVendorNames;
}

function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Single-token vendor names from /api/c/names that are too generic to match in free text (e.g. word "os"). */
const AMBIGUOUS_SINGLE_TOKEN_VENDORS = new Set([
  'os',
  'or',
  'go',
  'ai',
  'it',
  'no',
  'ok',
  'pc',
  'me',
  'we',
  'tv',
  'vr',
  'ar',
  'id',
]);

/**
 * Explicit OS product mentioned in the question (lowercase slug for findOSByProductAndDate).
 * Does not include bare "os" — that matched almost any vague sentence with the word "os".
 */
function inferOsProductKeyword(questionLower) {
  const q = String(questionLower || '');
  if (/\blinux-dist\b/.test(q) || /\blinux\s+dist\b/.test(q)) return 'linux-dist';
  if (/\bandroid\b/.test(q)) return 'android';
  if (/\bios\b/.test(q)) return 'ios';
  if (/\bwindows\b/.test(q)) return 'windows';
  if (/\bubuntu\b/.test(q)) return 'ubuntu';
  if (/\bmacos\b/.test(q)) return 'macos';
  if (/\boperating\s+system\b/.test(q)) return 'android';
  if (/\blinux\b/.test(q)) return 'linux';
  return null;
}

const OS_PRODUCT_DISPLAY = {
  android: 'Android',
  ios: 'iOS',
  windows: 'Windows',
  'linux-dist': 'linux-dist',
  ubuntu: 'Ubuntu',
  macos: 'macOS',
  linux: 'Linux',
};

function displayVendorForOsAnswer(resolvedVendor, productSlug) {
  if (resolvedVendor) return resolvedVendor;
  return OS_PRODUCT_DISPLAY[productSlug] || productSlug || 'Unknown';
}

/**
 * Single-token keyboard mash: repeated letters or only 1–2 distinct letters (e.g. gggg, ababab).
 * Skipped when caller already found a date or "patch for …" (structured elsewhere).
 */
function isLowEntropySingleTokenQuery(rawQuestion) {
  const t = String(rawQuestion || '').trim();
  if (!t || /\s/.test(t)) return false;
  const compact = t.toLowerCase();
  if (compact.length < 2 || compact.length > 64) return false;
  if (!/^[a-z0-9_-]+$/i.test(compact)) return false;
  const letters = compact.replace(/[-_0-9]/g, '');
  if (letters.length >= 2) {
    if (/^(.)\1+$/.test(letters)) return true;
    const uniq = new Set(letters).size;
    if (uniq <= 2 && letters.length >= 4) return true;
  } else if (/^\d+$/.test(compact) && compact.length < 8) {
    return true;
  }
  return false;
}

/**
 * True for empty, single-character, or repeated-character noise (e.g. "hhhh", "aaa").
 * Skipped when the query clearly has a date or "patch for …" structure.
 */
function isTrivialOrNonsenseQuery(rawQuestion, questionLower, parsedDate, parsedPatchVendor) {
  if (parsedDate || parsedPatchVendor) return false;
  const t = String(rawQuestion || '').trim();
  if (!t) return true;
  if (t.length <= 1) return true;
  const compact = t.toLowerCase().replace(/\s+/g, '');
  // Same character repeated (aa, gggg, …)
  if (compact.length >= 2 && /^(.)\1+$/.test(compact)) return true;
  if (isLowEntropySingleTokenQuery(rawQuestion)) return true;
  return false;
}

/**
 * Extra guard for single-token keyboard mash like "gdggdgcfbh".
 * Runs before fetching vendor names so we don't call ReleaseTrain `/api/c/names`
 * for obvious junk inputs.
 */
function isSingleTokenConsonantMash(rawQuestion) {
  const t = String(rawQuestion || '').trim();
  if (!t) return false;
  if (/\s/.test(t)) return false; // not a single token
  if (t.length < 6 || t.length > 24) return false;
  if (!/^[a-z0-9]+$/i.test(t)) return false;
  const lower = t.toLowerCase();
  const vowels = (lower.match(/[aeiou]/g) || []).length;
  // If there are essentially no vowels and the token is long enough, treat as gibberish.
  return vowels <= 1 && !/[0-9]/.test(lower);
}

/**
 * Longest vendor name from /api/c/names that appears as a real phrase in the question
 * (word boundaries / token boundaries), not as a substring inside random letters.
 */
function resolveVendorFromQuestion(question, vendorNames) {
  if (!question) return null;
  const q = String(question).toLowerCase();
  if (!Array.isArray(vendorNames) || vendorNames.length === 0) return null;
  const candidates = [...vendorNames]
    .map((raw) => String(raw || '').trim())
    .filter((name) => name.length >= 2);
  candidates.sort((a, b) => b.length - a.length);
  for (const name of candidates) {
    const lower = name.toLowerCase();
    const parts = lower.split(/\s+/).filter(Boolean);
    if (parts.length === 1 && AMBIGUOUS_SINGLE_TOKEN_VENDORS.has(parts[0])) continue;
    const body = parts.map((p) => escapeRegExp(p)).join('\\s+');
    const re = new RegExp(`(^|[^a-z0-9])${body}([^a-z0-9]|$)`, 'i');
    if (re.test(q)) return name;
  }
  return null;
}

/**
 * GET /api/vendors
 * Returns list of component/vendor names from ReleaseTrain.
 */
app.get('/api/vendors', async (req, res) => {
  try {
    const r = await fetch(VENDORS_NAMES_URL, { headers: { Accept: 'application/json' } });
    if (!r.ok) throw new Error(`Vendors API error: ${r.status}`);
    const data = await r.json();
    const list = Array.isArray(data) ? data : (data?.names ?? data?.data ?? []);
    return res.json(list);
  } catch (e) {
    console.error(e);
    return res.status(502).json([]);
  }
});

/**
 * GET /api/analytics/summary
 * In-memory analytics. Query: rangeDays=1|7|30, flow=all|os|patch, status=all|answer|abstain|error, vendor=text
 * KPI cards use the same filtered list as charts (period + vendor + flow + status).
 */
app.get('/api/analytics/summary', (req, res) => {
  const rangeDays = Math.min(90, Math.max(1, parseInt(String(req.query.rangeDays || req.query.range || '1'), 10) || 1));
  const flow = String(req.query.flow || 'all').toLowerCase();
  const status = String(req.query.status || 'all').toLowerCase();
  const vendor = String(req.query.vendor || '').trim();
  res.json({
    source: 'in-memory',
    summary: getAnalyticsSummaryQuery({ rangeDays, flow, status, vendor }),
  });
});

/**
 * GET /api/component?q=os
 * Proxies and shapes OS component data from ReleaseTrain.
 * Response: { main, versionNumber, additional: { versionReleaseNotes, versionProductLicense }, raw }
 */
app.get('/api/component', async (req, res) => {
  const q = (req.query.q || 'os').toLowerCase();
  try {
    if (q !== 'os') {
      return res.json({ error: 'Only q=os is supported', data: [] });
    }
    const components = await fetchOSComponents();
    const doc = findOSByProductAndDate(components, 'android', null);
    const formatted = formatOSResponse(doc);
    return res.json({ ...(formatted || {}), components });
  } catch (e) {
    console.error(e);
    return res.status(502).json({ error: e.message, main: null, additional: {} });
  }
});

/**
 * GET /api/debug/trace?question=...&fetch=1
 * Shows how the question is parsed and which branch (patch vs OS) would run.
 * Add fetch=1 to call ReleaseTrain and include itemsCount / first doc tags (slow).
 */
app.get('/api/debug/trace', async (req, res) => {
  const rawQuestion = String(req.query.question || '').trim();
  const question = rawQuestion.toLowerCase();
  const parsedPatchVendor = parsePatchVendor(question);
  const dateStr = parseDateToYYYYMMDD(question);
  let branch = 'os';
  if (parsedPatchVendor && parsedPatchVendor !== 'linux') {
    branch = 'patch_non_linux → idk';
  } else if (parsedPatchVendor || (!question.includes('android') && question.includes('linux'))) {
    branch = 'patch_linux';
  }
  const trace = {
    rawQuestion: rawQuestion || null,
    normalizedQuestion: question || null,
    parsedPatchVendor: parsedPatchVendor || null,
    parsedDateYYYYMMDD: dateStr || null,
    branch,
    hints: {
      patchVendorRegex: '/patch\\s+for\\s+([^,.?]+?)(?=\\s+on\\s+|\\s*[.?]|\\s*$)/i',
      dateRegex: '(MM)-(DD)-(YYYY) or (MM)/(DD)/(YYYY) → YYYYMMDD',
    },
  };
  if (req.query.fetch === '1' && branch.startsWith('patch_linux')) {
    const vendor = parsedPatchVendor || 'linux';
    try {
      const items = await fetchPatchSearch(vendor);
      const searchUrl = `https://releasetrain.io/api/v/search?q=${encodeURIComponent(vendor)}&channel=patch&limit=25&page=1`;
      const doc = findLinuxPatchByDate(items, dateStr);
      trace.patchSearch = {
        searchUrl,
        vendor,
        itemsCount: items.length,
        firstItemTags: items[0]?.versionSearchTags ?? null,
        firstItemProductName: items[0]?.versionProductName ?? null,
        matchedDocHasVendor: doc ? docHasVendor(doc, vendor) : null,
        matchedDocTags: doc?.versionSearchTags ?? null,
      };
    } catch (e) {
      trace.patchSearch = { error: e.message };
    }
  } else if (req.query.fetch === '1' && branch === 'os') {
    try {
      const components = await fetchOSComponents();
      const vendorNamesTrace = await getVendorNames();
      const resolvedV = resolveVendorFromQuestion(rawQuestion, vendorNamesTrace);
      let product = resolvedV ? resolvedV.toLowerCase() : inferOsProductKeyword(question);
      if (!product && dateStr) product = 'android';
      const doc = product
        ? findOSByProductAndDate(components, product, dateStr)
        : null;
      trace.os = {
        componentsCount: components.length,
        product,
        matchedDocSummary: doc
          ? {
              versionProductName: doc.versionProductName,
              versionReleaseDate: doc.versionReleaseDate,
              versionSearchTags: doc.versionSearchTags ?? null,
            }
          : null,
      };
    } catch (e) {
      trace.os = { error: e.message };
    }
  }
  res.json(trace);
});

/**
 * Parse date from prompt (e.g. "02-14-2026" or "2-14-2026") to YYYYMMDD.
 * @returns {string|null} e.g. "20260214" or null
 */
function parseDateToYYYYMMDD(question) {
  const dateMatch = question.match(/(\d{1,2})[-/](\d{1,2})[-/](\d{4})/);
  if (!dateMatch) return null;
  const [, m, d, y] = dateMatch;
  return `${y}${m.padStart(2, '0')}${d.padStart(2, '0')}`;
}

const DONT_KNOW_ANSWER = "I don't know about the question you asked.";

/**
 * Parse vendor from "patch for X" / "patch for X on date" in the question.
 * @returns {string|null} e.g. "Roblox", "Linux", or null
 */
function parsePatchVendor(question) {
  const m = question.match(/patch\s+for\s+([^,.?]+?)(?=\s+on\s+|\s*[.?]|\s*$)/i);
  return m ? m[1].trim() : null;
}

function normalizeDateYYYYMMDD(value) {
  if (!value) return '';
  return String(value).replace(/\D/g, '').slice(0, 8);
}

function isVendorInNames(vendor, vendorNames) {
  const v = String(vendor || '').trim().toLowerCase();
  if (!v || !Array.isArray(vendorNames)) return false;
  return vendorNames.some((name) => {
    const n = String(name || '').trim().toLowerCase();
    if (!n) return false;
    return n === v || n.includes(v) || v.includes(n);
  });
}

function findPatchDocInOSComponents(components, vendor, dateStr = null) {
  if (!Array.isArray(components) || components.length === 0) return null;
  const vendorNorm = String(vendor || '').trim().toLowerCase();
  const dateNorm = normalizeDateYYYYMMDD(dateStr);
  const candidates = components.filter((doc) => {
    const tags = Array.isArray(doc?.versionSearchTags)
      ? doc.versionSearchTags.map((t) => String(t).toLowerCase())
      : [];
    const name = String(doc?.versionProductName || '').toLowerCase();
    const brand = String(doc?.versionProductBrand || '').toLowerCase();
    const hasVendor = tags.some((t) => t === vendorNorm || t.includes(vendorNorm))
      || name === vendorNorm || name.includes(vendorNorm)
      || brand === vendorNorm || brand.includes(vendorNorm);
    const hasPatchTag = tags.includes('patch');
    return hasVendor && hasPatchTag;
  });
  if (candidates.length === 0) return null;
  if (dateNorm) {
    const dated = candidates.find((doc) => normalizeDateYYYYMMDD(doc?.versionReleaseDate) === dateNorm);
    if (dated) return dated;
  }
  const sorted = [...candidates].sort((a, b) => (b?.versionTimestamp || 0) - (a?.versionTimestamp || 0));
  return sorted[0] ?? null;
}

function findPatchByVendorAndDateFromTags(items, vendor, dateStr = null) {
  if (!Array.isArray(items) || items.length === 0) return null;
  const vendorNorm = String(vendor || '').trim().toLowerCase();
  const dateNorm = normalizeDateYYYYMMDD(dateStr);
  const candidates = items.filter((doc) => {
    const tags = Array.isArray(doc?.versionSearchTags)
      ? doc.versionSearchTags.map((t) => String(t).toLowerCase())
      : [];
    if (tags.length === 0) return false;
    const firstTag = tags[0] || '';
    const hasPatchTag = tags.includes('patch');
    // Avoid vendorNorm.includes(shortFirstTag): e.g. "fdcvdsvd".includes("os") would match wrongly.
    const vendorMatches =
      firstTag === vendorNorm
      || (vendorNorm.length >= 2 && firstTag.includes(vendorNorm))
      || (firstTag.length >= 4 && vendorNorm.includes(firstTag));
    return hasPatchTag && vendorMatches;
  });
  if (candidates.length === 0) return null;
  if (dateNorm) {
    const exactByTag = candidates.find((doc) => {
      const tags = Array.isArray(doc?.versionSearchTags)
        ? doc.versionSearchTags.map((t) => String(t))
        : [];
      return tags.includes(dateNorm);
    });
    if (exactByTag) return exactByTag;
    const exactByField = candidates.find((doc) => normalizeDateYYYYMMDD(doc?.versionReleaseDate) === dateNorm);
    if (exactByField) return exactByField;
  }
  const sorted = [...candidates].sort((a, b) => (b?.versionTimestamp || 0) - (a?.versionTimestamp || 0));
  return sorted[0] ?? null;
}

/**
 * 1 if the query matches a vendor/component name from ReleaseTrain /api/c/names, else 0 (for debug).
 */
function computeIsVendorFoundFlag(rawQuestion, vendorNames) {
  if (!rawQuestion || !Array.isArray(vendorNames) || vendorNames.length === 0) return 0;
  if (resolveVendorFromQuestion(rawQuestion, vendorNames)) return 1;
  const q = String(rawQuestion).toLowerCase();
  const pv = parsePatchVendor(q);
  if (pv) {
    const pvLower = pv.toLowerCase().trim();
    if (pvLower) {
      for (const rawName of vendorNames) {
        const nl = String(rawName).toLowerCase().trim();
        if (!nl) continue;
        if (nl === pvLower || nl.includes(pvLower)) return 1;
        // Avoid pvLower.includes("os") etc. when nl is a short substring of gibberish.
        if (nl.length >= 4 && pvLower.includes(nl)) return 1;
      }
    }
  }
  return 0;
}

/**
 * True when the question looks like an OS/version lookup intent.
 * Uses explicit product phrases only — not bare "os" (too many false positives on vague text).
 */
function isOsVersionIntent(questionLower, resolvedVendor, parsedDateYYYYMMDD) {
  const q = String(questionLower || '');
  const hasDate = Boolean(parsedDateYYYYMMDD);
  const inferredOsProduct = inferOsProductKeyword(q);
  const hasOsKeyword = Boolean(inferredOsProduct);
  if (hasDate || hasOsKeyword) return true;

  // If we only matched a vendor phrase, do NOT treat it as a valid OS/version request
  // unless the query explicitly asks for version/release/etc.
  if (resolvedVendor) {
    return /\b(version|latest|release|released|update|firmware|build|license|notes?)\b/i.test(q);
  }

  return false;
}

/**
 * True when the query has enough structure to run patch/OS ReleaseTrain logic.
 * Blocks random tokens (e.g. "fdcvdsvd") that would otherwise match patch tags via substring rules.
 */
function hasStructuredReleaseIntent(rawQuestion, questionLower, vendorNames) {
  const q = String(questionLower || '');
  if (parsePatchVendor(q)) return true;
  if (parseDateToYYYYMMDD(q)) return true;
  if (inferOsProductKeyword(q)) return true;
  if (q.includes('linux')) return true;
  if (Array.isArray(vendorNames) && vendorNames.length > 0 && resolveVendorFromQuestion(String(rawQuestion || ''), vendorNames)) {
    return true;
  }
  return false;
}

/**
 * POST /answer
 * Accepts { question } and returns main + additional from ReleaseTrain.
 * - Linux patch: "What is the patch for Linux on 02-14-2026?" → main = last of versionSearchTags, additional = versionUrl, versionReleaseNotes.
 * - OS/Android: "What is the version of OS Android on 02-14-2026?" → main = versionNumber, additional = versionReleaseNotes, versionProductLicense.
 */
app.post('/answer', async (req, res) => {
  const startedAt = Date.now();
  let analyticsFlow = 'os';
  let analyticsVendor = 'unknown';
  let isVendorFoundFlag = 0;
  const sendAnswer = (payload, code = 200) => {
    payload.isVendorFound = isVendorFoundFlag;
    // User rule: if vendor name was NOT found in ReleaseTrain `/api/c/names`,
    // force abstain even if the backend found a document via other heuristics.
    if (code < 500 && isVendorFoundFlag === 0) {
      payload = {
        ...payload,
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        sourceUrl: null,
        versionSearchTags: null,
      };
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      payload.isVendorFound = 0;
    }
    payload._debugFormation = {
      ...(payload._debugFormation && typeof payload._debugFormation === 'object' ? payload._debugFormation : {}),
      isVendorFound: isVendorFoundFlag,
    };
    const status = code >= 500 ? 'error' : (payload?.status || 'unknown');
    let reason = payload?._debugFormation?.reason ?? null;
    if (!reason && status === 'abstain' && analyticsFlow === 'patch' && payload?._debugFormation?.step === 'fetchPatchSearch') {
      reason = 'patch_api_error';
    }
    if (!reason && status === 'abstain' && analyticsFlow === 'os') {
      reason = 'os_no_match';
    }
    if (!reason && status === 'answer') reason = 'answered';
    if (!reason && status === 'error') reason = 'server_error';
    try {
      recordAnalyticsEvent({
        status,
        flow: analyticsFlow,
        vendor: analyticsVendor,
        latencyMs: Date.now() - startedAt,
        reason,
        isVendorFound: isVendorFoundFlag,
      });
    } catch (analyticsErr) {
      console.warn('recordAnalyticsEvent:', analyticsErr?.message);
    }
    try {
      return code >= 400 ? res.status(code).json(payload) : res.json(payload);
    } catch (jsonErr) {
      console.error('sendAnswer JSON error:', jsonErr);
      if (!res.headersSent) {
        return res.status(500).json({ status: 'error', error: 'Response serialization failed', answer: '' });
      }
      return undefined;
    }
  };
  try {
    // Accept multiple possible field names for the user question to avoid
    // silently falling back to the default Android answer when the client
    // sends e.g. { prompt: "..." } instead of { question: "..." }.
    const rawQuestion = (
      req.body?.question ??
      req.body?.prompt ??
      req.body?.q ??
      ''
    ).trim();
    await pushLastPrompt(rawQuestion);
    const question = (rawQuestion || '').toLowerCase();
    const preDateStr = parseDateToYYYYMMDD(question);
    const prePatchVendor = parsePatchVendor(question);
    if (isTrivialOrNonsenseQuery(rawQuestion, question, preDateStr, prePatchVendor)) {
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      isVendorFoundFlag = 0;
      return sendAnswer({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: { rawQuestion, normalizedQuestion: question, flow: 'unknown' },
          reason: 'vague_or_nonsense_query',
        },
      });
    }

    // Fast abstain: avoid fetching vendor names when the input is clearly gibberish
    // and there is no patch/date/explicit OS keyword evidence.
    const inferredOsProduct = inferOsProductKeyword(question);
    if (
      !preDateStr &&
      !prePatchVendor &&
      !inferredOsProduct &&
      !question.includes('linux') &&
      isSingleTokenConsonantMash(rawQuestion)
    ) {
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      isVendorFoundFlag = 0;
      return sendAnswer({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: { rawQuestion, normalizedQuestion: question, flow: 'unknown' },
          reason: 'no_structured_release_intent_gibberish_fast_abstain',
        },
      });
    }

    const vendorNames = await getVendorNames();
    if (!hasStructuredReleaseIntent(rawQuestion, question, vendorNames)) {
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      isVendorFoundFlag = 0;
      return sendAnswer({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: { rawQuestion, normalizedQuestion: question, flow: 'unknown' },
          reason: 'no_structured_release_intent',
        },
      });
    }

    isVendorFoundFlag = computeIsVendorFoundFlag(rawQuestion, vendorNames);

    if (process.env.DEBUG) {
      console.log('[answer] question:', rawQuestion);
    }

    // Queries containing "Linux" are answered ONLY from the patch search API:
    // https://releasetrain.io/api/v/search?q=linux&channel=patch&limit=25&page=1
    // (never from the OS component API).
    // Patch flow: "patch for X [on date]" or any question that contains "linux".
    // - Only Linux patch is supported via ReleaseTrain today.
    // - If user asks for a patch for some other vendor (e.g. Roblox), we answer "I don't know about the question you asked."
    const parsedPatchVendor = parsePatchVendor(question);
    const isLinuxVendor = (v) => (v || '').toLowerCase().trim() === 'linux';
    if (parsedPatchVendor) {
      analyticsFlow = 'patch';
      analyticsVendor = parsedPatchVendor;
      if (!isVendorInNames(parsedPatchVendor, vendorNames)) {
        return sendAnswer({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: parsedPatchVendor,
          source: 'ReleaseTrain',
          sourceUrl: VENDORS_NAMES_URL,
          versionSearchTags: null,
          _debugFormation: {
            step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: parsedPatchVendor },
            reason: 'patch_vendor_not_in_names',
          },
        });
      }
      const dateStr = parseDateToYYYYMMDD(question);
      let patchItems;
      try {
        patchItems = await fetchPatchSearchAll();
      } catch (e) {
        return sendAnswer({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: parsedPatchVendor,
          source: 'ReleaseTrain',
          sourceUrl: 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1',
          versionSearchTags: null,
          _debugFormation: {
            step: 'fetchPatchSearchAll',
            error: e.message,
            reason: 'patch_api_error',
          },
        });
      }
      const patchDoc = findPatchByVendorAndDateFromTags(patchItems, parsedPatchVendor, dateStr);
      if (!patchDoc) {
        return sendAnswer({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: parsedPatchVendor,
          source: 'ReleaseTrain',
          sourceUrl: 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1',
          versionSearchTags: null,
          _debugFormation: {
            step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: parsedPatchVendor, parsedDateYYYYMMDD: dateStr },
            step2_dataFetched: { sourceUrl: 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1', itemsCount: patchItems?.length ?? 0 },
            reason: 'patch_not_found_in_search_tags',
          },
        });
      }
      const tags = Array.isArray(patchDoc.versionSearchTags) ? patchDoc.versionSearchTags : [];
      const main = patchDoc.versionNumber ?? (tags.length > 0 ? String(tags[tags.length - 1]) : 'N/A');
      const deterministicAnswer = `Patch version: ${main}\n${patchDoc.versionUrl ? `URL: ${patchDoc.versionUrl}\n` : ''}Release notes: ${(patchDoc.versionReleaseNotes || 'N/A').slice(0, 200)}${(patchDoc.versionReleaseNotes || '').length > 200 ? '…' : ''}`;
      const phrased = await phraseAnswerWithGemini({
        rawQuestion,
        deterministicAnswer,
        status: 'answer',
        version: main,
        sourceUrl: patchDoc.versionUrl ?? 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1',
        notes: patchDoc.versionReleaseNotes ?? null,
        license: null,
      });
      return sendAnswer({
        answer: phrased.answer,
        status: 'answer',
        version: main,
        main,
        additional: {
          versionReleaseNotes: patchDoc.versionReleaseNotes ?? null,
          versionProductLicense: null,
          versionUrl: patchDoc.versionUrl ?? null,
        },
        vendor: patchDoc.versionProductName ?? parsedPatchVendor,
        source: 'ReleaseTrain',
        sourceUrl: 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1',
        versionSearchTags: tags,
        _debugFormation: {
          llm: phrased.llm,
          step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: parsedPatchVendor, parsedDateYYYYMMDD: dateStr },
          step2_dataFetched: { sourceUrl: 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1', itemsCount: patchItems?.length ?? 0 },
          step3_matchedDoc: {
            _id: patchDoc._id,
            versionId: patchDoc.versionId,
            versionReleaseDate: patchDoc.versionReleaseDate,
            versionSearchTags: patchDoc.versionSearchTags ?? null,
          },
        },
      });
    }

    // Use patch API only: when "patch for Linux" or when query contains "linux" (and not another patch vendor).
    const queryMentionsLinux = question.includes('linux');
    const patchVendor =
      (parsedPatchVendor && isLinuxVendor(parsedPatchVendor)) || (queryMentionsLinux && !question.includes('android'))
        ? 'linux'
        : parsedPatchVendor ?? null;
    if (patchVendor) {
      analyticsFlow = 'patch';
      analyticsVendor = patchVendor;
      const dateStr = parseDateToYYYYMMDD(question);
      // Linux patch queries are served only from this API (never from /api/component?q=os).
      const searchUrl = `https://releasetrain.io/api/v/search?q=${encodeURIComponent(patchVendor)}&channel=patch&limit=25&page=1`;
      let items;
      try {
        items = await fetchPatchSearch(patchVendor);
      } catch (e) {
        if (process.env.DEBUG) console.log('[answer] Patch search failed:', e.message);
        return sendAnswer({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: patchVendor,
          source: 'ReleaseTrain',
          sourceUrl: searchUrl,
          versionSearchTags: null,
          _debugFormation: { step: 'fetchPatchSearch', error: e.message, reason: 'patch_api_error' },
        });
      }
      if (!items || items.length === 0) {
        return sendAnswer({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: patchVendor,
          source: 'ReleaseTrain',
          sourceUrl: searchUrl,
          versionSearchTags: null,
          _debugFormation: {
            step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: patchVendor, parsedDateYYYYMMDD: dateStr },
            step2_dataFetched: { sourceUrl: searchUrl, itemsCount: 0 },
            reason: 'no_results',
          },
        });
      }
      const doc = findLinuxPatchByDate(items, dateStr);
      if (!doc || !docHasVendor(doc, patchVendor)) {
        return sendAnswer({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: patchVendor,
          source: 'ReleaseTrain',
          sourceUrl: searchUrl,
          versionSearchTags: doc?.versionSearchTags ?? null,
          _debugFormation: {
            step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: patchVendor, parsedDateYYYYMMDD: dateStr },
            step2_dataFetched: { sourceUrl: searchUrl, itemsCount: items?.length ?? 0 },
            step3_matchedDoc: doc ? { _id: doc._id, versionSearchTags: doc.versionSearchTags } : null,
            reason: 'versionSearchTags_does_not_have_provided_vendor',
          },
        });
      }
      const formatted = formatLinuxPatchResponse(doc);
      const main = formatted?.main ?? 'N/A';
      const additional = formatted?.additional ?? {};
      const notes = additional.versionReleaseNotes || 'N/A';
      const url = additional.versionUrl || '';
      const versionSearchTags = Array.isArray(doc?.versionSearchTags)
        ? doc.versionSearchTags
        : Array.isArray(formatted?.raw?.versionSearchTags)
          ? formatted.raw.versionSearchTags
          : null;
      const deterministicAnswer = `Patch version: ${main}\n${url ? `URL: ${url}\n` : ''}Release notes: ${notes.slice(0, 200)}${notes.length > 200 ? '…' : ''}`;
      const phrased = await phraseAnswerWithGemini({
        rawQuestion,
        deterministicAnswer,
        status: formatted ? 'answer' : 'abstain',
        version: main,
        sourceUrl: url || searchUrl,
        notes,
        license: null,
      });
      const payload = {
        answer: phrased.answer,
        status: formatted ? 'answer' : 'abstain',
        version: main,
        main,
        additional: {
          versionReleaseNotes: additional.versionReleaseNotes,
          versionProductLicense: null,
          versionUrl: additional.versionUrl,
        },
        vendor: formatted?.raw?.versionProductName ?? patchVendor,
        source: 'ReleaseTrain',
        sourceUrl: searchUrl,
        versionSearchTags,
      };
      payload._debugFormation = {
        llm: phrased.llm,
        versionSearchTags: { value: versionSearchTags, fromDoc: doc?.versionSearchTags ?? null },
        step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: patchVendor, parsedDateYYYYMMDD: dateStr },
        step2_dataFetched: { sourceUrl: searchUrl, itemsCount: items?.length ?? 0 },
        step3_matchedDoc: doc ? { _id: doc._id, versionSearchTags: doc.versionSearchTags } : null,
        step4_formatted: { main: formatted?.main ?? null },
      };
      if (process.env.DEBUG) console.log('[answer] Patch flow →', JSON.stringify(payload, null, 2));
      return sendAnswer(payload);
    }

    // OS flow: detect vendor from ReleaseTrain vendor names and match by date (or latest)
    const dateStr = parseDateToYYYYMMDD(question);
    const resolvedVendor = resolveVendorFromQuestion(rawQuestion, vendorNames);
    if (!isOsVersionIntent(question, resolvedVendor, dateStr)) {
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      return sendAnswer({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: { rawQuestion, normalizedQuestion: question, flow: 'unknown' },
          reason: 'unknown_intent_not_patch_or_os_version',
        },
      });
    }
    let product = resolvedVendor ? resolvedVendor.toLowerCase() : inferOsProductKeyword(question);
    if (!product && dateStr) product = 'android';
    if (!product) {
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      return sendAnswer({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: {
            rawQuestion,
            normalizedQuestion: question,
            flow: 'unknown',
            parsedDateYYYYMMDD: dateStr ?? null,
            resolvedVendor: resolvedVendor ?? null,
          },
          reason: 'os_no_resolved_vendor_or_product_keyword',
        },
      });
    }
    if (isLowEntropySingleTokenQuery(rawQuestion)) {
      analyticsFlow = 'unknown';
      analyticsVendor = 'unknown';
      isVendorFoundFlag = 0;
      return sendAnswer({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: 'Unknown',
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: { rawQuestion, normalizedQuestion: question, flow: 'unknown' },
          reason: 'low_entropy_gibberish_query',
        },
      });
    }
    const components = await fetchOSComponents();
    analyticsFlow = 'os';
    analyticsVendor = resolvedVendor ?? product;
    const doc = findOSByProductAndDate(components, product, dateStr);
    const formatted = formatOSResponse(doc);
    const main = formatted?.versionNumber ?? formatted?.main ?? 'N/A';
    const additional = formatted?.additional ?? {};
    const deterministicAnswerText = formatted
      ? `Version: ${main}. Release notes: ${additional.versionReleaseNotes || 'N/A'}. License: ${additional.versionProductLicense || 'N/A'}.`
      : 'No matching release found for that product or date.';
    const phrased = await phraseAnswerWithGemini({
      rawQuestion,
      deterministicAnswer: deterministicAnswerText,
      status: formatted ? 'answer' : 'abstain',
      version: main,
      sourceUrl: OS_COMPONENT_SOURCE_URL,
      notes: additional.versionReleaseNotes || null,
      license: additional.versionProductLicense || null,
    });
    let versionSearchTags = Array.isArray(doc?.versionSearchTags)
      ? doc.versionSearchTags
      : Array.isArray(formatted?.raw?.versionSearchTags)
        ? formatted.raw.versionSearchTags
        : null;
    if (doc && versionSearchTags == null && typeof doc === 'object') {
      versionSearchTags = doc.version_search_tags ?? doc.versionSearchTags ?? null;
    }
    const payload = {
      answer: phrased.answer,
      status: formatted ? 'answer' : 'abstain',
      version: main,
      main,
      additional: {
        versionReleaseNotes: additional.versionReleaseNotes,
        versionProductLicense: additional.versionProductLicense,
      },
      vendor: displayVendorForOsAnswer(resolvedVendor, product),
      source: 'ReleaseTrain',
      sourceUrl: OS_COMPONENT_SOURCE_URL,
      versionSearchTags,
    };
    if (doc && versionSearchTags == null && typeof doc === 'object') {
      payload._debugDocKeys = Object.keys(doc);
    }
    payload._debugFormation = {
      llm: phrased.llm,
      ...(formatted ? {} : { reason: 'os_no_match' }),
      versionSearchTags: {
        value: versionSearchTags,
        source: doc?.versionSearchTags != null ? 'matchedDoc.versionSearchTags' : formatted?.raw?.versionSearchTags != null ? 'formatted.raw.versionSearchTags' : 'null',
        fromDoc: doc?.versionSearchTags ?? null,
        fromFormattedRaw: formatted?.raw?.versionSearchTags ?? null,
      },
      step1_parsedInput: {
        rawQuestion,
        normalizedQuestion: question,
        flow: 'os',
        parsedDateYYYYMMDD: dateStr ?? null,
        product,
      },
      step2_dataFetched: {
        sourceUrl: OS_COMPONENT_SOURCE_URL,
        componentsCount: components?.length ?? 0,
      },
      step3_matchedDoc: doc
        ? {
            _id: doc._id,
            versionId: doc.versionId,
            versionReleaseDate: doc.versionReleaseDate,
            versionNumber: doc.versionNumber,
            versionProductName: doc.versionProductName,
            versionSearchTags: doc.versionSearchTags ?? null,
          }
        : null,
      step4_formatted: {
        main: formatted?.main ?? null,
        versionNumber: formatted?.versionNumber ?? null,
        additionalKeys: formatted?.additional ? Object.keys(formatted.additional) : [],
      },
      step5_finalResponse: {
        version: payload.version,
        main: payload.main,
        answerPreview: payload.answer?.slice(0, 80) + (payload.answer?.length > 80 ? '…' : ''),
      },
    };
    if (process.env.DEBUG) console.log('[answer] OS flow →', JSON.stringify(payload, null, 2));
    return sendAnswer(payload);
  } catch (e) {
    console.error(e);
    return sendAnswer({
      answer: '',
      status: 'error',
      error: e.message,
    }, 502);
  }
});

const PORT = process.env.PORT || 3000;
// Bind all interfaces (required for Docker, Render, Fly, Railway, etc.)
app.listen(PORT, '0.0.0.0', () => {
  console.log(`ReleaseHub API listening on 0.0.0.0:${PORT}`);
});
