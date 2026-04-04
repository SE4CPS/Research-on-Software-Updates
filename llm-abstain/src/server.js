import express from 'express';
import { createClient } from 'redis';
import {
  fetchOSComponents,
  findOSByProductAndDate,
  fetchPatchDetailsByVendorName,
  findPatchDetailsByVendorAndDate,
  formatPatchDetailsFromTags,
} from './releasetrain.js';
import { formatOSResponse } from './schema-os.js';

const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ ok: true, service: 'releasehub-api' });
});

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const VENDORS_NAMES_URL = 'https://releasetrain.io/api/c/names';
const REDIS_KEY_LAST_PROMPTS = 'releasehub:last_prompts';
const REDIS_KEY_ANALYTICS_EVENTS = 'releasehub:analytics_events';
const MAX_LAST_PROMPTS = 3;
const MAX_ANALYTICS_EVENTS = 5000;

let cachedVendorNames = null;
let cachedVendorNamesFetchedAt = 0;
const VENDORS_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
const analyticsEvents = [];

async function safeFetch(url, retries = 2) {
  for (let i = 0; i <= retries; i++) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);

      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timeout);

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      if (i === retries) throw e;
    }
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
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)
  );
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
    redisClient
      .lPush(REDIS_KEY_ANALYTICS_EVENTS, JSON.stringify(event))
      .then(() => redisClient?.lTrim(REDIS_KEY_ANALYTICS_EVENTS, 0, MAX_ANALYTICS_EVENTS - 1))
      .catch((e) => console.warn('Redis analytics event cache write:', e.message));
  }
}

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
    list = list.filter(
      (e) =>
        (e.vendor || '').toLowerCase() === v ||
        (e.vendor || '').toLowerCase().includes(v)
    );
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
      average: latencies.length
        ? Number((latencies.reduce((a, b) => a + b, 0) / latencies.length).toFixed(2))
        : null,
    },
    topVendors,
    topReasons,
    timeBuckets,
    maxVendorCount,
    maxDayCount,
  };
}

function buildKpiBreakdown(list) {
  const queriesAnswered = list.filter((e) => e.status === 'answer').length;
  const queriesAbstain = list.filter((e) => e.status === 'abstain').length;
  const vendorNotFoundCount = list.filter((e) => e.isVendorFound === 0).length;
  const osVersionQueryCount = list.filter((e) => e.flow === 'os').length;
  const patchQueryCount = list.filter((e) => e.flow === 'patch').length;

  return {
    queriesAnswered,
    queriesAbstain,
    respondedCount: queriesAnswered,
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
    redisClient
      .connect()
      .then(() => hydrateAnalyticsEventsFromCache())
      .catch(() => {
        redisClient = null;
      });
  } catch {
    redisClient = null;
  }
}

async function hydrateAnalyticsEventsFromCache() {
  if (!redisClient) return;

  try {
    const cached = await redisClient.lRange(
      REDIS_KEY_ANALYTICS_EVENTS,
      0,
      MAX_ANALYTICS_EVENTS - 1
    );
    if (!Array.isArray(cached) || cached.length === 0) return;

    const parsed = [];
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

  let timeout;

  try {
    const controller = new AbortController();
    timeout = setTimeout(() => controller.abort(), 5000);

    const r = await fetch(VENDORS_NAMES_URL, {
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    if (!r.ok) throw new Error(`Vendors API error: ${r.status}`);

    const data = await r.json();
    const list = Array.isArray(data) ? data : (data?.names ?? data?.data ?? []);

    cachedVendorNames = Array.isArray(list) ? list : [];
    cachedVendorNamesFetchedAt = now;
  } catch (e) {
    console.warn('getVendorNames failed:', e.message);
    cachedVendorNames = cachedVendorNames?.length
      ? cachedVendorNames
      : ['android', 'ios', 'windows', 'firefox', 'chrome', 'linux', 'macos'];
  } finally {
    if (timeout) clearTimeout(timeout);
  }

  return cachedVendorNames;
}

function resolveVendorFromQuestion(question, vendorNames) {
  if (!question) return null;
  const q = String(question).toLowerCase();
  if (!Array.isArray(vendorNames) || vendorNames.length === 0) return null;

  let best = null;
  for (const rawName of vendorNames) {
    if (!rawName) continue;
    const name = String(rawName).trim();
    if (!name) continue;
    const lower = name.toLowerCase();
    if (!lower || lower.length < 2) continue;

    if (q.includes(lower)) {
      if (!best || lower.length > best.toLowerCase().length) {
        best = name;
      }
    }
  }

  return best;
}

app.get('/api/vendors', async (req, res) => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);

    const r = await fetch(VENDORS_NAMES_URL, {
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!r.ok) throw new Error(`Vendors API error: ${r.status}`);
    const data = await r.json();
    const list = Array.isArray(data) ? data : (data?.names ?? data?.data ?? []);
    return res.json(Array.isArray(list) ? list : []);
  } catch (e) {
    console.warn('/api/vendors fallback:', e.message);
    return res.json([
      'android',
      'ios',
      'windows',
      'firefox',
      'chrome',
      'linux',
      'macos',
      'safari',
      'edge',
      'ubuntu'
    ]);
  }
});

app.get('/api/analytics/summary', (req, res) => {
  const rangeDays = Math.min(
    90,
    Math.max(1, parseInt(String(req.query.rangeDays || req.query.range || '1'), 10) || 1)
  );
  const flow = String(req.query.flow || 'all').toLowerCase();
  const status = String(req.query.status || 'all').toLowerCase();
  const vendor = String(req.query.vendor || '').trim();

  res.json({
    source: 'in-memory',
    summary: getAnalyticsSummaryQuery({ rangeDays, flow, status, vendor }),
  });
});

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

function parseDateToYYYYMMDD(question) {
  if (!question) return null;

  let m = question.match(/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/);
  if (m) {
    const [, y, month, day] = m;
    return `${y}${month.padStart(2, '0')}${day.padStart(2, '0')}`;
  }

  m = question.match(/(\d{1,2})[-/](\d{1,2})[-/](\d{4})/);
  if (m) {
    const [, month, day, y] = m;
    return `${y}${month.padStart(2, '0')}${day.padStart(2, '0')}`;
  }

  return null;
}

const DONT_KNOW_ANSWER = "I don't know about the question you asked.";



function parsePatchDetailsVendor(question) {
  const m = question.match(/patch\s+details\s+for\s+(.+?)(?=\s+on\s+|\s*\?$|\s*$)/i);
  return m ? m[1].trim() : null;
}

function normalizeDateYYYYMMDD(value) {
  if (!value) return '';
  return String(value).replace(/\D/g, '').slice(0, 8);
}


function computeIsVendorFoundFlag(rawQuestion, vendorNames) {
  if (!rawQuestion || !Array.isArray(vendorNames) || vendorNames.length === 0) return 0;
  if (resolveVendorFromQuestion(rawQuestion, vendorNames)) return 1;

  const q = String(rawQuestion).toLowerCase();

  const pv = parsePatchVendor(q);
  if (pv) {
    const pvLower = pv.toLowerCase().trim();
    for (const rawName of vendorNames) {
      const nl = String(rawName).toLowerCase().trim();
      if (!nl) continue;
      if (nl === pvLower || nl.includes(pvLower) || pvLower.includes(nl)) return 1;
    }
  }

  const pdv = parsePatchDetailsVendor(q);
  if (pdv) {
    const pdvLower = pdv.toLowerCase().trim();
    for (const rawName of vendorNames) {
      const nl = String(rawName).toLowerCase().trim();
      if (!nl) continue;
      if (nl === pdvLower || nl.includes(pdvLower) || pdvLower.includes(nl)) return 1;
    }
  }

  return 0;
}

function isOsVersionIntent(questionLower, resolvedVendor, parsedDateYYYYMMDD) {
  const q = String(questionLower || '');

  const osKeywords = /(android|ios|windows|macos|linux|ubuntu|operating system|\bos\b)/i;

  return osKeywords.test(q);
}

app.get('/api/debug/trace', async (req, res) => {
  const rawQuestion = String(req.query.question || '').trim();
  const question = rawQuestion.toLowerCase();
  const parsedPatchDetailsVendor = parsePatchDetailsVendor(question);
  
  const dateStr = parseDateToYYYYMMDD(question);

  let branch = 'os';
  if (parsedPatchDetailsVendor) {
    branch = 'patch_details';
  } else if (parsedPatchVendor && parsedPatchVendor !== 'linux') {
    branch = 'patch_non_linux → idk';
  } else if (parsedPatchVendor || (!question.includes('android') && question.includes('linux'))) {
    branch = 'patch_linux';
  }

  const trace = {
    rawQuestion: rawQuestion || null,
    normalizedQuestion: question || null,
    parsedPatchDetailsVendor: parsedPatchDetailsVendor || null,
    parsedPatchVendor: parsedPatchVendor || null,
    parsedDateYYYYMMDD: dateStr || null,
    branch,
  };

  if (req.query.fetch === '1' && branch === 'patch_details') {
    try {
      const items = await fetchPatchDetailsByVendorName(parsedPatchDetailsVendor);
      const docs = findPatchDetailsByVendorAndDate(items, parsedPatchDetailsVendor, dateStr);
      trace.patchDetails = {
        itemsCount: items.length,
        matchedCount: Array.isArray(docs) ? docs.length : 0,
        matchedDocs: Array.isArray(docs)
          ? docs.map((doc) => ({
              _id: doc._id,
              versionProductName: doc.versionProductName,
              versionReleaseDate: doc.versionReleaseDate,
              versionReleaseChannel: doc.versionReleaseChannel,
              versionSearchTags: doc.versionSearchTags ?? null,
              versionReleaseNotes: doc.versionReleaseNotes ?? null,
            }))
          : [],
      };
    } catch (e) {
      trace.patchDetails = { error: e.message };
    }
  }

  res.json(trace);
});

app.post('/answer', async (req, res) => {
  const startedAt = Date.now();
  let analyticsFlow = 'os';
  let analyticsVendor = 'unknown';
  let isVendorFoundFlag = 0;

  let rawQuestion = '';
  let question = '';

  const sendAnswer = (payload, code = 200) => {
    payload.isVendorFound = isVendorFoundFlag;
    payload._debug = {
      rawQuestion,
      parsedVendor: analyticsVendor,
      parsedDateYYYYMMDD: parseDateToYYYYMMDD(rawQuestion),
      isVendorFound: isVendorFoundFlag,
    };

    const status = code >= 500 ? 'error' : (payload?.status || 'unknown');
    const reason =
      status === 'answer'
        ? 'answered'
        : status === 'error'
          ? 'server_error'
          : 'abstain';

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

    return code >= 400 ? res.status(code).json(payload) : res.json(payload);
  };

  try {
    rawQuestion = (
      req.body?.question ??
      req.body?.prompt ??
      req.body?.q ??
      ''
    ).trim();

    question = rawQuestion.toLowerCase();

    await pushLastPrompt(rawQuestion);

    const vendorNames = await getVendorNames();
    if (!vendorNames || vendorNames.length === 0) {
  console.warn('Vendor API down → fallback mode');
}

isVendorFoundFlag = computeIsVendorFoundFlag(rawQuestion, vendorNames);

      
const parsedPatchDetailsVendor = parsePatchDetailsVendor(question);
if (parsedPatchDetailsVendor) {
  analyticsFlow = 'patch';
  analyticsVendor = parsedPatchDetailsVendor;

  const dateStr = parseDateToYYYYMMDD(question);
  const formedVendorUrl = `https://releasetrain.io/api/c/name/${encodeURIComponent(
    String(parsedPatchDetailsVendor || '').trim().toLowerCase()
  )}`;

  console.log('[DEBUG patch_details]', {
    rawQuestion,
    parsedPatchDetailsVendor,
    dateStr,
    formedVendorUrl,
  });

  if (!dateStr) {
    return sendAnswer({
      answer: DONT_KNOW_ANSWER,
      status: 'abstain',
      version: null,
      main: null,
      additional: {},
      vendor: parsedPatchDetailsVendor,
      source: 'ReleaseTrain',
      sourceUrl: null,
      versionSearchTags: null,
      records: [],
      debugTrace: {
        rawQuestion,
        parsedVendor: parsedPatchDetailsVendor,
        parsedDateYYYYMMDD: dateStr,
        formedVendorUrl,
        fetchedItemsCount: 0,
        matchedCount: 0,
      },
    });
  }

  let items;
  try {
    items = await fetchPatchDetailsByVendorName(parsedPatchDetailsVendor);
    console.log('[DEBUG patch_details fetched]', {
      itemsCount: Array.isArray(items) ? items.length : 0,
    });
  } catch (e) {
    return sendAnswer({
      answer: DONT_KNOW_ANSWER,
      status: 'abstain',
      version: null,
      main: null,
      additional: {},
      vendor: parsedPatchDetailsVendor,
      source: 'ReleaseTrain',
      sourceUrl: null,
      versionSearchTags: null,
      records: [],
      debugTrace: {
        rawQuestion,
        parsedVendor: parsedPatchDetailsVendor,
        parsedDateYYYYMMDD: dateStr,
        formedVendorUrl,
        fetchedItemsCount: 0,
        matchedCount: 0,
        fetchError: e.message,
      },
    });
  }

  const docs = findPatchDetailsByVendorAndDate(items, parsedPatchDetailsVendor, dateStr);

  console.log('[DEBUG patch_details matched]', {
    matchedCount: Array.isArray(docs) ? docs.length : 0,
    matchedDocs: Array.isArray(docs)
      ? docs.map((d) => ({
          versionProductName: d.versionProductName,
          versionReleaseDate: d.versionReleaseDate,
          versionSearchTags: d.versionSearchTags,
        }))
      : [],
  });

  if (!docs || docs.length === 0) {
    return sendAnswer({
      answer: DONT_KNOW_ANSWER,
      status: 'abstain',
      version: null,
      main: null,
      additional: {},
      vendor: parsedPatchDetailsVendor,
      source: 'ReleaseTrain',
      sourceUrl: null,
      versionSearchTags: null,
      records: [],
      debugTrace: {
        rawQuestion,
        parsedVendor: parsedPatchDetailsVendor,
        parsedDateYYYYMMDD: dateStr,
        formedVendorUrl,
        fetchedItemsCount: Array.isArray(items) ? items.length : 0,
        matchedCount: 0,
      },
    });
  }

  const formattedList = docs.map((doc) => formatPatchDetailsFromTags(doc));

  return sendAnswer({
    answer: formattedList
      .map(
        (f, i) =>
          `#${i + 1}\n` +
          `Version product name: ${f.additional.versionProductName}\n` +
          `Product type: ${f.additional.productType}\n` +
          `Version number: ${f.additional.versionNumber}\n` +
          `Release note URL: ${f.additional.releaseNoteUrl}`
      )
      .join('\n\n'),
    status: 'answer',
    version: formattedList.map((f) => f.additional.versionNumber),
    main: formattedList[0]?.main ?? null,
    additional: formattedList.map((f) => f.additional),
    vendor: parsedPatchDetailsVendor,
    source: 'ReleaseTrain',
    sourceUrl: formattedList.map((f) => f.additional.releaseNoteUrl),
    versionSearchTags: docs.map((d) => d.versionSearchTags ?? []),
    records: formattedList.map((f, i) => ({
      index: i + 1,
      ...f.additional,
    })),
    debugTrace: {
      rawQuestion,
      parsedVendor: parsedPatchDetailsVendor,
      parsedDateYYYYMMDD: dateStr,
      formedVendorUrl,
      fetchedItemsCount: Array.isArray(items) ? items.length : 0,
      matchedCount: docs.length,
    },
  });
}   

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
      });
    }

    const components = await fetchOSComponents();
    const product = resolvedVendor ? resolvedVendor.toLowerCase() : 'android';
    analyticsFlow = 'os';
    analyticsVendor = resolvedVendor ?? product;

    const doc = findOSByProductAndDate(components, product, dateStr);
    const formatted = formatOSResponse(doc);
    const main = formatted?.versionNumber ?? formatted?.main ?? 'N/A';
    const additional = formatted?.additional ?? {};
    const answerText = formatted
      ? `Version: ${main}. Release notes: ${additional.versionReleaseNotes || 'N/A'}. License: ${additional.versionProductLicense || 'N/A'}.`
      : 'No matching release found for that product or date.';

    const versionSearchTags = Array.isArray(doc?.versionSearchTags)
      ? doc.versionSearchTags
      : Array.isArray(formatted?.raw?.versionSearchTags)
        ? formatted.raw.versionSearchTags
        : null;

    return sendAnswer({
      answer: answerText,
      status: formatted ? 'answer' : 'abstain',
      version: main,
      main,
      additional: {
        versionReleaseNotes: additional.versionReleaseNotes,
        versionProductLicense: additional.versionProductLicense,
        releaseNoteUrl: additional.releaseNoteUrl,
        releaseDate: additional.releaseDate,
        releaseType: additional.releaseType,
      },
      vendor: formatted?.raw?.versionProductName ?? 'Unknown',
      source: 'ReleaseTrain',
      sourceUrl: additional.releaseNoteUrl || null,
      versionSearchTags,
    });
  } catch (e) {
    console.error(e);
    return sendAnswer(
      {
        answer: '',
        status: 'error',
        error: e.message,
      },
      502
    );
  }
});

const PORT = process.env.PORT || 3001;

app.listen(PORT, '0.0.0.0', () => {
  console.log(`ReleaseHub API listening on 0.0.0.0:${PORT}`);
});