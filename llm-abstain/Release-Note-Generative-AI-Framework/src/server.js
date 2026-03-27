import express from 'express';
import { createClient } from 'redis';
import {
  fetchOSComponents,
  findOSByProductAndDate,
  fetchPatchSearch,
  findLinuxPatchByDate,
  formatLinuxPatchResponse,
  docHasVendor,
} from './releasetrain.js';
import { formatOSResponse } from './schema-os.js';

const app = express();
app.use(express.json());

// Allow frontend on another origin (e.g. Vite dev server) to call the API
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  next();
});

const VENDORS_NAMES_URL = 'https://releasetrain.io/api/c/names';
const REDIS_KEY_LAST_PROMPTS = 'releasehub:last_prompts';
const MAX_LAST_PROMPTS = 3;

let cachedVendorNames = null;
let cachedVendorNamesFetchedAt = 0;
const VENDORS_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

let redisClient = null;
if (process.env.REDIS_URL) {
  try {
    redisClient = createClient({ url: process.env.REDIS_URL });
    redisClient.on('error', (err) => console.warn('Redis:', err.message));
    redisClient.connect().catch(() => { redisClient = null; });
  } catch {
    redisClient = null;
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
    const r = await fetch(VENDORS_NAMES_URL, { headers: { Accept: 'application/json' } });
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
      dateRegex: '(\\d{1,2})[-/](\\d{1,2})[-/](\\d{4}) → YYYYMMDD',
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
      const product = question.includes('android')
        ? 'android'
        : question.includes('ios')
          ? 'ios'
          : question.includes('windows')
            ? 'windows'
            : 'android';
      const doc = findOSByProductAndDate(components, product, dateStr);
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
 * Parse date from prompt (e.g. "14-02-2026" or "2-02-2026") to YYYYMMDD.
 * @returns {string|null} e.g. "20260214" or null
 */
function parseDateToYYYYMMDD(question) {
  const dateMatch = question.match(/(\d{1,2})[-/](\d{1,2})[-/](\d{4})/);
  if (!dateMatch) return null;
  const [, d, m, y] = dateMatch;
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

/**
 * POST /answer
 * Accepts { question } and returns main + additional from ReleaseTrain.
 * - Linux patch: "What is the patch for Linux on 14-02-2026?" → main = last of versionSearchTags, additional = versionUrl, versionReleaseNotes.
 * - OS/Android: "What is the version of OS Android on 2-02-2026?" → main = versionNumber, additional = versionReleaseNotes, versionProductLicense.
 */
app.post('/answer', async (req, res) => {
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
    if (parsedPatchVendor && !isLinuxVendor(parsedPatchVendor)) {
      res.json({
        answer: DONT_KNOW_ANSWER,
        status: 'abstain',
        version: null,
        main: null,
        additional: {},
        vendor: parsedPatchVendor,
        source: 'ReleaseTrain',
        sourceUrl: null,
        versionSearchTags: null,
        _debugFormation: {
          step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: parsedPatchVendor },
          reason: 'patch_for_non_linux_vendor_not_supported',
        },
      });
      return;
    }

    // Use patch API only: when "patch for Linux" or when query contains "linux" (and not another patch vendor).
    const queryMentionsLinux = question.includes('linux');
    const patchVendor =
      (parsedPatchVendor && isLinuxVendor(parsedPatchVendor)) || (queryMentionsLinux && !question.includes('android'))
        ? 'linux'
        : parsedPatchVendor ?? null;
    if (patchVendor) {
      const dateStr = parseDateToYYYYMMDD(question);
      // Linux patch queries are served only from this API (never from /api/component?q=os).
      const searchUrl = `https://releasetrain.io/api/v/search?q=${encodeURIComponent(patchVendor)}&channel=patch&limit=25&page=1`;
      let items;
      try {
        items = await fetchPatchSearch(patchVendor);
      } catch (e) {
        if (process.env.DEBUG) console.log('[answer] Patch search failed:', e.message);
        res.json({
          answer: DONT_KNOW_ANSWER,
          status: 'abstain',
          version: null,
          main: null,
          additional: {},
          vendor: patchVendor,
          source: 'ReleaseTrain',
          sourceUrl: searchUrl,
          versionSearchTags: null,
          _debugFormation: { step: 'fetchPatchSearch', error: e.message },
        });
        return;
      }
      if (!items || items.length === 0) {
        res.json({
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
        return;
      }
      const doc = findLinuxPatchByDate(items, dateStr);
      if (!doc || !docHasVendor(doc, patchVendor)) {
        res.json({
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
        return;
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
      const payload = {
        answer: `Patch version: ${main}.${url ? ` URL: ${url}.` : ''} Release notes: ${notes.slice(0, 200)}${notes.length > 200 ? '…' : ''}`,
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
        versionSearchTags: { value: versionSearchTags, fromDoc: doc?.versionSearchTags ?? null },
        step1_parsedInput: { rawQuestion, flow: 'patch', parsedVendor: patchVendor, parsedDateYYYYMMDD: dateStr },
        step2_dataFetched: { sourceUrl: searchUrl, itemsCount: items?.length ?? 0 },
        step3_matchedDoc: doc ? { _id: doc._id, versionSearchTags: doc.versionSearchTags } : null,
        step4_formatted: { main: formatted?.main ?? null },
      };
      if (process.env.DEBUG) console.log('[answer] Patch flow →', JSON.stringify(payload, null, 2));
      res.json(payload);
      return;
    }

    // OS flow: detect vendor from ReleaseTrain vendor names and match by date (or latest)
    const components = await fetchOSComponents();
    const dateStr = parseDateToYYYYMMDD(question);
    const vendorNames = await getVendorNames();
    const resolvedVendor = resolveVendorFromQuestion(rawQuestion, vendorNames);
    const product = resolvedVendor ? resolvedVendor.toLowerCase() : 'android';
    const doc = findOSByProductAndDate(components, product, dateStr);
    const formatted = formatOSResponse(doc);
    const main = formatted?.versionNumber ?? formatted?.main ?? 'N/A';
    const additional = formatted?.additional ?? {};
    const answerText = formatted
      ? `Version: ${main}. Release notes: ${additional.versionReleaseNotes || 'N/A'}. License: ${additional.versionProductLicense || 'N/A'}.`
      : 'No matching release found for that product or date.';
    let versionSearchTags = Array.isArray(doc?.versionSearchTags)
      ? doc.versionSearchTags
      : Array.isArray(formatted?.raw?.versionSearchTags)
        ? formatted.raw.versionSearchTags
        : null;
    if (doc && versionSearchTags == null && typeof doc === 'object') {
      versionSearchTags = doc.version_search_tags ?? doc.versionSearchTags ?? null;
    }
    const payload = {
      answer: answerText,
      status: formatted ? 'answer' : 'abstain',
      version: main,
      main,
      additional: {
        versionReleaseNotes: additional.versionReleaseNotes,
        versionProductLicense: additional.versionProductLicense,
      },
      vendor: formatted?.raw?.versionProductName ?? 'Unknown',
      source: 'ReleaseTrain',
      versionSearchTags,
    };
    if (doc && versionSearchTags == null && typeof doc === 'object') {
      payload._debugDocKeys = Object.keys(doc);
    }
    payload._debugFormation = {
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
        sourceUrl: 'https://releasetrain.io/api/component?q=os',
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
    res.json(payload);
  } catch (e) {
    console.error(e);
    res.status(502).json({
      answer: '',
      status: 'error',
      error: e.message,
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`ReleaseHub API listening on port ${PORT}`));
