// ============================================
// FILE: server/services/apiClient.js
// HTTP client for releasetrain.io
// Primary endpoint: GET /api/v/?q={software}
//   → same schema used by https://releasetrain.io/?q=Android
// ============================================

const axios = require("axios");

const BASE_URL = process.env.API_BASE_URL || "https://releasetrain.io/api";

const searchCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes


function debugSourceStats(versions) {
  const breakingCount = {};
  const securityCount = {};
  let cveCount = 0;

  for (const v of versions) {
    if (v.isCve) cveCount++;
    for (const t of v.classification?.breakingType || []) {
      breakingCount[t] = (breakingCount[t] || 0) + 1;
    }
    for (const t of v.classification?.securityType || []) {
      securityCount[t] = (securityCount[t] || 0) + 1;
    }
  }

  console.log(`[DEBUG][Source] CVE entries        : ${cveCount}`);
  if (Object.keys(breakingCount).length) {
    console.log(`[DEBUG][Source] Breaking types:`);
    for (const [k, v] of Object.entries(breakingCount)) {
      console.log(`  └─ ${k}: ${v}`);
    }
  }
  if (Object.keys(securityCount).length) {
    console.log(`[DEBUG][Source] Security types:`);
    for (const [k, v] of Object.entries(securityCount)) {
      console.log(`  └─ ${k}: ${v}`);
    }
  }
}

// GET /api/v/?q={query}  — releasetrain.io's own search used by its website
async function fetchVersionSearch(query) {
  const key = query.toLowerCase().trim();
  const now = Date.now();
  const cached = searchCache.get(key);

  if (cached && now - cached.fetchedAt < CACHE_TTL) {
    console.log(`[API Client] Cache hit for "${query}" (${cached.data.length} entries)`);
    return { success: true, data: cached.data, cached: true };
  }

  const encoded = encodeURIComponent(key);
  const fullUrl = `${BASE_URL}/v/?q=${encoded}`;

  console.log(`[DEBUG][Source] Fetching from : ${fullUrl}`);

  try {
    const response = await axios.get(fullUrl, {
      headers: { Accept: "application/json" },
      timeout: 15000,
    });

    const versions = response.data?.versions || [];
    console.log(`[DEBUG][Source] Status        : ${response.status} ${response.statusText}`);
    console.log(`[DEBUG][Source] Content-Type  : ${response.headers["content-type"]}`);
    console.log(`[DEBUG][Source] Total entries : ${versions.length}`);
    debugSourceStats(versions);

    searchCache.set(key, { data: versions, fetchedAt: now });
    return { success: true, data: versions, cached: false };
  } catch (error) {
    console.error(`[DEBUG][Source] FAILED for "${query}": ${error.message}`);
    if (cached) {
      console.warn(`[DEBUG][Source] Falling back to stale cache (${cached.data.length} entries)`);
      return { success: true, data: cached.data, cached: true, stale: true };
    }
    return { success: false, data: [], error: error.message };
  }
}

module.exports = { fetchVersionSearch };
