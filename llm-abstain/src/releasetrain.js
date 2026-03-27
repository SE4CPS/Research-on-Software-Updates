/**
 * Fetches OS component data from ReleaseTrain API.
 * Data source: https://releasetrain.io/api/component?q=os
 * Linux patch data: https://releasetrain.io/api/v/search?q=linux&channel=patch&limit=25&page=1
 */

const RELEASETRAIN_OS_URL = 'https://releasetrain.io/api/component?q=os';
const RELEASETRAIN_LINUX_PATCH_SEARCH_URL = 'https://releasetrain.io/api/v/search?q=linux&channel=patch&limit=25&page=1';

/**
 * Fetch OS components from ReleaseTrain.
 * @returns {Promise<Array>} Array of component documents (may be empty)
 */
export async function fetchOSComponents() {
  const res = await fetch(RELEASETRAIN_OS_URL, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`ReleaseTrain API error: ${res.status} ${res.statusText}`);
  }
  const data = await res.json();
  return Array.isArray(data) ? data : (data?.components ?? data?.data ?? []);
}

/**
 * Normalize API release date to YYYYMMDD for comparison (e.g. "2025[29]0213" -> "20250213").
 */
function normalizeReleaseDate(releaseDate) {
  if (!releaseDate) return '';
  const s = String(releaseDate).replace(/\s/g, '');
  const digitsOnly = s.replace(/\D/g, '');
  return digitsOnly.slice(0, 8);
}

/**
 * Find best-matching OS document for a query (e.g. Android, date 2-02-2026).
 * Uses versionSearchTags and versionReleaseDate. Picks by date when given, else latest by timestamp.
 */
export function findOSByProductAndDate(components, productName = 'android', dateStr = null) {
  if (!Array.isArray(components) || components.length === 0) return null;
  const product = productName.toLowerCase();
  const queryDateNorm = dateStr ? String(dateStr).replace(/-/g, '').replace(/\D/g, '').slice(0, 8) : null;

  const normalized = components.map((c) => ({
    doc: c,
    tags: (c.versionSearchTags || []).map((t) => String(t).toLowerCase()),
    name: (c.versionProductName || '').toLowerCase(),
    releaseDate: c.versionReleaseDate || '',
    releaseDateNorm: normalizeReleaseDate(c.versionReleaseDate),
    timestamp: c.versionTimestamp || 0,
  }));

  const forProduct = normalized.filter(
    (n) => n.name === product || n.tags.includes(product)
  );
  // Do not fall back to an unrelated component (e.g. Flutter when product is "android" but no row matches).
  if (forProduct.length === 0) return null;

  if (queryDateNorm) {
    const exact = forProduct.filter((n) => n.releaseDateNorm === queryDateNorm);
    if (exact.length > 0) {
      exact.sort((a, b) => (b.timestamp - a.timestamp));
      return exact[0].doc;
    }
    const withDate = forProduct.filter((n) => n.releaseDateNorm && n.releaseDateNorm.length >= 6);
    if (withDate.length > 0) {
      withDate.sort((a, b) => {
        const diffA = Math.abs(parseInt(a.releaseDateNorm, 10) - parseInt(queryDateNorm, 10));
        const diffB = Math.abs(parseInt(b.releaseDateNorm, 10) - parseInt(queryDateNorm, 10));
        return diffA - diffB || b.timestamp - a.timestamp;
      });
      return withDate[0].doc;
    }
  }

  forProduct.sort((a, b) => b.timestamp - a.timestamp);
  return forProduct[0].doc;
}

const RELEASETRAIN_PATCH_SEARCH_BASE = 'https://releasetrain.io/api/v/search';
const RELEASETRAIN_PATCH_SEARCH_ALL_URL = 'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1';

/**
 * Fetch patch search results from ReleaseTrain for a given vendor/component.
 * Response: { data: Array } — each item has versionSearchTags e.g. ["linux","patch","YYYYMMDD","version"].
 * @param {string} vendor - Query term (e.g. "linux", "roblox")
 * @returns {Promise<Array>} Array of patch documents (may be empty)
 */
export async function fetchPatchSearch(vendor) {
  const q = encodeURIComponent(String(vendor || 'linux').trim());
  const url = `${RELEASETRAIN_PATCH_SEARCH_BASE}?q=${q}&channel=patch&limit=25&page=1`;
  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    throw new Error(`ReleaseTrain search API error: ${res.status} ${res.statusText}`);
  }
  const json = await res.json();
  const list = json?.data ?? json?.components ?? (Array.isArray(json) ? json : []);
  return Array.isArray(list) ? list : [];
}

/**
 * Fetch global patch feed (no vendor query), then filter in app logic.
 * Endpoint: /api/v/search?channel=patch&limit=25&page=1
 * @returns {Promise<Array>} Array of patch documents (may be empty)
 */
export async function fetchPatchSearchAll() {
  const res = await fetch(RELEASETRAIN_PATCH_SEARCH_ALL_URL, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    throw new Error(`ReleaseTrain search API error: ${res.status} ${res.statusText}`);
  }
  const json = await res.json();
  const list = json?.data ?? json?.components ?? (Array.isArray(json) ? json : []);
  return Array.isArray(list) ? list : [];
}

/** @deprecated Use fetchPatchSearch('linux') */
export async function fetchLinuxPatchSearch() {
  return fetchPatchSearch('linux');
}

/**
 * Return true if doc's versionSearchTags or product name/brand contains the vendor (case-insensitive).
 * @param {object} doc - Patch document from search
 * @param {string} vendor - Asked vendor (e.g. "Roblox", "Linux")
 */
export function docHasVendor(doc, vendor) {
  if (!doc || !vendor) return false;
  const v = String(vendor).trim().toLowerCase();
  if (!v) return false;
  const tags = (doc.versionSearchTags || []).map((t) => String(t).toLowerCase());
  const name = (doc.versionProductName || '').toLowerCase();
  const brand = (doc.versionProductBrand || '').toLowerCase();
  return tags.some((t) => t === v || t.includes(v)) || name === v || name.includes(v) || brand === v || brand.includes(v);
}

/**
 * Find Linux patch document by date, or latest by timestamp when no date given.
 * @param {Array} items - From fetchLinuxPatchSearch()
 * @param {string|null} dateStr - YYYYMMDD (e.g. "20260214" for 14-02-2026), or null for latest
 * @returns {object|null} Matching document or null
 */
export function findLinuxPatchByDate(items, dateStr) {
  if (!Array.isArray(items) || items.length === 0) return null;
  if (!dateStr) {
    const sorted = [...items].sort((a, b) => (b.versionTimestamp || 0) - (a.versionTimestamp || 0));
    return sorted[0] ?? null;
  }
  const norm = String(dateStr).replace(/-/g, '').replace(/\D/g, '').slice(0, 8);
  if (!norm) return items[0] ?? null;
  const match = items.find((doc) => {
    const releaseDate = (doc.versionReleaseDate || '').toString().replace(/-/g, '');
    return releaseDate === norm || releaseDate.startsWith(norm) || norm.startsWith(releaseDate.slice(0, 8));
  });
  return match ?? items[0] ?? null;
}

/**
 * Format Linux patch doc for answer: main = last value of versionSearchTags; additional = versionUrl, versionReleaseNotes.
 */
export function formatLinuxPatchResponse(doc) {
  if (!doc) return null;
  const tags = doc.versionSearchTags || [];
  const main = tags.length > 0 ? String(tags[tags.length - 1]) : (doc.versionNumber ?? '');
  return {
    main,
    versionNumber: doc.versionNumber ?? main,
    additional: {
      versionUrl: doc.versionUrl ?? '',
      versionReleaseNotes: doc.versionReleaseNotes ?? '',
    },
    raw: {
      versionProductName: doc.versionProductName ?? 'Linux',
      versionReleaseDate: doc.versionReleaseDate,
      versionSearchTags: doc.versionSearchTags,
    },
  };
}
