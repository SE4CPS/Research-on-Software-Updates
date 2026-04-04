/**
 * ReleaseTrain data access + matching helpers
 */

const RELEASETRAIN_OS_URL = 'https://releasetrain.io/api/component?q=os';
const RELEASETRAIN_PATCH_SEARCH_BASE = 'https://releasetrain.io/api/v/search';
const RELEASETRAIN_PATCH_SEARCH_ALL_URL =
  'https://releasetrain.io/api/v/search?channel=patch&limit=25&page=1';

/**
 * Normalize date to YYYYMMDD.
 */
export function normalizeDateYYYYMMDD(value) {
  if (!value) return '';
  return String(value).replace(/\D/g, '').slice(0, 8);
}

/**
 * Fetch OS components from ReleaseTrain.
 * @returns {Promise<Array>}
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
 * Normalize API release date to YYYYMMDD.
 * Example: "2025[29]0213" -> "20250213"
 */
function normalizeReleaseDate(releaseDate) {
  if (!releaseDate) return '';
  const s = String(releaseDate).replace(/\s/g, '');
  const digitsOnly = s.replace(/\D/g, '');
  return digitsOnly.slice(0, 8);
}

/**
 * Normalize input date to YYYYMMDD.
 * Supports:
 * - 2026-04-03
 * - 20260403
 * - 04-03-2026
 */
export function normalizeDateToYYYYMMDD(input) {
  if (!input) return '';
  const s = String(input).trim();

  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
    return s.replaceAll('-', '');
  }

  const mdy = s.match(/^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$/);
  if (mdy) {
    const [, month, day, year] = mdy;
    return `${year}${month.padStart(2, '0')}${day.padStart(2, '0')}`;
  }

  return s.replace(/\D/g, '').slice(0, 8);
}

/**
 * Find best-matching OS document for a query.
 */
export function findOSByProductAndDate(components, productName = 'android', dateStr = null) {
  if (!Array.isArray(components) || components.length === 0) return null;

  const product = String(productName || 'android').toLowerCase();
  const queryDateNorm = dateStr ? normalizeDateYYYYMMDD(dateStr) : null;

  const normalized = components.map((c) => ({
    doc: c,
    tags: (c.versionSearchTags || []).map((t) => String(t).toLowerCase()),
    name: (c.versionProductName || '').toLowerCase(),
    releaseDateNorm: normalizeReleaseDate(c.versionReleaseDate),
    timestamp: c.versionTimestamp || 0,
  }));

  const forProduct = normalized.filter(
    (n) => n.name === product || n.tags.includes(product)
  );

  if (forProduct.length === 0) return normalized[0]?.doc ?? null;

  if (queryDateNorm) {
    const exact = forProduct.filter((n) => n.releaseDateNorm === queryDateNorm);
    if (exact.length > 0) {
      exact.sort((a, b) => b.timestamp - a.timestamp);
      return exact[0].doc;
    }

    const withDate = forProduct.filter(
      (n) => n.releaseDateNorm && n.releaseDateNorm.length >= 6
    );

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

/**
 * Fetch patch search results from ReleaseTrain for a given vendor/component.
 * @param {string} vendor
 * @returns {Promise<Array>}
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
 * Fetch global patch feed (legacy flow).
 * @returns {Promise<Array>}
 */
export async function fetchPatchSearchAll() {
  const res = await fetch(RELEASETRAIN_PATCH_SEARCH_ALL_URL, {
    headers: { Accept: 'application/json' },
  });

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
 * Vendor helper used by legacy Linux patch flow.
 */
export function docHasVendor(doc, vendor) {
  if (!doc || !vendor) return false;

  const v = String(vendor).trim().toLowerCase();
  if (!v) return false;

  const tags = Array.isArray(doc.versionSearchTags)
    ? doc.versionSearchTags.map((t) => String(t).toLowerCase())
    : [];

  const name = String(doc.versionProductName || '').toLowerCase();
  const brand = String(doc.versionProductBrand || '').toLowerCase();

  return (
    tags.some((t) => t === v || t.includes(v)) ||
    name === v ||
    name.includes(v) ||
    brand === v ||
    brand.includes(v)
  );
}

/**
 * Find Linux patch document by date, or latest by timestamp when no date given.
 */
export function findLinuxPatchByDate(items, dateStr) {
  if (!Array.isArray(items) || items.length === 0) return null;

  if (!dateStr) {
    const sorted = [...items].sort((a, b) => (b.versionTimestamp || 0) - (a.versionTimestamp || 0));
    return sorted[0] ?? null;
  }

  const norm = normalizeDateYYYYMMDD(dateStr);
  if (!norm) return items[0] ?? null;

  const match = items.find((doc) => {
    const releaseDate = normalizeDateYYYYMMDD(doc.versionReleaseDate);
    return (
      releaseDate === norm ||
      releaseDate.startsWith(norm) ||
      norm.startsWith(releaseDate.slice(0, 8))
    );
  });

  return match ?? items[0] ?? null;
}

/**
 * Format Linux patch doc for old flow.
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

/**
 * New approach:
 * Fetch vendor-specific records from:
 * https://releasetrain.io/api/c/name/<vendor>
 */
export async function fetchPatchDetailsByVendorName(vendor) {
  const safeVendor = encodeURIComponent(String(vendor || '').trim().toLowerCase());
  const url = `https://releasetrain.io/api/c/name/${safeVendor}`;

  const res = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!res.ok) {
    throw new Error(`ReleaseTrain vendor API error: ${res.status} ${res.statusText}`);
  }

  const json = await res.json();

  if (Array.isArray(json)) return json;

  const direct =
    json?.data ??
    json?.components ??
    json?.results ??
    json?.items ??
    json?.records;

  if (Array.isArray(direct)) return direct;

  const values = Object.values(json || {});
  const firstArray = values.find((v) => Array.isArray(v));

  return Array.isArray(firstArray) ? firstArray : [];
}

/**
 * Match vendor-specific records by vendor + date.
 *
 * Expected example:
 * versionSearchTags = ["firefox","minor","20260324","140.9.0"]
 *
 * index 0 -> vendor
 * index 1 -> release type
 * index 2 -> date
 * index 3 -> version
 */
function recordMatchesVendorAndDate(doc, vendor, dateStr) {
  const vendorNorm = String(vendor || '').trim().toLowerCase();
  const dateNorm = normalizeDateYYYYMMDD(dateStr);

  const tags = Array.isArray(doc?.versionSearchTags)
    ? doc.versionSearchTags.map((t) => String(t).trim().toLowerCase())
    : [];

  if (tags.length < 4) {
    return {
      matched: false,
      isPatch: false,
      isMinor: false,
      isMajor: false,
    };
  }

  const tagVendor = tags[0] || '';
  const tagChannel = tags[1] || '';
  const tagDate = tags[2] || '';
  const tagVersion = tags[3] || '';

  const productName = String(doc?.versionProductName || '').trim().toLowerCase();
  const productBrand = String(doc?.versionProductBrand || '').trim().toLowerCase();
  const releaseDate = normalizeDateYYYYMMDD(doc?.versionReleaseDate);

  const hasValidDate = /^\d{8}$/.test(tagDate) || /^\d{8}$/.test(releaseDate);
  const hasValidVersion = /\d/.test(tagVersion);

  const vendorMatches =
    tagVendor === vendorNorm ||
    productName === vendorNorm ||
    productBrand === vendorNorm ||
    tagVendor.includes(vendorNorm) ||
    vendorNorm.includes(tagVendor);

  const dateMatches =
    tagDate === dateNorm ||
    releaseDate === dateNorm;

  return {
    matched: vendorMatches && dateMatches && hasValidDate && hasValidVersion,
    isPatch: tagChannel === 'patch',
    isMinor: tagChannel === 'minor',
    isMajor: tagChannel === 'major',
  };
}

/**
 * Return ALL matching vendor records for a given date.
 */
export function findPatchDetailsByVendorAndDate(items, vendor, dateStr) {
  if (!Array.isArray(items) || items.length === 0) return [];

  const matches = items
    .map((doc) => {
      const result = recordMatchesVendorAndDate(doc, vendor, dateStr);
      return { doc, ...result };
    })
    .filter((x) => x.matched);

  if (matches.length === 0) return [];

  matches.sort((a, b) => {
    if (a.isPatch !== b.isPatch) return a.isPatch ? -1 : 1;
    if (a.isMinor !== b.isMinor) return a.isMinor ? -1 : 1;
    if (a.isMajor !== b.isMajor) return a.isMajor ? -1 : 1;
    return (b.doc?.versionTimestamp || 0) - (a.doc?.versionTimestamp || 0);
  });

  return matches.map((m) => m.doc);
}

/**
 * Format vendor-specific record.
 */
export function formatPatchDetailsFromTags(doc) {
  if (!doc) return null;

  const tags = Array.isArray(doc.versionSearchTags) ? doc.versionSearchTags : [];

  return {
    main: tags[3] ?? doc.versionNumber ?? '',
    versionNumber: tags[3] ?? doc.versionNumber ?? '',
    additional: {
      versionProductName: tags[0] ?? doc.versionProductName ?? '',
      productType: doc.versionProductType ?? '',
      versionNumber: tags[3] ?? doc.versionNumber ?? '',
      releaseNoteUrl: doc.versionReleaseNotes ?? '',
      releaseDate: tags[2] ?? doc.versionReleaseDate ?? '',
      releaseType: tags[1] ?? doc.versionReleaseChannel ?? '',
    },
    raw: {
      versionSearchTags: tags,
      versionProductName: doc.versionProductName ?? '',
      versionProductType: doc.versionProductType ?? '',
      versionReleaseNotes: doc.versionReleaseNotes ?? '',
      versionReleaseDate: doc.versionReleaseDate ?? '',
      versionReleaseChannel: doc.versionReleaseChannel ?? '',
    },
  };
}