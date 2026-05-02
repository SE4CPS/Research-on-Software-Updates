// ============================================
// FILE: server/services/componentNames.js
// Loads the authoritative component name list from releasetrain.io
// and provides fast entity matching for any of the ~6700 known names.
// ============================================

const axios = require("axios");

const BASE_URL = process.env.API_BASE_URL || "https://releasetrain.io/api";

// Names shorter than this are too generic to match safely (e.g. "A", "B", "AI", "AP")
const MIN_LENGTH = 4;

// Sorted longest-first so greedy match wins (e.g. "Visual Studio Code" beats "Visual Studio")
let sortedNames = [];

// First-word index: lower(firstWord) → [{ name, lower }]
// Lets us skip names whose first word doesn't appear in the query at all.
const namesByFirstWord = {};

async function loadComponentNames() {
  try {
    const res = await axios.get(`${BASE_URL}/c/names`, {
      headers: { Accept: "application/json" },
      timeout: 20000,
    });
    const raw = (res.data || []).filter(
      (n) => typeof n === "string" && n.trim().length >= MIN_LENGTH
    );

    sortedNames = [...raw].sort((a, b) => b.length - a.length);

    for (const name of sortedNames) {
      const fw = name.toLowerCase().split(/[\s\-_.]/)[0];
      if (!namesByFirstWord[fw]) namesByFirstWord[fw] = [];
      namesByFirstWord[fw].push({ name, lower: name.toLowerCase() });
    }

    console.log(`[ComponentNames] Loaded ${sortedNames.length} component names from releasetrain.io`);
    return true;
  } catch (err) {
    console.warn(`[ComponentNames] Could not load names: ${err.message} — falling back to built-in patterns`);
    return false;
  }
}

// Returns { name, confidence } for the longest component name found in text,
// or null if no known component is present.
function findComponentInText(text) {
  if (sortedNames.length === 0) return null;

  const lower = text.toLowerCase();

  // Collect candidate names whose first word appears in the query
  const words = new Set(lower.split(/[\s,.()\[\]"'!?;:/\\]+/).filter(Boolean));
  const candidates = [];
  for (const word of words) {
    for (const entry of (namesByFirstWord[word] || [])) {
      candidates.push(entry);
    }
  }

  // Check longest candidates first
  candidates.sort((a, b) => b.name.length - a.name.length);

  for (const { name, lower: nameLower } of candidates) {
    const idx = lower.indexOf(nameLower);
    if (idx === -1) continue;

    // Word-boundary check
    const charBefore = idx === 0 ? " " : lower[idx - 1];
    const charAfter =
      idx + nameLower.length >= lower.length ? " " : lower[idx + nameLower.length];
    const okBefore = idx === 0 || /[\s,.()\[\]"'!?;:/\\]/.test(charBefore);
    const okAfter =
      idx + nameLower.length >= lower.length || /[\s,.()\[\]"'!?;:/\\]/.test(charAfter);

    if (okBefore && okAfter) {
      return { name, confidence: 0.95 };
    }
  }

  return null;
}

// Hard-coded aliases for cases where the display name in a regex pattern differs
// from what releasetrain.io's API actually indexes.  Keep this list short.
const ENTITY_TO_SEARCH = new Map([
  ["visual studio code", "vscode"],
  ["vs code",            "vscode"],
  ["k8s",               "kubernetes"],
  ["node js",           "node"],
  ["nodejs",            "node"],
]);

// Given an entity name that returned 0 API results, try to resolve a better
// search term from the component list.  Returns the canonical name or null.
function resolveCanonicalSearchName(entityName, queryText) {
  if (sortedNames.length === 0) return null;

  // 0. Explicit alias map for known alternate display names.
  const alias = ENTITY_TO_SEARCH.get(entityName.toLowerCase());
  if (alias) return alias;

  // 1. Normalized comparison: strip separators and compare case-insensitively.
  //    "node.js" → "nodejs" matches "NodeJS" normalized "nodejs".
  const normalized = entityName.toLowerCase().replace(/[\s.\-_]/g, "");
  for (const name of sortedNames) {
    if (name.toLowerCase().replace(/[\s.\-_]/g, "") === normalized) {
      return name;
    }
  }

  // 2. Only when entity has NON-SPACE separators (e.g. "node.js", "vs-code"):
  //    try the first token as the API search term.  We deliberately skip multi-word
  //    names like "visual studio code" — splitting on spaces would pick "visual"
  //    which matches unrelated components starting with "Visual".
  const hasSeparator = /[\.\-_]/.test(entityName);
  if (hasSeparator) {
    const firstPart = entityName.toLowerCase().split(/[\.\-_]/)[0].trim();
    if (firstPart.length >= 3 && firstPart !== entityName.toLowerCase()) {
      const firstPartMatch = sortedNames.find(
        (n) => n.toLowerCase() === firstPart || n.toLowerCase().split(/[\.\-_]/)[0] === firstPart
      );
      if (firstPartMatch) return firstPartMatch;
      return firstPart; // try bare first token even if not in list (API may still match)
    }
  }

  // 3. Component text search — only accept when match length is within 60%–150%
  //    of entity length.  This filters out:
  //      - too-short fragments: "Studio" (6) for "visual studio code" (16) → ratio 0.38 < 0.60
  //      - over-specific extensions: "VS Code Extensions …" (38) for entity (16) → ratio 2.4 > 1.50
  const textMatch = findComponentInText(queryText);
  if (textMatch && textMatch.name.toLowerCase() !== entityName.toLowerCase()) {
    const ratio = textMatch.name.length / Math.max(entityName.length, 1);
    if (ratio >= 0.6 && ratio <= 1.5) return textMatch.name;
  }

  return null;
}

function isLoaded() {
  return sortedNames.length > 0;
}

module.exports = { loadComponentNames, findComponentInText, resolveCanonicalSearchName, isLoaded };
