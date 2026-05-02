// ============================================
// FILE: server/services/similarityMatcher.js
// Fuzzy matching extracted names against valid OS list
// ============================================

const stringSimilarity = require("string-similarity");
const { jaroWinkler } = require("../utils/similarity");

function findBestMatch(entityName, osList) {
  if (!entityName || !osList || osList.length === 0) {
    return {
      matched: false,
      score: 0,
      bestMatch: null,
      suggestions: [],
      method: "none",
    };
  }

  const query = entityName.toLowerCase().trim();

  // Extract names from OS list (handles string[] or object[])
  const validNames = osList.map((item) =>
    typeof item === "string" ? item : item.name || item.title || String(item)
  );
  const validNamesLower = validNames.map((n) => n.toLowerCase());

  // 1. Exact match
  const exactIdx = validNamesLower.indexOf(query);
  if (exactIdx !== -1) {
    return {
      matched: true,
      score: 1.0,
      bestMatch: validNames[exactIdx],
      suggestions: [],
      method: "exact",
    };
  }

  // 2. Contains match
  const containsMatches = validNames.filter(
    (name) =>
      name.toLowerCase().includes(query) ||
      query.includes(name.toLowerCase())
  );
  if (containsMatches.length === 1) {
    return {
      matched: true,
      score: 0.95,
      bestMatch: containsMatches[0],
      suggestions: [],
      method: "contains",
    };
  }

  // 3. Jaro-Winkler scores
  const jwScores = validNames.map((name) => ({
    name,
    score: jaroWinkler(query, name),
  }));

  // 4. Dice coefficient scores
  const diceResult = stringSimilarity.findBestMatch(query, validNamesLower);
  const diceScores = diceResult.ratings.map((r, i) => ({
    name: validNames[i],
    score: r.rating,
  }));

  // 5. Combine: 60% Jaro-Winkler + 40% Dice
  const combined = validNames.map((name) => {
    const jw = jwScores.find((s) => s.name === name)?.score || 0;
    const dice = diceScores.find((s) => s.name === name)?.score || 0;
    return { name, score: 0.6 * jw + 0.4 * dice, jwScore: jw, diceScore: dice };
  });
  combined.sort((a, b) => b.score - a.score);

  const best = combined[0];

  // High confidence (>= 0.9)
  if (best.score >= 0.9) {
    return {
      matched: true,
      score: best.score,
      bestMatch: best.name,
      suggestions: [],
      method: "fuzzy-high",
    };
  }

  // Partial match (0.6 – 0.9) → suggest alternatives
  if (best.score >= 0.6) {
    const suggestions = combined
      .filter((s) => s.score >= 0.5)
      .slice(0, 3)
      .map((s) => ({ name: s.name, score: Math.round(s.score * 100) / 100 }));
    return {
      matched: false,
      score: best.score,
      bestMatch: best.name,
      suggestions,
      method: "fuzzy-partial",
    };
  }

  // No match (< 0.6)
  return {
    matched: false,
    score: best.score,
    bestMatch: null,
    suggestions: combined
      .slice(0, 3)
      .map((s) => ({ name: s.name, score: Math.round(s.score * 100) / 100 })),
    method: "fuzzy-none",
  };
}

module.exports = { findBestMatch };