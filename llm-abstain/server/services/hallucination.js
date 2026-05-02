// ============================================
// FILE: server/services/hallucination.js
// BERTScore-based hallucination / grounding detector.
//
// Measures how well the pipeline's structured response is
// grounded in the raw releasetrain.io source entries.
//
// With NVIDIA embeddings (Triton → NIM):
//   Real BERTScore — pairwise cosine similarity → P / R / F1
//
// Without embeddings:
//   Lexical fallback — token overlap (ROUGE-1 style) → P / R / F1
// ============================================

const { embed } = require("./tritonClient");

// ── Public API ──────────────────────────────────

/**
 * Compute a BERTScore-style grounding score.
 *
 * When llmText is provided (RAG mode), it is used as the hypothesis —
 * this gives a real hallucination check on the generated text.
 * Otherwise falls back to structured data field phrases.
 *
 * @param {object}   structuredData  - Gate 5 output
 * @param {object[]} sourceEntries   - Raw releasetrain.io entries (Gate 3 pool)
 * @param {string}   [llmText]       - LLM-generated response text (optional)
 */
async function computeBERTScore(structuredData, sourceEntries, llmText = null) {
  if (!structuredData || !sourceEntries || sourceEntries.length === 0) return null;

  const hypothesis = llmText
    ? llmText.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 8)  // split LLM text into sentences
    : toHypothesisPhrases(structuredData);

  const references  = toReferencePhrases(sourceEntries.slice(0, 10));

  if (hypothesis.length === 0 || references.length === 0) return null;

  // ── Try embedding-based BERTScore ──────────────
  try {
    const allTexts  = [...hypothesis, ...references];
    const vectors   = await embed(allTexts, "query");

    const hypVecs   = vectors.slice(0, hypothesis.length);
    const refVecs   = vectors.slice(hypothesis.length);

    // Precision: each hypothesis phrase → max cosine sim over all reference phrases
    const precisionScores = hypVecs.map(h =>
      Math.max(...refVecs.map(r => cosine(h, r)))
    );

    // Recall: each reference phrase → max cosine sim over all hypothesis phrases
    const recallScores = refVecs.map(r =>
      Math.max(...hypVecs.map(h => cosine(r, h)))
    );

    const P  = avg(precisionScores);
    const R  = avg(recallScores);
    const F1 = harmonic(P, R);

    console.log(`[BERTScore] embedding — P:${P.toFixed(3)} R:${R.toFixed(3)} F1:${F1.toFixed(3)}`);

    return { precision: round2(P), recall: round2(R), f1: round2(F1), risk: riskLevel(F1, P), method: "embedding" };
  } catch (err) {
    console.warn(`[BERTScore] Embedding failed (${err.message}) — using lexical fallback`);
  }

  // ── Lexical fallback ────────────────────────────
  return lexicalScore(hypothesis, references);
}

// ── Phrase extraction ────────────────────────────

function toHypothesisPhrases(d) {
  const phrases = [];
  if (d.software)       phrases.push(`software ${d.software}`);
  if (d.version)        phrases.push(`version ${d.version}`);
  if (d.releaseDate)    phrases.push(`released ${d.releaseDate}`);
  if (d.releaseChannel) phrases.push(`${d.releaseChannel} release channel`);
  if (d.cveId)          phrases.push(`CVE identifier ${d.cveId}`);
  if (d.severity)       phrases.push(`severity ${d.severity}`);
  if (d.description)    phrases.push(d.description.slice(0, 200));

  (d.recentCves || []).slice(0, 3).forEach(c => {
    phrases.push(`${c.cveId || "CVE"} version ${c.version || ""} ${(c.notes || "").slice(0, 80)}`);
  });
  (d.recentPatches || []).slice(0, 3).forEach(p => {
    phrases.push(`patch ${p.version || ""} released ${p.date || ""}`);
  });
  (d.recentVersions || []).slice(0, 3).forEach(v => {
    phrases.push(`version ${v.version || ""} ${v.channel || ""} ${v.date || ""}`);
  });
  (d.entries || []).slice(0, 3).forEach(e => {
    phrases.push(`${e.version || ""} ${e.channel || ""} ${(e.notes || "").slice(0, 80)}`);
  });

  return phrases.filter(p => p.trim().length > 4).slice(0, 8);
}

function toReferencePhrases(entries) {
  return entries.map(e => {
    const parts = [
      e.versionProductBrand || e.versionProductName || "",
      e.versionNumber  ? `v${e.versionNumber}` : "",
      e.versionReleaseChannel || "",
      e.isCve ? "CVE security vulnerability" : "",
      (e.versionReleaseNotes || "").slice(0, 180),
    ].filter(Boolean);
    return parts.join(" ");
  }).filter(p => p.trim().length > 4);
}

// ── Lexical fallback (token overlap) ─────────────

function lexicalScore(hypothesis, references) {
  const hypTokens = tokenize(hypothesis.join(" "));
  const refTokens = tokenize(references.join(" "));
  const hypSet    = new Set(hypTokens);
  const refSet    = new Set(refTokens);

  const intersection = [...hypSet].filter(t => refSet.has(t));
  const P  = hypSet.size > 0 ? intersection.length / hypSet.size : 0;
  const R  = refSet.size > 0 ? intersection.length / refSet.size : 0;
  const F1 = harmonic(P, R);

  console.log(`[BERTScore] lexical — P:${P.toFixed(3)} R:${R.toFixed(3)} F1:${F1.toFixed(3)}`);

  return { precision: round2(P), recall: round2(R), f1: round2(F1), risk: riskLevel(F1, P), method: "lexical" };
}

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(t => t.length > 2);
}

// ── Risk level ───────────────────────────────────
// Based on precision (grounding), not F1.
// Recall is always low for summaries — don't penalise it.

function riskLevel(f1, precision) {
  const p = precision ?? f1;
  if (p >= 0.78) return "low";
  if (p >= 0.55) return "medium";
  return "high";
}

// ── Math helpers ─────────────────────────────────

function cosine(a, b) {
  let dot = 0, nA = 0, nB = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; nA += a[i]*a[i]; nB += b[i]*b[i]; }
  return dot / (Math.sqrt(nA) * Math.sqrt(nB) || 1);
}

function avg(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
function harmonic(p, r) { return (p + r) > 0 ? (2 * p * r) / (p + r) : 0; }
function round2(v) { return Math.round(v * 1000) / 1000; }

module.exports = { computeBERTScore };
