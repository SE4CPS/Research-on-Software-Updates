// ============================================
// FILE: server/services/nemoRetriever.js
// NVIDIA NeMo Reranker — cross-encoder re-ranking
//
// Primary  : nvidia/nv-rerankqa-mistral-4b-v3 (NVIDIA NIM ranking API)
// Fallback : Triton embeddings (local) → NVIDIA NIM embeddings + cosine similarity
//
// Falls back to original entry order when no backend is available.
// ============================================

const axios        = require("axios");
const { embed }    = require("./tritonClient");

const NIM_BASE      = "https://integrate.api.nvidia.com/v1";
const RERANK_MODEL  = "nvidia/nv-rerankqa-mistral-4b-v3";
const MAX_ENTRIES   = 30;   // cap to keep API calls fast
const ENABLED       = !!process.env.NVIDIA_API_KEY;

// ── Public API ──────────────────────────────────

/**
 * Re-ranks `entries` (releasetrain.io version objects) by relevance to `userQuery`.
 * Returns entries sorted best-first.
 *
 * Strategy:
 *  1. NeMo cross-encoder reranker  (nvidia/nv-rerankqa-mistral-4b-v3)
 *  2. Triton / NIM embeddings + cosine similarity
 *  3. Original order (silent no-op)
 */
async function rankEntriesByRelevance(userQuery, entries) {
  if (!entries || entries.length === 0) return entries;

  const pool = entries.slice(0, MAX_ENTRIES);

  // ── Path 1: NeMo cross-encoder reranker ──
  if (ENABLED) {
    try {
      const ranked = await rerank(userQuery, pool);
      console.log(
        `[NeMo Reranker] Cross-encoder ranked ${pool.length} entries — top: "${entryToPassage(ranked[0]).slice(0, 60)}"`
      );
      return [...ranked, ...entries.slice(MAX_ENTRIES)];
    } catch (err) {
      console.warn(`[NeMo Reranker] Reranker failed (${err.message}) — falling back to embeddings`);
    }
  } else {
    console.log("[NeMo Reranker] No NVIDIA_API_KEY — skipping reranker, trying embeddings");
  }

  // ── Path 2: embeddings + cosine similarity (Triton → NIM) ──
  try {
    const passages = pool.map(entryToPassage);
    const [queryVec, passageVecs] = await Promise.all([
      embed([userQuery], "query"),
      embed(passages, "passage"),
    ]);

    const scored = pool
      .map((entry, i) => ({ entry, score: cosineSimilarity(queryVec[0], passageVecs[i]) }))
      .sort((a, b) => b.score - a.score);

    const ranked = scored.map((s) => s.entry);
    console.log(
      `[NeMo Reranker] Embedding re-rank (${scored[0]?.score.toFixed(4)}) — top: "${entryToPassage(ranked[0]).slice(0, 60)}"`
    );
    return [...ranked, ...entries.slice(MAX_ENTRIES)];
  } catch (err) {
    console.warn(`[NeMo Reranker] Embedding fallback also failed (${err.message}) — using original order`);
    return entries;
  }
}

// ── NeMo Reranker (cross-encoder) ───────────────

async function rerank(query, entries) {
  const passages = entries.map((e) => ({ text: entryToPassage(e) }));

  const response = await axios.post(
    `${NIM_BASE}/ranking`,
    {
      model:    RERANK_MODEL,
      query:    { text: query },
      passages,
      truncate: "END",
    },
    {
      headers: {
        Authorization:  `Bearer ${process.env.NVIDIA_API_KEY}`,
        "Content-Type": "application/json",
      },
      timeout: 15000,
    }
  );

  // rankings is an array of { index, logit } sorted by relevance
  const rankings = response.data.rankings;
  if (!Array.isArray(rankings)) throw new Error("Unexpected reranker response shape");

  return rankings
    .sort((a, b) => b.logit - a.logit)
    .map((r) => entries[r.index]);
}

// ── Helpers ──────────────────────────────────────

function entryToPassage(entry) {
  const parts = [
    entry.versionProductBrand || entry.versionProductName || "",
    entry.versionNumber ? `v${entry.versionNumber}` : "",
    entry.versionReleaseChannel || "",
    entry.isCve ? "CVE security vulnerability" : "",
    (entry.versionReleaseNotes || "").slice(0, 200),
  ].filter(Boolean);
  return parts.join(" — ");
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

module.exports = {
  rankEntriesByRelevance,
  isEnabled: () => ENABLED,
  RERANK_MODEL,
};
