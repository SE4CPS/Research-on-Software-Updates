// ============================================
// FILE: server/services/nemoRetriever.js
// NVIDIA NeMo Retriever — semantic re-ranking
// Uses NIM embeddings API to rank releasetrain.io
// entries by relevance to the user's query.
//
// Falls back to original order if no API key set.
// ============================================

const axios = require("axios");

const NIM_BASE    = "https://integrate.api.nvidia.com/v1";
const EMBED_MODEL = "nvidia/nv-embedqa-e5-v5";
const MAX_ENTRIES = 30;   // cap to avoid slow/expensive batch calls
const ENABLED     = !!process.env.NVIDIA_API_KEY;

// ── Public API ──────────────────────────────────

// Re-ranks `entries` (releasetrain.io version objects) by semantic
// similarity to `userQuery`. Returns entries sorted best-first.
async function rankEntriesByRelevance(userQuery, entries) {
  if (!ENABLED || !entries || entries.length === 0) {
    if (!ENABLED) console.log("[NeMo] No NVIDIA_API_KEY — skipping semantic re-rank");
    return entries;
  }

  const pool = entries.slice(0, MAX_ENTRIES);

  try {
    // Build a short passage for each entry
    const passages = pool.map(entryToPassage);

    // Embed query and all passages in parallel
    const [queryEmbedding, passageEmbeddings] = await Promise.all([
      embed([userQuery], "query"),
      embed(passages, "passage"),
    ]);

    // Score each entry
    const scored = pool.map((entry, i) => ({
      entry,
      score: cosineSimilarity(queryEmbedding[0], passageEmbeddings[i]),
    }));

    scored.sort((a, b) => b.score - a.score);

    const ranked = scored.map((s) => s.entry);

    console.log(
      `[NeMo] Re-ranked ${pool.length} entries — top: "${entryToPassage(ranked[0]).slice(0, 60)}"`
    );

    // Return ranked pool + any remaining entries that weren't embedded
    return [...ranked, ...entries.slice(MAX_ENTRIES)];
  } catch (err) {
    console.error("[NeMo] Re-ranking failed, using original order:", err.message);
    return entries;
  }
}

// ── Internals ────────────────────────────────────

// Converts a releasetrain.io entry into a short text passage for embedding
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

// Calls NVIDIA NIM embeddings endpoint
async function embed(texts, inputType = "query") {
  const response = await axios.post(
    `${NIM_BASE}/embeddings`,
    {
      input: texts,
      model: EMBED_MODEL,
      input_type: inputType,
      encoding_format: "float",
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.NVIDIA_API_KEY}`,
        "Content-Type": "application/json",
      },
      timeout: 20000,
    }
  );

  // Sort by index to preserve order, then return vectors
  return response.data.data
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

// Standard cosine similarity between two vectors
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

module.exports = { rankEntriesByRelevance, isEnabled: () => ENABLED };
