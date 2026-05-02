// ============================================
// FILE: server/services/tritonClient.js
// Triton Inference Server client for NeMo embeddings.
//
// Priority:
//   1. Local Triton server  (TRITON_URL, default http://localhost:8000)
//   2. NVIDIA NIM cloud API (requires NVIDIA_API_KEY)
//
// Environment variables:
//   TRITON_URL   - Triton base URL  (default: http://localhost:8000)
//   TRITON_MODEL - Model name in Triton (default: nv-embedqa-e5-v5)
//   NVIDIA_API_KEY - Required for cloud fallback
// ============================================

const axios = require("axios");

const TRITON_URL   = process.env.TRITON_URL   || "http://localhost:8000";
const TRITON_MODEL = process.env.TRITON_MODEL || "nv-embedqa-e5-v5";
const NIM_BASE     = "https://integrate.api.nvidia.com/v1";
const NIM_MODEL    = "nvidia/nv-embedqa-e5-v5";

// Cached reachability: null = not yet tested, true/false = result
let tritonAvailable = null;

// ── Public API ──────────────────────────────────

/**
 * Embed texts using the best available backend:
 *   Triton (local)  →  NVIDIA NIM (cloud)
 *
 * @param {string[]} texts     - Texts to embed
 * @param {string}   inputType - "query" or "passage"
 * @returns {Promise<number[][]>} Embedding vectors (one per text)
 */
async function embed(texts, inputType = "query") {
  if (await isTritonReachable()) {
    try {
      const vectors = await embedWithTriton(texts, inputType);
      console.log(`[Triton] Embedded ${texts.length} text(s) via local server`);
      return vectors;
    } catch (err) {
      console.warn(`[Triton] Inference failed (${err.message}) — falling back to NVIDIA NIM`);
      tritonAvailable = false; // skip Triton for remainder of this session
    }
  }

  return embedWithNIM(texts, inputType);
}

/**
 * Returns which embedding backend is currently active.
 * @returns {"triton"|"nvidia-nim"|"unknown"}
 */
function activeBackend() {
  if (tritonAvailable === true)  return "triton";
  if (tritonAvailable === false) return "nvidia-nim";
  return "unknown";
}

// ── Triton backend ───────────────────────────────

/**
 * Check Triton server health and cache the result.
 */
async function isTritonReachable() {
  if (tritonAvailable !== null) return tritonAvailable;
  try {
    await axios.get(`${TRITON_URL}/v2/health/ready`, { timeout: 2000 });
    console.log(`[Triton] Server is ready at ${TRITON_URL}`);
    tritonAvailable = true;
  } catch {
    console.log(`[Triton] Not reachable at ${TRITON_URL} — will use NVIDIA NIM`);
    tritonAvailable = false;
  }
  return tritonAvailable;
}

/**
 * Call Triton HTTP inference API v2.
 * Expects the model to expose:
 *   input  "text_input"  BYTES [N, 1]
 *   input  "input_type"  BYTES [1, 1]
 *   output "embedding"   FP32  [N, dim]
 */
async function embedWithTriton(texts, inputType) {
  const payload = {
    inputs: [
      {
        name:     "text_input",
        shape:    [texts.length, 1],
        datatype: "BYTES",
        data:     texts,
      },
      {
        name:     "input_type",
        shape:    [1, 1],
        datatype: "BYTES",
        data:     [inputType],
      },
    ],
    outputs: [{ name: "embedding" }],
  };

  const response = await axios.post(
    `${TRITON_URL}/v2/models/${TRITON_MODEL}/infer`,
    payload,
    {
      headers: { "Content-Type": "application/json" },
      timeout: 20000,
    }
  );

  const output = response.data.outputs?.find((o) => o.name === "embedding");
  if (!output?.data) throw new Error("No 'embedding' tensor in Triton response");

  // Reshape flat float array into per-text vectors
  const flat   = output.data;
  const embDim = flat.length / texts.length;
  if (!Number.isInteger(embDim)) throw new Error(`Triton output length ${flat.length} not divisible by ${texts.length}`);

  return texts.map((_, i) => flat.slice(i * embDim, (i + 1) * embDim));
}

// ── NVIDIA NIM cloud fallback ────────────────────

async function embedWithNIM(texts, inputType) {
  if (!process.env.NVIDIA_API_KEY) {
    throw new Error("NVIDIA_API_KEY not set and Triton server is unavailable");
  }

  const response = await axios.post(
    `${NIM_BASE}/embeddings`,
    {
      input:           texts,
      model:           NIM_MODEL,
      input_type:      inputType,
      encoding_format: "float",
    },
    {
      headers: {
        Authorization:  `Bearer ${process.env.NVIDIA_API_KEY}`,
        "Content-Type": "application/json",
      },
      timeout: 20000,
    }
  );

  console.log(`[Triton→NIM] Embedded ${texts.length} text(s) via NVIDIA cloud`);
  return response.data.data
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

module.exports = { embed, activeBackend, isTritonReachable };
