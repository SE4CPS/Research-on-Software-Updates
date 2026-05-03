// ============================================
// FILE: server/routes/query.js
// POST /api/query - Accept raw user text,
// run NLP extraction + 5-gate abstain pipeline
// ============================================

const express = require("express");
const router = express.Router();
const { generatePromptFromText } = require("../services/nlpProcessor");
const { runPipeline } = require("../services/abstainPipeline");

router.post("/", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message || typeof message !== "string" || message.trim().length === 0) {
      return res.status(400).json({
        success: false,
        error: "Please provide a message to analyze.",
      });
    }

    console.log(`[Query] Received: "${message}"`);

    // Step 1: NLP Processing - extract entities from raw user text
    const processed = generatePromptFromText(message);

    if (!processed) {
      return res.json({
        success: true,
        decision: "abstain",
        compositeScore: 0,
        reason:
          "I couldn't identify any software or operating system in your question. Try mentioning a specific software name like Firefox, iOS, Windows, etc.",
        gates: [
          {
            gate: 2,
            name: "NLP extraction confidence",
            result: "fail",
            score: 0,
            reason: "No software/OS entity detected in the input",
          },
        ],
        data: null,
        suggestions: null,
        query: message,
        timestamp: new Date().toISOString(),
      });
    }

    if (processed.metadata?.isBroadQuery) {
      console.log(`[Query] Broad query detected — searching across popular software`);
    } else {
      console.log(
        `[Query] NLP extracted: "${processed.extraction.primaryEntity.name}" (${processed.extraction.primaryEntity.confidence})`
      );
    }

    // Step 2: Run 5-gate abstain pipeline
    const result = await runPipeline(processed);

    console.log(
      `[Query] Decision: ${result.decision} (composite: ${result.compositeScore})`
    );

    res.json({
      success: true,
      query: message,
      processedAs: processed.prompt,
      ...result,
    });
  } catch (error) {
    console.error("[Query] Error:", error);
    res.status(500).json({
      success: false,
      error: "Something went wrong processing your question. Please try again.",
    });
  }
});

module.exports = router;