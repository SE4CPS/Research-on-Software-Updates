# Good practices — ReleaseHub

## LLM / Claude (when you add an LLM layer)

These apply if you integrate **Claude** or any other LLM for parsing or summarization.

1. **Source of truth**  
   Never treat the model as authoritative for **version numbers**, **patch IDs**, **CVEs**, or **release dates**. Only values returned from **ReleaseTrain** (via your existing OS/patch logic) should be shown as factual product data.

2. **Ground the answer**  
   If the LLM rewrites or summarizes the response, base the summary strictly on the **JSON/text your backend already computed** from ReleaseTrain. Do not ask the model to “look up” versions.

3. **Structured extraction first**  
   Prefer the LLM output as **structured JSON** (intent, vendor, date, normalized question) and then run your **deterministic** `parsePatchVendor` / `parseDateToYYYYMMDD` / `findOSByProductAndDate` path — or validate LLM output against your rules before calling ReleaseTrain.

4. **Timeouts**  
   Set a **hard timeout** (e.g. 10–30s) on calls to Anthropic (or any LLM API). On timeout or HTTP errors, **fall back** to rule-based parsing and `POST /answer` behavior without the LLM.

5. **Fallback**  
   If the LLM returns invalid JSON or empty fields, **fall back** to the user’s raw question through the existing non-LLM pipeline.

6. **Secrets**  
   Store `ANTHROPIC_API_KEY` (or equivalent) in **environment variables** or a secrets manager. **Never** commit API keys to git. Add `.env` to `.gitignore` if you use local env files.

7. **Privacy & compliance**  
   Sending user questions to a third-party API may include **PII or confidential text**. Document this in your README; offer an **on-device / local model** option if required.

8. **Cost & rate limits**  
   Cache responses by **normalized question** where safe; debounce duplicate requests; respect provider **rate limits**.

9. **Observability**  
   Log LLM failures (without logging full API keys) and optionally tag analytics events as `llm_used` vs `rule_based` for debugging.

---

## API & backend

- **CORS** is currently permissive (`*`) for hackathon/demo use; tighten **origins** in production.
- **Redis** is optional; without `REDIS_URL`, last-prompt storage is simply skipped.
- **Analytics** are **in-memory** and reset on process restart; use Redis/DB for durable metrics in production.

---

## Docker

- Use **`docker compose up --build`** from the repo root for full stack; see **[DOCKER.md](./DOCKER.md)**.
- Do **not** bake API keys into images; pass secrets via **environment** or **Compose secrets** at runtime.
