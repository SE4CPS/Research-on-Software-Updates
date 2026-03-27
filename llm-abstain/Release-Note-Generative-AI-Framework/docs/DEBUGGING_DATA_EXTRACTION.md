# Debugging data extraction

This app extracts **vendor**, **date**, and **flow** from the user question, then fetches from ReleaseTrain and picks a document. Use the steps below when the wrong branch runs or the answer doesn’t match what you expect.

## 1. Frontend: Debug response (no server restart)

1. Run backend + frontend as usual.
2. Turn on **“Debug response”** in the chat UI.
3. Ask your question → expand **Debug: raw response**.
4. Inspect **`_debugFormation`**:
   - **`step1_parsedInput`** — normalized question, `flow` (`patch` vs `os`), `parsedVendor`, `parsedDateYYYYMMDD`, `product` (OS only).
   - **`step2_dataFetched`** — `sourceUrl`, `itemsCount` / `componentsCount`.
   - **`step3_matchedDoc`** — which document was chosen (`versionSearchTags`, `versionReleaseDate`, etc.).
   - **`reason`** — when you get “I don’t know…”, check `reason` (e.g. `patch_for_non_linux_vendor_not_supported`, `no_results`, `versionSearchTags_does_not_have_provided_vendor`).

## 2. Backend: Console logging

Start the API with `DEBUG` set:

```bash
cd /path/to/HackathonFeb27
DEBUG=1 node src/server.js
# or
DEBUG=1 npm run dev
```

Then `POST /answer` will log the full payload for patch/OS flows.

## 3. Trace endpoint (parse + branch only)

**Without** calling ReleaseTrain (fast):

```bash
curl -s "http://localhost:3000/api/debug/trace?question=What%20is%20the%20patch%20for%20Roblox%20on%2014-02-2024" | jq
```

Response fields:

| Field | Meaning |
|-------|--------|
| `normalizedQuestion` | Lowercased question used for parsing |
| `parsedPatchVendor` | Vendor after “patch for …” (e.g. `roblox`, `linux`), or `null` |
| `parsedDateYYYYMMDD` | Date from `DD-MM-YYYY` / `DD/MM/YYYY` → `YYYYMMDD`, or `null` |
| `branch` | Which code path runs: `patch_non_linux → idk`, `patch_linux`, or `os` |

## 4. Trace with fetch (hits ReleaseTrain)

```bash
curl -s "http://localhost:3000/api/debug/trace?question=What%20is%20the%20patch%20for%20Linux%20on%2014-02-2026&fetch=1" | jq
```

Adds:

- **Patch flow:** `patchSearch.itemsCount`, `firstItemTags`, `matchedDocTags`, `matchedDocHasVendor`.
- **OS flow:** `os.componentsCount`, `os.product`, `os.matchedDocSummary`.

## 5. Where extraction happens in code

| Step | Location | What it does |
|------|-----------|--------------|
| Patch vendor | `server.js` → `parsePatchVendor(question)` | Regex: `patch for <vendor>` until ` on ` / `.` / `?` / end |
| Date | `server.js` → `parseDateToYYYYMMDD(question)` | First `D-M-YYYY` or `D/M/YYYY` → `YYYYMMDD` |
| Patch doc by date | `releasetrain.js` → `findLinuxPatchByDate(items, dateStr)` | Match `versionReleaseDate` or fallback to first item |
| Vendor in doc | `releasetrain.js` → `docHasVendor(doc, vendor)` | Vendor must appear in `versionSearchTags` or product name/brand |
| OS product | `server.js` (OS branch) | `android` / `ios` / `windows` from question, else default `android` |
| OS doc | `releasetrain.js` → `findOSByProductAndDate(...)` | Filter by product + date or latest timestamp |

## 6. Common issues

| Symptom | Likely cause |
|--------|----------------|
| `parsedPatchVendor` is `null` but you expected a vendor | Question doesn’t match regex (e.g. missing “patch for …” or vendor has punctuation). |
| Always OS flow for a patch question | Question has no “patch for …” and no `linux` word → falls through to OS. |
| Wrong patch version | `findLinuxPatchByDate` fell back to `items[0]` when date didn’t match; check `step2_dataFetched.itemsCount` and `step3_matchedDoc.versionSearchTags`. |
| “I don’t know” for Linux | Search returned 0 items or matched doc doesn’t have `linux` in tags; use trace `fetch=1` to see `matchedDocHasVendor`. |

## 7. Direct API checks

```bash
# Vendors list (names only)
curl -s http://localhost:3000/api/vendors | head -c 200

# Full answer (same as UI)
curl -s -X POST http://localhost:3000/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the patch for Linux on 14-02-2026?"}' | jq
```
