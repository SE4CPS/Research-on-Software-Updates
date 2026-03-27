# ReleaseHub

Release intelligence UI + API: natural-language questions about OS versions and Linux patches, backed by [ReleaseTrain](https://releasetrain.io).

## Run locally

**Backend**

```bash
npm install
npm run dev
```

Default: **http://localhost:3000**

### Google Gemini (optional — answer phrasing only)

When `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set, successful **`status: answer`** responses can be **rewritten for readability** by Gemini. **ReleaseTrain data stays the source of truth**; if the model omits required literals (version / URL when present) or the call **errors or times out**, the API **falls back to the original deterministic text**.

```bash
export GEMINI_API_KEY="your_gemini_api_key"
# optional:
export GEMINI_MODEL="gemini-1.5-flash"
export GEMINI_TIMEOUT_MS=12000
```

**Frontend**

```bash
cd frontend && npm install && npm run dev
```

Set `frontend/.env` if the API is not proxied:

```env
VITE_API_URL=http://localhost:3000
```

## Docker — **yes, it is included**

From the **repository root**:

```bash
docker compose up --build
```

| Service | URL |
|--------|-----|
| Web (Nginx + React) | **http://localhost:8080** |
| API (direct) | **http://localhost:3000** |

Details: **[docs/DOCKER.md](./docs/DOCKER.md)**  
Files: `Dockerfile` (API), `docker-compose.yml`, `frontend/Dockerfile`, `frontend/nginx.conf`

## Good practices (LLM, API, Docker)

See **[docs/GOOD_PRACTICES.md](./docs/GOOD_PRACTICES.md)** — including guidance for **Gemini / LLM** integration (source of truth, timeouts, fallback, secrets, privacy) and production notes.

## Publish online

See **[docs/DEPLOY.md](./docs/DEPLOY.md)** (Render, Railway, Fly.io, VPS). Optional **[render.yaml](./render.yaml)** for Render Blueprints — set `VITE_API_URL` to your deployed API URL.

## Docs

- [docs/DEBUGGING_DATA_EXTRACTION.md](./docs/DEBUGGING_DATA_EXTRACTION.md)
- [docs/DATA_SCHEMA_OS.md](./docs/DATA_SCHEMA_OS.md)
- [docs/DATA_SCHEMA_LINUX_PATCH.md](./docs/DATA_SCHEMA_LINUX_PATCH.md)
- [docs/SYSTEM_ARCHITECTURE_AND_TEST_PROMPTS.txt](./docs/SYSTEM_ARCHITECTURE_AND_TEST_PROMPTS.txt)
