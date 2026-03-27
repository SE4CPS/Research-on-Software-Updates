# Publish ReleaseHub online

You need **two public URLs** (or one host that serves both):

1. **API** — Node/Express (`POST /answer`, `/api/*`)
2. **Frontend** — static files from `npm run build` in `frontend/`, built with **`VITE_API_URL`** pointing at your API (HTTPS)

---

## Before you deploy

- Push the project to **GitHub** (or GitLab/Bitbucket).
- **No secrets** are required for ReleaseTrain (public API). Optional: `REDIS_URL` for last-prompts / future persistence.
- **CORS** is open (`*`) in `server.js` — fine for demos; tighten for production.

---

## Option A — Render (simple, free tier)

### 1) API (Web Service)

1. [render.com](https://render.com) → **New +** → **Web Service**
2. Connect your repo, root directory: **/** (repo root)
3. **Runtime:** Node  
4. **Build command:** `npm install`  
5. **Start command:** `node src/server.js`  
6. **Instance type:** Free (spins down when idle — first request may be slow)
7. Deploy → copy URL, e.g. `https://releasehub-api.onrender.com`

### 2) Frontend (Static Site)

1. **New +** → **Static Site**
2. Same repo, **root directory:** `frontend`
3. **Build command:** `npm install && npm run build`
4. **Publish directory:** `dist`
5. **Environment** (important — used at build time):

   | Key | Value |
   |-----|--------|
   | `VITE_API_URL` | `https://releasehub-api.onrender.com` |

   Use your **exact** API URL, **no** trailing slash.

6. Deploy → open the static URL (e.g. `https://releasehub.onrender.com`)

### 3) Retry build after API exists

If the first frontend build ran before the API URL existed, trigger **Manual Deploy** on the static site after setting `VITE_API_URL`.

---

## Option B — Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Add **two services** from the same repo:

**Service 1 — API**

- Root: `/` (default)
- **Start:** `node src/server.js`
- **Build:** `npm install`
- Railway sets `PORT` — your app already uses `process.env.PORT || 3000` ✓
- Copy the **public URL** for the API

**Service 2 — Frontend** (or use “Empty service” + Nixpacks)

- Some teams use a second Railway service with root `frontend`, build `npm install && npm run build`, and serve `dist` with a static server or **Caddy** — Railway templates change often; easiest path is **Render static** for UI + **Railway** for API only.

Alternatively: **one Docker Compose** on a VPS (DigitalOcean, Hetzner) with `docker compose up -d` and a domain → see Option D.

---

## Option C — Fly.io (Docker)

You already have `Dockerfile` + `docker-compose.yml`.

1. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/)
2. **API app** from repo root:

   ```bash
   fly launch --name releasehub-api --dockerfile Dockerfile
   ```

3. **Web app** — deploy `frontend/` with a Dockerfile that builds Vite and serves with Nginx; set `VITE_API_URL` as a **build arg** in `fly.toml` / `docker build` to your API URL.

Fly can run both apps in one region; use `https://releasehub-api.fly.dev` for `VITE_API_URL`.

---

## Option D — Single VPS (Docker Compose)

1. Rent a small VPS (e.g. Ubuntu).
2. Install Docker + Docker Compose.
3. Clone repo, open ports **80** (and **443** with Caddy/Traefik + Let’s Encrypt).
4. Put a reverse proxy in front of `web` (port 8080) → `https://yourdomain.com`, optional API on subdomain or same host path.

Your `docker-compose` already proxies `/answer` and `/api` from **web** → **api**, so users only need **one domain** pointing at port **8080** (or 80 via proxy).

---

## Environment variables (cheat sheet)

| Where | Variable | Purpose |
|--------|-----------|--------|
| API host | `PORT` | Usually set by platform (default 3000) |
| API host | `REDIS_URL` | Optional |
| **Frontend build** | `VITE_API_URL` | Full origin of API, e.g. `https://api.example.com` — **no** trailing slash |

---

## “Backend error: 500” (or 502)

Your API code usually returns **502** when ReleaseTrain fails or an exception is thrown; **500** often comes from the **platform** or a **bad proxy**, not from the handler’s normal path.

| Symptom | What to check |
|--------|----------------|
| **500 / 502** right away | Is the API URL correct in **`VITE_API_URL`**? Rebuild the frontend after changing it. |
| **500** from static host | The UI may be **POSTing to the wrong host** (e.g. static site URL has no `/answer`). `VITE_API_URL` must be the **API** origin only. |
| **502 / bad gateway** | API process crashed, sleeping (free tier), or not listening on **`PORT`**. Open **`GET https://YOUR_API/health`** — should return `{"ok":true,...}`. |
| **Cold start** (Render free) | First request after idle can time out; retry once. |

Local check:

```bash
curl -sS -X POST http://localhost:3000/answer \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the version of OS Android on 02-14-2026?"}'
```

The server listens on **`0.0.0.0`** so Docker and PaaS can reach it.

---

## Checklist after deploy

- [ ] `GET https://YOUR_API/api/vendors` returns JSON  
- [ ] `POST https://YOUR_API/answer` with `{"question":"What is the patch for Linux on 02-14-2026?"}` works  
- [ ] UI loads and questions return answers (open browser **Network** tab if not)  
- [ ] If UI calls wrong host, rebuild frontend with correct `VITE_API_URL`

---

## `render.yaml` (optional)

The repo includes **`render.yaml`** for Render **Blueprints**. Create a Blueprint from the repo, then in the Render dashboard set **`VITE_API_URL`** on the static site to your API URL (e.g. `https://releasehub-api.onrender.com`) and redeploy the static site.

For client-side routing (if you add routes later), add a **rewrite** rule in the static site settings: `/*` → `/index.html`.
