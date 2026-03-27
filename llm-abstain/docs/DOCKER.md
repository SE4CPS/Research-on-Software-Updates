# Run ReleaseHub with Docker

## Full stack (API + UI)

From the project root:

```bash
docker compose up --build
```

- **UI:** http://localhost:8080  
- **API (direct):** http://localhost:3000  

The UI is built with an empty `VITE_API_URL`, so the browser calls `/answer` and `/api/*` on the same host (port 8080). Nginx forwards those paths to the `api` container.

## API only

```bash
docker compose up --build api
```

Or build and run the image manually:

```bash
docker build -t releasehub-api .
docker run --rm -p 3000:3000 releasehub-api
```

## Optional Redis

1. Uncomment `redis` service and `REDIS_URL` / `depends_on` in `docker-compose.yml`.
2. `docker compose up --build`.

## Development without Docker UI

Run the API in Docker and the frontend with Vite locally (proxies to localhost:3000):

```bash
docker compose up --build api
cd frontend && npm install && npm run dev
```

Vite’s `vite.config.js` already proxies `/answer` and `/api` to `http://localhost:3000`.
