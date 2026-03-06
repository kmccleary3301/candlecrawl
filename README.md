# CandleCrawl (Robyn)

A production-ready, modular web scraping and crawling service built with Robyn.

## Features
- HTTP + Playwright rendering with heuristics and fallbacks
- Markdown/HTML/links extraction plus PDF/image metadata fast paths
- Concurrent, depth-aware crawler with include/exclude path filters, robots.txt politeness, and budgets
- v1 + v2 Firecrawl-compatible endpoints
- Hermes provider endpoints (Serper, Scrape.do, OpenRouter)
- Redis-backed or in-memory job/cache/idempotency storage

## API (high level)
- `/health`
- `/v1/*` scrape/crawl/map/batch/hermes endpoints
- `/v2/*` scrape/crawl/map/batch/search/extract endpoints

See `app/models.py` and `contracts/openapi-v1.yaml` for schemas.

## Configuration
Environment variables are loaded from `.env` and process env via `pydantic-settings`.
Key settings:
- `HOST`, `PORT`, `REDIS_URL`
- `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`
- `DEFAULT_TIMEOUT`, `MAX_CONCURRENT_REQUESTS`, `USER_AGENT`
- provider keys: `SERPER_DEV_API_KEY`, `SCRAPE_DO_API_KEY`, `OPENROUTER_API_KEY`

## Development workflow (uv)
```bash
uv sync
uv run python -m app.main
```
Default port is `3002`; set `PORT` to override.

Canonical checks:
```bash
uv run ruff check .
uv run pytest -q
uv run python -m app.contract_check
```

## Docker
```bash
docker build -t candlecrawl .
docker run --rm -p 3010:3010 candlecrawl
```

## Optional provider smoke tests
```bash
uv run python -m app.scripts.provider_smoketests
```
