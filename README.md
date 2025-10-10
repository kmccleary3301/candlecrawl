# Firecrawl FastAPI (Compacted)

A production-ready, modular web scraping and crawling service built with FastAPI.

## Features
- HTTP + Playwright rendering with heuristics and fallbacks
- Markdown/HTML/links extraction; PDF/image metadata fast-paths
- Concurrent, depth-aware crawler with:
  - include/exclude path filters, domain/subdomain policy
  - robots.txt and crawl-delay politeness
  - global budgets (max_pages, max_time, max_bytes)
  - blocked detection (captcha, ip_block, rate_limited)
  - cancel/export endpoints
- Resilient HTTP client with retries/backoff and Retry-After support
- Frontier abstraction (in-memory) for forward-compatible Redis frontier
- Lifespan startup/shutdown

## API
- POST /v1/scrape
- POST /v1/scrape/bulk
- GET  /v1/scrape/{url}
- POST /v1/crawl
- GET  /v1/crawl/{id}
- POST /v1/crawl/{id}/cancel
- GET  /v1/crawl/{id}/export?format=jsonl
- POST /v1/map

See `app/models.py` for request/response models.

## Config (env)
- HOST, PORT, REDIS_URL
- RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
- default_timeout, max_concurrent_requests, user_agent
- retry_max_attempts, backoff_base_ms, backoff_max_ms

## Run
```
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 3010
```

## Hermes Providers (Serper.dev, Scrape.do, OpenRouter)

Environment variables (loaded via `.env` or process env):
- `SERPER_DEV_API_KEY` — Serper.dev API key
- `SCRAPE_DO_API_KEY` — Scrape.do API key
- `OPENROUTER_API_KEY` — OpenRouter API key

Smoke tests (optional, use your keys):
```
python -m app.scripts.provider_smoketests
```

Provider classes live in `app/providers/` with Pydantic request/response models.

## Hermes API (prototype)

Endpoints:
- POST `/v1/hermes/leads/search` → Serper.dev search
- POST `/v1/hermes/leads/enrich` → scrape domains via internal scraper
- POST `/v1/hermes/external-scrape` → Scrape.do fallback
- POST `/v1/hermes/compose` → OpenRouter chat completion

Example:
```
curl -X POST http://localhost:3010/v1/hermes/leads/search -H 'Content-Type: application/json' -d '{"query":"best devops startups","limit":5}'
```

## Tests
```
pytest -q
```

