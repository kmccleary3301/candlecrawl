# Getting Started

This guide is the shortest practical path from clone to a healthy local CandleCrawl instance.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| Python `3.12` | Matches CI |
| `pip` / virtualenv | Standard Python workflow |
| Playwright Chromium runtime | Required for browser-backed scrape paths |
| Optional Redis | Nice to have for crawl job persistence; not required for local smoke use |

## Installation

```bash
git clone https://github.com/kmccleary3301/candlecrawl.git
cd candlecrawl

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

## Running The API

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 3010
```

If you want an alternate port:

```bash
PORT=3010 python -m uvicorn app.main:app --host 0.0.0.0 --port 3010
```

## Health Verification

```bash
curl http://127.0.0.1:3010/health | jq
```

Healthy example:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "browserReady": true,
  "browserError": null
}
```

Degraded example when Playwright Chromium is missing:

```json
{
  "status": "degraded",
  "version": "1.0.0",
  "browserReady": false,
  "browserError": "Playwright Chromium runtime missing; run `python -m playwright install chromium`"
}
```

## First Scrape

```bash
curl -sS http://127.0.0.1:3010/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown", "links"],
    "onlyMainContent": true
  }' | jq
```

## Optional Provider Configuration

Provider-backed routes are optional. They are only needed for search, external fallback scraping, or Hermes BCAS flows.

```bash
export SERPER_DEV_API_KEY=...
export SCRAPE_DO_API_KEY=...
export OPENROUTER_API_KEY=...
```

Quick provider smoke test:

```bash
python -m app.scripts.provider_smoketests
```

## Useful Development Commands

Run the full test suite:

```bash
pytest -q
```

Run focused endpoint/runtime tests:

```bash
pytest -q tests/test_api.py tests/test_scraper_runtime.py
```

## Common Failure Modes

### Browser runtime missing

Symptom:

- `/health` reports `browserReady: false`
- browser-rendered scrape requests fail immediately

Fix:

```bash
python -m playwright install chromium
```

### Redis unavailable

Symptom:

- startup logs indicate Redis is unavailable

Impact:

- crawl jobs fall back to in-memory storage
- local dev is still usable

### Provider routes failing

Symptom:

- `/v2/search` or Hermes provider-backed routes return auth/provider errors

Fix:

- verify the corresponding API key is set,
- re-run `python -m app.scripts.provider_smoketests`.

## Next Reading

- [API And Operations](./API_AND_OPERATIONS.md)
- [Architecture](./ARCHITECTURE.md)
- [Branch Protection Policy](./BRANCH_PROTECTION_POLICY.md)
