# API And Operations

This document is the operational companion to the README. It focuses on endpoint inventory, request examples, and the behaviors that matter when you are actually running CandleCrawl.

## Endpoint Inventory

### Core

| Endpoint | Method | Notes |
| --- | --- | --- |
| `/health` | `GET` | Health plus browser runtime readiness |
| `/v1/scrape` | `POST` | Single scrape |
| `/v2/scrape` | `POST` | Firecrawl-style `v2` scrape |
| `/v1/scrape/bulk` | `POST` | Bulk scrape |
| `/v2/batch/scrape` | `POST` | Async batch scrape |
| `/v1/map` | `POST` | URL discovery |
| `/v2/map` | `POST` | URL discovery with `v2` payload |
| `/v1/crawl` | `POST` | Crawl start |
| `/v2/crawl` | `POST` | Crawl start with `v2` contract |
| `/v2/search` | `POST` | Provider-backed search |
| `/v2/extract` | `POST` | Multi-URL scrape/extract surface |

### Hermes bridge

| Endpoint | Method | Notes |
| --- | --- | --- |
| `/v1/hermes/leads/search` | `POST` | Serper-backed lead/entity search |
| `/v1/hermes/leads/enrich` | `POST` | Domain enrichment by scraping |
| `/v1/hermes/external-scrape` | `POST` | Scrape.do-backed fallback |
| `/v1/hermes/compose` | `POST` | OpenRouter-backed composition |
| `/v1/hermes/research` | `POST` | BCAS research flow |
| `/v1/hermes/costs/*` | `GET` | Cost telemetry |

## Health Semantics

`/health` does more than return `"ok"`.

Important fields:

| Field | Meaning |
| --- | --- |
| `status` | `healthy` or `degraded` |
| `browserReady` | whether Playwright Chromium is available |
| `browserError` | concrete operational error if browser runtime is missing |

This is intentional. Browser failure should show up in health, not only when a request later trips a rendering path.

## Example Requests

### `v2/scrape`

```bash
curl -sS http://127.0.0.1:3010/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown", "links"],
    "onlyMainContent": true,
    "waitFor": 1000
  }'
```

### `v2/search`

```bash
curl -sS http://127.0.0.1:3010/v2/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "site:openai.com reasoning models",
    "limit": 5,
    "sources": [{"type": "web"}, {"type": "news"}]
  }'
```

### `v2/crawl`

```bash
curl -sS http://127.0.0.1:3010/v2/crawl \
  -H 'Content-Type: application/json' \
  -H 'X-Idempotency-Key: crawl-demo-001' \
  -d '{
    "url": "https://example.com",
    "limit": 20,
    "maxDepth": 2,
    "includeSubdomains": false
  }'
```

## Operational Notes

### Redis optionality

If Redis is unavailable, CandleCrawl falls back to in-memory job storage. That is acceptable for local development and smoke runs, but you should not confuse it with durable multi-process orchestration.

### Browser runtime failures

If Playwright Chromium is missing:

- `/health` degrades,
- browser-rendered requests fail with a clear diagnostic,
- fix is:

```bash
python -m playwright install chromium
```

### Provider failures

Provider-backed routes fail independently from core scrape/crawl routes. A broken Serper or Scrape.do key should not be interpreted as a broken core scrape service.

Recommended check:

```bash
python -m app.scripts.provider_smoketests
```

## Contract And Compatibility

The contract file lives at:

- [`contracts/openapi-v1.yaml`](../contracts/openapi-v1.yaml)

CI currently validates that the file exists and contains required path entries. That is a minimal but intentional contract discipline layer.

## Release And Governance Links

- [Branch Protection Policy](./BRANCH_PROTECTION_POLICY.md)
- [Contributing](../CONTRIBUTING.md)
- [Versioning](../VERSIONING.md)
- [Changelog](../CHANGELOG.md)
