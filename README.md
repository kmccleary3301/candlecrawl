# CandleCrawl

<p align="left">
  Open-source crawling, scraping, mapping, search, and extraction infrastructure for developer-heavy research systems.
</p>

<p align="left">
  <a href="https://github.com/kmccleary3301/candlecrawl/actions/workflows/ci.yml"><img src="https://github.com/kmccleary3301/candlecrawl/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
  <a href="https://github.com/kmccleary3301/candlecrawl/actions/workflows/security.yml"><img src="https://github.com/kmccleary3301/candlecrawl/actions/workflows/security.yml/badge.svg" alt="Security scan status"></a>
  <a href="https://github.com/kmccleary3301/candlecrawl/actions/workflows/release-artifact.yml"><img src="https://github.com/kmccleary3301/candlecrawl/actions/workflows/release-artifact.yml/badge.svg" alt="Release artifact status"></a>
  <a href="https://github.com/kmccleary3301/candlecrawl/releases"><img src="https://img.shields.io/github/v/release/kmccleary3301/candlecrawl?display_name=tag" alt="Latest release"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/github/license/kmccleary3301/candlecrawl" alt="License"></a>
</p>

<p align="left">
  <a href="./contracts/openapi-v1.yaml"><img src="https://img.shields.io/badge/contract-openapi%20v1%20draft-1f6feb" alt="OpenAPI contract"></a>
  <img src="https://img.shields.io/badge/python-3.12-3776ab" alt="Python 3.12">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688" alt="FastAPI">
  <img src="https://img.shields.io/badge/browser-Playwright-45ba63" alt="Playwright">
  <img src="https://img.shields.io/badge/license-MIT-black" alt="MIT license">
</p>

CandleCrawl is a self-hostable web ingestion service for teams that need more than a thin scrape endpoint. It combines:

- direct HTTP retrieval with retry, backoff, and `Retry-After` handling,
- Playwright rendering for JS-heavy pages and action-driven workflows,
- depth-aware crawling with politeness controls and export paths,
- Firecrawl-style `v1` and `v2` compatibility surfaces,
- provider-backed search and extraction helpers,
- optional Hermes-facing BCAS and cost-tracking endpoints,
- a repo structure intended to remain useful as a standalone system or as a substrate inside larger research stacks.

The target audience here is not "people who want a demo". It is engineers, infra-minded researchers, and platform builders who need a service they can inspect, adapt, and embed into higher-level intelligence systems.

## Table Of Contents

- [What CandleCrawl Is For](#what-candlecrawl-is-for)
- [Capability Snapshot](#capability-snapshot)
- [Quick Start](#quick-start)
- [Installation And Setup](#installation-and-setup)
- [Configuration](#configuration)
- [API Surface](#api-surface)
- [Usage Examples](#usage-examples)
- [Architecture Notes](#architecture-notes)
- [Repository Map](#repository-map)
- [Testing, CI, And Release Discipline](#testing-ci-and-release-discipline)
- [Documentation Index](#documentation-index)
- [Current Design Boundaries](#current-design-boundaries)
- [Contributing](#contributing)
- [License](#license)

## What CandleCrawl Is For

CandleCrawl exists for cases where "fetch me a page" is too small an abstraction and "run a full browser farm with a huge control plane" is too much overhead.

Typical uses:

- building research agents that need repeatable scrape, map, crawl, and extract primitives,
- standing up a local or internal alternative to managed crawl/scrape APIs,
- feeding downstream retrieval, RAG, indexing, or dossier-generation systems,
- giving another system a stable HTTP boundary over page acquisition, browser rendering, and crawl policy.

In practice the repo currently serves three adjacent roles:

1. A standalone crawl/scrape/search/extract API.
2. A Firecrawl-style compatibility layer for `v1` / `v2` request shapes.
3. A provider-enabled substrate used by Hermes for BCAS-style research and enrichment flows.

## Capability Snapshot

| Area | Status | What exists now | Notes |
| --- | --- | --- | --- |
| Single-page scrape | 🟢 | `POST /v1/scrape`, `POST /v2/scrape` | HTTP-first, Playwright when required |
| Batch scrape | 🟢 | `POST /v1/scrape/bulk`, `POST /v2/batch/scrape` | Concurrent and cancelable |
| Site map discovery | 🟢 | `POST /v1/map`, `POST /v2/map` | Includes sitemap probing and link discovery |
| Async crawl jobs | 🟢 | `POST /v1/crawl`, `POST /v2/crawl` | In-memory or Redis-backed job state |
| Crawl export | 🟢 | `/v1/crawl/{id}/export` | JSONL export path exists today |
| Search aggregation | 🟢 | `POST /v2/search` | Serper-backed web/news/image search |
| Structured extraction | 🟢 | `POST /v2/extract` | Current implementation is scrape-first, structure-second |
| JS rendering | 🟢 | Playwright-backed | Startup preflight now surfaces browser readiness |
| Crawl politeness | 🟢 | robots.txt, crawl delay, path rules, budgets | Designed for extension, not just happy path demos |
| Provider abstraction | 🟢 | Serper, Scrape.do, OpenRouter | Useful both standalone and for Hermes integration |
| Cost telemetry | 🟢 | `/v1/hermes/costs/*` | Research-job oriented, not generic billing |
| Hermes BCAS research | 🟡 | `/v1/hermes/research` and helpers | Powerful, but still a compatibility/bridge surface |
| Public contract discipline | 🟡 | `contracts/openapi-v1.yaml` | Draft contract, versioned intentionally |
| Distributed frontier | 🔵 | `MemoryFrontier` today | Redis/Ray-ready direction documented, not fully shipped |

## Quick Start

### Local Development

```bash
git clone https://github.com/kmccleary3301/candlecrawl.git
cd candlecrawl

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -m playwright install chromium
python -m uvicorn app.main:app --host 0.0.0.0 --port 3010
```

### Package-Oriented Development

CandleCrawl also exposes a lightweight installable package surface for SDK,
CLI, and Hermes integration work:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[service,browser,pdf,ocr,test]"

candlecrawl version
candlecrawl doctor
candlecrawl serve --host 0.0.0.0 --port 3010
```

For base SDK-only usage, install without service extras:

```bash
pip install -e .
python -c "import candlecrawl, candlecrawl.client; print(candlecrawl.__version__)"
```

The package CLI is the preferred integration direction for downstream systems.
Direct `uvicorn app.main:app` remains available as a legacy development path
while the server internals are being moved behind the public `candlecrawl`
package boundary.

Health check:

```bash
curl http://127.0.0.1:3010/health | jq
```

Expected shape:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "browserReady": true,
  "browserError": null
}
```

### Docker

```bash
docker build -t candlecrawl:local .
docker run --rm -p 3010:3010 candlecrawl:local
```

### First Useful Request

```bash
curl -sS http://127.0.0.1:3010/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown", "links"],
    "onlyMainContent": true
  }' | jq
```

## Installation And Setup

### Requirements

| Requirement | Why it matters |
| --- | --- |
| Python `3.12` | Matches CI and tested runtime |
| Playwright Chromium runtime | Required for browser-rendered paths |
| Optional Redis | Used for queue/state when available |
| Optional provider keys | Required only for provider-backed search/extract/BCAS paths |

### Environment Variables

Core runtime settings come from `app/config.py` and are loaded via `.env` or process environment.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | API bind host |
| `PORT` | `3002` | API port |
| `REDIS_URL` | local Redis URL | Crawl queue/job persistence when Redis is available |
| `RATE_LIMIT_REQUESTS` | `100` | Rate limit numerator |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `DEFAULT_TIMEOUT` | `30` | Default HTTP timeout |
| `MAX_CONCURRENT_REQUESTS` | `5` | Concurrency cap for internal async work |
| `USER_AGENT` | Chrome-like UA | Default fetch/browser UA |
| `RETRY_MAX_ATTEMPTS` | `3` | HTTP retry ceiling |
| `BACKOFF_BASE_MS` | `200` | Retry backoff base |
| `BACKOFF_MAX_MS` | `3000` | Retry backoff cap |
| `CACHE_ENABLED` | `true` | Cache toggle for Firecrawl-style semantics |
| `CACHE_DEFAULT_MAX_AGE_MS` | `172800000` | Default cache TTL in milliseconds |

Optional provider configuration:

| Variable | Used by | Needed for |
| --- | --- | --- |
| `SERPER_DEV_API_KEY` | `v2/search`, Hermes search flows | Provider-backed search |
| `SCRAPE_DO_API_KEY` | external fallback routes | Secondary fetch/render fallback |
| `OPENROUTER_API_KEY` | Hermes compose/BCAS flows | LLM-backed research synthesis |
| `OPENAI_API_KEY` | optional extract-related work | Future extraction integrations |

### Browser Runtime Preflight

The service now checks Playwright Chromium availability at startup. If the browser runtime is missing, `/health` degrades cleanly and reports a concrete fix:

> `Playwright Chromium runtime missing; run python -m playwright install chromium`

That is deliberate. A broken browser runtime should be obvious before you discover it mid-scrape.

For more operational detail, see [docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md).

## Configuration

### Request Shaping

Scrape and crawl requests support:

- format selection: `markdown`, `html`, `rawHtml`, `links`, `screenshot`,
- tag inclusion/exclusion,
- main-content filtering,
- custom headers,
- wait/timeout controls,
- mobile emulation,
- optional browser actions for `v2/scrape`,
- crawl include/exclude paths,
- subdomain/external-link policy,
- dedupe and query-parameter handling,
- time/byte/page/concurrency budgets.

### Storage And Queue Behavior

| Mode | Behavior |
| --- | --- |
| Redis available | Crawl job metadata and queues use Redis/RQ |
| Redis unavailable | Service falls back to in-memory job storage |
| Browser unavailable | Browser-backed routes degrade and health exposes the cause |

### Compatibility Positioning

| Surface | Intent |
| --- | --- |
| `v1/*` | Lightweight Firecrawl-style compatibility and existing callers |
| `v2/*` | Cleaner evolving contract surface |
| `/v1/hermes/*` | Internal compatibility bridge for Hermes research and enrichment flows |

## API Surface

### Core Endpoints

| Endpoint | Method | Purpose | Notes |
| --- | --- | --- | --- |
| `/health` | `GET` | Service health | Includes browser runtime readiness |
| `/v1/scrape` | `POST` | Scrape one URL | Classic Firecrawl-style request |
| `/v2/scrape` | `POST` | Scrape one URL with richer `v2` semantics | Supports actions and Firecrawl-style payload mapping |
| `/v1/scrape/bulk` | `POST` | Bulk scrape | Returns per-URL responses |
| `/v2/batch/scrape` | `POST` | Async batch scrape | Poll, cancel, inspect errors |
| `/v1/map` | `POST` | Discover links from a root URL | Lightweight mapping surface |
| `/v2/map` | `POST` | Discover links with `v2` payload | Search/filter aware |
| `/v1/crawl` | `POST` | Start crawl job | Async crawl lifecycle |
| `/v2/crawl` | `POST` | Start crawl job with `v2` contract | Includes idempotency support |
| `/v2/search` | `POST` | Provider-backed search | Web, news, image search today |
| `/v2/extract` | `POST` | Extract structured docs from URLs | Current implementation is URL scrape aggregation |

### Crawl Lifecycle Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/v1/crawl/{job_id}` | `GET` | Poll crawl status |
| `/v1/crawl/{job_id}/cancel` | `POST` | Cancel crawl |
| `/v1/crawl/{job_id}/export` | `GET` | Export results |
| `/v2/crawl/{job_id}` | `GET` | Poll `v2` crawl job |
| `/v2/crawl/{job_id}` | `DELETE` | Cancel `v2` crawl job |
| `/v2/crawl/{job_id}/errors` | `GET` | Error inspection |
| `/v2/batch/scrape/{job_id}` | `GET` | Poll batch scrape |
| `/v2/batch/scrape/{job_id}` | `DELETE` | Cancel batch scrape |
| `/v2/batch/scrape/{job_id}/errors` | `GET` | Error inspection |

### Hermes Bridge Endpoints

These are not the main public story of CandleCrawl, but they are real and useful.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/v1/hermes/leads/search` | `POST` | Search via Serper for lead/entity discovery |
| `/v1/hermes/leads/enrich` | `POST` | Scrape/enrich a set of domains |
| `/v1/hermes/external-scrape` | `POST` | Provider-backed fallback scrape |
| `/v1/hermes/compose` | `POST` | LLM composition helper |
| `/v1/hermes/research` | `POST` | BCAS-style research orchestration |
| `/v1/hermes/costs/*` | `GET` | Cost and provider-usage telemetry |

### Contract Artifact

The evolving draft contract lives at:

- [`contracts/openapi-v1.yaml`](./contracts/openapi-v1.yaml)

That file is part of CI, release discipline, and consumer-compatibility signaling. It is not filler.

## Usage Examples

### 1. Minimal `v2` scrape

```bash
curl -sS http://127.0.0.1:3010/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown", "html", "links"],
    "onlyMainContent": true,
    "timeout": 15000
  }'
```

### 2. Action-driven browser scrape

```bash
curl -sS http://127.0.0.1:3010/v2/scrape \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com",
    "formats": ["markdown"],
    "actions": [
      {
        "type": "evaluate",
        "script": "document.body.insertAdjacentHTML(\"beforeend\", \"<p>runtime-marker</p>\");"
      }
    ]
  }'
```

### 3. Start an async crawl

```bash
curl -sS http://127.0.0.1:3010/v2/crawl \
  -H 'Content-Type: application/json' \
  -H 'X-Idempotency-Key: demo-crawl-001' \
  -d '{
    "url": "https://example.com",
    "limit": 25,
    "maxDepth": 2,
    "includeSubdomains": false,
    "allowExternalLinks": false
  }'
```

Poll it:

```bash
curl -sS http://127.0.0.1:3010/v2/crawl/<job_id> | jq
```

### 4. Discover URLs with `map`

```bash
curl -sS http://127.0.0.1:3010/v2/map \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com",
    "limit": 250,
    "search": "blog"
  }'
```

### 5. Provider-backed search

```bash
curl -sS http://127.0.0.1:3010/v2/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "site:openai.com reasoning models",
    "limit": 5,
    "sources": [{"type": "web"}, {"type": "news"}]
  }'
```

### 6. Multi-URL extract

```bash
curl -sS http://127.0.0.1:3010/v2/extract \
  -H 'Content-Type: application/json' \
  -d '{
    "urls": [
      "https://example.com",
      "https://www.iana.org/domains/reserved"
    ],
    "scrapeOptions": {
      "formats": ["markdown"],
      "onlyMainContent": true
    }
  }'
```

### 7. Hermes BCAS research call

```bash
curl -sS http://127.0.0.1:3010/v1/hermes/research \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Map recent memory-augmented model work associated with Ali Behrouz",
    "tier": "TARGETED",
    "max_searches": 4,
    "model": "openai/gpt-5-nano",
    "use_preplanning": true
  }'
```

### Python Example

```python
import httpx

payload = {
    "url": "https://example.com",
    "formats": ["markdown", "links"],
    "onlyMainContent": True,
}

with httpx.Client(base_url="http://127.0.0.1:3010", timeout=30.0) as client:
    response = client.post("/v2/scrape", json=payload)
    response.raise_for_status()
    data = response.json()

print(data["success"])
print(data["data"]["metadata"]["title"])
print(data["data"]["markdown"][:200])
```

## Architecture Notes

At a high level, CandleCrawl is built around a deliberately small set of moving parts:

1. **FastAPI request layer**
   - request validation,
   - compatibility shims,
   - rate limiting,
   - job lifecycle endpoints.
2. **Scraping service**
   - HEAD/GET probing,
   - HTML/file routing,
   - Playwright escalation,
   - markdown/html/link extraction.
3. **Crawl frontier**
   - currently in-memory,
   - intentionally shaped for Redis-backed replacement.
4. **Provider adapters**
   - Serper for search,
   - Scrape.do for fallback scraping,
   - OpenRouter for Hermes composition/BCAS paths.
5. **Operational layers**
   - browser runtime preflight,
   - queue fallback when Redis is absent,
   - cost tracking,
   - contract and CI discipline.

Request flow, simplified:

```text
Client
  -> FastAPI endpoint
    -> request normalization / rate limiting
      -> scrape | map | crawl | search | extract path
        -> HTTP client and/or Playwright
          -> extraction + normalization
            -> response envelope / job persistence / export
```

The service intentionally keeps the frontier, provider clients, and browser rendering logic separate enough that you can swap internals without changing the HTTP contract every week.

For deeper discussion, see:

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- [QueryLake_Integration_Plan.md](./QueryLake_Integration_Plan.md)

## Repository Map

```text
candlecrawl/
├── app/
│   ├── main.py                  # FastAPI app, compatibility routes, crawl lifecycle
│   ├── scraper.py               # HTTP-first scraping + Playwright escalation
│   ├── models.py                # Pydantic request/response contracts
│   ├── frontier.py              # Crawl frontier abstraction; in-memory implementation today
│   ├── http_client.py           # Retry/backoff HTTP client wrapper
│   ├── config.py                # Environment-backed runtime settings
│   ├── metrics.py               # Prometheus metric primitives
│   ├── cost_tracking.py         # Provider and stage-level cost accounting
│   ├── cost_endpoints.py        # Hermes cost telemetry API
│   ├── model_pricing.py         # Model cost estimation helpers
│   ├── chunking.py              # Search/research-oriented chunking helpers
│   ├── hermes_bcas.py           # BCAS-style research orchestration bridge
│   ├── providers/
│   │   ├── base.py              # Shared provider exception types
│   │   ├── serper.py            # Search/news/image provider client
│   │   ├── scrapedo.py          # Scrape.do fallback client
│   │   └── openrouter.py        # LLM provider client for BCAS and compose flows
│   └── scripts/
│       └── provider_smoketests.py
├── contracts/
│   └── openapi-v1.yaml          # Draft public contract artifact tracked by CI
├── docs/
│   ├── GETTING_STARTED.md       # Installation, env, smoke checks, troubleshooting
│   ├── ARCHITECTURE.md          # Internal structure, request flow, queue model
│   ├── API_AND_OPERATIONS.md    # Endpoint catalog, examples, operational behaviors
│   └── BRANCH_PROTECTION_POLICY.md
├── tests/
│   ├── test_api.py              # Endpoint and response-shape coverage
│   ├── test_frontier.py         # Frontier behavior
│   ├── test_http_client.py      # Retry/backoff semantics
│   ├── test_crawl_policies.py   # Crawl policy rules
│   ├── test_crawl_extreme.py    # Stress-ish edge cases
│   ├── test_hermes_endpoints.py # Hermes bridge routes
│   ├── test_providers.py        # Provider client coverage
│   ├── test_performance.py      # Performance-oriented checks
│   └── test_scraper_runtime.py  # Browser runtime preflight and error normalization
├── .github/workflows/
│   ├── ci.yml
│   ├── security.yml
│   ├── release-artifact.yml
│   └── consumer-compat-dispatch.yml
├── Dockerfile
├── QueryLake_Integration_Plan.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── VERSIONING.md
└── README.md
```

## Testing, CI, And Release Discipline

### Local Test Commands

```bash
pytest -q
```

Focused commands:

```bash
pytest -q tests/test_api.py tests/test_frontier.py tests/test_http_client.py
pytest -q tests/test_scraper_runtime.py
```

Optional provider smoke tests:

```bash
python -m app.scripts.provider_smoketests
```

### CI And Policy Surface

| Check | Purpose |
| --- | --- |
| `CandleCrawl CI / unit-and-api` | Core unit/API behavior |
| `CandleCrawl CI / contract-validation` | Contract artifact presence and required path validation |
| `CandleCrawl Security Scan / pip-audit` | Dependency audit |
| `CandleCrawl Release Artifact` | Image build + artifact export on tags |
| `CandleCrawl Consumer Compatibility Dispatch` | Optional downstream compatibility signaling |

Branch protection policy is documented at [docs/BRANCH_PROTECTION_POLICY.md](./docs/BRANCH_PROTECTION_POLICY.md).

### Versioning

CandleCrawl follows semantic versioning. See [VERSIONING.md](./VERSIONING.md).

## Documentation Index

| Document | What it covers |
| --- | --- |
| [docs/GETTING_STARTED.md](./docs/GETTING_STARTED.md) | Setup, env vars, first boot, smoke checks |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Runtime design, module responsibilities, queue and render model |
| [docs/API_AND_OPERATIONS.md](./docs/API_AND_OPERATIONS.md) | Endpoint catalog, request examples, health and troubleshooting |
| [QueryLake_Integration_Plan.md](./QueryLake_Integration_Plan.md) | Planned QueryLake/Ray alignment |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Contribution workflow and contract-change discipline |
| [CHANGELOG.md](./CHANGELOG.md) | Release history |
| [VERSIONING.md](./VERSIONING.md) | Versioning policy |

## Current Design Boundaries

Some constraints are deliberate:

- CandleCrawl is not pretending to be a full browser-farm orchestration platform.
- The current frontier implementation is intentionally simple and forward-compatible rather than prematurely distributed.
- `v2/extract` is useful today, but it is still closer to scrape aggregation than a final-form schema-first extraction engine.
- Hermes compatibility routes exist because they are operationally useful, not because they define the whole repo.
- The contract artifact is still marked draft; compatibility matters, but the surface is still evolving.

That is the right tradeoff for the current project stage. The system is already useful, inspectable, and extensible without pretending it is finished.

## Contributing

Start with [CONTRIBUTING.md](./CONTRIBUTING.md). The short version:

- branch from `main`,
- add tests for behavior changes,
- keep contract-impacting changes explicit,
- run the relevant test suite before opening a PR.

If your change affects endpoint semantics, update [`contracts/openapi-v1.yaml`](./contracts/openapi-v1.yaml) and note the compatibility implications.

## License

CandleCrawl is released under the [MIT License](./LICENSE).
