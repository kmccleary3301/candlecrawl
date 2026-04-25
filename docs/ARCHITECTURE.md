# Architecture

This document describes CandleCrawl as it exists today, not as a hypothetical future control plane.

## High-Level Shape

CandleCrawl is a FastAPI service with four main internal layers:

1. request and compatibility handling,
2. retrieval and rendering,
3. crawl scheduling/state,
4. provider-backed auxiliary services.

In practical terms:

```text
HTTP request
  -> FastAPI endpoint
    -> request normalization / compatibility mapping
      -> scrape | crawl | map | search | extract path
        -> HTTP fetch and/or Playwright render
          -> extraction / link discovery / response shaping
            -> optional queue persistence / export / cost tracking
```

## Core Modules

| Module | Responsibility |
| --- | --- |
| `app/main.py` | FastAPI app, endpoint definitions, lifecycle, queue fallback logic |
| `app/scraper.py` | Core scraping service: file/HTML routing, browser escalation, browser runtime preflight |
| `app/models.py` | Pydantic request/response contracts |
| `app/frontier.py` | Crawl frontier abstraction with in-memory implementation |
| `app/http_client.py` | Retry/backoff HTTP wrapper |
| `app/config.py` | Runtime settings and environment parsing |
| `app/providers/*` | Search, scrape fallback, and LLM provider clients |
| `legacy/hermes_bcas.py` | Quarantined historical BCAS bridge; not included in installable packages |
| `app/cost_tracking.py` | Provider and stage-level cost accounting |

## Scraping Strategy

The scraper is deliberately layered.

### 1. HEAD probe

For many requests the service first performs a `HEAD` request to inspect content type and avoid unnecessarily launching a browser.

### 2. Route by content type

- `text/html` -> HTML processing path
- non-HTML -> file metadata / file-handling path

### 3. Escalate only when needed

The scraper chooses Playwright when the request actually requires it:

- browser actions are present,
- screenshot capture is requested,
- mobile emulation is requested,
- wait conditions imply JS completion,
- or HTTP retrieval clearly returns a JS-stub shell.

### 4. Normalize output

Responses are normalized into `FirecrawlDocument`-style payloads so callers do not need to understand whether content came from direct HTTP or browser rendering.

## Browser Runtime Preflight

The service now preflights Playwright Chromium at startup.

Why this matters:

- missing browsers are a deployment problem, not an application mystery,
- health now reports `browserReady` and `browserError`,
- runtime failures are expressed as a concrete operational fix rather than a vague file-not-found error.

## Crawl Model

The crawl system is intentionally small and practical.

Current behavior:

- crawl jobs are async,
- Redis/RQ is used when available,
- otherwise the service falls back to in-memory job storage,
- the frontier abstraction is isolated enough to be replaced later.

The current frontier implementation is `MemoryFrontier`, but the interface was written to preserve a path toward Redis-backed scheduling without rewriting every crawl endpoint.

## Crawl Policy Controls

The crawl path already includes several controls that matter in real systems:

- robots.txt awareness,
- crawl-delay politeness,
- include/exclude path filters,
- subdomain and external-link policy,
- max pages, max time, and max byte budgets,
- duplicate handling via canonicalization and query normalization options.

## Provider Layer

CandleCrawl uses dedicated provider adapters rather than mixing provider logic into endpoints.

| Provider | Current use |
| --- | --- |
| Serper | `v2/search`, Hermes entity search, news/image search |
| Scrape.do | external fallback scraping |
| OpenRouter | Hermes compose and BCAS research flows |

This is intentionally modular. Provider changes should primarily affect `app/providers/*`, not every endpoint.

## Hermes Bridge

The Hermes-facing routes exist because CandleCrawl has been used as a substrate for higher-level research workflows.

Those routes are useful, but they should be read as bridge surfaces rather than the primary public identity of the repo. The main public story remains scrape/crawl/map/search/extract infrastructure.

## Contract Discipline

The draft contract artifact at `contracts/openapi-v1.yaml` is part of CI. That is important because the repo is being used as a producer dependency for other systems.

Current contract stance:

- `v1/*` and `v2/*` routes coexist,
- `contracts/openapi-v1.yaml` tracks the evolving public-facing shape,
- compatibility is taken seriously,
- but the contract is still explicitly marked draft.

## Intended Evolution

The most obvious medium-term architecture direction is:

- Redis-backed frontier implementation,
- stronger distributed crawl orchestration,
- more explicit extraction contract depth,
- cleaner separation between public API and Hermes-specific bridge surfaces.

The current implementation is already useful. The goal is to extend it without needing to throw away the runtime shape that already works.
