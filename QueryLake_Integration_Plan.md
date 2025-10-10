# QueryLake Integration Plan

This document outlines how to compact the Firecrawl FastAPI app into a standalone repo and integrate it into QueryLake (Ray-based) at scale.

## 1. Compacting the Repo
- Include:
  - `app/` (all `.py` files)
  - `tests/` (pytest suites)
  - `requirements.txt`
  - `README.md` (this project spec)
  - `QueryLake_Integration_Plan.md` (this file)
- Exclude: upstream project files, node_modules, unrelated examples.
- Keep package paths `app.*` stable for import.

## 2. Architecture Alignment with QueryLake
- Stateless API service exposes HTTP endpoints.
- Crawling executed by workers (future Ray actors) consuming a frontier.
- Frontier abstraction: swap `MemoryFrontier` -> Redis-backed (in QueryLake):
  - Per-host queues (Redis lists) and global priority (sorted set).
  - Visited/inflight dedupe (sets/Bloom).
  - Leases/heartbeats for lost task requeue.
- HTTP layer remains with retry/backoff; add proxy support later via adapter interface.

## 3. Ray Integration Phases
### Phase A: Local Ray Orchestration
- Wrap crawl jobs in Ray tasks/actors:
  - `CrawlCoordinatorActor`: manages job budgets, pushes seeds to frontier, monitors metrics.
  - `CrawlWorkerActor`: pulls from frontier, fetches/extracts, pushes discovered links.
- Use Redis (QueryLake infra) for state.
- API `POST /v1/crawl` enqueues a job to coordinator; status polled from Redis.

### Phase B: Horizontal Scaling
- Scale `CrawlWorkerActor` pool across nodes.
- Enforce per-host politeness with per-host actor/semaphore and crawl-delay.
- Use backpressure: if Playwright pool saturated, reduce pull rate.

### Phase C: Data Plane & Exports
- Stream results (JSONL) to QueryLake object storage (S3/MinIO bucket) with presigned URLs.
- Store page artifacts (screenshots) in a bucket path per job.

## 4. Interfaces to Implement
- Frontier interface (done), Redis implementation (next):
  - Methods: enqueue, dequeue (with lease), mark_done (ack), seen, size, inflight_size.
- HTTP client adapters: allow proxies/IP pools later (provider adapters).
- Render adapter: pool of Playwright contexts; configurable.

## 5. Observability (QueryLake)
- Structured logs (JSON) with job_id/url/host/depth/duration/status.
- Metrics via QueryLake’s monitoring (no Prometheus required initially).
- Tracing optional via OpenTelemetry.

## 6. Security & Compliance
- robots.txt and crawl-delay respected; user_agent configurable.
- PII-safe logs; redact query params if configured.

## 7. Testing & CI in QueryLake
- Keep pytest unit/integration suites.
- Add mocked Redis-based frontier tests.
- Add integration tests with minimal Ray cluster in CI.

## 8. Migration Steps
1. Zip current compacted folder; publish internal artifact.
2. New repo in QueryLake org:
   - Push contents; set up CI (pytest matrix).
3. Implement RedisFrontier with identical interface.
4. Introduce Ray actors with feature flag; keep local execution path.
5. Add S3 export adapter; wire `GET /export` to presigned URL.

## 9. Backlog after Migration
- Pause/resume via frontier lease manipulation.
- Retries/backoff policy tuning; circuit breakers per host.
- Playwright pooling; JS-render heuristics tuning.
- Proxy adapter integration if needed.

