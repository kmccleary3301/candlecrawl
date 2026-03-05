
import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable
from urllib.parse import unquote

from loguru import logger
from pydantic import BaseModel, ValidationError
from robyn import ALLOW_CORS, Headers, Request, Response, Robyn, jsonify
from robyn.openapi import OpenAPI

from app import cost_endpoints as cost_api
from app import service
from app.config import settings
from app.cost_tracking import TierBudgetConfig
from app.framework_shims import HTTPException, RawResponse
from app.models import (
    BatchScrapeRequest,
    CrawlRequest,
    HermesComposeRequest,
    HermesEnrichRequest,
    HermesExternalScrapeRequest,
    HermesSearchRequest,
    MapRequest,
    ScrapeRequest,
)

OPENAPI = OpenAPI()
OPENAPI.openapi_file_override = True
app = Robyn(__file__, openapi=OPENAPI)
ALLOW_CORS(app, origins="*", headers="*")


@dataclass(slots=True)
class ServiceRequest:
    headers: dict[str, str]


class BackgroundTaskShim:
    def add_task(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
        asyncio.create_task(fn(*args, **kwargs))


class SimpleRateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    def limit(self, rule: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        # rule format: "<requests>/<window>seconds"
        parts = rule.split("/")
        max_requests = int(parts[0]) if len(parts) >= 1 else settings.rate_limit_requests
        window_seconds = settings.rate_limit_window
        if len(parts) >= 2 and parts[1].endswith("seconds"):
            raw = parts[1].replace("seconds", "")
            window_seconds = int(raw)

        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                request = _find_request(*args, **kwargs)
                identity = "unknown"
                if request is not None:
                    identity = getattr(request, "ip_addr", None) or "unknown"
                bucket_key = f"{identity}:{fn.__name__}"

                now = time.time()
                async with self._lock:
                    bucket = self._buckets[bucket_key]
                    while bucket and (now - bucket[0]) > window_seconds:
                        bucket.popleft()
                    if len(bucket) >= max_requests:
                        return _status_json({"detail": "Rate limit exceeded"}, 429)
                    bucket.append(now)

                return await fn(*args, **kwargs)

            return wrapper

        return decorator


limiter = SimpleRateLimiter()


def _find_request(*args: Any, **kwargs: Any) -> Request | None:
    for arg in args:
        if isinstance(arg, Request):
            return arg
    req = kwargs.get("request")
    return req if isinstance(req, Request) else None


def _headers_from_request(request: Request) -> dict[str, str]:
    try:
        return {str(k): str(v) for k, v in request.headers.items()}
    except Exception:
        return {}


def _service_request(request: Request) -> ServiceRequest:
    return ServiceRequest(headers=_headers_from_request(request))


def _query_param(request: Request, key: str, default: Any = None) -> Any:
    qp = request.query_params
    if hasattr(qp, "get"):
        try:
            val = qp.get(key, None)
        except TypeError:
            # Compatibility for query param implementations that don't support a default arg.
            val = qp.get(key)
        return default if val is None else val
    return default


def _query_param_int(request: Request, key: str, default: int, *, minimum: int | None = None) -> int:
    raw = _query_param(request, key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Query parameter '{key}' must be an integer") from exc
    if minimum is not None and value < minimum:
        raise HTTPException(status_code=422, detail=f"Query parameter '{key}' must be >= {minimum}")
    return value


def _request_json(request: Request) -> dict[str, Any]:
    try:
        value = request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=422, detail="JSON body must be an object")
    return value


def _parse_model(request: Request, model_cls: type[BaseModel]) -> BaseModel:
    payload = _request_json(request)
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _status_json(payload: Any, status_code: int) -> Response:
    return Response(
        status_code=status_code,
        headers=Headers({"Content-Type": "application/json"}),
        description=jsonify(_jsonable(payload)),
    )


async def _invoke(coro: Awaitable[Any]) -> Any:
    try:
        result = await coro
        if isinstance(result, RawResponse):
            return Response(
                status_code=result.status_code,
                headers=Headers({"Content-Type": result.media_type}),
                description=result.content.encode("utf-8"),
            )
        return _jsonable(result)
    except HTTPException as exc:
        return _status_json({"detail": exc.detail}, int(exc.status_code))


async def _invoke_with_model(
    request: Request,
    model_cls: type[BaseModel],
    cb: Callable[[BaseModel], Awaitable[Any]],
) -> Any:
    async def runner() -> Any:
        model = _parse_model(request, model_cls)
        return await cb(model)

    return await _invoke(runner())


async def _invoke_with_json(
    request: Request,
    cb: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Any:
    async def runner() -> Any:
        payload = _request_json(request)
        return await cb(payload)

    return await _invoke(runner())


# --- API routes ---


@app.get("/health")
async def health_check() -> Any:
    return await _invoke(service.health_check())


@app.post("/v1/scrape")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def scrape_url(request: Request) -> Any:
    return await _invoke_with_model(
        request,
        ScrapeRequest,
        lambda model: service.scrape_url(_service_request(request), model, ""),
    )


@app.post("/v2/scrape")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_scrape(request: Request) -> Any:
    return await _invoke_with_json(
        request,
        lambda payload: service.v2_scrape(_service_request(request), payload, ""),
    )


@app.post("/v1/scrape/bulk")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def bulk_scrape(request: Request) -> Any:
    return await _invoke_with_model(
        request,
        BatchScrapeRequest,
        lambda model: service.bulk_scrape(_service_request(request), model, ""),
    )


@app.get("/v1/scrape/*url")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def scrape_url_get(request: Request) -> Any:
    url = unquote(request.path_params.get("url", ""))
    return await _invoke(service.scrape_url_get(_service_request(request), url, ""))


@app.post("/v1/crawl")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def crawl_url(request: Request) -> Any:
    tasks = BackgroundTaskShim()
    return await _invoke_with_model(
        request,
        CrawlRequest,
        lambda model: service.crawl_url(_service_request(request), model, tasks, ""),
    )


@app.get("/v1/crawl/:job_id")
async def get_crawl_status(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.get_crawl_status(job_id))


@app.post("/v1/crawl/:job_id/cancel")
async def cancel_crawl(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.cancel_crawl(job_id))


@app.get("/v1/crawl/:job_id/export")
async def export_crawl(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    fmt = _query_param(request, "format", "jsonl")
    return await _invoke(service.export_crawl(job_id, str(fmt)))


@app.post("/v2/crawl")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_crawl(request: Request) -> Any:
    tasks = BackgroundTaskShim()
    return await _invoke_with_json(
        request,
        lambda payload: service.v2_crawl(_service_request(request), payload, tasks, ""),
    )


@app.get("/v2/crawl/:job_id")
async def v2_get_crawl_status(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.v2_get_crawl_status(job_id))


@app.delete("/v2/crawl/:job_id")
async def v2_cancel_crawl(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.v2_cancel_crawl(job_id))


@app.get("/v2/crawl/:job_id/errors")
async def v2_crawl_errors(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.v2_crawl_errors(job_id))


@app.post("/v1/batch-scrape")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def batch_scrape(request: Request) -> Any:
    tasks = BackgroundTaskShim()
    return await _invoke_with_model(
        request,
        BatchScrapeRequest,
        lambda model: service.batch_scrape(_service_request(request), model, tasks, ""),
    )


@app.get("/v1/batch-scrape/:job_id")
async def get_batch_status(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.get_batch_status(job_id))


@app.post("/v1/map")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def map_url(request: Request) -> Any:
    return await _invoke_with_model(
        request,
        MapRequest,
        lambda model: service.map_url(_service_request(request), model, ""),
    )


@app.post("/v2/map")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_map(request: Request) -> Any:
    return await _invoke_with_json(
        request,
        lambda payload: service.v2_map(_service_request(request), payload, ""),
    )


@app.post("/v2/batch/scrape")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_batch_scrape(request: Request) -> Any:
    tasks = BackgroundTaskShim()
    return await _invoke_with_json(
        request,
        lambda payload: service.v2_batch_scrape(_service_request(request), payload, tasks, ""),
    )


@app.get("/v2/batch/scrape/:job_id")
async def v2_get_batch_status(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.v2_get_batch_status(job_id))


@app.delete("/v2/batch/scrape/:job_id")
async def v2_cancel_batch(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.v2_cancel_batch(job_id))


@app.get("/v2/batch/scrape/:job_id/errors")
async def v2_batch_errors(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    return await _invoke(service.v2_batch_errors(job_id))


@app.post("/v2/search")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_search(request: Request) -> Any:
    return await _invoke_with_json(
        request,
        lambda payload: service.v2_search(_service_request(request), payload, ""),
    )


@app.post("/v2/extract")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_extract(request: Request) -> Any:
    return await _invoke_with_json(
        request,
        lambda payload: service.v2_extract(_service_request(request), payload, ""),
    )


@app.post("/v1/hermes/leads/search")
async def hermes_search(request: Request) -> Any:
    return await _invoke_with_model(request, HermesSearchRequest, service.hermes_search)


@app.post("/v1/hermes/compose")
async def hermes_compose(request: Request) -> Any:
    return await _invoke_with_model(request, HermesComposeRequest, service.hermes_compose)


@app.post("/v1/hermes/external-scrape")
async def hermes_external_scrape(request: Request) -> Any:
    return await _invoke_with_model(request, HermesExternalScrapeRequest, service.hermes_external_scrape)


@app.post("/v1/hermes/leads/enrich")
async def hermes_enrich(request: Request) -> Any:
    return await _invoke_with_model(request, HermesEnrichRequest, service.hermes_enrich)


@app.post("/v1/hermes/research")
async def hermes_research(request: Request) -> Any:
    return await _invoke_with_json(request, service.hermes_research)


# --- Cost endpoints ---


@app.get("/v1/hermes/costs/job/:job_id")
async def cost_job(request: Request) -> Any:
    job_id = request.path_params.get("job_id", "")
    include_trace_raw = _query_param(request, "include_trace", "false")
    include_trace = str(include_trace_raw).strip().lower() in {"1", "true", "yes", "on"}
    return await _invoke(cost_api.get_job_costs(job_id=job_id, include_trace=include_trace))


@app.get("/v1/hermes/costs/summary")
async def cost_summary(request: Request) -> Any:
    async def runner() -> Any:
        tier = _query_param(request, "tier", None)
        days = _query_param_int(request, "days", 7, minimum=1)
        return await cost_api.get_cost_summary(days=days, tier=tier)

    return await _invoke(runner())


@app.get("/v1/hermes/costs/providers")
async def cost_providers() -> Any:
    return await _invoke(cost_api.get_provider_costs())


@app.get("/v1/hermes/costs/active")
async def cost_active() -> Any:
    return await _invoke(cost_api.get_active_jobs())


@app.get("/v1/hermes/costs/budget")
async def cost_budget_get() -> Any:
    return await _invoke(cost_api.get_budget_config())


@app.post("/v1/hermes/costs/budget")
async def cost_budget_post(request: Request) -> Any:
    return await _invoke_with_model(request, TierBudgetConfig, cost_api.update_budget_config)


@app.get("/v1/hermes/costs/alerts")
async def cost_alerts() -> Any:
    return await _invoke(cost_api.get_budget_alerts())


@app.post("/v1/hermes/costs/cleanup")
async def cost_cleanup(request: Request) -> Any:
    async def runner() -> Any:
        max_age_hours = _query_param_int(request, "max_age_hours", 24, minimum=1)
        return await cost_api.cleanup_old_trackers(max_age_hours=max_age_hours)

    return await _invoke(runner())


@app.get("/v1/hermes/costs/efficiency/tier/:tier")
async def cost_efficiency(request: Request) -> Any:
    tier = request.path_params.get("tier", "")
    return await _invoke(cost_api.get_tier_efficiency_analysis(tier))


@app.shutdown_handler
async def shutdown() -> None:
    try:
        await service.scraper.close_browser()
    except Exception as exc:  # pragma: no cover - shutdown best effort
        logger.warning(f"Browser shutdown failed: {exc}")


if __name__ == "__main__":
    app.start(host=settings.host, port=settings.port)




