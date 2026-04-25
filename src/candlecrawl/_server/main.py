from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response
from contextlib import asynccontextmanager
from dataclasses import asdict
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import redis
from rq import Queue
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
import xml.etree.ElementTree as ET
import httpx
import hashlib
import hmac
import json
import ipaddress
import socket
import time
from loguru import logger

from candlecrawl._server.config import settings
from candlecrawl._server.models import (
    ScrapeRequest, ScrapeResponse, HealthResponse,
    CrawlRequest, CrawlResponse, CrawlStatusResponse,
    BatchScrapeRequest, BatchScrapeResponse,
    MapRequest, MapResponse,
    ActionCapability, ActionFieldCapability, ActionsCapabilitiesResponse,
    FirecrawlDocument, DocumentMetadata, ScrapeOptions,
    utc_now,
)
from candlecrawl._server.scraper import scraper
from candlecrawl._server.frontier import MemoryFrontier
from candlecrawl._server.providers.serper import SerperClient, SerperImageRequest, SerperNewsRequest, SerperSearchRequest
from candlecrawl._server.metrics import (
    metrics_response,
    record_budget_guardrail,
    record_kill_switch,
    record_security_deny,
)
from candlecrawl._server.compat.firecrawl_v2 import apply_v2_contract_headers, json_error
from candlecrawl._server.artifacts import ArtifactSinkSelectorConfig, select_artifact_sink
from candlecrawl._server.querylake_files_sink import QueryLakeFilesArtifactSink, QueryLakeFilesSinkConfig
from candlecrawl.schemas import (
    BatchScrapeRequest as PublicBatchScrapeRequest,
    CancelResponse as PublicCancelResponse,
    CrawlRequest as PublicCrawlRequest,
    ExtractRequest as PublicExtractRequest,
    JobCreateResponse as PublicJobCreateResponse,
    JobErrorsResponse as PublicJobErrorsResponse,
    JobStatusResponse as PublicJobStatusResponse,
    MapRequest as PublicMapRequest,
    MapResponse as PublicMapResponse,
    ScrapeRequest as PublicScrapeRequest,
    ScrapeResponse as PublicScrapeResponse,
    SearchRequest as PublicSearchRequest,
    SearchResponse as PublicSearchResponse,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Firecrawl API...")
    print(f"Redis available: {redis_available}")
    browser_ready, browser_error = await scraper.preflight_browser_runtime()
    if browser_ready:
        print("Browser runtime ready: True")
    else:
        logger.error(f"Browser runtime preflight failed: {browser_error}")
        print(f"Browser runtime ready: False ({browser_error})")
    try:
        yield
    finally:
        await scraper.close_browser()
        print("Firecrawl API shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Firecrawl API",
    description="Self-hosted web scraping and crawling service",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _v2_configured_api_keys() -> set[str]:
    raw = getattr(settings, "v2_api_keys", "") or ""
    return {k.strip() for k in raw.split(",") if k.strip()}


def _extract_v2_auth_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization", "")
    if isinstance(auth_header, str) and auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            return token
    api_key = request.headers.get("x-api-key")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    return None


def _resolve_request_tenant_id(request: Request, auth_token: str | None = None) -> str:
    tenant_header = request.headers.get("x-tenant-id")
    if isinstance(tenant_header, str) and tenant_header.strip():
        return tenant_header.strip()
    if auth_token:
        return f"token-{hashlib.sha256(auth_token.encode('utf-8')).hexdigest()[:12]}"
    return settings.v2_default_tenant_id


def _enforce_v2_auth(request: Request) -> Response | None:
    if not settings.v2_auth_enabled:
        request.state.tenant_id = _resolve_request_tenant_id(request)
        return None

    configured_keys = _v2_configured_api_keys()
    if not configured_keys:
        record_security_deny("v2_auth", "misconfigured")
        return json_error(
            "Authentication is enabled but no API keys are configured",
            status_code=500,
            code="AUTH_MISCONFIGURED",
        )

    token = _extract_v2_auth_token(request)
    if not token or token not in configured_keys:
        record_security_deny("v2_auth", "unauthorized")
        return json_error("Unauthorized", status_code=401, code="UNAUTHORIZED")

    request.state.tenant_id = _resolve_request_tenant_id(request, token)
    return None


def _request_tenant_id(request: Request) -> str:
    tenant_id = getattr(request.state, "tenant_id", None)
    if isinstance(tenant_id, str) and tenant_id.strip():
        return tenant_id.strip()
    return settings.v2_default_tenant_id


def _enforce_job_tenant(request: Request, job_data: Dict[str, Any]) -> Response | None:
    job_tenant = (job_data or {}).get("tenant_id") or settings.v2_default_tenant_id
    if job_tenant != _request_tenant_id(request):
        record_security_deny("v2_jobs", "tenant_forbidden")
        return json_error("Forbidden", status_code=403, code="TENANT_FORBIDDEN")
    return None


def _ip_is_private_or_local(ip_value: ipaddress._BaseAddress) -> bool:
    return any(
        [
            ip_value.is_private,
            ip_value.is_loopback,
            ip_value.is_link_local,
            ip_value.is_reserved,
            ip_value.is_multicast,
            ip_value.is_unspecified,
        ]
    )


def _host_resolves_to_private_network(host: str) -> bool:
    normalized = (host or "").strip().lower()
    if not normalized:
        return True
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        literal_ip = ipaddress.ip_address(normalized)
        return _ip_is_private_or_local(literal_ip)
    except ValueError:
        pass

    try:
        addrinfos = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
    except Exception:
        # If DNS resolution fails, leave validation to downstream URL handling.
        return False

    for addrinfo in addrinfos:
        try:
            resolved_ip = ipaddress.ip_address(addrinfo[4][0])
        except Exception:
            continue
        if _ip_is_private_or_local(resolved_ip):
            return True
    return False


def _is_ssrf_blocked_url(url: str) -> bool:
    if not settings.v2_ssrf_protection_enabled:
        return False
    if settings.v2_allow_private_network:
        return False

    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return True
    if (host == "localhost" or host.endswith(".localhost")) and settings.v2_allow_localhost:
        return False
    return _host_resolves_to_private_network(host)


def _v2_status_payload(fields: Dict[str, Any]) -> Dict[str, Any]:
    if settings.strict_firecrawl_v2:
        return fields
    return {"success": True, **fields}


def _v2_errors_payload(errors: List[Dict[str, Any]] | List[str], robots_blocked: List[str]) -> Dict[str, Any]:
    payload = {"errors": errors, "robotsBlocked": robots_blocked}
    if settings.strict_firecrawl_v2:
        return payload
    return {"success": True, "data": payload}


def _make_job_error_item(url: str, error: str) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": utc_now().isoformat(),
        "url": url,
        "error": error,
    }


def _v2_cancel_payload(message: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"success": True, "message": message}
    if not settings.strict_firecrawl_v2:
        payload["status"] = "cancelled"
    return payload


def _parse_status_cursor(cursor_raw: str | None) -> int | None:
    if cursor_raw is None:
        return 0
    cursor = cursor_raw.strip()
    if not cursor:
        return 0
    if cursor.lower().startswith("offset:"):
        cursor = cursor.split(":", 1)[1].strip()
    try:
        value = int(cursor)
    except Exception:
        return None
    return value if value >= 0 else None


def _normalize_webhook_config(raw_webhook: Any) -> Dict[str, Any] | None:
    if isinstance(raw_webhook, str):
        url = raw_webhook.strip()
        if not url:
            return None
        return {"url": url, "headers": {}, "metadata": {}}
    if not isinstance(raw_webhook, dict):
        return None
    url = raw_webhook.get("url")
    if not isinstance(url, str) or not url.strip():
        return None

    normalized_headers: Dict[str, str] = {}
    headers = raw_webhook.get("headers")
    if isinstance(headers, dict):
        for key, value in headers.items():
            if isinstance(key, str) and key.strip() and isinstance(value, str):
                normalized_headers[key.strip()] = value

    normalized_metadata: Dict[str, str] = {}
    metadata = raw_webhook.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(key, str) and key.strip() and isinstance(value, str):
                normalized_metadata[key.strip()] = value

    return {
        "url": url.strip(),
        "headers": normalized_headers,
        "metadata": normalized_metadata,
    }


def _webhook_event_name(job_type: str, status: str) -> str:
    event_status = (status or "unknown").strip().lower() or "unknown"
    return f"{job_type}.{event_status}"


def _webhook_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


async def _deliver_job_webhook(
    *,
    job_type: str,
    job_id: str,
    job_data: Dict[str, Any],
) -> Dict[str, Any]:
    request_payload = job_data.get("request") if isinstance(job_data, dict) else None
    webhook_cfg = _normalize_webhook_config((request_payload or {}).get("webhook") if isinstance(request_payload, dict) else None)
    if not webhook_cfg:
        return {
            "status": "skipped",
            "reason": "not_configured",
            "attempts": 0,
            "event": _webhook_event_name(job_type, str(job_data.get("status") or "")),
        }

    status_value = str(job_data.get("status") or "unknown")
    event_name = _webhook_event_name(job_type, status_value)
    payload = {
        "event": event_name,
        "job": {
            "id": job_id,
            "type": job_type,
            "status": status_value,
            "completed": job_data.get("completed"),
            "total": job_data.get("total"),
            "creditsUsed": job_data.get("credits_used"),
            "expiresAt": job_data.get("expires_at").isoformat()
            if isinstance(job_data.get("expires_at"), datetime)
            else job_data.get("expires_at"),
            "error": job_data.get("error"),
            "next": job_data.get("next"),
        },
        "errorsCount": len(job_data.get("errors") or []) if isinstance(job_data.get("errors"), list) else 0,
        "robotsBlockedCount": len(job_data.get("robots_blocked") or [])
        if isinstance(job_data.get("robots_blocked"), list)
        else 0,
        "metadata": webhook_cfg.get("metadata") or {},
        "timestamp": utc_now().isoformat(),
    }
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    ts_epoch_ms = str(int(time.time() * 1000))

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "candlecrawl-webhook/1.0",
        "X-CandleCrawl-Event": event_name,
        "X-CandleCrawl-Job-Id": job_id,
        "X-CandleCrawl-Timestamp": ts_epoch_ms,
    }
    for key, value in (webhook_cfg.get("headers") or {}).items():
        if isinstance(key, str) and key and isinstance(value, str):
            headers[key] = value
    if settings.webhook_signing_secret:
        headers["X-CandleCrawl-Signature"] = _webhook_signature(settings.webhook_signing_secret, body)

    max_retries = max(0, int(settings.webhook_max_retries))
    timeout_seconds = float(settings.webhook_timeout_seconds)
    base_delay_ms = max(10, int(settings.webhook_retry_base_delay_ms))

    attempts = 0
    last_error: str | None = None
    last_status_code: int | None = None
    delivery_ok = False

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for attempt in range(max_retries + 1):
            attempts = attempt + 1
            try:
                response = await client.post(webhook_cfg["url"], content=body, headers=headers)
                last_status_code = int(response.status_code)
                if 200 <= response.status_code < 300:
                    delivery_ok = True
                    break

                # Retry transient failures only.
                if response.status_code in {408, 409, 425, 429} or response.status_code >= 500:
                    if attempt < max_retries:
                        await asyncio.sleep((base_delay_ms * (2 ** attempt)) / 1000.0)
                        continue
                text = response.text if isinstance(response.text, str) else ""
                last_error = f"webhook_http_{response.status_code}:{text[:300]}"
                break
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    await asyncio.sleep((base_delay_ms * (2 ** attempt)) / 1000.0)
                    continue
                break

    return {
        "status": "delivered" if delivery_ok else "failed",
        "attempts": attempts,
        "event": event_name,
        "url": webhook_cfg["url"],
        "last_status_code": last_status_code,
        "last_error": last_error,
        "delivered_at": utc_now().isoformat() if delivery_ok else None,
    }


def _artifact_mode_from_payload(payload: Dict[str, Any]) -> str | None:
    mode = payload.get("artifact_mode")
    if mode is None:
        mode = payload.get("artifactMode")
    return mode if isinstance(mode, str) else None


def _trace_headers_for_upstream(request: Request) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    request_id = getattr(request.state, "request_id", None)
    audit_id = getattr(request.state, "audit_id", None)
    trace_id = getattr(request.state, "trace_id", None)
    if isinstance(request_id, str) and request_id.strip():
        headers["X-Request-Id"] = request_id.strip()
    if isinstance(audit_id, str) and audit_id.strip():
        headers["X-Audit-Id"] = audit_id.strip()
    if isinstance(trace_id, str) and trace_id.strip():
        headers["X-Trace-Id"] = trace_id.strip()
    tenant_id = _request_tenant_id(request)
    if tenant_id:
        headers["X-Tenant-Id"] = tenant_id
    idem_key = _get_idempotency_key(request)
    if idem_key:
        headers["X-Idempotency-Key"] = idem_key
    return headers


def _artifact_selector_config_for_request(request: Request) -> ArtifactSinkSelectorConfig:
    querylake_sink = None
    if settings.artifact_querylake_enabled and settings.querylake_files_base_url:
        querylake_sink = QueryLakeFilesArtifactSink(
            config=QueryLakeFilesSinkConfig(
                base_url=settings.querylake_files_base_url,
                ingestion_path=settings.querylake_files_ingestion_path,
                auth_header=settings.querylake_files_auth_header,
                auth_token=settings.querylake_files_auth_token,
                timeout_seconds=settings.querylake_files_timeout_seconds,
                max_retries=settings.querylake_files_max_retries,
                retry_base_delay_ms=settings.querylake_files_retry_base_delay_ms,
                tenant_id=_request_tenant_id(request),
                collection_id=settings.querylake_files_collection_id,
                source="candlecrawl_v2_scrape",
            )
        )
    return ArtifactSinkSelectorConfig(
        default_mode=settings.artifact_mode_default,
        allow_fallback_to_inline=settings.artifact_allow_fallback_to_inline,
        local_root=settings.artifact_local_root,
        inline_max_bytes=settings.artifact_inline_max_bytes,
        querylake_enabled=settings.artifact_querylake_enabled,
        querylake_sink=querylake_sink,
    )


async def _maybe_externalize_v2_artifacts(
    *,
    request: Request,
    payload: Dict[str, Any],
    data: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    requested_mode = _artifact_mode_from_payload(payload)
    selection = select_artifact_sink(requested_mode, config=_artifact_selector_config_for_request(request))
    artifacts: Dict[str, Any] = {}
    transformed = dict(data)
    externalized_fields = {
        "markdown": ("artifact_markdown.md", "text/markdown"),
        "html": ("artifact_html.html", "text/html"),
        "rawHtml": ("artifact_raw_html.html", "text/html"),
    }

    try:
        if selection.resolved_mode != "inline":
            for field_name, (filename, content_type) in externalized_fields.items():
                value = transformed.get(field_name)
                if not isinstance(value, str) or not value:
                    continue
                ref = await selection.sink.put_bytes(
                    content=value.encode("utf-8"),
                    filename=filename,
                    content_type=content_type,
                    trace_headers=_trace_headers_for_upstream(request),
                )
                artifacts[field_name] = asdict(ref)
                transformed.pop(field_name, None)
    finally:
        close_fn = getattr(selection.sink, "aclose", None)
        if callable(close_fn):
            await close_fn()

    metadata = {
        "artifactMode": selection.resolved_mode,
        "artifactFallback": selection.fallback_reason,
        "artifacts": artifacts or None,
    }
    return transformed, metadata


@app.middleware("http")
async def attach_v2_contract_headers(request: Request, call_next):
    """Stamp trace/contract headers and enforce v2 auth."""
    request_id = request.headers.get("x-request-id") or request.headers.get("X-Request-Id") or str(uuid.uuid4())
    audit_id = request.headers.get("x-audit-id") or request.headers.get("X-Audit-Id") or request_id
    trace_id = request.headers.get("x-trace-id") or request.headers.get("X-Trace-Id") or request_id or audit_id
    request.state.request_id = request_id
    request.state.audit_id = audit_id
    request.state.trace_id = trace_id
    inbound_idempotency = _get_idempotency_key(request)

    if request.url.path.startswith("/v2/"):
        auth_error = _enforce_v2_auth(request)
        if auth_error is not None:
            auth_error.headers["X-Request-Id"] = request_id
            auth_error.headers["X-Audit-Id"] = audit_id
            auth_error.headers["X-Trace-Id"] = trace_id
            if inbound_idempotency:
                auth_error.headers["X-Idempotency-Key"] = inbound_idempotency
            tenant_id = _request_tenant_id(request)
            if tenant_id:
                auth_error.headers["X-Tenant-Id"] = tenant_id
            apply_v2_contract_headers(auth_error)
            return auth_error
    response = await call_next(request)

    response.headers["X-Request-Id"] = request_id
    response.headers["X-Audit-Id"] = audit_id
    response.headers["X-Trace-Id"] = trace_id
    if inbound_idempotency:
        response.headers["X-Idempotency-Key"] = inbound_idempotency
    tenant_id = _request_tenant_id(request)
    if tenant_id:
        response.headers["X-Tenant-Id"] = tenant_id

    if request.url.path.startswith("/v2/"):
        apply_v2_contract_headers(response)
    return response

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


def _rate_limit_spec(requests: int, window_seconds: int) -> str:
    req = max(1, int(requests))
    window = max(1, int(window_seconds))
    return f"{req}/{window}seconds"


def _rate_limit_exceeded_handler_with_metrics(request: Request, exc: RateLimitExceeded):
    record_security_deny(request.url.path, "rate_limit_exceeded")
    return _rate_limit_exceeded_handler(request, exc)


V2_LIMIT_SCRAPE = _rate_limit_spec(settings.v2_rate_limit_scrape_requests, settings.v2_rate_limit_window_seconds)
V2_LIMIT_MAP = _rate_limit_spec(settings.v2_rate_limit_map_requests, settings.v2_rate_limit_window_seconds)
V2_LIMIT_SEARCH = _rate_limit_spec(settings.v2_rate_limit_search_requests, settings.v2_rate_limit_window_seconds)
V2_LIMIT_CRAWL = _rate_limit_spec(settings.v2_rate_limit_crawl_requests, settings.v2_rate_limit_window_seconds)
V2_LIMIT_BATCH_SCRAPE = _rate_limit_spec(
    settings.v2_rate_limit_batch_scrape_requests,
    settings.v2_rate_limit_window_seconds,
)
V2_LIMIT_EXTRACT = _rate_limit_spec(settings.v2_rate_limit_extract_requests, settings.v2_rate_limit_window_seconds)

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler_with_metrics)
app.add_middleware(SlowAPIMiddleware)

# Redis connection for job queue
try:
    # If a password is supplied in env var it should be included in settings.redis_url
    redis_client = redis.from_url(settings.redis_url)
    # Test the connection
    redis_client.ping()
    job_queue = Queue(connection=redis_client)
    redis_available = True
    print("Redis connection successful")
except Exception as e:
    print(f"Redis not available: {e}")
    redis_available = False
    job_queue = None
    redis_client = None

# In-memory storage for jobs when Redis is not available
job_storage: Dict[str, Dict[str, Any]] = {}
_cancelled_jobs: set[str] = set()
_cancelled_batch_jobs: set[str] = set()

# In-memory cache fallback (used when Redis is unavailable).
_scrape_cache_memory: Dict[str, Dict[str, Any]] = {}

# In-memory idempotency store fallback (used when Redis is unavailable).
_idempotency_memory: Dict[str, Dict[str, Any]] = {}

def serialize_job_data(data: Dict[str, Any]) -> str:
    """Serialize job data to JSON string, handling datetime objects"""
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=json_serializer)

def deserialize_job_data(data: str) -> Dict[str, Any]:
    """Deserialize job data from JSON string, handling datetime objects"""
    job_data = json.loads(data)

    # Convert ISO datetime strings back to datetime objects
    if 'expires_at' in job_data and isinstance(job_data['expires_at'], str):
        job_data['expires_at'] = datetime.fromisoformat(job_data['expires_at'])

    return job_data


def _ms_to_seconds(value: Any) -> int | None:
    if value is None:
        return None
    try:
        # Firecrawl v2 semantics use milliseconds; internal scraper expects seconds.
        ms = float(value)
    except Exception:
        return None
    if ms <= 0:
        return None
    # Round up to avoid 0s timeouts.
    return max(1, int((ms + 999) // 1000))


def _normalize_v2_format_token(token: str) -> str | None:
    normalized = token.strip().lower()
    mapping = {
        "markdown": "markdown",
        "md": "markdown",
        "content": "markdown",
        "html": "html",
        "rawhtml": "rawHtml",
        "raw_html": "rawHtml",
        "raw-html": "rawHtml",
        "links": "links",
        "link": "links",
        "screenshot": "screenshot",
        "screenshot@fullpage": "screenshot",
        "screenshot@full_page": "screenshot",
    }
    return mapping.get(normalized)


def _requested_v2_format_types(payload: Dict[str, Any]) -> set[str]:
    formats_in = payload.get("formats")
    requested: set[str] = set()
    if not isinstance(formats_in, list):
        return requested

    def _norm_type(raw: str) -> str:
        t = raw.strip().lower()
        aliases = {
            "raw_html": "rawhtml",
            "raw-html": "rawhtml",
            "changetracking": "changetracking",
            "change-tracking": "changetracking",
        }
        return aliases.get(t, t)

    for item in formats_in:
        if isinstance(item, str):
            requested.add(_norm_type(item))
            continue
        if isinstance(item, dict) and isinstance(item.get("type"), str):
            requested.add(_norm_type(item["type"]))
    return requested


def _parse_pdf_requested(payload: Dict[str, Any]) -> bool:
    if _coerce_bool(payload.get("parsePDF"), False) or _coerce_bool(payload.get("parsePdf"), False):
        return True
    parsers_in = payload.get("parsers")
    if not isinstance(parsers_in, list):
        return False

    for item in parsers_in:
        if isinstance(item, str):
            normalized = item.strip().lower()
            if normalized in {"pdf", "application/pdf"}:
                return True
        elif isinstance(item, dict):
            parser_type = item.get("type")
            if isinstance(parser_type, str) and parser_type.strip().lower() in {"pdf", "application/pdf"}:
                return True
    return False


def _parse_ocr_config(payload: Dict[str, Any]) -> tuple[bool, str | None, str | None]:
    enabled = False
    provider: str | None = None
    prompt: str | None = None

    ocr_payload = payload.get("ocr")
    if isinstance(ocr_payload, bool):
        enabled = ocr_payload
    elif isinstance(ocr_payload, dict):
        enabled = _coerce_bool(ocr_payload.get("enabled"), True)
        provider_raw = ocr_payload.get("provider")
        if isinstance(provider_raw, str) and provider_raw.strip():
            provider = provider_raw.strip()
        prompt_raw = ocr_payload.get("prompt")
        if isinstance(prompt_raw, str) and prompt_raw.strip():
            prompt = prompt_raw.strip()

    provider_raw = payload.get("ocrProvider")
    if isinstance(provider_raw, str) and provider_raw.strip():
        provider = provider_raw.strip()
        enabled = True

    prompt_raw = payload.get("ocrPrompt")
    if isinstance(prompt_raw, str) and prompt_raw.strip():
        prompt = prompt_raw.strip()

    parsers_in = payload.get("parsers")
    if isinstance(parsers_in, list):
        for item in parsers_in:
            parser_token: str | None = None
            if isinstance(item, str):
                parser_token = item.strip().lower()
            elif isinstance(item, dict) and isinstance(item.get("type"), str):
                parser_token = item.get("type", "").strip().lower()
            if not parser_token:
                continue
            if parser_token in {"ocr", "vision"}:
                enabled = True
            elif parser_token in {"mistral", "mistral_ocr", "mistral-ocr"}:
                enabled = True
                provider = "mistral"
            elif parser_token in {"querylake", "chandra", "querylake_chandra", "querylake-chandra"}:
                enabled = True
                provider = "querylake_chandra"

    return enabled, provider, prompt


def _scrape_options_from_v2(payload: Dict[str, Any]) -> ScrapeOptions:
    formats_in = payload.get("formats")
    requested_types = _requested_v2_format_types(payload)
    formats: list[str] = []
    seen_formats: set[str] = set()
    if isinstance(formats_in, list):
        for item in formats_in:
            if isinstance(item, str):
                normalized = _normalize_v2_format_token(item)
                if normalized and normalized not in seen_formats:
                    formats.append(normalized)
                    seen_formats.add(normalized)
            elif isinstance(item, dict):
                item_type = item.get("type")
                if isinstance(item_type, str):
                    normalized = _normalize_v2_format_token(item_type)
                    if normalized and normalized not in seen_formats:
                        formats.append(normalized)
                        seen_formats.add(normalized)
                # Non-render output variants (json/summary/etc.) are derived later.
    if not formats:
        formats = ["markdown"]
    elif (
        requested_types.intersection({"json", "summary", "branding", "changetracking"})
        and "markdown" not in seen_formats
    ):
        # Ensure we have source text available to derive richer output formats.
        formats.append("markdown")

    ocr_enabled, ocr_provider, ocr_prompt = _parse_ocr_config(payload)

    return ScrapeOptions(
        formats=formats,
        headers=payload.get("headers"),
        include_tags=payload.get("includeTags"),
        exclude_tags=payload.get("excludeTags"),
        only_main_content=payload.get("onlyMainContent"),
        wait_for=_ms_to_seconds(payload.get("waitFor")),
        timeout=_ms_to_seconds(payload.get("timeout")),
        location=payload.get("location"),
        mobile=payload.get("mobile"),
        skip_tls_verification=payload.get("skipTlsVerification"),
        remove_base64_images=payload.get("removeBase64Images"),
        use_relative_links=payload.get("useRelativeLinks"),
        include_file_body=payload.get("includeFileBody"),
        parse_pdf=_parse_pdf_requested(payload),
        ocr=ocr_enabled,
        ocr_provider=ocr_provider,
        ocr_prompt=ocr_prompt,
        actions=payload.get("actions") if isinstance(payload.get("actions"), list) else None,
    )


def _derive_summary_text(data: Dict[str, Any]) -> str | None:
    source = data.get("markdown") or data.get("html")
    if not isinstance(source, str):
        return None
    text = " ".join(source.strip().split())
    if not text:
        return None
    # Deterministic lightweight summary: first ~320 characters.
    return text[:320]


def _augment_v2_data_with_format_objects(
    data: Dict[str, Any],
    payload: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str]]:
    requested_types = _requested_v2_format_types(payload)
    if not requested_types:
        return data, []

    out = dict(data)
    warnings: list[str] = []

    if "summary" in requested_types and out.get("summary") is None:
        summary = _derive_summary_text(out)
        if summary is not None:
            out["summary"] = summary

    if "json" in requested_types and out.get("json") is None:
        source_text = out.get("markdown") or out.get("html")
        if isinstance(source_text, str):
            out["json"] = {
                "content": source_text,
                "metadata": out.get("metadata") if isinstance(out.get("metadata"), dict) else {},
            }

    if "branding" in requested_types and out.get("branding") is None:
        source_html = out.get("html") or out.get("rawHtml") or ""
        colors: list[str] = []
        fonts: list[str] = []
        if isinstance(source_html, str) and source_html:
            colors = list(dict.fromkeys(re.findall(r"#[0-9a-fA-F]{3,8}", source_html)))[:10]
            fonts = list(
                dict.fromkeys(
                    [m.strip().strip("'\"") for m in re.findall(r"font-family\\s*:\\s*([^;]+);", source_html, flags=re.IGNORECASE)]
                )
            )[:10]
        out["branding"] = {"colors": colors, "fonts": fonts}

    if "changetracking" in requested_types and out.get("changeTracking") is None:
        source_text = out.get("markdown") or out.get("html") or ""
        snapshot_hash = (
            hashlib.sha256(source_text.encode("utf-8")).hexdigest()
            if isinstance(source_text, str) and source_text
            else None
        )
        out["changeTracking"] = {
            "snapshotHash": snapshot_hash,
            "capturedAt": utc_now().isoformat(),
            "mode": "snapshot",
        }

    known_types = {
        "markdown",
        "md",
        "content",
        "html",
        "rawhtml",
        "links",
        "link",
        "screenshot",
        "json",
        "summary",
        "branding",
        "changetracking",
    }
    unknown = sorted(requested_types.difference(known_types))
    if unknown:
        warnings.append("Unknown format types ignored: " + ", ".join(unknown))

    return out, warnings


def _v2_document_from_v1(doc: FirecrawlDocument | Dict[str, Any]) -> Dict[str, Any]:
    data = doc.model_dump() if hasattr(doc, "model_dump") else (doc or {})
    meta_in = data.get("metadata") or {}
    meta_out: Dict[str, Any] = {}

    if isinstance(meta_in, dict):
        if meta_in.get("title") is not None:
            meta_out["title"] = meta_in.get("title")
        if meta_in.get("description") is not None:
            meta_out["description"] = meta_in.get("description")
        if meta_in.get("language") is not None:
            meta_out["language"] = meta_in.get("language")
        if meta_in.get("source_url") is not None:
            meta_out["sourceURL"] = meta_in.get("source_url")
        if meta_in.get("status_code") is not None:
            meta_out["statusCode"] = meta_in.get("status_code")
        if meta_in.get("content_type") is not None:
            meta_out["contentType"] = meta_in.get("content_type")
        if meta_in.get("file_metadata") is not None:
            meta_out["fileMetadata"] = meta_in.get("file_metadata")
        if meta_in.get("blocked") is not None:
            meta_out["blocked"] = meta_in.get("blocked")
        if meta_in.get("blocked_reason") is not None:
            meta_out["blockedReason"] = meta_in.get("blocked_reason")

    out: Dict[str, Any] = {}
    if data.get("markdown") is not None:
        out["markdown"] = data.get("markdown")
    if data.get("html") is not None:
        out["html"] = data.get("html")
    if data.get("raw_html") is not None:
        out["rawHtml"] = data.get("raw_html")
    if data.get("rawHtml") is not None:
        out["rawHtml"] = data.get("rawHtml")
    if data.get("links") is not None:
        out["links"] = data.get("links")
    if data.get("screenshot") is not None:
        out["screenshot"] = data.get("screenshot")
    if data.get("content_base64") is not None:
        out["contentBase64"] = data.get("content_base64")
    if data.get("contentBase64") is not None:
        out["contentBase64"] = data.get("contentBase64")
    if meta_out:
        out["metadata"] = meta_out
    else:
        # Preserve any existing v2-shaped metadata if present.
        if isinstance(meta_in, dict) and meta_in:
            out["metadata"] = meta_in

    return out


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _estimate_document_credits(doc: FirecrawlDocument | Dict[str, Any]) -> int:
    data = doc.model_dump() if hasattr(doc, "model_dump") else (doc or {})
    if not isinstance(data, dict):
        return 1

    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    content_type = meta.get("content_type") or meta.get("contentType")
    if isinstance(content_type, str) and content_type.lower().startswith("application/pdf"):
        file_meta = meta.get("file_metadata") or meta.get("fileMetadata")
        if isinstance(file_meta, dict):
            page_count = _coerce_positive_int(file_meta.get("page_count") or file_meta.get("pageCount"))
            if page_count is not None:
                return max(1, page_count)
    return 1


def _estimate_documents_credits(documents: List[FirecrawlDocument | Dict[str, Any]]) -> int:
    return sum(_estimate_document_credits(doc) for doc in documents)


def _canonicalize_url(
    url: str,
    *,
    ignore_query_parameters: bool = False,
    deduplicate_similar_urls: bool = False,
) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()

    # Drop default ports.
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parsed.path or "/"
    if deduplicate_similar_urls and path != "/" and path.endswith("/"):
        path = path[:-1]

    query = parsed.query or ""
    if ignore_query_parameters:
        query = ""
    elif deduplicate_similar_urls and query:
        # Stable query ordering to reduce duplicates like ?b=2&a=1 vs ?a=1&b=2
        query = urlencode(sorted(parse_qsl(query, keep_blank_values=True)))

    # Fragments never affect content for HTTP requests.
    fragment = ""

    return urlunparse((scheme, netloc, path, parsed.params, query, fragment))


def _remove_empty_top_level(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, dict) and len(v) == 0:
            continue
        out[k] = v
    return out


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _scrape_cache_key(url: str, payload: Dict[str, Any]) -> str:
    # Keep this stable and conservative: include any request field that could
    # impact output once we fully implement the v2 surface.
    fingerprint = _remove_empty_top_level(
        {
            "url": _canonicalize_url(url),
            "formats": payload.get("formats"),
            "onlyMainContent": payload.get("onlyMainContent"),
            "includeTags": payload.get("includeTags"),
            "excludeTags": payload.get("excludeTags"),
            "headers": payload.get("headers"),
            "waitFor": payload.get("waitFor"),
            "timeout": payload.get("timeout"),
            "location": payload.get("location"),
            "mobile": payload.get("mobile"),
            "skipTlsVerification": payload.get("skipTlsVerification"),
            "removeBase64Images": payload.get("removeBase64Images"),
            "useRelativeLinks": payload.get("useRelativeLinks"),
            "includeFileBody": payload.get("includeFileBody"),
            "parsePDF": payload.get("parsePDF"),
            "parsePdf": payload.get("parsePdf"),
            "actions": payload.get("actions"),
            "parsers": payload.get("parsers"),
            "proxy": payload.get("proxy"),
            "blockAds": payload.get("blockAds"),
            "fastMode": payload.get("fastMode"),
            "artifact_mode": payload.get("artifact_mode") or payload.get("artifactMode"),
        }
    )
    raw = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"cache:scrape:{digest}"


def _get_cached_scrape(cache_key: str, max_age_ms: int | None) -> Dict[str, Any] | None:
    if not getattr(settings, "cache_enabled", True):
        return None
    if max_age_ms is None or max_age_ms <= 0:
        return None

    entry: Dict[str, Any] | None = None
    if redis_available:
        try:
            raw = redis_client.get(cache_key)
            if raw:
                entry = json.loads(raw)
        except Exception:
            entry = None
    else:
        entry = _scrape_cache_memory.get(cache_key)

    if not entry or not isinstance(entry, dict):
        return None

    cached_at_ms = entry.get("cached_at_ms")
    if not isinstance(cached_at_ms, int):
        return None
    if (_now_ms() - cached_at_ms) > max_age_ms:
        return None
    data = entry.get("data")
    if not isinstance(data, dict):
        return None
    # Never serve cached error/blocked responses: these can be transient
    # (e.g., bot mitigation, UA changes, temporary upstream failures) and
    # would otherwise "poison" the cache for successful retries.
    meta = data.get("metadata")
    if isinstance(meta, dict):
        status = meta.get("statusCode")
        if isinstance(status, int) and status >= 400:
            return None
        if meta.get("blocked"):
            return None
    return data


def _set_cached_scrape(cache_key: str, data: Dict[str, Any]) -> None:
    if not getattr(settings, "cache_enabled", True):
        return
    entry = {"cached_at_ms": _now_ms(), "data": data}
    ttl_seconds = int(getattr(settings, "cache_ttl_seconds", 172800))
    if redis_available:
        try:
            redis_client.set(cache_key, json.dumps(entry, ensure_ascii=False), ex=ttl_seconds)
        except Exception:
            pass
    else:
        _scrape_cache_memory[cache_key] = entry


def _is_cacheable_v2_scrape_data(data: Dict[str, Any]) -> bool:
    """Return True if this v2 scrape response should be cached.

    We intentionally avoid caching error/blocked responses since those are
    often transient and can prevent the system from recovering on retry.
    """
    if not isinstance(data, dict):
        return False
    # Require at least one material content field. Metadata-only responses
    # often represent transient upstream/network failures and should not be cached.
    has_material_content = any(
        data.get(field) not in (None, "", [])
        for field in ("markdown", "html", "rawHtml", "links", "screenshot", "content_base64", "contentBase64")
    )
    if not has_material_content:
        return False
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        return True
    status = meta.get("statusCode")
    if isinstance(status, int) and status >= 400:
        return False
    if meta.get("blocked"):
        return False
    return True


def _v2_file_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        return {}
    file_meta = meta.get("fileMetadata")
    return file_meta if isinstance(file_meta, dict) else {}


def _v2_action_results(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    file_meta = _v2_file_metadata(data)
    action_results = file_meta.get("action_results")
    if isinstance(action_results, list):
        return [r for r in action_results if isinstance(r, dict)]
    return []


def _v2_has_action_generated_pdf(data: Dict[str, Any]) -> bool:
    file_meta = _v2_file_metadata(data)
    return bool(file_meta.get("generated_pdf"))


def _v2_has_action_screenshot(data: Dict[str, Any]) -> bool:
    for result in _v2_action_results(data):
        action_type = result.get("type")
        status = result.get("status")
        if isinstance(action_type, str) and action_type.strip().lower() == "screenshot" and status == "success":
            return True
    return False


def _extract_action_payloads(data: Dict[str, Any]) -> tuple[list[Dict[str, Any]] | None, Dict[str, Any] | None]:
    action_results = _v2_action_results(data)
    if not action_results:
        return None, None

    file_meta = _v2_file_metadata(data)
    artifacts: Dict[str, Any] = {}
    if _v2_has_action_generated_pdf(data) and isinstance(data.get("contentBase64"), str):
        artifacts["generatedPdf"] = {
            "encoding": "base64",
            "location": "data.contentBase64",
            "sizeBytes": file_meta.get("generated_pdf_bytes"),
        }
    if _v2_has_action_screenshot(data) and isinstance(data.get("screenshot"), str):
        artifacts["actionScreenshot"] = {
            "encoding": "base64",
            "location": "data.screenshot",
        }

    return action_results, (artifacts or None)


def _filter_v2_data_to_requested_formats(
    data: Dict[str, Any],
    options: ScrapeOptions,
    *,
    include_file_body: bool = False,
) -> Dict[str, Any]:
    filtered = dict(data or {})
    requested = set(options.formats or ["markdown"])

    if "markdown" not in requested and "content" not in requested:
        filtered.pop("markdown", None)
    if "html" not in requested:
        filtered.pop("html", None)
    if "rawHtml" not in requested:
        filtered.pop("rawHtml", None)
    if "links" not in requested:
        filtered.pop("links", None)
    if "screenshot" not in requested and not _v2_has_action_screenshot(filtered):
        filtered.pop("screenshot", None)
    if not include_file_body and not _v2_has_action_generated_pdf(filtered):
        filtered.pop("contentBase64", None)

    return filtered


async def _v2_scrape_data(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    zero_data_retention = _coerce_bool(payload.get("zeroDataRetention"), False)
    max_age_raw = payload.get("maxAge")
    if max_age_raw is None:
        max_age_ms = int(getattr(settings, "cache_default_max_age_ms", 172800000))
    else:
        try:
            max_age_ms = int(max_age_raw)
        except Exception:
            max_age_ms = int(getattr(settings, "cache_default_max_age_ms", 172800000))

    store_in_cache = _coerce_bool(payload.get("storeInCache"), True)
    if zero_data_retention:
        max_age_ms = 0
        store_in_cache = False

    cache_key = _scrape_cache_key(url, payload)
    cached = _get_cached_scrape(cache_key, max_age_ms)
    if cached is not None:
        return cached

    options = _scrape_options_from_v2(payload)
    document = await scraper.scrape_url(url, options)
    data = _v2_document_from_v1(document)
    data = _filter_v2_data_to_requested_formats(
        data,
        options,
        include_file_body=bool(options.include_file_body),
    )
    if store_in_cache and _is_cacheable_v2_scrape_data(data):
        _set_cached_scrape(cache_key, data)
    return data


def _get_idempotency_key(request: Request) -> str | None:
    key = request.headers.get("x-idempotency-key") or request.headers.get("X-Idempotency-Key")
    if not key:
        return None
    key = key.strip()
    return key or None


def _idempotency_storage_key(scope: str, key: str) -> str:
    # `scope` is per-endpoint, e.g. "v2:crawl" or "v2:batch"
    return f"idempotency:{scope}:{key}"


def _payload_hash_for_idempotency(payload: Dict[str, Any]) -> str:
    # Exclude request fields that can change between retries but shouldn't
    # affect idempotency (origin/integration are advisory).
    cleaned = {k: v for k, v in payload.items() if k not in {"origin", "integration"}}
    raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_idempotent_response(scope: str, key: str, payload_hash: str) -> Dict[str, Any] | None:
    storage_key = _idempotency_storage_key(scope, key)
    entry: Dict[str, Any] | None = None
    if redis_available:
        try:
            raw = redis_client.get(storage_key)
            if raw:
                entry = json.loads(raw)
        except Exception:
            entry = None
    else:
        entry = _idempotency_memory.get(storage_key)

    if not entry:
        return None
    if entry.get("payload_hash") != payload_hash:
        raise HTTPException(status_code=409, detail="Idempotency key already used with different request payload")
    response = entry.get("response")
    if not isinstance(response, dict):
        return None
    return response


def _set_idempotent_response(scope: str, key: str, payload_hash: str, response: Dict[str, Any]) -> None:
    storage_key = _idempotency_storage_key(scope, key)
    entry = {"payload_hash": payload_hash, "response": response, "created_at_ms": _now_ms()}
    ttl_seconds = 86400  # match job expiry window
    if redis_available:
        try:
            redis_client.set(storage_key, json.dumps(entry, ensure_ascii=False), ex=ttl_seconds)
        except Exception:
            pass
    else:
        _idempotency_memory[storage_key] = entry

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    browser_status = scraper.get_browser_runtime_status()
    return HealthResponse(
        status="healthy" if browser_status["browser_ready"] else "degraded",
        browserReady=bool(browser_status["browser_ready"]),
        browserError=browser_status["browser_error"],
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    content, content_type = metrics_response()
    return Response(content=content, media_type=content_type)

@app.post("/v1/scrape", response_model=ScrapeResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def scrape_url(request: Request, scrape_request: ScrapeRequest, remote_addr: str = Depends(get_remote_address)):
    """Scrape a single URL"""
    parsed_url = urlparse(scrape_request.url)
    if not (parsed_url.scheme and parsed_url.netloc):
        raise HTTPException(status_code=400, detail=f"Invalid URL: {scrape_request.url}. Please provide a valid URL with a scheme (e.g., http:// or https://).")

    try:
        # Convert request to scrape options
        options = ScrapeOptions(
            formats=scrape_request.formats,
            headers=scrape_request.headers,
            include_tags=scrape_request.include_tags,
            exclude_tags=scrape_request.exclude_tags,
            only_main_content=scrape_request.only_main_content,
            wait_for=scrape_request.wait_for,
            timeout=scrape_request.timeout,
            location=scrape_request.location,
            mobile=scrape_request.mobile,
            skip_tls_verification=scrape_request.skip_tls_verification,
            remove_base64_images=scrape_request.remove_base64_images,
            use_relative_links=scrape_request.use_relative_links,
            include_file_body=scrape_request.include_file_body
        )

        # Scrape the URL
        document = await scraper.scrape_url(scrape_request.url, options)

        return ScrapeResponse(
            success=True,
            data=document
        )

    except Exception as e:
        return ScrapeResponse(
            success=False,
            error=str(e)
        )


# --- Firecrawl v2 compatibility endpoints (for firecrawl-js v2 / firecrawl-mcp) ---

def _v2_actions_capabilities() -> ActionsCapabilitiesResponse:
    common_selector_field = ActionFieldCapability(
        name="selector",
        valueType="string",
        required=False,
        description="CSS selector for the target element.",
    )
    return ActionsCapabilitiesResponse(
        actions=[
            ActionCapability(type="wait", aliases=["sleep"], requestFields=[ActionFieldCapability(name="milliseconds", valueType="integer", description="Delay in milliseconds.")]),
            ActionCapability(type="waitForSelector", aliases=["wait_for_selector"], requestFields=[common_selector_field]),
            ActionCapability(type="waitForNavigation", aliases=["wait_for_navigation"], requestFields=[ActionFieldCapability(name="milliseconds", valueType="integer", description="Optional action timeout override in milliseconds.")]),
            ActionCapability(type="click", requestFields=[common_selector_field, ActionFieldCapability(name="button", valueType="string", description="Mouse button: left/right/middle.")]),
            ActionCapability(type="doubleClick", aliases=["double_click", "dblclick"], requestFields=[common_selector_field]),
            ActionCapability(type="hover", requestFields=[common_selector_field]),
            ActionCapability(type="type", aliases=["fill", "write"], requestFields=[common_selector_field, ActionFieldCapability(name="text", valueType="string", description="Text payload to type.")]),
            ActionCapability(type="clear", aliases=["clearInput", "clear_input"], requestFields=[common_selector_field]),
            ActionCapability(type="selectOption", aliases=["select", "select_option"], requestFields=[common_selector_field, ActionFieldCapability(name="value", valueType="string", description="Option value/text to select.")]),
            ActionCapability(type="check", aliases=["tick"], requestFields=[common_selector_field]),
            ActionCapability(type="uncheck", aliases=["untick"], requestFields=[common_selector_field]),
            ActionCapability(type="press", aliases=["keyPress"], requestFields=[common_selector_field, ActionFieldCapability(name="key", valueType="string", description="Keyboard key, e.g. Enter or Shift+A.")]),
            ActionCapability(type="keyDown", aliases=["key_down"], requestFields=[ActionFieldCapability(name="key", valueType="string", required=True)]),
            ActionCapability(type="keyUp", aliases=["key_up"], requestFields=[ActionFieldCapability(name="key", valueType="string", required=True)]),
            ActionCapability(type="scroll", requestFields=[ActionFieldCapability(name="direction", valueType="string", description="up/down/left/right/top/bottom"), ActionFieldCapability(name="milliseconds", valueType="integer")]),
            ActionCapability(type="focus", requestFields=[common_selector_field]),
            ActionCapability(type="blur", requestFields=[common_selector_field]),
            ActionCapability(
                type="dragAndDrop",
                aliases=["drag_and_drop"],
                requestFields=[
                    common_selector_field,
                    ActionFieldCapability(name="targetSelector", valueType="string", required=True, description="CSS selector for the drop target."),
                ],
            ),
            ActionCapability(
                type="navigate",
                aliases=["goto", "go_to"],
                requestFields=[ActionFieldCapability(name="url", valueType="string", description="Absolute URL to navigate to.")],
                outputs=["output.url"],
            ),
            ActionCapability(
                type="evaluate",
                aliases=["script", "executeJavascript", "execute_javascript"],
                requestFields=[ActionFieldCapability(name="script", valueType="string", required=True)],
                outputs=["output"],
            ),
            ActionCapability(
                type="screenshot",
                requestFields=[common_selector_field, ActionFieldCapability(name="fullPage", valueType="boolean", description="For page-level screenshot when selector is not set.")],
                outputs=["actionArtifacts.actionScreenshot", "data.screenshot"],
            ),
            ActionCapability(
                type="generatePdf",
                aliases=["generate_pdf"],
                outputs=["actionArtifacts.generatedPdf", "data.contentBase64"],
            ),
            ActionCapability(type="scrape", notes="No-op marker action; final extraction occurs at end of /v2/scrape."),
        ],
        documentedRequestFields=[
            "type",
            "selector",
            "targetSelector",
            "text",
            "value",
            "url",
            "key",
            "button",
            "script",
            "milliseconds",
            "direction",
            "fullPage",
        ],
        documentedResponseFields=[
            "actionResults[].index",
            "actionResults[].type",
            "actionResults[].status",
            "actionResults[].reason",
            "actionResults[].error",
            "actionResults[].output",
            "actionArtifacts.generatedPdf.encoding",
            "actionArtifacts.generatedPdf.location",
            "actionArtifacts.generatedPdf.sizeBytes",
            "actionArtifacts.actionScreenshot.encoding",
            "actionArtifacts.actionScreenshot.location",
        ],
    )


@app.get("/v2/actions/capabilities", response_model=ActionsCapabilitiesResponse)
@limiter.limit(V2_LIMIT_MAP)
async def v2_actions_capabilities(request: Request, remote_addr: str = Depends(get_remote_address)):
    _ = request, remote_addr
    return _v2_actions_capabilities()


@app.post("/v2/scrape", response_model=PublicScrapeResponse)
@limiter.limit(V2_LIMIT_SCRAPE)
async def v2_scrape(request: Request, payload: PublicScrapeRequest, remote_addr: str = Depends(get_remote_address)):
    payload = payload.model_dump(by_alias=True, exclude_none=True)
    if settings.kill_switch_v2_scrape:
        record_kill_switch("v2_scrape", "scrape")
        return json_error(
            "Scrape temporarily disabled by kill-switch",
            status_code=503,
            code="KILL_SWITCH_SCRAPE",
        )
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return json_error("Missing url", code="MISSING_URL")
    parsed_url = urlparse(url)
    if not (parsed_url.scheme and parsed_url.netloc):
        return json_error(
            f"Invalid URL: {url}. Please include http:// or https://.",
            code="INVALID_URL",
        )
    if _is_ssrf_blocked_url(url):
        record_security_deny("v2_scrape", "ssrf_blocked")
        return json_error("Target URL blocked by SSRF policy", status_code=403, code="SSRF_BLOCKED")
    try:
        data = await _v2_scrape_data(url, payload)
        data, format_warnings = _augment_v2_data_with_format_objects(data, payload)
        zero_data_retention = _coerce_bool(payload.get("zeroDataRetention"), False)
        artifact_payload = payload
        retention_warnings: list[str] = []
        if zero_data_retention:
            requested_mode = _artifact_mode_from_payload(payload)
            if requested_mode and requested_mode.strip().lower() != "inline":
                artifact_payload = dict(payload)
                artifact_payload["artifact_mode"] = "inline"
                retention_warnings.append(
                    "artifact_mode forced to inline because zeroDataRetention=true"
                )
        transformed, artifact_meta = await _maybe_externalize_v2_artifacts(
            request=request,
            payload=artifact_payload,
            data=data,
        )
        response_payload: Dict[str, Any] = {"success": True, "data": transformed, "artifactMode": artifact_meta["artifactMode"]}
        action_results, action_artifacts = _extract_action_payloads(transformed)
        if action_results is not None:
            response_payload["actionResults"] = action_results
        if action_artifacts is not None:
            response_payload["actionArtifacts"] = action_artifacts
        merged_warnings = format_warnings + retention_warnings
        if merged_warnings:
            response_payload["warning"] = "; ".join(merged_warnings)
        if artifact_meta.get("artifactFallback"):
            response_payload["artifactFallback"] = artifact_meta["artifactFallback"]
        if artifact_meta.get("artifacts"):
            response_payload["artifacts"] = artifact_meta["artifacts"]
        if zero_data_retention:
            response_payload["zeroDataRetentionApplied"] = True
        return response_payload
    except ValueError as e:
        return json_error(str(e), code="INVALID_ARTIFACT_MODE")
    except Exception as e:
        return json_error(str(e), status_code=500, code="INTERNAL_ERROR")

@app.post("/v1/scrape/bulk", response_model=List[ScrapeResponse])
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def bulk_scrape(request: Request, bulk_request: BatchScrapeRequest, remote_addr: str = Depends(get_remote_address)):
    """Scrape a list of URLs in bulk"""

    async def scrape_url_wrapper(url: str, options: ScrapeOptions):
        try:
            parsed_url = urlparse(url)
            if not (parsed_url.scheme and parsed_url.netloc):
                 return ScrapeResponse(success=False, error=f"Invalid URL: {url}")

            document = await scraper.scrape_url(url, options)
            return ScrapeResponse(success=True, data=document)
        except Exception as e:
            return ScrapeResponse(success=False, error=str(e))

    # Convert request to scrape options
    options = ScrapeOptions(
        formats=bulk_request.formats,
        headers=bulk_request.headers,
        include_tags=bulk_request.include_tags,
        exclude_tags=bulk_request.exclude_tags,
        only_main_content=bulk_request.only_main_content,
        wait_for=bulk_request.wait_for,
        timeout=bulk_request.timeout,
        location=bulk_request.location,
        mobile=bulk_request.mobile,
        skip_tls_verification=bulk_request.skip_tls_verification,
        remove_base64_images=bulk_request.remove_base64_images,
        use_relative_links=bulk_request.use_relative_links,
        include_file_body=bulk_request.include_file_body
    )

    tasks = [scrape_url_wrapper(url, options) for url in bulk_request.urls]
    results = await asyncio.gather(*tasks)
    return results

@app.get("/v1/scrape/{url:path}", response_model=ScrapeResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def scrape_url_get(request: Request, url: str, remote_addr: str = Depends(get_remote_address)):
    """Scrape a single URL via GET request"""
    try:
        # Use default options for GET requests
        options = ScrapeOptions()

        # Scrape the URL
        document = await scraper.scrape_url(url, options)

        return ScrapeResponse(
            success=True,
            data=document
        )

    except Exception as e:
        return ScrapeResponse(
            success=False,
            error=str(e)
        )

@app.post("/v1/crawl", response_model=CrawlResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def crawl_url(request: Request, crawl_request: CrawlRequest, background_tasks: BackgroundTasks, remote_addr: str = Depends(get_remote_address)):
    """Start a crawl job"""
    try:
        job_id = str(uuid.uuid4())

        # Create job data
        job_data = {
            "id": job_id,
            "url": crawl_request.url,
            "status": "scraping",
            "completed": 0,
            "total": 0,
            "credits_used": 0,
            "expires_at": utc_now() + timedelta(hours=24),
            "data": [],
            "request": crawl_request.model_dump()
        }

        # Store job data
        if redis_available:
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)  # 24 hours
        else:
            job_storage[job_id] = job_data

        # Start background crawl task
        background_tasks.add_task(crawl_background_task, job_id, crawl_request)

        return CrawlResponse(
            success=True,
            id=job_id,
            url=crawl_request.url
        )

    except Exception as e:
        return CrawlResponse(
            success=False,
            error=str(e)
        )

@app.get("/v1/crawl/{job_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(job_id: str):
    """Get crawl job status"""
    try:
        # Get job data
        if redis_available:
            job_data_str = redis_client.get(f"crawl:{job_id}")
            if job_data_str:
                job_data = deserialize_job_data(job_data_str.decode('utf-8'))
            else:
                job_data = None
        else:
            job_data = job_storage.get(job_id)

        if not job_data:
            raise HTTPException(status_code=404, detail="Crawl job not found")

        return CrawlStatusResponse(**job_data)

    except HTTPException:
        raise
    except Exception as e:
        return CrawlStatusResponse(
            success=False,
            error=str(e)
        )

@app.post("/v1/crawl/{job_id}/cancel")
async def cancel_crawl(job_id: str):
    """Cancel a running crawl job"""
    _cancelled_jobs.add(job_id)
    # Update status if present
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode('utf-8'))
            job_data.update({"status": "cancelled"})
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
    else:
        job_data = job_storage.get(job_id)
        if job_data:
            job_data.update({"status": "cancelled"})
            job_storage[job_id] = job_data
    return {"success": True}

@app.get("/v1/crawl/{job_id}/export")
async def export_crawl(job_id: str, format: str = "jsonl"):
    """Export crawl results (simple JSONL stream)."""
    from fastapi import Response
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        if not job_data_str:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        job_data = deserialize_job_data(job_data_str.decode('utf-8'))
    else:
        job_data = job_storage.get(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Crawl job not found")
    docs = job_data.get("data", [])
    if format == "jsonl":
        lines = "\n".join(json.dumps(d) for d in docs)
        return Response(content=lines, media_type="application/x-ndjson")
    return {"success": True, "data": docs}


@app.post("/v2/crawl", response_model=PublicJobCreateResponse)
@limiter.limit(V2_LIMIT_CRAWL)
async def v2_crawl(request: Request, payload: PublicCrawlRequest, background_tasks: BackgroundTasks, remote_addr: str = Depends(get_remote_address)):
    payload = payload.model_dump(by_alias=True, exclude_none=True)
    if settings.kill_switch_v2_crawl:
        record_kill_switch("v2_crawl", "crawl")
        return json_error(
            "Crawl temporarily disabled by kill-switch",
            status_code=503,
            code="KILL_SWITCH_CRAWL",
        )
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return json_error("Missing url", code="MISSING_URL")
    if _is_ssrf_blocked_url(url):
        record_security_deny("v2_crawl", "ssrf_blocked")
        return json_error("Target URL blocked by SSRF policy", status_code=403, code="SSRF_BLOCKED")
    try:
        idem_key = _get_idempotency_key(request)
        payload_hash = _payload_hash_for_idempotency(payload) if idem_key else ""
        if idem_key:
            existing = _get_idempotent_response("v2:crawl", idem_key, payload_hash)
            if existing is not None:
                return existing

        sitemap = payload.get("sitemap")
        ignore_sitemap = sitemap == "skip"

        requested_limit = payload.get("limit") or 50
        try:
            requested_limit = int(requested_limit)
        except Exception:
            requested_limit = 50
        if requested_limit > settings.budget_v2_crawl_limit_cap:
            record_budget_guardrail("v2_crawl", "crawl_limit_cap", "blocked")
            return json_error(
                f"Requested crawl limit exceeds budget cap ({settings.budget_v2_crawl_limit_cap})",
                code="BUDGET_LIMIT_EXCEEDED",
                maxLimit=settings.budget_v2_crawl_limit_cap,
            )

        crawl_request = CrawlRequest(
            url=url,
            include_paths=payload.get("includePaths"),
            exclude_paths=payload.get("excludePaths"),
            max_depth=payload.get("maxDiscoveryDepth") or 2,
            limit=requested_limit,
            allow_external_links=payload.get("allowExternalLinks") or False,
            include_subdomains=payload.get("allowSubdomains") or False,
            ignore_sitemap=ignore_sitemap,
            ignore_query_parameters=_coerce_bool(payload.get("ignoreQueryParameters"), False),
            deduplicate_similar_urls=_coerce_bool(payload.get("deduplicateSimilarURLs"), False),
            scrape_options=_scrape_options_from_v2(payload.get("scrapeOptions") or {}),
            webhook=payload.get("webhook"),
            delay=payload.get("delay") or 0,
            max_concurrency=payload.get("maxConcurrency") or 10,
        )

        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "url": crawl_request.url,
            "status": "scraping",
            "completed": 0,
            "total": 0,
            "credits_used": 0,
            "expires_at": utc_now() + timedelta(hours=24),
            "data": [],
            "errors": [],
            "robots_blocked": [],
            "zero_data_retention": _coerce_bool(payload.get("zeroDataRetention"), False),
            "request": crawl_request.model_dump(),
            "v2_sitemap_mode": sitemap,
            "tenant_id": _request_tenant_id(request),
        }

        if redis_available:
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_id] = job_data

        background_tasks.add_task(crawl_background_task, job_id, crawl_request)

        response = {"success": True, "id": job_id, "url": crawl_request.url}
        if idem_key:
            _set_idempotent_response("v2:crawl", idem_key, payload_hash, response)
        return response
    except HTTPException:
        raise
    except Exception as e:
        return json_error(str(e), status_code=500, code="INTERNAL_ERROR")


@app.get("/v2/crawl/{job_id}", response_model=PublicJobStatusResponse)
async def v2_get_crawl_status(request: Request, job_id: str):
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
        else:
            job_data = None
    else:
        job_data = job_storage.get(job_id)

    if not job_data:
        return json_error("Crawl job not found", status_code=404, code="JOB_NOT_FOUND")
    tenant_error = _enforce_job_tenant(request, job_data)
    if tenant_error is not None:
        return tenant_error

    cursor_value = request.query_params.get("cursor") or request.query_params.get("next")
    offset = _parse_status_cursor(cursor_value)
    if offset is None:
        return json_error("Invalid cursor", code="INVALID_CURSOR")
    limit_raw = request.query_params.get("limit")
    if limit_raw is None or not str(limit_raw).strip():
        page_limit = 100
    else:
        try:
            page_limit = int(limit_raw)
        except Exception:
            return json_error("Invalid limit", code="INVALID_LIMIT")
    page_limit = min(1000, max(1, page_limit))

    expires_at = job_data.get("expires_at")
    expires_at_str = expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
    docs_in = job_data.get("data") or []
    docs_out = [_v2_document_from_v1(d) for d in docs_in]
    paged_data = docs_out[offset : offset + page_limit]
    next_cursor = f"offset:{offset + page_limit}" if (offset + page_limit) < len(docs_out) else None

    return _v2_status_payload(
        {
        "status": job_data.get("status"),
        "completed": job_data.get("completed", 0),
        "total": job_data.get("total", 0),
        "creditsUsed": job_data.get("credits_used"),
        "expiresAt": expires_at_str,
        "next": next_cursor,
        "data": paged_data,
        "zeroDataRetentionApplied": bool(job_data.get("zero_data_retention_applied")),
        }
    )


@app.delete("/v2/crawl/{job_id}", response_model=PublicCancelResponse)
async def v2_cancel_crawl(request: Request, job_id: str):
    _cancelled_jobs.add(job_id)
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
            tenant_error = _enforce_job_tenant(request, job_data)
            if tenant_error is not None:
                return tenant_error
            job_data.update({"status": "cancelled"})
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
    else:
        job_data = job_storage.get(job_id)
        if job_data:
            tenant_error = _enforce_job_tenant(request, job_data)
            if tenant_error is not None:
                return tenant_error
            job_data.update({"status": "cancelled"})
            job_storage[job_id] = job_data
    return _v2_cancel_payload("Crawl cancelled.")


@app.get("/v2/crawl/{job_id}/errors", response_model=PublicJobErrorsResponse)
async def v2_crawl_errors(request: Request, job_id: str):
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        job_data = deserialize_job_data(job_data_str.decode("utf-8")) if job_data_str else None
    else:
        job_data = job_storage.get(job_id)
    if not job_data:
        return json_error("Crawl job not found", status_code=404, code="JOB_NOT_FOUND")
    tenant_error = _enforce_job_tenant(request, job_data)
    if tenant_error is not None:
        return tenant_error
    errors = job_data.get("errors") if isinstance(job_data.get("errors"), list) else []
    robots_blocked = (
        job_data.get("robots_blocked")
        if isinstance(job_data.get("robots_blocked"), list)
        else []
    )
    return _v2_errors_payload(errors=errors, robots_blocked=robots_blocked)

@app.post("/v1/batch-scrape", response_model=BatchScrapeResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def batch_scrape(request: Request, batch_request: BatchScrapeRequest, background_tasks: BackgroundTasks, remote_addr: str = Depends(get_remote_address)):
    """Start a batch scraping job"""
    try:
        job_id = str(uuid.uuid4())

        # Validate URLs
        invalid_urls = []
        valid_urls = []

        for url in batch_request.urls:
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc:
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            except:
                invalid_urls.append(url)

        if not valid_urls:
            return BatchScrapeResponse(
                success=False,
                error="No valid URLs provided",
                invalid_urls=invalid_urls
            )

        # Create job data
        job_data = {
            "id": job_id,
            "status": "scraping",
            "completed": 0,
            "total": len(valid_urls),
            "credits_used": 0,
            "expires_at": utc_now() + timedelta(hours=24),
            "data": [],
            "request": batch_request.model_dump(),
            "valid_urls": valid_urls
        }

        # Store job data
        if redis_available:
            redis_client.set(f"batch:{job_id}", serialize_job_data(job_data), ex=86400)
        else:
            job_storage[f"batch:{job_id}"] = job_data

        # Start background batch task
        background_tasks.add_task(batch_scrape_background_task, job_id, valid_urls, batch_request)

        return BatchScrapeResponse(
            success=True,
            id=job_id,
            invalid_urls=invalid_urls if invalid_urls else None
        )

    except Exception as e:
        return BatchScrapeResponse(
            success=False,
            error=str(e)
        )

@app.get("/v1/batch-scrape/{job_id}")
async def get_batch_status(job_id: str):
    """Get batch scrape job status"""
    try:
        # Get job data
        if redis_available:
            job_data_str = redis_client.get(f"batch:{job_id}")
            if job_data_str:
                job_data = deserialize_job_data(job_data_str.decode('utf-8'))
            else:
                job_data = None
        else:
            job_data = job_storage.get(f"batch:{job_id}")

        if not job_data:
            raise HTTPException(status_code=404, detail="Batch job not found")

        return job_data

    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/v1/map", response_model=MapResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def map_url(request: Request, map_request: MapRequest, remote_addr: str = Depends(get_remote_address)):
    """Map a website to get all its links"""
    try:
        links = []
        base_domain = urlparse(map_request.url).netloc

        # Try to get sitemap first
        if not map_request.ignore_sitemap:
            sitemap_links = await get_sitemap_links(map_request.url)
            links.extend(sitemap_links)

        # If no sitemap or we want to crawl for more links
        if not links or not map_request.ignore_sitemap:
            # Scrape the main page for links
            options = ScrapeOptions(formats=["links"])
            document = await scraper.scrape_url(map_request.url, options)

            if document.links:
                for link in document.links:
                    parsed_link = urlparse(link)

                    # Filter based on subdomain settings
                    if map_request.include_subdomains:
                        if base_domain not in parsed_link.netloc:
                            continue
                    else:
                        if parsed_link.netloc != base_domain:
                            continue

                    if link not in links:
                        links.append(link)

        # Apply limit
        if map_request.limit and len(links) > map_request.limit:
            links = links[:map_request.limit]

        return MapResponse(
            success=True,
            links=links
        )

    except Exception as e:
        return MapResponse(
            success=False,
            error=str(e)
        )

async def get_sitemap_links(url: str) -> List[str]:
    """Extract links from sitemap"""
    links = []
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

    sitemap_urls = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{url.rstrip('/')}/sitemap.xml"
    ]

    async with httpx.AsyncClient() as client:
        for sitemap_url in sitemap_urls:
            try:
                response = await client.get(sitemap_url, timeout=10)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)

                    # Handle sitemap namespace
                    ns = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                    # Extract URLs from sitemap
                    for url_elem in root.findall('.//sitemap:url/sitemap:loc', ns):
                        if url_elem.text:
                            links.append(url_elem.text)

                    # If this is a sitemap index, get URLs from sub-sitemaps
                    for sitemap_elem in root.findall('.//sitemap:sitemap/sitemap:loc', ns):
                        if sitemap_elem.text:
                            sub_links = await get_sitemap_links(sitemap_elem.text)
                            links.extend(sub_links)

                    break  # Found a sitemap, no need to try others
            except:
                continue

    return links


@app.post("/v2/map", response_model=PublicMapResponse)
@limiter.limit(V2_LIMIT_MAP)
async def v2_map(request: Request, payload: PublicMapRequest, remote_addr: str = Depends(get_remote_address)):
    payload = payload.model_dump(by_alias=True, exclude_none=True)
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return json_error("Missing url", code="MISSING_URL")
    if _is_ssrf_blocked_url(url):
        record_security_deny("v2_map", "ssrf_blocked")
        return json_error("Target URL blocked by SSRF policy", status_code=403, code="SSRF_BLOCKED")
    try:
        links: list[str] = []
        seen: set[str] = set()
        base_domain = urlparse(_canonicalize_url(url)).netloc
        sitemap_mode = payload.get("sitemap")
        include_subdomains = _coerce_bool(payload.get("includeSubdomains"), False)
        limit = payload.get("limit")
        search_filter = payload.get("search")
        ignore_query_parameters = _coerce_bool(payload.get("ignoreQueryParameters"), False)

        ignore_sitemap = sitemap_mode == "skip"
        only_sitemap = sitemap_mode == "only"
        if not ignore_sitemap:
            try:
                for link in await get_sitemap_links(url):
                    link = _canonicalize_url(link, ignore_query_parameters=ignore_query_parameters)
                    if link not in seen:
                        seen.add(link)
                        links.append(link)
            except Exception:
                pass

        if not only_sitemap and (not links or not ignore_sitemap):
            options = ScrapeOptions(formats=["links"])
            document = await scraper.scrape_url(url, options)
            if document.links:
                for link in document.links:
                    link = _canonicalize_url(link, ignore_query_parameters=ignore_query_parameters)
                    parsed_link = urlparse(link)
                    if parsed_link.scheme.lower() not in {"http", "https"} or not parsed_link.netloc:
                        continue

                    if include_subdomains:
                        if parsed_link.netloc != base_domain and not parsed_link.netloc.endswith("." + base_domain):
                            continue
                    else:
                        if parsed_link.netloc != base_domain:
                            continue

                    if link not in seen:
                        seen.add(link)
                        links.append(link)

        if isinstance(search_filter, str) and search_filter.strip():
            token = search_filter.strip()
            links = [l for l in links if token in l]

        if isinstance(limit, int) and limit > 0 and len(links) > limit:
            links = links[:limit]

        link_objects: list[Dict[str, Any]] = [
            {
                "url": link,
                "title": None,
                "description": None,
            }
            for link in links
        ]
        response_payload: Dict[str, Any] = {"success": True, "links": link_objects}
        if not settings.strict_firecrawl_v2:
            # Transitional compatibility path for legacy internal consumers.
            response_payload["links_text"] = links
        return response_payload
    except Exception as e:
        return json_error(str(e), status_code=500, code="INTERNAL_ERROR")


@app.post("/v2/batch/scrape", response_model=PublicJobCreateResponse)
@limiter.limit(V2_LIMIT_BATCH_SCRAPE)
async def v2_batch_scrape(request: Request, payload: PublicBatchScrapeRequest, background_tasks: BackgroundTasks, remote_addr: str = Depends(get_remote_address)):
    payload = payload.model_dump(by_alias=True, exclude_none=True)
    urls = payload.get("urls")
    if not isinstance(urls, list) or not urls:
        return json_error("No URLs provided", code="MISSING_URLS")

    idem_key = _get_idempotency_key(request)
    payload_hash = _payload_hash_for_idempotency(payload) if idem_key else ""
    if idem_key:
        existing = _get_idempotent_response("v2:batch", idem_key, payload_hash)
        if existing is not None:
            return existing

    # Map v2 scrape options to v1 BatchScrapeRequest
    options_payload: Dict[str, Any] = payload.copy()
    options_payload.pop("urls", None)
    scrape_options = _scrape_options_from_v2(options_payload)

    batch_request = BatchScrapeRequest(
        urls=[str(u) for u in urls],
        formats=scrape_options.formats,
        headers=scrape_options.headers,
        include_tags=scrape_options.include_tags,
        exclude_tags=scrape_options.exclude_tags,
        only_main_content=bool(scrape_options.only_main_content) if scrape_options.only_main_content is not None else False,
        wait_for=scrape_options.wait_for,
        timeout=scrape_options.timeout,
        location=scrape_options.location,
        mobile=bool(scrape_options.mobile) if scrape_options.mobile is not None else False,
        skip_tls_verification=bool(scrape_options.skip_tls_verification) if scrape_options.skip_tls_verification is not None else False,
        remove_base64_images=bool(scrape_options.remove_base64_images) if scrape_options.remove_base64_images is not None else True,
        max_concurrency=int(payload.get("maxConcurrency") or 5),
    )

    invalid_urls: list[str] = []
    blocked_urls: list[str] = []
    valid_urls: list[str] = []
    for u in batch_request.urls:
        parsed = urlparse(u)
        if parsed.scheme and parsed.netloc and not _is_ssrf_blocked_url(u):
            valid_urls.append(u)
        else:
            invalid_urls.append(u)
            if parsed.scheme and parsed.netloc:
                blocked_urls.append(u)
    if not valid_urls:
        if blocked_urls and len(blocked_urls) == len(invalid_urls):
            record_security_deny("v2_batch_scrape", "ssrf_blocked")
            return json_error("All URLs blocked by SSRF policy", status_code=403, code="SSRF_BLOCKED", invalidURLs=invalid_urls)
        return json_error("No valid URLs provided", code="NO_VALID_URLS", invalidURLs=invalid_urls)

    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "status": "scraping",
        "completed": 0,
        "total": len(valid_urls),
        "credits_used": 0,
        "expires_at": utc_now() + timedelta(hours=24),
        "data": [],
        "errors": [],
        "robots_blocked": [],
        "zero_data_retention": _coerce_bool(payload.get("zeroDataRetention"), False),
        "request": payload,
        "valid_urls": valid_urls,
        "tenant_id": _request_tenant_id(request),
    }

    job_key = f"batch:{job_id}"
    if redis_available:
        redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
    else:
        job_storage[job_key] = job_data

    background_tasks.add_task(batch_scrape_background_task, job_id, valid_urls, batch_request)

    response = {
        "success": True,
        "id": job_id,
        "url": valid_urls[0],
        "invalidURLs": invalid_urls or None,
    }
    if idem_key:
        _set_idempotent_response("v2:batch", idem_key, payload_hash, response)
    return response


@app.get("/v2/batch/scrape/{job_id}", response_model=PublicJobStatusResponse)
async def v2_get_batch_status(request: Request, job_id: str):
    job_key = f"batch:{job_id}"
    if redis_available:
        job_data_str = redis_client.get(job_key)
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
        else:
            job_data = None
    else:
        job_data = job_storage.get(job_key)

    if not job_data:
        return json_error("Batch job not found", status_code=404, code="JOB_NOT_FOUND")
    tenant_error = _enforce_job_tenant(request, job_data)
    if tenant_error is not None:
        return tenant_error

    cursor_value = request.query_params.get("cursor") or request.query_params.get("next")
    offset = _parse_status_cursor(cursor_value)
    if offset is None:
        return json_error("Invalid cursor", code="INVALID_CURSOR")
    limit_raw = request.query_params.get("limit")
    if limit_raw is None or not str(limit_raw).strip():
        page_limit = 100
    else:
        try:
            page_limit = int(limit_raw)
        except Exception:
            return json_error("Invalid limit", code="INVALID_LIMIT")
    page_limit = min(1000, max(1, page_limit))

    expires_at = job_data.get("expires_at")
    expires_at_str = expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
    docs_in = job_data.get("data") or []
    docs_out = [_v2_document_from_v1(d) for d in docs_in]
    paged_data = docs_out[offset : offset + page_limit]
    next_cursor = f"offset:{offset + page_limit}" if (offset + page_limit) < len(docs_out) else None

    return _v2_status_payload(
        {
        "status": job_data.get("status"),
        "completed": job_data.get("completed", 0),
        "total": job_data.get("total", 0),
        "creditsUsed": job_data.get("credits_used"),
        "expiresAt": expires_at_str,
        "next": next_cursor,
        "data": paged_data,
        "zeroDataRetentionApplied": bool(job_data.get("zero_data_retention_applied")),
        }
    )


@app.delete("/v2/batch/scrape/{job_id}", response_model=PublicCancelResponse)
async def v2_cancel_batch(request: Request, job_id: str):
    _cancelled_batch_jobs.add(job_id)
    job_key = f"batch:{job_id}"
    if redis_available:
        job_data_str = redis_client.get(job_key)
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
            tenant_error = _enforce_job_tenant(request, job_data)
            if tenant_error is not None:
                return tenant_error
            job_data.update({"status": "cancelled"})
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
    else:
        job_data = job_storage.get(job_key)
        if job_data:
            tenant_error = _enforce_job_tenant(request, job_data)
            if tenant_error is not None:
                return tenant_error
            job_data.update({"status": "cancelled"})
            job_storage[job_key] = job_data
    return _v2_cancel_payload("Batch scrape cancelled.")


@app.get("/v2/batch/scrape/{job_id}/errors", response_model=PublicJobErrorsResponse)
async def v2_batch_errors(request: Request, job_id: str):
    job_key = f"batch:{job_id}"
    if redis_available:
        job_data_str = redis_client.get(job_key)
        job_data = deserialize_job_data(job_data_str.decode("utf-8")) if job_data_str else None
    else:
        job_data = job_storage.get(job_key)
    if not job_data:
        return json_error("Batch job not found", status_code=404, code="JOB_NOT_FOUND")
    tenant_error = _enforce_job_tenant(request, job_data)
    if tenant_error is not None:
        return tenant_error
    errors = job_data.get("errors") if isinstance(job_data.get("errors"), list) else []
    robots_blocked = (
        job_data.get("robots_blocked")
        if isinstance(job_data.get("robots_blocked"), list)
        else []
    )
    return _v2_errors_payload(errors=errors, robots_blocked=robots_blocked)


@app.post("/v2/search", response_model=PublicSearchResponse)
@limiter.limit(V2_LIMIT_SEARCH)
async def v2_search(request: Request, payload: PublicSearchRequest, remote_addr: str = Depends(get_remote_address)):
    payload = payload.model_dump(by_alias=True, exclude_none=True)
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return json_error(
            "Missing query",
            code="MISSING_QUERY",
            warning=None,
            id=str(uuid.uuid4()),
            creditsUsed=0,
        )
    try:
        request_id = str(uuid.uuid4())
        warnings: list[str] = []
        credits_used = 0
        sources_succeeded = 0

        try:
            limit = int(payload.get("limit") or 5)
        except Exception:
            limit = 5
            warnings.append("Invalid limit ignored; defaulting to 5")
        if limit <= 0:
            limit = 5
            warnings.append("Non-positive limit ignored; defaulting to 5")
        # Cap by configured budget guardrail to prevent runaway spend.
        if limit > settings.budget_v2_search_limit_cap:
            record_budget_guardrail("v2_search", "search_limit_cap", "capped")
            warnings.append(
                f"Requested limit capped by budget guardrail ({settings.budget_v2_search_limit_cap})"
            )
        limit = min(limit, settings.budget_v2_search_limit_cap)

        supported_sources = {"web", "news", "images"}
        raw_sources = payload.get("sources")
        parsed_sources: set[str] = set()
        if isinstance(raw_sources, list):
            for idx, source_item in enumerate(raw_sources):
                candidate: str | None = None
                if isinstance(source_item, str):
                    candidate = source_item.strip().lower()
                elif isinstance(source_item, dict) and isinstance(source_item.get("type"), str):
                    candidate = source_item["type"].strip().lower()
                else:
                    warnings.append(f"Invalid source entry at index {idx} ignored")
                    continue

                if not candidate:
                    warnings.append(f"Empty source entry at index {idx} ignored")
                elif candidate not in supported_sources:
                    warnings.append(f"Unsupported source '{candidate}' ignored")
                else:
                    parsed_sources.add(candidate)

        categories_in = payload.get("categories")
        if isinstance(categories_in, list):
            category_to_source = {
                "web": "web",
                "news": "news",
                "image": "images",
                "images": "images",
            }
            for raw_category in categories_in:
                if not isinstance(raw_category, str):
                    continue
                mapped_source = category_to_source.get(raw_category.strip().lower())
                if mapped_source is not None:
                    parsed_sources.add(mapped_source)

        if not parsed_sources:
            parsed_sources = {"web"}

        data: Dict[str, Any] = {"web": [], "news": [], "images": []}

        location = payload.get("location")
        location_country = location.get("country") if isinstance(location, dict) else None
        location_languages = location.get("languages") if isinstance(location, dict) else None
        gl = payload.get("country") or payload.get("gl") or location_country
        hl = payload.get("lang") or payload.get("hl")
        if not hl and isinstance(location_languages, list):
            first_language = next((x for x in location_languages if isinstance(x, str) and x.strip()), None)
            if first_language is not None:
                hl = first_language.strip()

        tbs = payload.get("tbs")
        ignore_invalid_urls = _coerce_bool(payload.get("ignoreInvalidURLs"), False)
        timeout_in = payload.get("timeout")
        timeout_seconds = _ms_to_seconds(timeout_in)
        if timeout_in is not None and timeout_seconds is None:
            warnings.append("Invalid timeout ignored")

        serper = SerperClient()

        if "web" in parsed_sources:
            try:
                res, _ = await serper.search(SerperSearchRequest(q=query, gl=gl, hl=hl, num=limit, tbs=tbs))
                credits_used += int(getattr(res, "credits_used", 1) or 1)
                sources_succeeded += 1
                web_results: list[Dict[str, Any]] = []
                if res.organic:
                    for r in res.organic[:limit]:
                        if not r.link:
                            if not ignore_invalid_urls:
                                warnings.append("Dropped web result with missing url")
                            continue
                        web_results.append({"url": r.link, "title": r.title, "description": r.snippet})

                scrape_opts_payload = payload.get("scrapeOptions") or {}
                if not isinstance(scrape_opts_payload, dict):
                    scrape_opts_payload = {}
                    warnings.append("Invalid scrapeOptions ignored")
                if timeout_in is not None and "timeout" not in scrape_opts_payload and timeout_seconds is not None:
                    scrape_opts_payload["timeout"] = timeout_in

                formats = scrape_opts_payload.get("formats")
                if isinstance(formats, list) and len(formats) > 0:
                    semaphore = asyncio.Semaphore(min(settings.max_concurrent_requests, 5))

                    async def scrape_one(u: str):
                        async with semaphore:
                            try:
                                scrape_payload = {"url": u, **scrape_opts_payload}
                                return await _v2_scrape_data(u, scrape_payload)
                            except Exception:
                                return {"metadata": {"sourceURL": u, "statusCode": None}}

                    data["web"] = await asyncio.gather(*[scrape_one(r["url"]) for r in web_results])
                else:
                    data["web"] = web_results
            except Exception as e:
                warnings.append(f"web source failed: {e}")

        if "news" in parsed_sources:
            try:
                news_res, _ = await serper.news(SerperNewsRequest(q=query, gl=gl, hl=hl, num=limit))
                credits_used += 1
                sources_succeeded += 1
                news_items = news_res.items[:limit] if isinstance(news_res.items, list) else []
                out_news: list[Dict[str, Any]] = []
                for idx, item in enumerate(news_items):
                    if not isinstance(item, dict):
                        continue
                    out_news.append(
                        _remove_empty_top_level(
                            {
                                "title": item.get("title"),
                                "url": item.get("link") or item.get("url"),
                                "snippet": item.get("snippet") or item.get("description"),
                                "date": item.get("date"),
                                "imageUrl": item.get("imageUrl") or item.get("image"),
                                "position": item.get("position") or (idx + 1),
                                "category": item.get("source") or item.get("category"),
                            }
                        )
                    )
                data["news"] = out_news
            except Exception as e:
                warnings.append(f"news source failed: {e}")

        if "images" in parsed_sources:
            try:
                img_res, _ = await serper.images(SerperImageRequest(q=query, gl=gl, hl=hl, num=limit))
                credits_used += 1
                sources_succeeded += 1
                img_items = img_res.items[:limit] if isinstance(img_res.items, list) else []
                out_images: list[Dict[str, Any]] = []
                for idx, item in enumerate(img_items):
                    if not isinstance(item, dict):
                        continue
                    out_images.append(
                        _remove_empty_top_level(
                            {
                                "title": item.get("title"),
                                "imageUrl": item.get("imageUrl") or item.get("image"),
                                "imageWidth": item.get("imageWidth") or item.get("imageWidthPx") or item.get("width"),
                                "imageHeight": item.get("imageHeight") or item.get("imageHeightPx") or item.get("height"),
                                "url": item.get("link") or item.get("url"),
                                "position": item.get("position") or (idx + 1),
                            }
                        )
                    )
                data["images"] = out_images
            except Exception as e:
                warnings.append(f"images source failed: {e}")

        deduped_warnings = list(dict.fromkeys(warnings))
        warning_value = "; ".join(deduped_warnings) if deduped_warnings else None
        if sources_succeeded == 0:
            return json_error(
                "Search failed",
                status_code=502,
                code="UPSTREAM_SEARCH_FAILED",
                warning=warning_value,
                id=request_id,
                creditsUsed=credits_used,
            )

        return {
            "success": True,
            "data": data,
            "warning": warning_value,
            "id": request_id,
            "creditsUsed": credits_used,
        }
    except Exception as e:
        return json_error(
            str(e),
            status_code=500,
            code="INTERNAL_ERROR",
            warning=None,
            id=str(uuid.uuid4()),
            creditsUsed=0,
        )


@app.post("/v2/extract", response_model=PublicJobCreateResponse)
@limiter.limit(V2_LIMIT_EXTRACT)
async def v2_extract(
    request: Request,
    payload: PublicExtractRequest,
    background_tasks: BackgroundTasks,
    remote_addr: str = Depends(get_remote_address),
):
    payload = payload.model_dump(by_alias=True, exclude_none=True)
    urls = payload.get("urls")
    if not isinstance(urls, list) or not urls:
        return json_error("No URLs provided", code="MISSING_URLS")
    if len(urls) > settings.budget_v2_extract_url_cap:
        record_budget_guardrail("v2_extract", "extract_url_cap", "blocked")
        return json_error(
            f"Requested URL count exceeds budget cap ({settings.budget_v2_extract_url_cap})",
            code="EXTRACT_BUDGET_EXCEEDED",
            maxURLs=settings.budget_v2_extract_url_cap,
        )

    idem_key = _get_idempotency_key(request)
    payload_hash = _payload_hash_for_idempotency(payload) if idem_key else ""
    if idem_key:
        existing = _get_idempotent_response("v2:extract", idem_key, payload_hash)
        if existing is not None:
            return existing

    invalid_urls: list[str] = []
    blocked_urls: list[str] = []
    valid_urls: list[str] = []
    for u in urls:
        parsed = urlparse(str(u))
        if parsed.scheme and parsed.netloc and not _is_ssrf_blocked_url(str(u)):
            valid_urls.append(str(u))
        else:
            invalid_urls.append(str(u))
            if parsed.scheme and parsed.netloc:
                blocked_urls.append(str(u))

    if not valid_urls:
        if blocked_urls and len(blocked_urls) == len(invalid_urls):
            record_security_deny("v2_extract", "ssrf_blocked")
            return json_error("All URLs blocked by SSRF policy", status_code=403, code="SSRF_BLOCKED", invalidURLs=invalid_urls)
        return json_error("No valid URLs provided", code="NO_VALID_URLS", invalidURLs=invalid_urls)

    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "status": "processing",
        "completed": 0,
        "total": len(valid_urls),
        "credits_used": 0,
        "expires_at": utc_now() + timedelta(hours=24),
        "data": {"documents": [], "errors": []},
        "invalid_urls": invalid_urls,
        "zero_data_retention": _coerce_bool(payload.get("zeroDataRetention"), False),
        "request": payload,
        "valid_urls": valid_urls,
        "tenant_id": _request_tenant_id(request),
    }

    extract_job_key = f"extract:{job_id}"
    if redis_available:
        redis_client.set(extract_job_key, serialize_job_data(job_data), ex=86400)
    else:
        job_storage[extract_job_key] = job_data

    background_tasks.add_task(extract_background_task, job_id, valid_urls, payload)

    response = {
        "success": True,
        "id": job_id,
        "invalidURLs": invalid_urls or None,
    }
    if idem_key:
        _set_idempotent_response("v2:extract", idem_key, payload_hash, response)
    return response


@app.get("/v2/extract/{job_id}", response_model=PublicJobStatusResponse)
async def v2_extract_status(request: Request, job_id: str):
    extract_job_key = f"extract:{job_id}"
    if redis_available:
        job_data_str = redis_client.get(extract_job_key)
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
        else:
            job_data = None
    else:
        job_data = job_storage.get(extract_job_key)

    if not job_data:
        return json_error("Extract job not found", status_code=404, code="JOB_NOT_FOUND")
    tenant_error = _enforce_job_tenant(request, job_data)
    if tenant_error is not None:
        return tenant_error

    expires_at = job_data.get("expires_at")
    expires_at_str = expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
    return _v2_status_payload(
        {
        "id": job_data.get("id", job_id),
        "status": job_data.get("status"),
        "completed": job_data.get("completed", 0),
        "total": job_data.get("total", 0),
        "creditsUsed": job_data.get("credits_used", 0),
        "expiresAt": expires_at_str,
        "next": job_data.get("next"),
        "data": job_data.get("data"),
        "invalidURLs": job_data.get("invalid_urls"),
        "error": job_data.get("error"),
        "zeroDataRetentionApplied": bool(job_data.get("zero_data_retention_applied")),
        }
    )


# --- Retired Hermes compatibility surface ---

@app.api_route(
    "/v1/hermes/{legacy_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def retired_hermes_compat_endpoint(legacy_path: str):
    return json_error(
        "CandleCrawl no longer serves /v1/hermes compatibility endpoints. "
        "Use the CandleCrawl-native /v2 API surface instead.",
        status_code=410,
        code="HERMES_COMPAT_RETIRED",
        legacyPath=f"/v1/hermes/{legacy_path}",
        migrationTarget="/v2",
    )

async def crawl_background_task(job_id: str, crawl_request: CrawlRequest):
    """Background task for crawling with depth and concurrency control."""
    start_time = time.time()
    job_data: Dict[str, Any] = {"id": job_id}
    try:
        # Get job data from storage
        if redis_available:
            job_data_str = redis_client.get(f"crawl:{job_id}")
            if not job_data_str:
                logger.error(f"Crawl job {job_id} not found in Redis.")
                return
            job_data = deserialize_job_data(job_data_str.decode('utf-8'))
        else:
            job_data = job_storage.get(job_id)
            if not job_data:
                logger.error(f"Crawl job {job_id} not found in memory.")
                return

        # --- Crawler Setup ---
        options = crawl_request.scrape_options or ScrapeOptions(only_main_content=True)
        ignore_query_parameters = bool(getattr(crawl_request, "ignore_query_parameters", False))
        deduplicate_similar_urls = bool(getattr(crawl_request, "deduplicate_similar_urls", False))

        def canon(u: str) -> str:
            return _canonicalize_url(
                u,
                ignore_query_parameters=ignore_query_parameters,
                deduplicate_similar_urls=deduplicate_similar_urls,
            )

        frontier = MemoryFrontier()
        crawled_urls: set[str] = set()
        results: list[FirecrawlDocument] = []
        crawl_errors: list[Dict[str, Any]] = []
        robots_blocked: set[str] = set()
        error_lock = asyncio.Lock()
        start_parsed = urlparse(canon(crawl_request.url))
        base_domain = start_parsed.netloc
        start_job_time = time.time()
        total_bytes: int = 0

        # robots.txt setup per origin
        robots_cache: dict[str, RobotFileParser] = {}
        host_semaphores: dict[str, asyncio.Semaphore] = {}
        host_locks: dict[str, asyncio.Lock] = {}
        host_last_fetch: dict[str, float] = {}

        async def get_robots_parser_for_url(u: str) -> RobotFileParser:
            parsed = urlparse(u)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin in robots_cache:
                return robots_cache[origin]
            rp = RobotFileParser()
            robots_url = urljoin(origin, "/robots.txt")
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        rp.parse([])
            except Exception:
                rp.parse([])
            robots_cache[origin] = rp
            return rp

        def domain_allowed(u: str) -> bool:
            parsed = urlparse(u)
            if crawl_request.allow_external_links:
                return True
            if crawl_request.include_subdomains:
                return parsed.netloc == base_domain or parsed.netloc.endswith("." + base_domain)
            return parsed.netloc == base_domain

        def path_allowed(u: str) -> bool:
            parsed = urlparse(u)
            path = parsed.path or "/"
            if crawl_request.include_paths:
                if not any(path.startswith(p) for p in (crawl_request.include_paths or [])):
                    return False
            if crawl_request.exclude_paths:
                if any(path.startswith(p) for p in (crawl_request.exclude_paths or [])):
                    return False
            return True

        async def enqueue(u: str, depth: int):
            parsed = urlparse(u)
            if parsed.scheme.lower() not in ["http", "https"] or not parsed.netloc:
                return
            u = canon(u)
            if u in crawled_urls:
                return
            if not domain_allowed(u) or not path_allowed(u):
                return
            if crawl_request.limit and (len(results) + frontier.size()) >= crawl_request.limit:
                return
            await frontier.enqueue(u, depth)

        async def _record_crawl_error(url: str, error: str, *, blocked_reason: str | None = None) -> None:
            async with error_lock:
                crawl_errors.append(_make_job_error_item(url, error))
                if blocked_reason == "robots_disallow":
                    robots_blocked.add(url)

        # Seed with start URL unconditionally (bypass path/domain filters)
        await frontier.enqueue(canon(crawl_request.url), 0)
        # Seed with sitemap links if allowed
        if not crawl_request.ignore_sitemap:
            try:
                sitemap_links = await get_sitemap_links(crawl_request.url)
                for link in sitemap_links:
                    await enqueue(link, 0)
                    if crawl_request.limit and (len(results) + frontier.size()) >= crawl_request.limit:
                        break
            except Exception:
                pass

        # Determine desired concurrency up-front for budget-aware enqueue decisions
        concurrency = max(1, min(crawl_request.max_concurrency or 5, settings.max_concurrent_requests))

        # --- Worker Definition ---
        async def worker(worker_id: int):
            while True:
                url: str | None = None
                rp: RobotFileParser | None = None
                try:
                    url, depth = await frontier.dequeue()
                    nonlocal total_bytes

                    if url in crawled_urls or (crawl_request.limit and len(results) >= crawl_request.limit):
                        continue

                    if depth > crawl_request.max_depth:
                        logger.info(f"Worker {worker_id}: Skipping {url}, max depth exceeded.")
                        continue

                    # robots.txt check
                    try:
                        rp = await get_robots_parser_for_url(url)
                        ua = (options.headers or {}).get("User-Agent") or settings.user_agent or "*"
                        if not rp.can_fetch(ua, url):
                            logger.info(f"Worker {worker_id}: Disallowed by robots.txt {url}")
                            await _record_crawl_error(
                                url,
                                "Blocked by robots.txt policy",
                                blocked_reason="robots_disallow",
                            )
                            continue
                    except Exception:
                        # If robots cannot be fetched/parsed, proceed
                        rp = None

                    # Per-host politeness (crawl-delay and concurrency=1 per host)
                    host = urlparse(url).netloc
                    if host not in host_semaphores:
                        host_semaphores[host] = asyncio.Semaphore(1)
                        host_locks[host] = asyncio.Lock()

                    # Budget/cancel checks before doing expensive work
                    if job_id in _cancelled_jobs:
                        break
                    if crawl_request.max_time and (time.time() - start_job_time) > crawl_request.max_time:
                        break
                    if crawl_request.max_bytes and total_bytes >= crawl_request.max_bytes:
                        break

                    crawled_urls.add(url)
                    logger.info(f"Worker {worker_id}: Crawling {url} at depth {depth}")

                    try:
                        # enforce crawl-delay if declared
                        try:
                            ua = (options.headers or {}).get("User-Agent") or settings.user_agent or "*"
                            delay_val = 0.0
                            try:
                                delay_from_robots = rp.crawl_delay(ua) if rp else None
                            except Exception:
                                delay_from_robots = None
                            if delay_from_robots:
                                delay_val = float(delay_from_robots)
                            # use host lock to coordinate timing
                            async with host_semaphores[host]:
                                async with host_locks[host]:
                                    last = host_last_fetch.get(host, 0.0)
                                    now = time.time()
                                    wait_needed = max(0.0, (last + delay_val) - now)
                                    if wait_needed > 0:
                                        await asyncio.sleep(wait_needed)
                                    host_last_fetch[host] = time.time()
                                # perform fetch under host semaphore
                                doc = await scraper.scrape_url(url, options)
                        except Exception:
                            # fall back without politeness if something goes wrong
                            doc = await scraper.scrape_url(url, options)
                        # Update counters
                        if doc.markdown:
                            total_bytes += len(doc.markdown.encode("utf-8"))
                        elif doc.html:
                            total_bytes += len(doc.html.encode("utf-8"))
                        results.append(doc)

                        # Budget guard: if we've hit the byte budget, stop enqueuing
                        if crawl_request.max_bytes and total_bytes >= crawl_request.max_bytes:
                            logger.info(f"Worker {worker_id}: max_bytes threshold met; halting further enqueue")
                            break

                        if depth < crawl_request.max_depth and doc.links:
                            # Under max_bytes budgeting, throttle enqueues to at most (concurrency-1)
                            # to reduce overshoot while still allowing children to start.
                            if crawl_request.max_bytes:
                                inflight_future = max(0, frontier.inflight_size() - 1)  # discount current URL
                                max_new = max(0, (concurrency - 1) - frontier.size() - inflight_future)
                            else:
                                max_new = None  # no throttle
                            new_added = 0
                            for link in doc.links:
                                parsed_link = urlparse(link)
                                if parsed_link.scheme.lower() not in ['http', 'https']:
                                    continue
                                link = canon(link)
                                if not domain_allowed(link) or not path_allowed(link):
                                    continue
                                if link in crawled_urls or frontier.seen(link):
                                    continue
                                if crawl_request.limit and (len(results) + frontier.size()) >= crawl_request.limit:
                                    break
                                if max_new is not None and new_added >= max_new:
                                    break
                                if job_id in _cancelled_jobs:
                                    break
                                await frontier.enqueue(link, depth + 1)
                                new_added += 1
                    except Exception as e:
                        logger.error(f"Worker {worker_id}: Error scraping {url}: {e}")
                        # Heuristic blocked detection
                        meta = DocumentMetadata(source_url=url, status_code=None)
                        msg = str(e).lower()
                        blocked_reason = None
                        if any(k in msg for k in ["captcha", "recaptcha"]):
                            blocked_reason = "captcha"
                            meta.blocked = True
                            meta.blocked_reason = blocked_reason
                        elif any(k in msg for k in ["forbidden", "403", "blocked"]):
                            blocked_reason = "ip_block"
                            meta.blocked = True
                            meta.blocked_reason = blocked_reason
                        elif any(k in msg for k in ["too many requests", "429", "rate limit"]):
                            blocked_reason = "rate_limited"
                            meta.blocked = True
                            meta.blocked_reason = blocked_reason
                        await _record_crawl_error(url, str(e), blocked_reason=blocked_reason)
                        results.append(FirecrawlDocument(url=url, metadata=meta))
                    finally:
                        # Optional polite delay between tasks per worker
                        if crawl_request.delay and crawl_request.delay > 0:
                            try:
                                await asyncio.sleep(crawl_request.delay)
                            except Exception:
                                pass

                except asyncio.CancelledError:
                    break
                finally:
                    if url is not None:
                        try:
                            await frontier.mark_done(url)
                        except Exception:
                            pass

        # --- Start and Manage Workers ---
        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]

        # Wait for the queue to be fully processed with job-level budgets
        final_status = "completed"
        while True:
            await asyncio.sleep(0.1)
            if job_id in _cancelled_jobs:
                final_status = "cancelled"
                break
            if crawl_request.max_time and (time.time() - start_job_time) > crawl_request.max_time:
                job_data["warning"] = "max_time reached"
                break
            if crawl_request.max_bytes and total_bytes >= crawl_request.max_bytes:
                job_data["warning"] = "max_bytes reached"
                break
            if frontier.size() == 0 and frontier.inflight_size() == 0:
                break

        # Cancel all worker tasks
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        # --- Finalize Job ---
        zero_data_retention = bool(job_data.get("zero_data_retention"))
        persisted_docs = [] if zero_data_retention else [doc.model_dump() for doc in results]
        credits_used = _estimate_documents_credits(results)
        job_data.update({
            "status": final_status,
            "completed": len(results),
            "total": len(crawled_urls), # Total unique URLs encountered
            "credits_used": credits_used,
            "data": persisted_docs,
            "errors": crawl_errors,
            "robots_blocked": sorted(robots_blocked),
            "zero_data_retention_applied": zero_data_retention,
        })
        job_data["webhook_delivery"] = await _deliver_job_webhook(
            job_type="crawl",
            job_id=job_id,
            job_data=job_data,
        )

        if redis_available:
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_id] = job_data

        duration = time.time() - start_time
        logger.info(f"Crawl job {job_id} completed in {duration:.2f} seconds. Crawled {len(results)} pages.")

    except Exception as e:
        logger.error(f"Crawl job {job_id} failed: {e}", exc_info=True)
        job_data.update({"status": "failed", "error": str(e)})
        job_data["webhook_delivery"] = await _deliver_job_webhook(
            job_type="crawl",
            job_id=job_id,
            job_data=job_data,
        )

        if redis_available:
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_id] = job_data

async def batch_scrape_background_task(job_id: str, urls: List[str], batch_request: BatchScrapeRequest):
    """Background task for batch scraping"""
    job_data: Dict[str, Any] = {"id": job_id}
    try:
        # Convert request to scrape options
        options = ScrapeOptions(
            formats=batch_request.formats,
            headers=batch_request.headers,
            include_tags=batch_request.include_tags,
            exclude_tags=batch_request.exclude_tags,
            only_main_content=batch_request.only_main_content,
            wait_for=batch_request.wait_for,
            timeout=batch_request.timeout,
            location=batch_request.location,
            mobile=batch_request.mobile,
            skip_tls_verification=batch_request.skip_tls_verification,
            remove_base64_images=batch_request.remove_base64_images
        )

        results = []
        batch_errors: list[Dict[str, Any]] = []
        robots_blocked: set[str] = set()

        # Process URLs with concurrency control
        semaphore = asyncio.Semaphore(batch_request.max_concurrency or 5)

        async def scrape_with_semaphore(url: str):
            async with semaphore:
                try:
                    return {"url": url, "document": await scraper.scrape_url(url, options), "error": None}
                except Exception as e:
                    return {"url": url, "document": None, "error": str(e)}

        # Execute all scraping tasks
        tasks = [scrape_with_semaphore(url) for url in urls]
        gathered = await asyncio.gather(*tasks)
        for item in gathered:
            url = item.get("url")
            document = item.get("document")
            error = item.get("error")
            if document is not None:
                try:
                    meta = getattr(document, "metadata", None)
                    blocked = bool(getattr(meta, "blocked", False)) if meta is not None else False
                    blocked_reason = getattr(meta, "blocked_reason", None) if meta is not None else None
                    if blocked:
                        batch_errors.append(
                            _make_job_error_item(
                                str(url),
                                f"Blocked during scrape ({blocked_reason or 'unknown'})",
                            )
                        )
                    if blocked_reason == "robots_disallow":
                        robots_blocked.add(str(url))
                except Exception:
                    pass
                results.append(document)
                continue

            batch_errors.append(_make_job_error_item(str(url), str(error or "Unknown scrape error")))

        # Update job with results
        job_key = f"batch:{job_id}"
        if redis_available:
            job_data_str = redis_client.get(job_key)
            if job_data_str:
                job_data = deserialize_job_data(job_data_str.decode('utf-8'))
            else:
                return  # Job not found
        else:
            job_data = job_storage.get(job_key)
            if not job_data:
                return  # Job not found

        # Avoid overwriting a cancelled job with "completed".
        if job_id in _cancelled_batch_jobs or job_data.get("status") == "cancelled":
            return

        zero_data_retention = bool(job_data.get("zero_data_retention"))
        persisted_docs = [] if zero_data_retention else [doc.model_dump() for doc in results]
        credits_used = _estimate_documents_credits(results)
        job_data.update({
            "status": "completed",
            "completed": len(results),
            "credits_used": credits_used,
            "data": persisted_docs,
            "errors": batch_errors,
            "robots_blocked": sorted(robots_blocked),
            "zero_data_retention_applied": zero_data_retention,
        })
        job_data["webhook_delivery"] = await _deliver_job_webhook(
            job_type="batch_scrape",
            job_id=job_id,
            job_data=job_data,
        )

        # Store updated job data
        if redis_available:
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_key] = job_data

    except Exception as e:
        # Mark job as failed
        job_key = f"batch:{job_id}"
        if redis_available:
            job_data_str = redis_client.get(job_key)
            if job_data_str:
                job_data = deserialize_job_data(job_data_str.decode('utf-8'))
            else:
                job_data = {"id": job_id, "status": "failed", "error": "Job data not found"}
        else:
            job_data = job_storage.get(job_key, {"id": job_id, "status": "failed", "error": "Job data not found"})

        job_data.update({
            "status": "failed",
            "error": str(e)
        })
        job_data["webhook_delivery"] = await _deliver_job_webhook(
            job_type="batch_scrape",
            job_id=job_id,
            job_data=job_data,
        )

        if redis_available:
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_key] = job_data


async def extract_background_task(job_id: str, urls: List[str], payload: Dict[str, Any]):
    job_key = f"extract:{job_id}"
    job_data: Dict[str, Any] = {"id": job_id}
    try:
        scrape_opts_payload = payload.get("scrapeOptions") or {}
        if not isinstance(scrape_opts_payload, dict):
            scrape_opts_payload = {}
        options = _scrape_options_from_v2(scrape_opts_payload)

        show_sources = _coerce_bool(payload.get("showSources"), False)
        schema_payload = payload.get("schema")
        prompt_payload = payload.get("prompt")
        results: list[Dict[str, Any]] = []
        errors: list[Dict[str, Any]] = []
        credits_used = 0

        for u in urls:
            try:
                doc = await scraper.scrape_url(str(u), options)
                source_doc = _v2_document_from_v1(doc)
                extracted_payload: Dict[str, Any] = {
                    "url": str(u),
                    "content": source_doc.get("markdown") or source_doc.get("html"),
                }
                # Preserve user intent for later upgraded extraction implementation.
                if isinstance(schema_payload, dict) and schema_payload:
                    extracted_payload["schema"] = schema_payload
                if isinstance(prompt_payload, str) and prompt_payload.strip():
                    extracted_payload["prompt"] = prompt_payload
                row: Dict[str, Any] = {"url": str(u), "data": _remove_empty_top_level(extracted_payload)}
                if show_sources:
                    row["source"] = source_doc
                results.append(row)
                credits_used += _estimate_document_credits(source_doc)
            except Exception as e:
                errors.append({"url": str(u), "error": str(e)})

            if redis_available:
                job_data_str = redis_client.get(job_key)
                job_data = deserialize_job_data(job_data_str.decode("utf-8")) if job_data_str else None
            else:
                job_data = job_storage.get(job_key)
            if not job_data:
                continue

            zero_data_retention = bool(job_data.get("zero_data_retention"))
            job_data.update(
                {
                    "status": "processing",
                    "completed": len(results) + len(errors),
                    "credits_used": credits_used,
                    "data": {"documents": ([] if zero_data_retention else results), "errors": errors},
                    "zero_data_retention_applied": zero_data_retention,
                }
            )
            if redis_available:
                redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
            else:
                job_storage[job_key] = job_data

        if redis_available:
            job_data_str = redis_client.get(job_key)
            job_data = deserialize_job_data(job_data_str.decode("utf-8")) if job_data_str else None
        else:
            job_data = job_storage.get(job_key)
        if not job_data:
            return

        final_status = "completed" if results else "failed"
        zero_data_retention = bool(job_data.get("zero_data_retention"))
        job_data.update(
            {
                "status": final_status,
                "completed": len(results) + len(errors),
                "credits_used": credits_used,
                "data": {"documents": ([] if zero_data_retention else results), "errors": errors},
                "zero_data_retention_applied": zero_data_retention,
            }
        )
        if final_status == "failed" and not job_data.get("error"):
            job_data["error"] = "Extraction failed for all URLs"
        job_data["webhook_delivery"] = await _deliver_job_webhook(
            job_type="extract",
            job_id=job_id,
            job_data=job_data,
        )

        if redis_available:
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_key] = job_data
    except Exception as e:
        if redis_available:
            job_data_str = redis_client.get(job_key)
            job_data = deserialize_job_data(job_data_str.decode("utf-8")) if job_data_str else None
        else:
            job_data = job_storage.get(job_key)
        if not job_data:
            job_data = {"id": job_id}
        job_data.update({"status": "failed", "error": str(e)})
        job_data["webhook_delivery"] = await _deliver_job_webhook(
            job_type="extract",
            job_id=job_id,
            job_data=job_data,
        )
        if redis_available:
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_key] = job_data

# Startup/shutdown handled by lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "candlecrawl._server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
