from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from contextlib import asynccontextmanager
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
import json
import time
from loguru import logger

from app.config import settings
from app.models import (
    ScrapeRequest, ScrapeResponse, HealthResponse,
    CrawlRequest, CrawlResponse, CrawlStatusResponse,
    BatchScrapeRequest, BatchScrapeResponse,
    MapRequest, MapResponse,
    FirecrawlDocument, DocumentMetadata, ScrapeOptions,
    HermesSearchRequest, HermesSearchResponse,
    HermesComposeRequest, HermesComposeResponse,
    HermesExternalScrapeRequest, HermesExternalScrapeResponse,
    HermesEnrichRequest, HermesEnrichResponse,
    utc_now,
)
from app.scraper import scraper
from app.frontier import MemoryFrontier
from app.providers.serper import SerperClient, SerperImageRequest, SerperNewsRequest, SerperSearchRequest
from app.providers.openrouter import OpenRouterClient, OpenRouterChatRequest, ORMessage
from app.providers.scrapedo import ScrapeDoClient, ScrapeDoRequest
from app.cost_endpoints import router as cost_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Firecrawl API...")
    print(f"Redis available: {redis_available}")
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

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Include cost tracking router
app.include_router(cost_router)

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


def _scrape_options_from_v2(payload: Dict[str, Any]) -> ScrapeOptions:
    formats_in = payload.get("formats")
    formats: list[str] = []
    if isinstance(formats_in, list):
        for item in formats_in:
            if isinstance(item, str):
                if item in {"markdown", "html", "rawHtml", "links", "screenshot"}:
                    formats.append(item)
                elif item == "screenshot@fullPage":
                    formats.append("screenshot")
            elif isinstance(item, dict):
                if item.get("type") == "screenshot":
                    formats.append("screenshot")
                # json/extract/branding/changeTracking/summary are ignored (not supported by this server)
    if not formats:
        formats = ["markdown"]

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
        actions=payload.get("actions") if isinstance(payload.get("actions"), list) else None,
    )


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
    if meta_out:
        out["metadata"] = meta_out
    else:
        # Preserve any existing v2-shaped metadata if present.
        if isinstance(meta_in, dict) and meta_in:
            out["metadata"] = meta_in

    return out


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
            "actions": payload.get("actions"),
            "parsers": payload.get("parsers"),
            "proxy": payload.get("proxy"),
            "blockAds": payload.get("blockAds"),
            "fastMode": payload.get("fastMode"),
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
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        return True
    status = meta.get("statusCode")
    if isinstance(status, int) and status >= 400:
        return False
    if meta.get("blocked"):
        return False
    return True


async def _v2_scrape_data(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    max_age_raw = payload.get("maxAge")
    if max_age_raw is None:
        max_age_ms = int(getattr(settings, "cache_default_max_age_ms", 172800000))
    else:
        try:
            max_age_ms = int(max_age_raw)
        except Exception:
            max_age_ms = int(getattr(settings, "cache_default_max_age_ms", 172800000))

    store_in_cache = _coerce_bool(payload.get("storeInCache"), True)

    cache_key = _scrape_cache_key(url, payload)
    cached = _get_cached_scrape(cache_key, max_age_ms)
    if cached is not None:
        return cached

    options = _scrape_options_from_v2(payload)
    document = await scraper.scrape_url(url, options)
    data = _v2_document_from_v1(document)
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
    return HealthResponse()

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

@app.post("/v2/scrape")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_scrape(request: Request, payload: Dict[str, Any], remote_addr: str = Depends(get_remote_address)):
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return {"success": False, "error": "Missing url"}
    parsed_url = urlparse(url)
    if not (parsed_url.scheme and parsed_url.netloc):
        return {"success": False, "error": f"Invalid URL: {url}. Please include http:// or https://."}
    try:
        data = await _v2_scrape_data(url, payload)
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}

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


@app.post("/v2/crawl")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_crawl(request: Request, payload: Dict[str, Any], background_tasks: BackgroundTasks, remote_addr: str = Depends(get_remote_address)):
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return {"success": False, "error": "Missing url"}
    try:
        idem_key = _get_idempotency_key(request)
        payload_hash = _payload_hash_for_idempotency(payload) if idem_key else ""
        if idem_key:
            existing = _get_idempotent_response("v2:crawl", idem_key, payload_hash)
            if existing is not None:
                return existing

        sitemap = payload.get("sitemap")
        ignore_sitemap = sitemap == "skip"

        crawl_request = CrawlRequest(
            url=url,
            include_paths=payload.get("includePaths"),
            exclude_paths=payload.get("excludePaths"),
            max_depth=payload.get("maxDiscoveryDepth") or 2,
            limit=payload.get("limit") or 50,
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
            "request": crawl_request.model_dump(),
            "v2_sitemap_mode": sitemap,
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
        return {"success": False, "error": str(e)}


@app.get("/v2/crawl/{job_id}")
async def v2_get_crawl_status(job_id: str):
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
        else:
            job_data = None
    else:
        job_data = job_storage.get(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Crawl job not found")

    expires_at = job_data.get("expires_at")
    expires_at_str = expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
    docs_in = job_data.get("data") or []
    docs_out = [_v2_document_from_v1(d) for d in docs_in]

    return {
        "success": True,
        "status": job_data.get("status"),
        "completed": job_data.get("completed", 0),
        "total": job_data.get("total", 0),
        "creditsUsed": job_data.get("credits_used"),
        "expiresAt": expires_at_str,
        "next": job_data.get("next"),
        "data": docs_out,
    }


@app.delete("/v2/crawl/{job_id}")
async def v2_cancel_crawl(job_id: str):
    _cancelled_jobs.add(job_id)
    if redis_available:
        job_data_str = redis_client.get(f"crawl:{job_id}")
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
            job_data.update({"status": "cancelled"})
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
    else:
        job_data = job_storage.get(job_id)
        if job_data:
            job_data.update({"status": "cancelled"})
            job_storage[job_id] = job_data
    return {"status": "cancelled"}


@app.get("/v2/crawl/{job_id}/errors")
async def v2_crawl_errors(job_id: str):
    # Minimal compatibility: return empty error set.
    return {"success": True, "data": {"errors": [], "robotsBlocked": []}}

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


@app.post("/v2/map")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_map(request: Request, payload: Dict[str, Any], remote_addr: str = Depends(get_remote_address)):
    url = payload.get("url")
    if not isinstance(url, str) or not url.strip():
        return {"success": False, "error": "Missing url"}
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

        return {"success": True, "links": links}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/v2/batch/scrape")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_batch_scrape(request: Request, payload: Dict[str, Any], background_tasks: BackgroundTasks, remote_addr: str = Depends(get_remote_address)):
    urls = payload.get("urls")
    if not isinstance(urls, list) or not urls:
        return {"success": False, "error": "No URLs provided"}

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
    valid_urls: list[str] = []
    for u in batch_request.urls:
        parsed = urlparse(u)
        if parsed.scheme and parsed.netloc:
            valid_urls.append(u)
        else:
            invalid_urls.append(u)
    if not valid_urls:
        return {"success": False, "error": "No valid URLs provided", "invalidURLs": invalid_urls}

    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "status": "scraping",
        "completed": 0,
        "total": len(valid_urls),
        "credits_used": 0,
        "expires_at": utc_now() + timedelta(hours=24),
        "data": [],
        "request": payload,
        "valid_urls": valid_urls,
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


@app.get("/v2/batch/scrape/{job_id}")
async def v2_get_batch_status(job_id: str):
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
        raise HTTPException(status_code=404, detail="Batch job not found")

    expires_at = job_data.get("expires_at")
    expires_at_str = expires_at.isoformat() if isinstance(expires_at, datetime) else expires_at
    docs_in = job_data.get("data") or []
    docs_out = [_v2_document_from_v1(d) for d in docs_in]

    return {
        "success": True,
        "status": job_data.get("status"),
        "completed": job_data.get("completed", 0),
        "total": job_data.get("total", 0),
        "creditsUsed": job_data.get("credits_used"),
        "expiresAt": expires_at_str,
        "next": job_data.get("next"),
        "data": docs_out,
    }


@app.delete("/v2/batch/scrape/{job_id}")
async def v2_cancel_batch(job_id: str):
    _cancelled_batch_jobs.add(job_id)
    job_key = f"batch:{job_id}"
    if redis_available:
        job_data_str = redis_client.get(job_key)
        if job_data_str:
            job_data = deserialize_job_data(job_data_str.decode("utf-8"))
            job_data.update({"status": "cancelled"})
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
    else:
        job_data = job_storage.get(job_key)
        if job_data:
            job_data.update({"status": "cancelled"})
            job_storage[job_key] = job_data
    return {"status": "cancelled"}


@app.get("/v2/batch/scrape/{job_id}/errors")
async def v2_batch_errors(job_id: str):
    return {"success": True, "data": {"errors": [], "robotsBlocked": []}}


@app.post("/v2/search")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_search(request: Request, payload: Dict[str, Any], remote_addr: str = Depends(get_remote_address)):
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return {"success": False, "error": "Missing query"}
    try:
        limit = int(payload.get("limit") or 5)
        if limit <= 0:
            limit = 5
        # Serper hard-limits are higher, but keep this conservative by default.
        limit = min(limit, 20)
        sources_in = payload.get("sources") or [{"type": "web"}]
        sources: set[str] = set()
        if isinstance(sources_in, list):
            for s in sources_in:
                if isinstance(s, str):
                    sources.add(s)
                elif isinstance(s, dict) and isinstance(s.get("type"), str):
                    sources.add(s["type"])
        if not sources:
            sources = {"web"}

        data: Dict[str, Any] = {"web": [], "news": [], "images": []}

        gl = payload.get("country") or payload.get("gl")
        hl = payload.get("lang") or payload.get("hl")
        tbs = payload.get("tbs")

        serper = SerperClient()

        if "web" in sources:
            res, _ = await serper.search(SerperSearchRequest(q=query, gl=gl, hl=hl, num=limit, tbs=tbs))
            web_results: list[Dict[str, Any]] = []
            if res.organic:
                for r in res.organic[:limit]:
                    if not r.link:
                        continue
                    web_results.append({"url": r.link, "title": r.title, "description": r.snippet})

            scrape_opts_payload = payload.get("scrapeOptions") or {}
            formats = scrape_opts_payload.get("formats") if isinstance(scrape_opts_payload, dict) else None
            if isinstance(formats, list) and len(formats) > 0:
                semaphore = asyncio.Semaphore(min(settings.max_concurrent_requests, 5))

                async def scrape_one(u: str):
                    async with semaphore:
                        try:
                            scrape_payload = {"url": u, **(scrape_opts_payload if isinstance(scrape_opts_payload, dict) else {})}
                            return await _v2_scrape_data(u, scrape_payload)
                        except Exception:
                            return {"metadata": {"sourceURL": u, "statusCode": None}}

                data["web"] = await asyncio.gather(*[scrape_one(r["url"]) for r in web_results])
            else:
                data["web"] = web_results

        if "news" in sources:
            news_res, _ = await serper.news(SerperNewsRequest(q=query, gl=gl, hl=hl, num=limit))
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

        if "images" in sources:
            img_res, _ = await serper.images(SerperImageRequest(q=query, gl=gl, hl=hl, num=limit))
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

        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/v2/extract")
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window}seconds")
async def v2_extract(request: Request, payload: Dict[str, Any], remote_addr: str = Depends(get_remote_address)):
    urls = payload.get("urls")
    if not isinstance(urls, list) or not urls:
        return {"success": False, "error": "No URLs provided"}
    try:
        scrape_opts_payload = payload.get("scrapeOptions") or {}
        options = _scrape_options_from_v2(scrape_opts_payload)
        semaphore = asyncio.Semaphore(min(settings.max_concurrent_requests, 5))

        async def scrape_one(u: str):
            async with semaphore:
                try:
                    doc = await scraper.scrape_url(str(u), options)
                    return _v2_document_from_v1(doc)
                except Exception:
                    return {"metadata": {"sourceURL": str(u), "statusCode": None}}

        docs = await asyncio.gather(*[scrape_one(u) for u in urls])
        return {"success": True, "status": "completed", "data": {"documents": docs}}
    except Exception as e:
        return {"success": False, "status": "failed", "error": str(e)}


# --- Hermes endpoints ---

@app.post("/v1/hermes/leads/search", response_model=HermesSearchResponse)
async def hermes_search(req: HermesSearchRequest):
    try:
        serper = SerperClient()
        res, _ = await serper.search(SerperSearchRequest(q=req.query, gl=req.gl, hl=req.hl, num=req.limit))
        results = []
        if res.organic:
            for r in res.organic:
                results.append({"title": r.title, "link": r.link, "snippet": r.snippet, "date": r.date})
        return HermesSearchResponse(success=True, leads=results, total=len(results))
    except Exception as e:
        return HermesSearchResponse(success=False, error=str(e))


@app.post("/v1/hermes/compose", response_model=HermesComposeResponse)
async def hermes_compose(req: HermesComposeRequest):
    try:
        openrouter = OpenRouterClient()
        messages = []
        if req.system:
            messages.append(ORMessage(role="system", content=req.system))
        messages.append(ORMessage(role="user", content=req.prompt))
        chat, _ = await openrouter.chat_completions(OpenRouterChatRequest(
            model=req.model, messages=messages, max_tokens=req.max_tokens, temperature=req.temperature
        ))
        text = chat.choices[0].message.content if chat.choices else None
        return HermesComposeResponse(success=True, content=text)
    except Exception as e:
        return HermesComposeResponse(success=False, error=str(e))


@app.post("/v1/hermes/external-scrape", response_model=HermesExternalScrapeResponse)
async def hermes_external_scrape(req: HermesExternalScrapeRequest):
    try:
        client = ScrapeDoClient()
        out, _ = await client.fetch(ScrapeDoRequest(url=req.url, render_js=bool(req.render_js), timeout_ms=int(req.timeout_ms or 30000), geo_code=req.geo_code))
        return HermesExternalScrapeResponse(success=True, status_code=out.status_code, content_type=out.content_type, content=out.content)
    except Exception as e:
        # Surface ProviderError details if available
        status = getattr(e, "status_code", None)
        payload = getattr(e, "payload", None)
        
        # Check for specific Scrape.do plan limitations
        if status == 401 and payload:
            if "Geo Targeting is not included" in payload:
                msg = "Scrape.do API: GeoCode feature not available in current plan"
            elif "JS Render is not included" in payload:
                msg = "Scrape.do API: JavaScript rendering not available in current plan"
            else:
                if len(payload) > 400:
                    payload = payload[:400] + "..."
                msg = f"Scrape.do error: {payload}"
        else:
            if payload and isinstance(payload, str) and len(payload) > 400:
                payload = payload[:400] + "..."
            msg = str(e)
            if status:
                msg = f"{msg} (status={status})"
        return HermesExternalScrapeResponse(success=False, status_code=status, content_type=None, content=None, error=msg)


@app.post("/v1/hermes/leads/enrich", response_model=HermesEnrichResponse)
async def hermes_enrich(req: HermesEnrichRequest):
    try:
        opts_dict = req.scrapeOptions or {}
        # Map incoming options dict into ScrapeOptions fields where possible
        options = ScrapeOptions(**{k: v for k, v in opts_dict.items() if k in ScrapeOptions.model_fields})
        docs = []
        for domain in req.domains:
            url = domain if domain.startswith("http") else f"https://{domain}"
            doc = await scraper.scrape_url(url, options)
            docs.append(doc)
        return HermesEnrichResponse(success=True, domains=docs)
    except Exception as e:
        return HermesEnrichResponse(success=False, error=str(e))


@app.post("/v1/hermes/research")
async def hermes_research(body: dict):
    try:
        # Keep Hermes compatibility path optional in standalone CandleCrawl.
        from app.hermes_bcas import HermesBStar04

        question = body.get("question") or body.get("query")
        max_searches = int(body.get("max_searches", 3))
        model = body.get("model", "openai/gpt-5-nano")
        
        # Cost tracking parameters
        job_id = body.get("job_id", str(uuid.uuid4()))
        tier = body.get("tier", "TARGETED")  # Default tier
        
        # Contact extraction configuration
        contact_extraction = None
        if body.get("contact_extraction"):
            from app.models import ContactExtractionConfig
            contact_extraction = ContactExtractionConfig(**body["contact_extraction"])
        
        # Optional budgets and toggles (native to BCAS)
        method_kwargs = {
            "max_total_tokens": int(body.get("max_total_tokens", 8000)),
            "max_responses": int(body.get("max_responses", 25)),
            "tell_search_limit": bool(body.get("tell_search_limit", True)),
            "tell_context_limit": bool(body.get("tell_context_limit", True)),
            "search_limit": int(body.get("search_limit", 5)),
            "use_intermittent_reasoning": bool(body.get("use_intermittent_reasoning", False)),
            "use_preplanning": bool(body.get("use_preplanning", True)),
            "max_completion_tokens": int(body.get("max_completion_tokens", 2048)),
            "enforce_tier_length": bool(body.get("enforce_tier_length", False)),
            "debug_trace_level": str(body.get("debug_trace_level", "summary")),
            "force_direct_scrape": bool(body.get("force_direct_scrape", False)),
        }
        debug_trace = bool(body.get("debug_trace", False))
        
        # Initialize engine with cost tracking
        engine = HermesBStar04(
            model=model, 
            job_id=job_id,
            tier=tier,
            contact_extraction=contact_extraction,
            **method_kwargs
        )
        
        result = await engine.run_async(question, max_searches)
        if debug_trace and engine.cost_tracker:
            result["trace"] = engine.cost_tracker.get_trace()
        return {"success": True, **result}
    except Exception as e:
        err = {"success": False, "error": str(e)}
        # Surface ProviderError details when available
        status = getattr(e, "status_code", None)
        payload = getattr(e, "payload", None)
        if status or payload:
            if isinstance(payload, str) and len(payload) > 800:
                payload = payload[:800] + "..."
            err["error_details"] = {"status_code": status, "payload": payload}
        # Attach trace if any
        try:
            if 'engine' in locals() and engine and engine.cost_tracker:
                err["trace"] = engine.cost_tracker.get_trace()
        except Exception:
            pass
        return err

async def crawl_background_task(job_id: str, crawl_request: CrawlRequest):
    """Background task for crawling with depth and concurrency control."""
    start_time = time.time()
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
                        if any(k in msg for k in ["captcha", "recaptcha"]):
                            meta.blocked = True; meta.blocked_reason = "captcha"
                        elif any(k in msg for k in ["forbidden", "403", "blocked"]):
                            meta.blocked = True; meta.blocked_reason = "ip_block"
                        elif any(k in msg for k in ["too many requests", "429", "rate limit"]):
                            meta.blocked = True; meta.blocked_reason = "rate_limited"
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
        job_data.update({
            "status": final_status,
            "completed": len(results),
            "total": len(crawled_urls), # Total unique URLs encountered
            "data": [doc.model_dump() for doc in results]
        })
        
        if redis_available:
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_id] = job_data
        
        duration = time.time() - start_time
        logger.info(f"Crawl job {job_id} completed in {duration:.2f} seconds. Crawled {len(results)} pages.")
            
    except Exception as e:
        logger.error(f"Crawl job {job_id} failed: {e}", exc_info=True)
        job_data.update({"status": "failed", "error": str(e)})
        
        if redis_available:
            redis_client.set(f"crawl:{job_id}", serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_id] = job_data

async def batch_scrape_background_task(job_id: str, urls: List[str], batch_request: BatchScrapeRequest):
    """Background task for batch scraping"""
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
        completed = 0
        
        # Process URLs with concurrency control
        semaphore = asyncio.Semaphore(batch_request.max_concurrency or 5)
        
        async def scrape_with_semaphore(url: str):
            async with semaphore:
                try:
                    return await scraper.scrape_url(url, options)
                except Exception as e:
                    return FirecrawlDocument(
                        url=url,
                        metadata=DocumentMetadata(source_url=url, status_code=None)
                    )
        
        # Execute all scraping tasks
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
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
        
        job_data.update({
            "status": "completed",
            "completed": len(results),
            "data": [doc.model_dump() for doc in results]
        })
        
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
        
        if redis_available:
            redis_client.set(job_key, serialize_job_data(job_data), ex=86400)
        else:
            job_storage[job_key] = job_data

# Startup/shutdown handled by lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    ) 
