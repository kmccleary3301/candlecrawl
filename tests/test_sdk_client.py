from __future__ import annotations

import json

import httpx
import pytest

from candlecrawl.client import AsyncCandleCrawlClient, CandleCrawlClient
from candlecrawl.errors import CandleCrawlAuthError, CandleCrawlRateLimitError
from candlecrawl.schemas import ExtractRequest, ScrapeRequest, SearchRequest
from candlecrawl.trace import TraceContext


@pytest.mark.asyncio
async def test_async_sdk_shapes_core_v2_requests_and_headers() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        seen.append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": body,
                "trace": request.headers.get("X-Trace-Id"),
                "auth": request.headers.get("Authorization"),
            }
        )
        if request.url.path == "/v2/scrape":
            return httpx.Response(200, json={"success": True, "data": {"markdown": "ok"}, "creditsUsed": 1})
        if request.url.path == "/v2/search":
            return httpx.Response(200, json={"success": True, "data": {"web": []}, "id": "s1", "creditsUsed": 1})
        if request.url.path == "/v2/extract":
            return httpx.Response(200, json={"success": True, "id": "job-1"})
        if request.url.path == "/v2/extract/job-1":
            return httpx.Response(200, json={"status": "completed", "data": []})
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = AsyncCandleCrawlClient(
            base_url="https://candlecrawl.test",
            api_key="secret",
            trace=TraceContext(trace_id="trace-1"),
            client=http_client,
        )

        scrape = await client.scrape(ScrapeRequest(url="https://example.com", onlyMainContent=True))
        search = await client.search(SearchRequest(query="q", sources=[{"type": "web"}]))
        extract = await client.extract(ExtractRequest(urls=["https://example.com"], schema_={"type": "object"}))
        status = await client.get_extract(extract.id)

    assert scrape.data == {"markdown": "ok"}
    assert search.credits_used == 1
    assert status.status == "completed"
    assert seen[0]["body"] == {"url": "https://example.com", "onlyMainContent": True}
    assert seen[1]["body"] == {"query": "q", "sources": [{"type": "web"}]}
    assert seen[2]["body"] == {"urls": ["https://example.com"], "schema": {"type": "object"}}
    assert {item["trace"] for item in seen} == {"trace-1"}
    assert {item["auth"] for item in seen} == {"Bearer secret"}


def test_sync_sdk_maps_v2_job_and_error_responses() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/crawl":
            return httpx.Response(200, json={"success": True, "id": "crawl-1"})
        if request.url.path == "/v2/crawl/crawl-1":
            return httpx.Response(200, json={"status": "scraping", "total": 3, "completed": 1})
        if request.url.path == "/v2/crawl/crawl-1/errors":
            return httpx.Response(200, json={"errors": [], "robotsBlocked": ["https://blocked.example"]})
        if request.url.path == "/v2/search":
            return httpx.Response(
                429,
                json={"success": False, "error": "rate limited", "code": "RATE_LIMITED"},
                headers={"X-Trace-Id": "trace-rate"},
            )
        if request.url.path == "/v2/scrape":
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(404, json={"error": "not found"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = CandleCrawlClient(base_url="https://candlecrawl.test", client=http_client)
        crawl = client.crawl({"url": "https://example.com"})
        status = client.get_crawl(crawl.id)
        errors = client.crawl_errors(crawl.id)

        with pytest.raises(CandleCrawlRateLimitError) as rate_error:
            client.search({"query": "q"})
        with pytest.raises(CandleCrawlAuthError):
            client.scrape({"url": "https://example.com"})

    assert crawl.id == "crawl-1"
    assert status.completed == 1
    assert errors.robots_blocked == ["https://blocked.example"]
    assert rate_error.value.status_code == 429
    assert rate_error.value.retryable is True
    assert rate_error.value.trace_id == "trace-rate"
