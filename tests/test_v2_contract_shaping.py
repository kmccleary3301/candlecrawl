from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import asyncio

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.artifacts import ArtifactRef
from app.models import BatchScrapeRequest, CrawlRequest, DocumentMetadata, FirecrawlDocument

client = TestClient(app)


def test_v2_contract_headers_present_on_success_response() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="ok",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post("/v2/scrape", json={"url": "https://example.com"})

    assert response.status_code == 200
    assert response.headers.get("X-Contract-Version")
    assert response.headers.get("X-CandleCrawl-Profile") in {"strict", "compat"}
    assert response.headers.get("X-Request-Id")
    assert response.headers.get("X-Audit-Id")
    assert response.headers.get("X-Trace-Id")
    assert response.json().get("artifactMode") == "inline"


def test_v2_scrape_missing_url_uses_error_envelope() -> None:
    response = client.post("/v2/scrape", json={})
    payload = response.json()

    assert response.status_code == 400
    assert response.headers.get("X-Request-Id")
    assert response.headers.get("X-Audit-Id")
    assert response.headers.get("X-Trace-Id")
    assert payload["success"] is False
    assert payload["code"] == "MISSING_URL"
    assert "error" in payload


def test_v2_contract_headers_not_added_to_v1_response() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="ok",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post("/v1/scrape", json={"url": "https://example.com"})

    assert response.status_code == 200
    assert "X-Contract-Version" not in response.headers


def test_v2_trace_headers_are_echoed_when_provided() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="ok",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            headers={
                "X-Request-Id": "req-abc",
                "X-Audit-Id": "aud-abc",
                "X-Trace-Id": "trace-abc",
                "X-Idempotency-Key": "idem-abc",
                "X-Tenant-ID": "tenant-abc",
            },
            json={"url": "https://example.com"},
        )

    assert response.status_code == 200
    assert response.headers.get("X-Request-Id") == "req-abc"
    assert response.headers.get("X-Audit-Id") == "aud-abc"
    assert response.headers.get("X-Trace-Id") == "trace-abc"
    assert response.headers.get("X-Idempotency-Key") == "idem-abc"
    assert response.headers.get("X-Tenant-Id") == "tenant-abc"


def test_v2_map_returns_link_objects_in_strict_shape() -> None:
    async def _fake_sitemap(_url: str):
        return ["https://example.com/sitemap-page"]

    async def _fake_scrape(_url: str, _options):
        return FirecrawlDocument(
            url="https://example.com",
            links=["https://example.com/from-page"],
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )

    with patch("app.main.get_sitemap_links", new=_fake_sitemap), patch(
        "app.main.scraper.scrape_url", new_callable=AsyncMock
    ) as mock_scrape:
        mock_scrape.side_effect = _fake_scrape
        response = client.post("/v2/map", json={"url": "https://example.com"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert isinstance(payload["links"], list)
    assert payload["links"], "Expected at least one link object"
    assert all(isinstance(item, dict) for item in payload["links"])
    assert all("url" in item for item in payload["links"])
    assert all("title" in item and "description" in item for item in payload["links"])
    assert all(not isinstance(item, str) for item in payload["links"])


def test_v2_scrape_supports_summary_and_json_formats() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="This is a sample markdown body used for summary and json format derivation.",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": [{"type": "summary"}, {"type": "json"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert isinstance(payload["data"].get("summary"), str)
    assert isinstance(payload["data"].get("json"), dict)
    assert "content" in payload["data"]["json"]


def test_v2_scrape_derives_branding_and_change_tracking_formats() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="content",
            html="<html><body><div style='color:#112233;font-family:Inter;'>x</div></body></html>",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": [{"type": "branding"}, {"type": "changeTracking"}, {"type": "markdown"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "branding" in payload["data"]
    assert isinstance(payload["data"]["branding"], dict)
    assert "changeTracking" in payload["data"]
    assert isinstance(payload["data"]["changeTracking"], dict)


def test_v2_scrape_warns_on_unknown_format_types() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="content",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": [{"type": "mysteryFormat"}, {"type": "markdown"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "warning" in payload
    assert "Unknown format types ignored" in payload["warning"]


def test_v2_scrape_zero_data_retention_disables_cache(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "redis_available", False)
    main_module._scrape_cache_memory.clear()

    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="cache bypass",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        payload = {
            "url": "https://example.com",
            "formats": ["markdown"],
            "maxAge": 172800000,
            "storeInCache": True,
            "zeroDataRetention": True,
        }
        first = client.post("/v2/scrape", json=payload)
        second = client.post("/v2/scrape", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert mock_scrape.call_count == 2
    assert first.json().get("zeroDataRetentionApplied") is True


def test_v2_scrape_zero_data_retention_forces_inline_artifact_mode() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="artifact retention",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": ["markdown"],
                "artifact_mode": "local",
                "zeroDataRetention": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifactMode"] == "inline"
    assert payload.get("zeroDataRetentionApplied") is True
    assert "warning" in payload
    assert "forced to inline" in payload["warning"]


def test_v2_search_returns_id_warning_and_credits_used() -> None:
    class FakeSerperClient:
        async def search(self, req):
            _ = req
            return (
                SimpleNamespace(
                    organic=[
                        SimpleNamespace(link="https://example.com/a", title="A", snippet="alpha"),
                        SimpleNamespace(link=None, title="B", snippet="beta"),
                    ],
                    credits_used=2,
                ),
                0.001,
            )

        async def news(self, req):
            _ = req
            return (SimpleNamespace(items=[{"title": "News", "link": "https://news.example.com/x"}]), 0.001)

        async def images(self, req):
            _ = req
            return (
                SimpleNamespace(items=[{"title": "Image", "imageUrl": "https://img.example.com/a.png", "link": "https://example.com/a"}]),
                0.001,
            )

    with patch("app.main.SerperClient", new=FakeSerperClient):
        response = client.post(
            "/v2/search",
            json={
                "query": "test query",
                "sources": [{"type": "web"}, {"type": "news"}, {"type": "images"}],
                "ignoreInvalidURLs": True,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert isinstance(payload.get("id"), str) and payload["id"]
    assert payload.get("warning") is None
    assert payload.get("creditsUsed") == 4
    assert len(payload["data"]["web"]) == 1
    assert len(payload["data"]["news"]) == 1
    assert len(payload["data"]["images"]) == 1


def test_v2_search_request_parsing_uses_location_and_warns_on_unsupported_sources() -> None:
    captured = {}

    class FakeSerperClient:
        async def search(self, req):
            captured["gl"] = req.gl
            captured["hl"] = req.hl
            captured["num"] = req.num
            return (SimpleNamespace(organic=[SimpleNamespace(link="https://example.com/a", title="A", snippet="alpha")], credits_used=1), 0.001)

        async def news(self, req):
            _ = req
            return (SimpleNamespace(items=[]), 0.001)

        async def images(self, req):
            _ = req
            return (SimpleNamespace(items=[]), 0.001)

    with patch("app.main.SerperClient", new=FakeSerperClient):
        response = client.post(
            "/v2/search",
            json={
                "query": "test query",
                "sources": [{"type": "web"}, {"type": "videos"}],
                "location": {"country": "us", "languages": ["en"]},
                "limit": 0,
                "timeout": "not-a-timeout",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload.get("creditsUsed") == 1
    assert "Unsupported source 'videos' ignored" in (payload.get("warning") or "")
    assert "Invalid timeout ignored" in (payload.get("warning") or "")
    assert captured["gl"] == "us"
    assert captured["hl"] == "en"
    assert captured["num"] == 5


def test_v2_extract_is_async_job_create_plus_status() -> None:
    with patch("app.main.extract_background_task", new_callable=AsyncMock):
        create_resp = client.post(
            "/v2/extract",
            json={"urls": ["https://example.com/a", "not-a-url"], "prompt": "extract summary"},
        )
    assert create_resp.status_code == 200
    create_payload = create_resp.json()
    assert create_payload["success"] is True
    assert isinstance(create_payload.get("id"), str) and create_payload["id"]
    assert create_payload.get("invalidURLs") == ["not-a-url"]

    status_resp = client.get(f"/v2/extract/{create_payload['id']}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert "success" not in status_payload
    assert status_payload["id"] == create_payload["id"]
    assert status_payload["status"] in {"processing", "completed", "failed"}
    assert status_payload["total"] == 1
    assert status_payload["completed"] in {0, 1}
    assert "data" in status_payload


def test_v2_extract_rejects_all_invalid_urls() -> None:
    response = client.post("/v2/extract", json={"urls": ["invalid-1", "invalid-2"]})
    assert response.status_code == 400
    payload = response.json()
    assert payload["success"] is False
    assert payload["code"] == "NO_VALID_URLS"
    assert payload["error"] == "No valid URLs provided"
    assert payload["invalidURLs"] == ["invalid-1", "invalid-2"]


def test_v2_missing_required_fields_use_400_status_and_error_codes() -> None:
    cases = [
        ("/v2/crawl", {}, "MISSING_URL"),
        ("/v2/map", {}, "MISSING_URL"),
        ("/v2/search", {}, "MISSING_QUERY"),
        ("/v2/batch/scrape", {}, "MISSING_URLS"),
        ("/v2/extract", {}, "MISSING_URLS"),
    ]
    for route, payload, expected_code in cases:
        response = client.post(route, json=payload)
        assert response.status_code == 400
        body = response.json()
        assert body["success"] is False
        assert body["code"] == expected_code
        assert "error" in body


def test_v2_status_endpoints_use_not_found_error_envelope() -> None:
    for route in ("/v2/crawl/not-real", "/v2/batch/scrape/not-real", "/v2/extract/not-real"):
        response = client.get(route)
        assert response.status_code == 404
        payload = response.json()
        assert payload["success"] is False
        assert payload["code"] == "JOB_NOT_FOUND"


def test_v2_crawl_errors_returns_persisted_errors_and_robots_blocked(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "redis_available", False)
    job_id = "crawl-errors-job"
    main_module.job_storage[job_id] = {
        "id": job_id,
        "status": "completed",
        "completed": 1,
        "total": 1,
        "credits_used": 1,
        "data": [],
        "tenant_id": main_module.settings.v2_default_tenant_id,
        "errors": [
            {
                "id": "err-1",
                "timestamp": "2026-02-26T00:00:00+00:00",
                "url": "https://example.com/blocked",
                "error": "Blocked by robots.txt policy",
            }
        ],
        "robots_blocked": ["https://example.com/blocked"],
    }
    try:
        response = client.get(f"/v2/crawl/{job_id}/errors")
        assert response.status_code == 200
        payload = response.json()
        assert "success" not in payload
        assert payload["errors"][0]["url"] == "https://example.com/blocked"
        assert payload["robotsBlocked"] == ["https://example.com/blocked"]
    finally:
        main_module.job_storage.pop(job_id, None)


def test_v2_batch_errors_returns_persisted_errors_and_robots_blocked(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "redis_available", False)
    job_id = "batch-errors-job"
    key = f"batch:{job_id}"
    main_module.job_storage[key] = {
        "id": job_id,
        "status": "completed",
        "completed": 1,
        "total": 1,
        "credits_used": 1,
        "data": [],
        "tenant_id": main_module.settings.v2_default_tenant_id,
        "errors": [
            {
                "id": "err-1",
                "timestamp": "2026-02-26T00:00:00+00:00",
                "url": "https://example.com/blocked",
                "error": "Blocked during scrape (robots_disallow)",
            }
        ],
        "robots_blocked": ["https://example.com/blocked"],
    }
    try:
        response = client.get(f"/v2/batch/scrape/{job_id}/errors")
        assert response.status_code == 200
        payload = response.json()
        assert "success" not in payload
        assert payload["errors"][0]["url"] == "https://example.com/blocked"
        assert payload["robotsBlocked"] == ["https://example.com/blocked"]
    finally:
        main_module.job_storage.pop(key, None)


def test_v2_crawl_status_pagination_cursor_and_next(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "redis_available", False)
    job_id = "crawl-pagination-job"
    main_module.job_storage[job_id] = {
        "id": job_id,
        "status": "completed",
        "completed": 3,
        "total": 3,
        "credits_used": 3,
        "data": [
            {"markdown": "doc-1", "metadata": {"source_url": "https://example.com/1"}},
            {"markdown": "doc-2", "metadata": {"source_url": "https://example.com/2"}},
            {"markdown": "doc-3", "metadata": {"source_url": "https://example.com/3"}},
        ],
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        first = client.get(f"/v2/crawl/{job_id}?limit=2")
        assert first.status_code == 200
        first_payload = first.json()
        assert len(first_payload["data"]) == 2
        assert first_payload["next"] == "offset:2"

        second = client.get(f"/v2/crawl/{job_id}?cursor=offset:2&limit=2")
        assert second.status_code == 200
        second_payload = second.json()
        assert len(second_payload["data"]) == 1
        assert second_payload["next"] is None
    finally:
        main_module.job_storage.pop(job_id, None)


def test_v2_batch_status_pagination_cursor_and_next(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "redis_available", False)
    job_id = "batch-pagination-job"
    key = f"batch:{job_id}"
    main_module.job_storage[key] = {
        "id": job_id,
        "status": "completed",
        "completed": 3,
        "total": 3,
        "credits_used": 3,
        "data": [
            {"markdown": "doc-1", "metadata": {"source_url": "https://example.com/1"}},
            {"markdown": "doc-2", "metadata": {"source_url": "https://example.com/2"}},
            {"markdown": "doc-3", "metadata": {"source_url": "https://example.com/3"}},
        ],
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        first = client.get(f"/v2/batch/scrape/{job_id}?limit=2")
        assert first.status_code == 200
        first_payload = first.json()
        assert len(first_payload["data"]) == 2
        assert first_payload["next"] == "offset:2"

        second = client.get(f"/v2/batch/scrape/{job_id}?cursor=2&limit=2")
        assert second.status_code == 200
        second_payload = second.json()
        assert len(second_payload["data"]) == 1
        assert second_payload["next"] is None
    finally:
        main_module.job_storage.pop(key, None)


def test_webhook_delivery_retries_and_signs_payload(monkeypatch) -> None:
    class _Resp:
        def __init__(self, status_code: int, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

    calls = {"count": 0, "headers": []}

    class _Client:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb

        async def post(self, url, content=None, headers=None):
            _ = url, content
            calls["count"] += 1
            calls["headers"].append(headers or {})
            if calls["count"] == 1:
                return _Resp(500, "upstream error")
            return _Resp(200, "ok")

    monkeypatch.setattr(main_module.settings, "webhook_signing_secret", "test-secret")
    monkeypatch.setattr(main_module.settings, "webhook_retry_base_delay_ms", 1)
    monkeypatch.setattr(main_module.settings, "webhook_max_retries", 2)
    monkeypatch.setattr(main_module.httpx, "AsyncClient", _Client)

    job_data = {
        "id": "job-1",
        "status": "completed",
        "completed": 1,
        "total": 1,
        "credits_used": 1,
        "request": {"webhook": {"url": "https://example.com/hook"}},
    }
    result = asyncio.run(
        main_module._deliver_job_webhook(
            job_type="crawl",
            job_id="job-1",
            job_data=job_data,
        )
    )
    assert result["status"] == "delivered"
    assert result["attempts"] == 2
    assert calls["count"] == 2
    assert "X-CandleCrawl-Signature" in calls["headers"][-1]
    assert calls["headers"][-1]["X-CandleCrawl-Event"] == "crawl.completed"


def test_extract_background_records_webhook_delivery(monkeypatch) -> None:
    class _Resp:
        status_code = 200
        text = "ok"

    class _Client:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb

        async def post(self, url, content=None, headers=None):
            _ = url, content, headers
            return _Resp()

    async def _fake_scrape(url, options):
        _ = options
        return FirecrawlDocument(
            url=url,
            markdown="ok",
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    monkeypatch.setattr(main_module, "redis_available", False)
    monkeypatch.setattr(main_module.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(main_module.scraper, "scrape_url", _fake_scrape)

    job_id = "extract-webhook-job"
    key = f"extract:{job_id}"
    main_module.job_storage[key] = {
        "id": job_id,
        "status": "processing",
        "completed": 0,
        "total": 1,
        "credits_used": 0,
        "expires_at": main_module.utc_now(),
        "data": {"documents": [], "errors": []},
        "request": {"webhook": {"url": "https://example.com/hook"}},
        "valid_urls": ["https://example.com/a"],
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        asyncio.run(
            main_module.extract_background_task(
                job_id,
                ["https://example.com/a"],
                {"webhook": {"url": "https://example.com/hook"}},
            )
        )
        updated = main_module.job_storage[key]
        assert updated["status"] == "completed"
        assert updated.get("webhook_delivery", {}).get("status") == "delivered"
    finally:
        main_module.job_storage.pop(key, None)


def test_extract_background_zero_data_retention_redacts_documents(monkeypatch) -> None:
    async def _fake_scrape(url, options):
        _ = options
        return FirecrawlDocument(
            url=url,
            markdown="sensitive document content",
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    monkeypatch.setattr(main_module, "redis_available", False)
    monkeypatch.setattr(main_module.scraper, "scrape_url", _fake_scrape)

    job_id = "extract-zero-retention-job"
    key = f"extract:{job_id}"
    main_module.job_storage[key] = {
        "id": job_id,
        "status": "processing",
        "completed": 0,
        "total": 1,
        "credits_used": 0,
        "expires_at": main_module.utc_now(),
        "data": {"documents": [], "errors": []},
        "request": {},
        "valid_urls": ["https://example.com/a"],
        "zero_data_retention": True,
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        asyncio.run(
            main_module.extract_background_task(
                job_id,
                ["https://example.com/a"],
                {"zeroDataRetention": True},
            )
        )
        updated = main_module.job_storage[key]
        assert updated["status"] == "completed"
        assert updated["data"]["documents"] == []
        assert updated.get("zero_data_retention_applied") is True
    finally:
        main_module.job_storage.pop(key, None)


def test_batch_background_updates_credits_used(monkeypatch) -> None:
    async def _fake_scrape(url, options):
        _ = options
        return FirecrawlDocument(
            url=url,
            markdown="ok",
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    monkeypatch.setattr(main_module, "redis_available", False)
    monkeypatch.setattr(main_module.scraper, "scrape_url", _fake_scrape)

    job_id = "batch-credits-job"
    key = f"batch:{job_id}"
    urls = ["https://example.com/a", "https://example.com/b"]
    main_module.job_storage[key] = {
        "id": job_id,
        "status": "scraping",
        "completed": 0,
        "total": 2,
        "credits_used": 0,
        "expires_at": main_module.utc_now(),
        "data": [],
        "request": {},
        "valid_urls": urls,
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        asyncio.run(
            main_module.batch_scrape_background_task(
                job_id,
                urls,
                BatchScrapeRequest(urls=urls),
            )
        )
        updated = main_module.job_storage[key]
        assert updated["status"] == "completed"
        assert updated["credits_used"] == 2
    finally:
        main_module.job_storage.pop(key, None)


def test_batch_background_pdf_page_count_drives_credits(monkeypatch) -> None:
    async def _fake_scrape(url, options):
        _ = options
        if url.endswith(".pdf"):
            return FirecrawlDocument(
                url=url,
                markdown="pdf text",
                metadata=DocumentMetadata(
                    source_url=url,
                    status_code=200,
                    content_type="application/pdf",
                    file_metadata={"page_count": 3},
                ),
            )
        return FirecrawlDocument(
            url=url,
            markdown="html text",
            metadata=DocumentMetadata(source_url=url, status_code=200, content_type="text/html"),
        )

    monkeypatch.setattr(main_module, "redis_available", False)
    monkeypatch.setattr(main_module.scraper, "scrape_url", _fake_scrape)

    job_id = "batch-credits-pdf-job"
    key = f"batch:{job_id}"
    urls = ["https://example.com/a.pdf", "https://example.com/b"]
    main_module.job_storage[key] = {
        "id": job_id,
        "status": "scraping",
        "completed": 0,
        "total": 2,
        "credits_used": 0,
        "expires_at": main_module.utc_now(),
        "data": [],
        "request": {},
        "valid_urls": urls,
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        asyncio.run(
            main_module.batch_scrape_background_task(
                job_id,
                urls,
                BatchScrapeRequest(urls=urls),
            )
        )
        updated = main_module.job_storage[key]
        assert updated["status"] == "completed"
        assert updated["credits_used"] == 4
    finally:
        main_module.job_storage.pop(key, None)


def test_crawl_background_updates_credits_used(monkeypatch) -> None:
    async def _fake_scrape(url, options):
        _ = options
        return FirecrawlDocument(
            url=url,
            markdown="ok",
            links=[],
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    class _Resp:
        def __init__(self, status_code: int = 200, text: str = "") -> None:
            self.status_code = status_code
            self.text = text

    class _Client:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb

        async def get(self, url):
            _ = url
            return _Resp(404, "")

        async def post(self, url, content=None, headers=None):
            _ = url, content, headers
            return _Resp(200, "ok")

    monkeypatch.setattr(main_module, "redis_available", False)
    monkeypatch.setattr(main_module.scraper, "scrape_url", _fake_scrape)
    monkeypatch.setattr(main_module.httpx, "AsyncClient", _Client)

    job_id = "crawl-credits-job"
    main_module.job_storage[job_id] = {
        "id": job_id,
        "url": "https://example.com",
        "status": "scraping",
        "completed": 0,
        "total": 0,
        "credits_used": 0,
        "expires_at": main_module.utc_now(),
        "data": [],
        "request": {},
        "tenant_id": main_module.settings.v2_default_tenant_id,
    }
    try:
        asyncio.run(
            main_module.crawl_background_task(
                job_id,
                CrawlRequest(url="https://example.com", ignore_sitemap=True, max_depth=0, max_concurrency=1),
            )
        )
        updated = main_module.job_storage[job_id]
        assert updated["status"] in {"completed", "cancelled"}
        assert updated["credits_used"] >= 1
    finally:
        main_module.job_storage.pop(job_id, None)


def test_estimate_document_credits_pdf_page_count() -> None:
    doc = {
        "metadata": {
            "contentType": "application/pdf",
            "fileMetadata": {"page_count": 7},
        }
    }
    assert main_module._estimate_document_credits(doc) == 7
    assert main_module._estimate_document_credits({"metadata": {"contentType": "text/html"}}) == 1


def test_v2_auth_gate_blocks_when_enabled_and_allows_valid_key() -> None:
    with patch.object(main_module.settings, "v2_auth_enabled", True), patch.object(
        main_module.settings, "v2_api_keys", "secret-key"
    ):
        unauthorized = client.post("/v2/extract", json={})
        assert unauthorized.status_code == 401
        unauthorized_payload = unauthorized.json()
        assert unauthorized_payload["success"] is False
        assert unauthorized_payload["code"] == "UNAUTHORIZED"

        authorized = client.post(
            "/v2/extract",
            json={},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert authorized.status_code == 400
        authorized_payload = authorized.json()
        assert authorized_payload["success"] is False
        assert authorized_payload["code"] == "MISSING_URLS"


def test_v2_auth_gate_records_security_deny_telemetry() -> None:
    with patch.object(main_module.settings, "v2_auth_enabled", True), patch.object(
        main_module.settings, "v2_api_keys", "secret-key"
    ), patch("app.main.record_security_deny") as mock_security_deny:
        unauthorized = client.post("/v2/extract", json={})
    assert unauthorized.status_code == 401
    assert ("v2_auth", "unauthorized") in [tuple(c.args) for c in mock_security_deny.call_args_list]


def test_v2_extract_tenant_scoping_blocks_cross_tenant_status_reads() -> None:
    with patch("app.main.extract_background_task", new_callable=AsyncMock):
        create_resp = client.post(
            "/v2/extract",
            json={"urls": ["https://example.com/a"]},
            headers={"X-Tenant-ID": "tenant-a"},
        )
    assert create_resp.status_code == 200
    job_id = create_resp.json()["id"]

    allowed = client.get(f"/v2/extract/{job_id}", headers={"X-Tenant-ID": "tenant-a"})
    assert allowed.status_code == 200

    denied = client.get(f"/v2/extract/{job_id}", headers={"X-Tenant-ID": "tenant-b"})
    assert denied.status_code == 403
    denied_payload = denied.json()
    assert denied_payload["success"] is False
    assert denied_payload["code"] == "TENANT_FORBIDDEN"


def test_v2_tenant_denies_record_security_telemetry() -> None:
    with patch("app.main.extract_background_task", new_callable=AsyncMock):
        create_resp = client.post(
            "/v2/extract",
            json={"urls": ["https://example.com/a"]},
            headers={"X-Tenant-ID": "tenant-a"},
        )
    assert create_resp.status_code == 200
    job_id = create_resp.json()["id"]
    with patch("app.main.record_security_deny") as mock_security_deny:
        denied = client.get(f"/v2/extract/{job_id}", headers={"X-Tenant-ID": "tenant-b"})
    assert denied.status_code == 403
    assert ("v2_jobs", "tenant_forbidden") in [tuple(c.args) for c in mock_security_deny.call_args_list]


def test_v2_ssrf_policy_blocks_private_targets() -> None:
    scrape_resp = client.post("/v2/scrape", json={"url": "http://127.0.0.1/internal"})
    assert scrape_resp.status_code == 403
    scrape_payload = scrape_resp.json()
    assert scrape_payload["success"] is False
    assert scrape_payload["code"] == "SSRF_BLOCKED"

    extract_resp = client.post("/v2/extract", json={"urls": ["http://127.0.0.1/internal"]})
    assert extract_resp.status_code == 403
    extract_payload = extract_resp.json()
    assert extract_payload["success"] is False
    assert extract_payload["code"] == "SSRF_BLOCKED"
    assert extract_payload["invalidURLs"] == ["http://127.0.0.1/internal"]


def test_v2_ssrf_blocks_record_security_telemetry() -> None:
    with patch("app.main.record_security_deny") as mock_security_deny:
        scrape_resp = client.post("/v2/scrape", json={"url": "http://127.0.0.1/internal"})
        extract_resp = client.post("/v2/extract", json={"urls": ["http://127.0.0.1/internal"]})
    assert scrape_resp.status_code == 403
    assert extract_resp.status_code == 403
    seen = [tuple(c.args) for c in mock_security_deny.call_args_list]
    assert ("v2_scrape", "ssrf_blocked") in seen
    assert ("v2_extract", "ssrf_blocked") in seen


def test_v2_budget_guardrails_and_kill_switches() -> None:
    with patch.object(main_module.settings, "kill_switch_v2_scrape", True):
        scrape_resp = client.post("/v2/scrape", json={"url": "https://example.com"})
    assert scrape_resp.status_code == 503
    scrape_payload = scrape_resp.json()
    assert scrape_payload["success"] is False
    assert scrape_payload["code"] == "KILL_SWITCH_SCRAPE"

    with patch.object(main_module.settings, "kill_switch_v2_crawl", True):
        crawl_resp = client.post("/v2/crawl", json={"url": "https://example.com"})
    assert crawl_resp.status_code == 503
    crawl_payload = crawl_resp.json()
    assert crawl_payload["success"] is False
    assert crawl_payload["code"] == "KILL_SWITCH_CRAWL"

    with patch.object(main_module.settings, "budget_v2_crawl_limit_cap", 1):
        crawl_limit_resp = client.post("/v2/crawl", json={"url": "https://example.com", "limit": 2})
    assert crawl_limit_resp.status_code == 400
    crawl_limit_payload = crawl_limit_resp.json()
    assert crawl_limit_payload["success"] is False
    assert crawl_limit_payload["code"] == "BUDGET_LIMIT_EXCEEDED"

    with patch.object(main_module.settings, "budget_v2_extract_url_cap", 1):
        extract_resp = client.post(
            "/v2/extract",
            json={"urls": ["https://example.com/a", "https://example.com/b"]},
        )
    assert extract_resp.status_code == 400
    extract_payload = extract_resp.json()
    assert extract_payload["success"] is False
    assert extract_payload["code"] == "EXTRACT_BUDGET_EXCEEDED"

    captured = {}

    class FakeSerperClient:
        async def search(self, req):
            captured["num"] = req.num
            return (SimpleNamespace(organic=[], credits_used=0), 0.001)

        async def news(self, req):
            _ = req
            return (SimpleNamespace(items=[]), 0.001)

        async def images(self, req):
            _ = req
            return (SimpleNamespace(items=[]), 0.001)

    with patch.object(main_module.settings, "budget_v2_search_limit_cap", 3), patch(
        "app.main.SerperClient", new=FakeSerperClient
    ):
        search_resp = client.post("/v2/search", json={"query": "budget test", "limit": 9})
    assert search_resp.status_code == 200
    search_payload = search_resp.json()
    assert search_payload["success"] is True
    assert captured["num"] == 3
    assert "budget guardrail" in (search_payload.get("warning") or "")

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_body = metrics_resp.text
    kill_lines = [
        line for line in metrics_body.splitlines() if line.startswith("candlecrawl_kill_switch_events_total{")
    ]
    budget_lines = [
        line
        for line in metrics_body.splitlines()
        if line.startswith("candlecrawl_budget_guardrail_events_total{")
    ]
    assert any('endpoint="v2_scrape"' in line and 'kill_switch="scrape"' in line for line in kill_lines)
    assert any('endpoint="v2_crawl"' in line and 'kill_switch="crawl"' in line for line in kill_lines)
    assert any(
        'endpoint="v2_crawl"' in line and 'guardrail="crawl_limit_cap"' in line and 'action="blocked"' in line
        for line in budget_lines
    )
    assert any(
        'endpoint="v2_extract"' in line and 'guardrail="extract_url_cap"' in line and 'action="blocked"' in line
        for line in budget_lines
    )
    assert any(
        'endpoint="v2_search"' in line and 'guardrail="search_limit_cap"' in line and 'action="capped"' in line
        for line in budget_lines
    )


def test_v2_strict_status_errors_and_cancel_shapes_for_crawl_and_batch() -> None:
    with patch("app.main.crawl_background_task", new_callable=AsyncMock):
        crawl_create = client.post("/v2/crawl", json={"url": "https://example.com"})
    assert crawl_create.status_code == 200
    crawl_id = crawl_create.json()["id"]

    crawl_status = client.get(f"/v2/crawl/{crawl_id}")
    assert crawl_status.status_code == 200
    crawl_status_payload = crawl_status.json()
    assert "success" not in crawl_status_payload
    for key in ("status", "completed", "total", "creditsUsed", "expiresAt", "next", "data"):
        assert key in crawl_status_payload

    crawl_errors = client.get(f"/v2/crawl/{crawl_id}/errors")
    assert crawl_errors.status_code == 200
    crawl_errors_payload = crawl_errors.json()
    assert "success" not in crawl_errors_payload
    assert "errors" in crawl_errors_payload
    assert "robotsBlocked" in crawl_errors_payload

    crawl_cancel = client.delete(f"/v2/crawl/{crawl_id}")
    assert crawl_cancel.status_code == 200
    crawl_cancel_payload = crawl_cancel.json()
    assert crawl_cancel_payload["success"] is True
    assert crawl_cancel_payload["message"] == "Crawl cancelled."

    with patch("app.main.batch_scrape_background_task", new_callable=AsyncMock):
        batch_create = client.post("/v2/batch/scrape", json={"urls": ["https://example.com"]})
    assert batch_create.status_code == 200
    batch_id = batch_create.json()["id"]

    batch_status = client.get(f"/v2/batch/scrape/{batch_id}")
    assert batch_status.status_code == 200
    batch_status_payload = batch_status.json()
    assert "success" not in batch_status_payload
    for key in ("status", "completed", "total", "creditsUsed", "expiresAt", "next", "data"):
        assert key in batch_status_payload

    batch_errors = client.get(f"/v2/batch/scrape/{batch_id}/errors")
    assert batch_errors.status_code == 200
    batch_errors_payload = batch_errors.json()
    assert "success" not in batch_errors_payload
    assert "errors" in batch_errors_payload
    assert "robotsBlocked" in batch_errors_payload

    batch_cancel = client.delete(f"/v2/batch/scrape/{batch_id}")
    assert batch_cancel.status_code == 200
    batch_cancel_payload = batch_cancel.json()
    assert batch_cancel_payload["success"] is True
    assert batch_cancel_payload["message"] == "Batch scrape cancelled."


def test_v2_scrape_querylake_files_mode_falls_back_to_inline_when_unavailable() -> None:
    with patch.object(main_module.settings, "artifact_querylake_enabled", False), patch.object(
        main_module.settings, "artifact_allow_fallback_to_inline", True
    ), patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="inline body",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            json={"url": "https://example.com", "artifact_mode": "querylake_files"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["artifactMode"] == "inline"
    assert payload["artifactFallback"] == "querylake_unavailable"
    assert payload["data"]["markdown"] == "inline body"


def test_v2_scrape_querylake_files_mode_externalizes_markdown_when_enabled() -> None:
    with patch.object(main_module.settings, "artifact_querylake_enabled", True), patch.object(
        main_module.settings, "querylake_files_base_url", "https://querylake.local"
    ), patch.object(main_module.settings, "artifact_allow_fallback_to_inline", False), patch(
        "app.main.scraper.scrape_url",
        new_callable=AsyncMock,
    ) as mock_scrape, patch(
        "app.main.QueryLakeFilesArtifactSink.put_bytes",
        new_callable=AsyncMock,
    ) as mock_put:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="externalize me",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        mock_put.return_value = ArtifactRef(
            mode="querylake_files",
            uri="querylake://files/doc-123",
            content_type="text/markdown",
            size_bytes=14,
        )
        response = client.post(
            "/v2/scrape",
            json={"url": "https://example.com", "artifact_mode": "querylake_files"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["artifactMode"] == "querylake_files"
    assert payload["artifacts"]["markdown"]["uri"] == "querylake://files/doc-123"
    assert "markdown" not in payload["data"]
    assert mock_put.await_count >= 1


def test_v2_scrape_invalid_artifact_mode_returns_400() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="body",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post(
            "/v2/scrape",
            json={"url": "https://example.com", "artifact_mode": "not_real"},
        )

    assert response.status_code == 400
    payload = response.json()
    assert payload["success"] is False
    assert payload["code"] == "INVALID_ARTIFACT_MODE"


def test_v2_scrape_action_generated_pdf_is_preserved_and_shaped() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="body",
            content_base64="UERGREFUQQ==",
            metadata=DocumentMetadata(
                source_url="https://example.com",
                status_code=200,
                file_metadata={
                    "generated_pdf": True,
                    "generated_pdf_bytes": 7,
                    "action_results": [{"index": 0, "type": "generatePdf", "status": "success"}],
                },
            ),
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": ["markdown"],
                "actions": [{"type": "generatePdf"}],
                "maxAge": 0,
                "storeInCache": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["contentBase64"] == "UERGREFUQQ=="
    assert payload["actionResults"] == [{"index": 0, "type": "generatePdf", "status": "success"}]
    assert set(payload["actionResults"][0].keys()) == {"index", "type", "status"}
    assert set(payload["actionArtifacts"].keys()) == {"generatedPdf"}
    assert set(payload["actionArtifacts"]["generatedPdf"].keys()) == {"encoding", "location", "sizeBytes"}
    assert payload["actionArtifacts"]["generatedPdf"]["location"] == "data.contentBase64"
    assert payload["actionArtifacts"]["generatedPdf"]["sizeBytes"] == 7


def test_v2_scrape_action_screenshot_is_preserved_without_screenshot_format() -> None:
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="body",
            screenshot="c2NyZWVuc2hvdA==",
            metadata=DocumentMetadata(
                source_url="https://example.com",
                status_code=200,
                file_metadata={
                    "action_results": [{"index": 0, "type": "screenshot", "status": "success"}],
                },
            ),
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": ["markdown"],
                "actions": [{"type": "screenshot"}],
                "maxAge": 0,
                "storeInCache": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["screenshot"] == "c2NyZWVuc2hvdA=="
    assert payload["actionResults"] == [{"index": 0, "type": "screenshot", "status": "success"}]
    assert set(payload["actionResults"][0].keys()) == {"index", "type", "status"}
    assert set(payload["actionArtifacts"].keys()) == {"actionScreenshot"}
    assert set(payload["actionArtifacts"]["actionScreenshot"].keys()) == {"encoding", "location"}
    assert payload["actionArtifacts"]["actionScreenshot"]["location"] == "data.screenshot"


def test_v2_actions_capabilities_endpoint_has_stable_shape() -> None:
    response = client.get("/v2/actions/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "success",
        "contractVersion",
        "actions",
        "documentedRequestFields",
        "documentedResponseFields",
    }
    assert payload["success"] is True
    assert payload["contractVersion"] == "candlecrawl.actions.v1"
    assert isinstance(payload["actions"], list)
    assert isinstance(payload["documentedRequestFields"], list)
    assert isinstance(payload["documentedResponseFields"], list)
    assert "type" in payload["documentedRequestFields"]
    assert "actionResults[].status" in payload["documentedResponseFields"]
    action_types = {entry.get("type") for entry in payload["actions"] if isinstance(entry, dict)}
    for required in (
        "wait",
        "click",
        "type",
        "evaluate",
        "screenshot",
        "generatePdf",
        "dragAndDrop",
        "navigate",
    ):
        assert required in action_types
