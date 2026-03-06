import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.framework_shims import HTTPException
from app.main import _request_json
from app.models import FirecrawlDocument


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_scrape_success(client):
    with patch("app.service.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="Example Domain",
            links=[],
        )
        response = client.post("/v1/scrape", json={"url": "https://example.com"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "markdown" in data["data"]


def test_v2_scrape_actions_evaluate(client):
    with patch("app.service.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="codex-action-test",
            links=[],
        )
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": ["markdown"],
                "onlyMainContent": True,
                "actions": [
                    {
                        "type": "evaluate",
                        "script": "document.body.insertAdjacentHTML('beforeend','<p>codex-action-test</p>');",
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data


def test_scrape_get_captures_full_url_path(client):
    with patch("app.service.scrape_url_get", new_callable=AsyncMock) as mock_scrape_get:
        mock_scrape_get.return_value = {"success": True, "data": {"url": "https://example.com/a/b"}}

        response = client.get("/v1/scrape/https://example.com/a/b")

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert mock_scrape_get.await_count == 1
        assert mock_scrape_get.await_args.args[1] == "https://example.com/a/b"


def test_cors_preflight_returns_headers(client):
    response = client.options(
        "/v2/scrape",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 204
    assert response.headers["access-control-allow-origin"] == "https://example.com"
    assert "POST" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "/v2/scrape",
        "/v2/crawl",
        "/v2/map",
        "/v2/batch/scrape",
        "/v2/search",
        "/v2/extract",
        "/v1/hermes/research",
    ],
)
def test_invalid_json_returns_400(endpoint, client):
    response = client.post(endpoint, content="{not-json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "Invalid JSON body" in response.json()["detail"]


def test_request_json_reads_body_bytes_before_request_json():
    class RequestStub:
        body = b'{"url":"https://example.com"}'

        def json(self):
            raise RuntimeError("request.json() should not be called")

    assert _request_json(RequestStub()) == {"url": "https://example.com"}


def test_request_json_rejects_non_object_body():
    class RequestStub:
        body = b'["https://example.com"]'

        def json(self):
            raise RuntimeError("request.json() should not be called")

    with pytest.raises(HTTPException) as exc_info:
        _request_json(RequestStub())

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "JSON body must be an object"


def test_scrape_missing_url(client):
    response = client.post("/v1/scrape", json={})
    assert response.status_code == 422


def test_scrape_invalid_url(client):
    response = client.post("/v1/scrape", json={"url": "not-a-url"})
    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]


def test_cost_summary_invalid_days_returns_422(client):
    response = client.get("/v1/hermes/costs/summary?days=abc")
    assert response.status_code == 422
    assert "days" in response.json()["detail"]


def test_cost_cleanup_invalid_max_age_hours_returns_422(client):
    response = client.post("/v1/hermes/costs/cleanup?max_age_hours=abc")
    assert response.status_code == 422
    assert "max_age_hours" in response.json()["detail"]


@pytest.mark.asyncio
async def test_crawl_success(client, mocker):
    mock_scrape = mocker.patch("app.service.scraper.scrape_url", new_callable=AsyncMock)
    mock_scrape.side_effect = [
        FirecrawlDocument(
            url="https://example.com",
            content="This is a test.",
            markdown="This is a test.",
            links=["https://example.com/other_page"],
        ),
        FirecrawlDocument(
            url="https://example.com/other_page",
            content="This is another test.",
            markdown="This is another test.",
        ),
    ]

    response = client.post(
        "/v1/crawl",
        json={"url": "https://example.com", "limit": 2, "max_depth": 1, "max_concurrency": 1},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    timeout = time.time() + 30
    while time.time() < timeout:
        status_response = client.get(f"/v1/crawl/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        if status_data["status"] == "completed":
            assert len(status_data["data"]) == 2
            return
        await asyncio.sleep(0.2)

    assert False, "Crawl job did not complete within timeout."


@pytest.mark.asyncio
async def test_crawl_wikipedia(client, mocker):
    mock_scrape = mocker.patch("app.service.scraper.scrape_url", new_callable=AsyncMock)
    mock_scrape.side_effect = [
        FirecrawlDocument(
            url="https://en.wikipedia.org/wiki/Mossad",
            markdown="Mossad page content",
            links=[
                "https://en.wikipedia.org/wiki/Israel",
                "https://en.wikipedia.org/wiki/Shin_Bet",
                "https://www.external-link.com/page",
            ],
        ),
        FirecrawlDocument(url="https://en.wikipedia.org/wiki/Israel", markdown="Israel page"),
        FirecrawlDocument(url="https://en.wikipedia.org/wiki/Shin_Bet", markdown="Shin Bet page"),
    ]

    response = client.post(
        "/v1/crawl",
        json={"url": "https://en.wikipedia.org/wiki/Mossad", "limit": 3, "max_depth": 1, "max_concurrency": 2},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    timeout = time.time() + 30
    while time.time() < timeout:
        status_response = client.get(f"/v1/crawl/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        if status_data["status"] == "completed":
            assert len(status_data["data"]) == 3
            return
        await asyncio.sleep(0.2)

    assert False, "Crawl job did not complete within timeout."


def test_batch_scrape(client):
    urls = ["https://example.com", "https://firecrawl.dev"]
    with patch("app.service.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = [
            FirecrawlDocument(url=urls[0], markdown="one", links=[]),
            FirecrawlDocument(url=urls[1], markdown="two", links=[]),
        ]
        response = client.post("/v1/scrape/bulk", json={"urls": urls})
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(r["success"] for r in results)


def test_batch_status_serializes_datetime_fields(client):
    with patch("app.service.get_batch_status", new_callable=AsyncMock) as mock_get_batch_status:
        mock_get_batch_status.return_value = {
            "id": "job-123",
            "status": "completed",
            "expires_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        }

        response = client.get("/v1/batch-scrape/job-123")

        assert response.status_code == 200
        assert response.json()["expires_at"] == "2026-01-02T03:04:05+00:00"
