import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

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


def test_scrape_missing_url(client):
    response = client.post("/v1/scrape", json={})
    assert response.status_code == 422


def test_scrape_invalid_url(client):
    response = client.post("/v1/scrape", json={"url": "not-a-url"})
    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]


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
