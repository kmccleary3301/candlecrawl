import pytest
from fastapi.testclient import TestClient
from app.main import app
import asyncio
import time
from unittest.mock import AsyncMock, patch
from app.models import FirecrawlDocument

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_scrape_success():
    response = client.post("/v1/scrape", json={"url": "https://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    assert "data" in data
    assert "markdown" in data["data"]
    assert "metadata" in data["data"]
    assert "Example Domain" in data["data"]["markdown"]

def test_v2_scrape_actions_evaluate():
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
    assert data["success"] == True
    assert "data" in data
    assert "markdown" in data["data"]
    assert "codex-action-test" in data["data"]["markdown"]

def test_scrape_missing_url():
    response = client.post("/v1/scrape", json={})
    assert response.status_code == 422

def test_scrape_invalid_url():
    response = client.post("/v1/scrape", json={"url": "not-a-url"})
    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]

@pytest.mark.asyncio
async def test_crawl_success(mocker):
    """Test successful crawl initiation and completion."""
    # Mock the scraper's scrape_url method
    mock_scrape = mocker.patch(
        'app.main.scraper.scrape_url',
        new_callable=AsyncMock
    )
    mock_scrape.side_effect = [
        FirecrawlDocument(
            url="https://example.com",
            content="This is a test.",
            markdown="This is a test.",
            links=["https://example.com/other_page"]
        ),
        FirecrawlDocument(
            url="https://example.com/other_page",
            content="This is another test.",
            markdown="This is another test."
        )
    ]


    # 1. Start the crawl
    response = client.post("/v1/crawl", json={
        "url": "https://example.com",
        "limit": 2,
        "max_depth": 1,
        "max_concurrency": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    job_id = data["id"]

    # 2. Poll for completion
    timeout = time.time() + 180  # 3-minute timeout
    while time.time() < timeout:
        status_response = client.get(f"/v1/crawl/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        if status_data["status"] == "completed":
            assert len(status_data["data"]) == 2  # Both original and linked page
            assert status_data["data"][0]["url"] == "https://example.com"
            assert status_data["data"][1]["url"] == "https://example.com/other_page"
            return  # Test success
        await asyncio.sleep(0.5)

    assert False, "Crawl job did not complete within timeout."


@pytest.mark.asyncio
async def test_crawl_wikipedia(mocker):
    """Test crawling a Wikipedia page."""
    # Mock the scraper's scrape_url method to simulate finding a few links
    mock_scrape = mocker.patch(
        'app.main.scraper.scrape_url',
        new_callable=AsyncMock,
    )
    mock_scrape.side_effect = [
        FirecrawlDocument(
            url="https://en.wikipedia.org/wiki/Mossad",
            content="Mossad page content",
            markdown="Mossad page content",
            links=[
                "https://en.wikipedia.org/wiki/Israel",
                "https://en.wikipedia.org/wiki/Shin_Bet",
                "https://www.external-link.com/page" # This should be ignored
            ]
        ),
        FirecrawlDocument(url="https://en.wikipedia.org/wiki/Israel", content="Israel page", markdown="Israel page"),
        FirecrawlDocument(url="https://en.wikipedia.org/wiki/Shin_Bet", content="Shin Bet page", markdown="Shin Bet page"),
    ]

    # 1. Start the crawl
    response = client.post("/v1/crawl", json={
        "url": "https://en.wikipedia.org/wiki/Mossad",
        "limit": 3,
        "max_depth": 1,
        "max_concurrency": 2
    })
    assert response.status_code == 200
    job_id = response.json()["id"]

    # 2. Poll for completion
    timeout = time.time() + 180  # 3-minute timeout
    while time.time() < timeout:
        status_response = client.get(f"/v1/crawl/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        if status_data["status"] == "completed":
            assert len(status_data["data"]) == 3  # Original + 2 same-domain links
            urls_crawled = {d["url"] for d in status_data["data"]}
            assert "https://en.wikipedia.org/wiki/Mossad" in urls_crawled
            assert "https://en.wikipedia.org/wiki/Israel" in urls_crawled
            assert "https://en.wikipedia.org/wiki/Shin_Bet" in urls_crawled
            assert "https://www.external-link.com/page" not in urls_crawled
            return  # Test success
        await asyncio.sleep(0.5)
    
    assert False, "Crawl job did not complete within timeout."


def test_batch_scrape():
    urls = ["https://example.com", "https://firecrawl.dev"]
    response = client.post("/v1/scrape/bulk", json={"urls": urls})
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2
    for res in results:
        assert res["success"] == True
        assert "data" in res
        assert "markdown" in res["data"]
        assert "metadata" in res["data"] 
