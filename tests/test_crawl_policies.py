import pytest
from fastapi.testclient import TestClient
from app.main import app
import asyncio
import time
from unittest.mock import AsyncMock
from app.models import FirecrawlDocument

client = TestClient(app)

@pytest.mark.asyncio
async def test_crawl_include_exclude_paths(mocker):
    base = "https://example.com"
    # Mock scraping: root has 3 links
    mock_scrape = mocker.patch('app.main.scraper.scrape_url', new_callable=AsyncMock)
    mock_scrape.side_effect = [
        FirecrawlDocument(url=f"{base}", links=[f"{base}/allowed/a", f"{base}/blocked/b", f"{base}/allowed/c"]),
        FirecrawlDocument(url=f"{base}/allowed/a"),
        FirecrawlDocument(url=f"{base}/allowed/c"),
    ]

    resp = client.post("/v1/crawl", json={
        "url": base,
        "limit": 10,
        "max_depth": 2,
        "include_paths": ["/allowed"],
        "exclude_paths": ["/blocked"],
        "max_concurrency": 2
    })
    assert resp.status_code == 200
    job_id = resp.json()["id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        s = client.get(f"/v1/crawl/{job_id}")
        assert s.status_code == 200
        data = s.json()
        if data["status"] == "completed":
            urls = {d["url"] for d in data["data"]}
            assert f"{base}/allowed/a" in urls
            assert f"{base}/allowed/c" in urls
            assert f"{base}/blocked/b" not in urls
            return
        await asyncio.sleep(0.2)
    assert False, "timeout"

@pytest.mark.asyncio
async def test_crawl_budget_limits(mocker):
    base = "https://example.org"
    mock_scrape = mocker.patch('app.main.scraper.scrape_url', new_callable=AsyncMock)
    # Make each page heavy to hit max_bytes quickly
    heavy_md = "x" * 1024 * 64
    mock_scrape.side_effect = [
        FirecrawlDocument(url=f"{base}", markdown=heavy_md, links=[f"{base}/p1", f"{base}/p2", f"{base}/p3"]),
        FirecrawlDocument(url=f"{base}/p1", markdown=heavy_md),
        FirecrawlDocument(url=f"{base}/p2", markdown=heavy_md),
        FirecrawlDocument(url=f"{base}/p3", markdown=heavy_md),
    ]

    resp = client.post("/v1/crawl", json={
        "url": base,
        "limit": 100,
        "max_depth": 2,
        "max_concurrency": 3,
        "max_bytes": 128 * 1024  # stop around 2 heavy pages
    })
    assert resp.status_code == 200
    job_id = resp.json()["id"]

    deadline = time.time() + 30
    while time.time() < deadline:
        s = client.get(f"/v1/crawl/{job_id}")
        assert s.status_code == 200
        data = s.json()
        if data["status"] == "completed":
            # Should have cut off early due to max_bytes
            assert len(data["data"]) <= 3
            return
        await asyncio.sleep(0.2)
    assert False, "timeout"




