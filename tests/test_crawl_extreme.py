import pytest
from fastapi.testclient import TestClient
from app.main import app
import asyncio
import time
from unittest.mock import AsyncMock
from app.models import FirecrawlDocument

client = TestClient(app)

@pytest.mark.asyncio
async def test_crawl_mossad_extreme(mocker):
    """Test crawling a Wikipedia page with a high limit."""
    # Mock the scraper's scrape_url method to simulate a deep crawl
    mock_scrape = mocker.patch(
        'app.main.scraper.scrape_url',
        new_callable=AsyncMock,
    )

    # Generate 50 unique but predictable links
    base_url = "https://en.wikipedia.org/wiki/Mossad"
    links_to_find = [f"{base_url}_{i}" for i in range(49)]

    # First response contains all the links
    first_response = FirecrawlDocument(
        url=base_url,
        content="Mossad page content",
        markdown="Mossad page content",
        links=links_to_find
    )

    # Subsequent responses are for the linked pages
    other_responses = [
        FirecrawlDocument(url=link, content=f"Page content for {link}", markdown=f"Markdown for {link}")
        for link in links_to_find
    ]

    mock_scrape.side_effect = [first_response] + other_responses

    # 1. Start the crawl
    response = client.post("/v1/crawl", json={
        "url": base_url,
        "limit": 50,
        "max_depth": 1,
        "max_concurrency": 10
    })
    assert response.status_code == 200
    job_id = response.json()["id"]

    # 2. Poll for completion
    timeout = time.time() + 180  # 3-minute timeout for this larger test
    while time.time() < timeout:
        status_response = client.get(f"/v1/crawl/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        if status_data["status"] == "completed":
            assert mock_scrape.call_count == 50
            assert len(status_data["data"]) == 50
            urls_crawled = {d["url"] for d in status_data["data"]}
            assert base_url in urls_crawled
            for link in links_to_find:
                assert link in urls_crawled
            return  # Test success
        await asyncio.sleep(1)

    assert False, "Crawl job did not complete within timeout." 
    
if __name__ == "__main__":
    asyncio.run(test_crawl_mossad_extreme())