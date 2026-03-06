import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from app.models import FirecrawlDocument


@pytest.mark.asyncio
async def test_crawl_mossad_extreme(client, mocker):
    mock_scrape = mocker.patch("app.service.scraper.scrape_url", new_callable=AsyncMock)

    base_url = "https://en.wikipedia.org/wiki/Mossad"
    links_to_find = [f"{base_url}_{i}" for i in range(49)]

    first_response = FirecrawlDocument(
        url=base_url,
        markdown="Mossad page content",
        links=links_to_find,
    )
    other_responses = [FirecrawlDocument(url=link, markdown=f"Markdown for {link}") for link in links_to_find]

    mock_scrape.side_effect = [first_response] + other_responses

    response = client.post(
        "/v1/crawl",
        json={"url": base_url, "limit": 50, "max_depth": 1, "max_concurrency": 10},
    )
    assert response.status_code == 200
    job_id = response.json()["id"]

    timeout = time.time() + 60
    while time.time() < timeout:
        status_response = client.get(f"/v1/crawl/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        if status_data["status"] == "completed":
            assert mock_scrape.call_count == 50
            assert len(status_data["data"]) == 50
            return
        await asyncio.sleep(0.5)

    assert False, "Crawl job did not complete within timeout."
