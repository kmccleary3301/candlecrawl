import pytest
import time
from fastapi.testclient import TestClient
from app.main import app
import asyncio

client = TestClient(app)

NUM_URLS_SMALL = 5
NUM_URLS_LARGE = 20
CRAWL_URL = "https://firecrawl.dev"


@pytest.mark.performance
def test_single_scrape_performance():
    start_time = time.time()
    response = client.post("/v1/scrape", json={"url": "https://example.com"})
    end_time = time.time()
    
    assert response.status_code == 200
    print(f"Single scrape took {end_time - start_time:.2f} seconds.")


@pytest.mark.performance
def test_batch_scrape_small_performance():
    urls = [f"https://example.com?i={i}" for i in range(NUM_URLS_SMALL)]
    
    start_time = time.time()
    response = client.post("/v1/scrape/bulk", json={"urls": urls})
    end_time = time.time()
    
    assert response.status_code == 200
    print(f"Batch scrape with {NUM_URLS_SMALL} URLs took {end_time - start_time:.2f} seconds.")


@pytest.mark.performance
def test_batch_scrape_large_performance():
    urls = [f"https://example.com?i={i}" for i in range(NUM_URLS_LARGE)]
    
    start_time = time.time()
    response = client.post("/v1/scrape/bulk", json={"urls": urls})
    end_time = time.time()
    
    assert response.status_code == 200
    print(f"Batch scrape with {NUM_URLS_LARGE} URLs took {end_time - start_time:.2f} seconds.")


@pytest.mark.asyncio
@pytest.mark.performance
async def test_crawl_performance():
    start_time = time.time()
    
    response = client.post("/v1/crawl", json={"url": CRAWL_URL})
    assert response.status_code == 200
    job_id = response.json()["id"]
    
    while True:
        await asyncio.sleep(5)
        status_response = client.get(f"/v1/crawl/{job_id}")
        if status_response.status_code == 200:
            data = status_response.json()
            if data["status"] == "completed":
                break
            elif data["status"] == "failed":
                pytest.fail("Crawl job failed.")
    
    end_time = time.time()
    print(f"Crawl of {CRAWL_URL} took {end_time - start_time:.2f} seconds.") 