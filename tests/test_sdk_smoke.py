from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import DocumentMetadata, FirecrawlDocument


client = TestClient(app)


def test_firecrawl_style_sdk_smoke_for_p0_v2_endpoints() -> None:
    class FakeSerperClient:
        async def search(self, req):
            _ = req
            return (
                SimpleNamespace(
                    organic=[SimpleNamespace(link="https://example.com/result", title="Result", snippet="snippet")],
                    credits_used=1,
                ),
                0.001,
            )

        async def news(self, req):
            _ = req
            return (SimpleNamespace(items=[{"title": "News", "link": "https://news.example.com"}]), 0.001)

        async def images(self, req):
            _ = req
            return (SimpleNamespace(items=[{"title": "Image", "link": "https://img.example.com", "imageUrl": "https://img.example.com/a.png"}]), 0.001)

    async def _fake_scrape(url: str, options) -> FirecrawlDocument:
        _ = options
        return FirecrawlDocument(
            url=url,
            markdown=f"content for {url}",
            links=["https://example.com/child"],
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    async def _fake_sitemap(url: str) -> list[str]:
        _ = url
        return ["https://example.com/sitemap-a"]

    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape, patch(
        "app.main.get_sitemap_links", new=_fake_sitemap
    ), patch("app.main.SerperClient", new=FakeSerperClient), patch(
        "app.main.extract_background_task", new_callable=AsyncMock
    ), patch("app.main.crawl_background_task", new_callable=AsyncMock):
        mock_scrape.side_effect = _fake_scrape

        scrape_resp = client.post("/v2/scrape", json={"url": "https://example.com", "formats": ["markdown"]})
        assert scrape_resp.status_code == 200
        scrape_payload = scrape_resp.json()
        assert scrape_payload["success"] is True
        assert "data" in scrape_payload

        map_resp = client.post("/v2/map", json={"url": "https://example.com"})
        assert map_resp.status_code == 200
        map_payload = map_resp.json()
        assert map_payload["success"] is True
        assert isinstance(map_payload["links"], list)

        search_resp = client.post(
            "/v2/search",
            json={"query": "sdk smoke", "sources": [{"type": "web"}, {"type": "news"}, {"type": "images"}]},
        )
        assert search_resp.status_code == 200
        search_payload = search_resp.json()
        assert search_payload["success"] is True
        assert "data" in search_payload
        assert "id" in search_payload
        assert "creditsUsed" in search_payload

        extract_create = client.post("/v2/extract", json={"urls": ["https://example.com"]})
        assert extract_create.status_code == 200
        extract_create_payload = extract_create.json()
        assert extract_create_payload["success"] is True
        extract_id = extract_create_payload["id"]

        extract_status = client.get(f"/v2/extract/{extract_id}")
        assert extract_status.status_code == 200
        extract_status_payload = extract_status.json()
        assert "status" in extract_status_payload
        assert "data" in extract_status_payload

        crawl_create = client.post("/v2/crawl", json={"url": "https://example.com"})
        assert crawl_create.status_code == 200
        crawl_id = crawl_create.json()["id"]

        crawl_status = client.get(f"/v2/crawl/{crawl_id}")
        assert crawl_status.status_code == 200
        assert "status" in crawl_status.json()

        crawl_errors = client.get(f"/v2/crawl/{crawl_id}/errors")
        assert crawl_errors.status_code == 200

        crawl_cancel = client.delete(f"/v2/crawl/{crawl_id}")
        assert crawl_cancel.status_code == 200
