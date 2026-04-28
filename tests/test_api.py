import pytest
from fastapi.testclient import TestClient
from app.main import app
import app.main as main_module
import asyncio
import time
from unittest.mock import AsyncMock, patch
from app.models import FirecrawlDocument, DocumentMetadata

client = TestClient(app)

def test_health_check():
    with patch("app.main.scraper.get_browser_runtime_status", return_value={"browser_ready": False, "browser_error": "missing browser"}):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["browserReady"] is False
    assert response.json()["browserError"] == "missing browser"

def test_health_check_healthy_when_browser_ready():
    with patch("app.main.scraper.get_browser_runtime_status", return_value={"browser_ready": True, "browser_error": None}):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["browserReady"] is True
    assert response.json()["browserError"] is None

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert b"crawl_jobs_total" in response.content

def test_scrape_success():
    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.return_value = FirecrawlDocument(
            url="https://example.com",
            markdown="Example Domain - deterministic test payload",
            metadata=DocumentMetadata(source_url="https://example.com", status_code=200),
        )
        response = client.post("/v1/scrape", json={"url": "https://example.com"})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "data" in data
        assert "markdown" in data["data"]
        assert "metadata" in data["data"]
        assert "Example Domain" in data["data"]["markdown"]

def test_v2_scrape_actions_evaluate():
    captured = {}

    async def _fake_scrape(url, options):
        captured["url"] = url
        captured["options"] = options
        return FirecrawlDocument(
            url=url,
            markdown="Example Domain codex-action-test",
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = _fake_scrape
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": ["markdown"],
                "onlyMainContent": True,
                "maxAge": 0,
                "storeInCache": False,
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
        assert captured["url"] == "https://example.com"
        assert captured["options"].actions is not None
        assert captured["options"].actions[0].type == "evaluate"
        assert captured["options"].actions[0].script is not None

def test_scrape_missing_url():
    response = client.post("/v1/scrape", json={})
    assert response.status_code == 422

def test_scrape_invalid_url():
    response = client.post("/v1/scrape", json={"url": "not-a-url"})
    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]


def test_v2_cacheability_requires_material_content():
    assert main_module._is_cacheable_v2_scrape_data({"metadata": {"sourceURL": "https://example.com"}}) is False
    assert (
        main_module._is_cacheable_v2_scrape_data(
            {"markdown": "Example Domain", "metadata": {"sourceURL": "https://example.com", "statusCode": 200}}
        )
        is True
    )


def test_v2_structured_formats_are_normalized():
    options = main_module._scrape_options_from_v2(
        {
            "formats": [
                {"type": "markdown"},
                {"type": "raw_html"},
                {"type": "links"},
                {"type": "screenshot"},
                {"type": "extract"},  # unsupported
                "MD",
            ]
        }
    )
    assert options.formats == ["markdown", "rawHtml", "links", "screenshot"]


def test_v2_structured_formats_fallback_to_markdown():
    options = main_module._scrape_options_from_v2({"formats": [{"type": "extract"}, {"type": "summary"}]})
    assert options.formats == ["markdown"]


def test_v2_pdf_parser_flag_from_parsers_and_explicit_flags():
    parsers_options = main_module._scrape_options_from_v2(
        {
            "formats": [{"type": "markdown"}],
            "parsers": ["pdf"],
        }
    )
    assert parsers_options.parse_pdf is True

    parse_pdf_flag_options = main_module._scrape_options_from_v2(
        {
            "formats": ["markdown"],
            "parsePDF": True,
        }
    )
    assert parse_pdf_flag_options.parse_pdf is True

    parse_pdf_camel_flag_options = main_module._scrape_options_from_v2(
        {
            "formats": ["markdown"],
            "parsePdf": "true",
        }
    )
    assert parse_pdf_camel_flag_options.parse_pdf is True

    default_options = main_module._scrape_options_from_v2(
        {
            "formats": ["markdown"],
        }
    )
    assert default_options.parse_pdf is False


def test_v2_ocr_options_from_payload_and_parsers():
    options = main_module._scrape_options_from_v2(
        {
            "formats": ["markdown"],
            "ocr": {"enabled": True, "provider": "mistral", "prompt": "Extract text"},
        }
    )
    assert options.ocr is True
    assert options.ocr_provider == "mistral"
    assert options.ocr_prompt == "Extract text"

    parser_options = main_module._scrape_options_from_v2(
        {
            "formats": ["markdown"],
            "parsers": ["chandra"],
        }
    )
    assert parser_options.ocr is True
    assert parser_options.ocr_provider == "querylake_chandra"

    explicit_provider_options = main_module._scrape_options_from_v2(
        {
            "formats": ["markdown"],
            "ocrProvider": "querylake_chandra",
            "ocrPrompt": "custom prompt",
        }
    )
    assert explicit_provider_options.ocr is True
    assert explicit_provider_options.ocr_provider == "querylake_chandra"
    assert explicit_provider_options.ocr_prompt == "custom prompt"


def test_v2_document_maps_content_base64_and_metadata():
    doc = FirecrawlDocument(
        url="https://example.com/file.pdf",
        content_base64="ZmFrZS1iYXNlNjQ=",
        metadata=DocumentMetadata(
            source_url="https://example.com/file.pdf",
            status_code=200,
            content_type="application/pdf",
        ),
    )
    out = main_module._v2_document_from_v1(doc)
    assert out["contentBase64"] == "ZmFrZS1iYXNlNjQ="
    assert out["metadata"]["sourceURL"] == "https://example.com/file.pdf"
    assert out["metadata"]["statusCode"] == 200
    assert out["metadata"]["contentType"] == "application/pdf"


def test_v2_scrape_structured_formats_endpoint():
    captured = {}

    async def _fake_scrape(url, options):
        captured["formats"] = options.formats
        return FirecrawlDocument(
            url=url,
            markdown="Structured markdown body",
            raw_html="<html><body>Structured markdown body</body></html>",
            links=["https://example.com/page"],
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = _fake_scrape
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": [{"type": "markdown"}, {"type": "raw_html"}, {"type": "links"}],
                "maxAge": 0,
                "storeInCache": False,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["success"] is True
        assert payload["data"]["markdown"] == "Structured markdown body"
        assert payload["data"]["rawHtml"] == "<html><body>Structured markdown body</body></html>"
        assert payload["data"]["links"] == ["https://example.com/page"]
        assert captured["formats"] == ["markdown", "rawHtml", "links"]


def test_v2_scrape_only_requested_formats_are_returned():
    async def _fake_scrape(url, options):
        return FirecrawlDocument(
            url=url,
            markdown="Primary markdown",
            html="<html><body>Primary markdown</body></html>",
            raw_html="<html><body>Raw html body</body></html>",
            links=["https://example.com/extra"],
            screenshot="base64-image",
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = _fake_scrape
        response = client.post(
            "/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": [{"type": "markdown"}],
                "maxAge": 0,
                "storeInCache": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["markdown"] == "Primary markdown"
    assert "rawHtml" not in payload["data"]
    assert "html" not in payload["data"]
    assert "links" not in payload["data"]
    assert "screenshot" not in payload["data"]


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
    async def _fake_scrape(url, options):
        return FirecrawlDocument(
            url=url,
            markdown=f"Deterministic markdown for {url}",
            metadata=DocumentMetadata(source_url=url, status_code=200),
        )

    with patch("app.main.scraper.scrape_url", new_callable=AsyncMock) as mock_scrape:
        mock_scrape.side_effect = _fake_scrape
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
