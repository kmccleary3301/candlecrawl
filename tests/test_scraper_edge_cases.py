import base64
import httpx
import pytest
from unittest.mock import AsyncMock, patch

from app.models import FirecrawlDocument, ScrapeAction, ScrapeOptions
from app.providers.document_ocr import OCRExtractionResult
from app.scraper import ScrapingService


@pytest.mark.asyncio
async def test_initialize_browser_surfaces_missing_playwright_runtime():
    service = ScrapingService()

    class DummyChromium:
        async def launch(self, *args, **kwargs):
            raise RuntimeError("BrowserType.launch: Executable doesn't exist at /missing/chromium\nRun `playwright install`")

    class DummyPlaywright:
        def __init__(self):
            self.chromium = DummyChromium()
            self.stopped = False

        async def stop(self):
            self.stopped = True

    dummy_playwright = DummyPlaywright()

    class DummyAsyncPlaywrightStarter:
        async def start(self):
            return dummy_playwright

    with patch("app.scraper.async_playwright", return_value=DummyAsyncPlaywrightStarter()):
        with pytest.raises(RuntimeError, match="Playwright Chromium runtime missing; run `python -m playwright install chromium`"):
            await service.initialize_browser()

    status = service.get_browser_runtime_status()
    assert status["browser_ready"] is False
    assert status["browser_error"] == "Playwright Chromium runtime missing; run `python -m playwright install chromium`"
    assert dummy_playwright.stopped is True


@pytest.mark.asyncio
async def test_scrape_url_routes_xhtml_to_html_handler():
    service = ScrapingService()
    expected = FirecrawlDocument(url="https://example.com/test.xhtml", markdown="xhtml body")
    service._scrape_html_page = AsyncMock(return_value=expected)
    service._get_file_metadata = AsyncMock()

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def head(self, url, headers=None, follow_redirects=True):
            return httpx.Response(
                200,
                request=httpx.Request("HEAD", url),
                headers={"content-type": "application/xhtml+xml; charset=utf-8"},
            )

    with patch("app.scraper.httpx.AsyncClient", DummyAsyncClient):
        result = await service.scrape_url("https://example.com/test.xhtml", ScrapeOptions())

    assert result.markdown == "xhtml body"
    service._scrape_html_page.assert_awaited_once()
    service._get_file_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_scrape_url_routes_html_extension_when_content_type_is_text_plain():
    service = ScrapingService()
    expected = FirecrawlDocument(url="https://example.com/index.html", markdown="html body")
    service._scrape_html_page = AsyncMock(return_value=expected)
    service._get_file_metadata = AsyncMock()

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def head(self, url, headers=None, follow_redirects=True):
            return httpx.Response(
                200,
                request=httpx.Request("HEAD", url),
                headers={"content-type": "text/plain"},
            )

    with patch("app.scraper.httpx.AsyncClient", DummyAsyncClient):
        result = await service.scrape_url("https://example.com/index.html", ScrapeOptions())

    assert result.markdown == "html body"
    service._scrape_html_page.assert_awaited_once()
    service._get_file_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_scrape_url_falls_back_when_head_transport_fails():
    service = ScrapingService()
    expected = FirecrawlDocument(url="https://example.com", markdown="fallback body")
    service._scrape_html_page = AsyncMock(return_value=expected)

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def head(self, url, headers=None, follow_redirects=True):
            raise httpx.ConnectError("head transport failure", request=httpx.Request("HEAD", url))

    with patch("app.scraper.httpx.AsyncClient", DummyAsyncClient):
        result = await service.scrape_url("https://example.com", ScrapeOptions())

    assert result.markdown == "fallback body"
    service._scrape_html_page.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrape_html_page_hides_raw_html_when_not_requested():
    service = ScrapingService()
    service._scrape_with_http = AsyncMock(
        return_value=FirecrawlDocument(
            url="https://example.com",
            markdown="x" * 150,
            raw_html="<html><body>raw body</body></html>",
        )
    )
    service._scrape_with_browser = AsyncMock()

    result = await service._scrape_html_page(
        "https://example.com",
        ScrapeOptions(formats=["markdown"]),
    )

    assert result.raw_html is None
    service._scrape_with_browser.assert_not_awaited()


@pytest.mark.asyncio
async def test_scrape_html_page_keeps_raw_html_when_requested():
    service = ScrapingService()
    service._scrape_with_http = AsyncMock(
        return_value=FirecrawlDocument(
            url="https://example.com",
            markdown="x" * 150,
            raw_html="<html><body>raw body</body></html>",
        )
    )
    service._scrape_with_browser = AsyncMock()

    result = await service._scrape_html_page(
        "https://example.com",
        ScrapeOptions(formats=["markdown", "rawHtml"]),
    )

    assert result.raw_html == "<html><body>raw body</body></html>"
    service._scrape_with_browser.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_actions_generate_pdf_returns_base64_payload():
    service = ScrapingService()

    class DummyPage:
        async def pdf(self, **kwargs):
            _ = kwargs
            return b"%PDF-test-binary%"

    outputs = await service._apply_actions(
        DummyPage(),
        [ScrapeAction(type="generatePdf")],
        timeout_ms=1000,
    )
    assert "generated_pdf_base64" in outputs
    assert base64.b64decode(outputs["generated_pdf_base64"]) == b"%PDF-test-binary%"
    assert outputs["action_results"] == [
        {"index": 0, "type": "generatePdf", "status": "success", "output": "generated_pdf_base64"}
    ]


@pytest.mark.asyncio
async def test_apply_actions_records_screenshot_and_unknown_action_status():
    service = ScrapingService()

    class DummyPage:
        async def screenshot(self, **kwargs):
            _ = kwargs
            return b"fake-screenshot-bytes"

    outputs = await service._apply_actions(
        DummyPage(),
        [ScrapeAction(type="screenshot"), ScrapeAction(type="totallyUnknownAction")],
        timeout_ms=1000,
    )
    assert base64.b64decode(outputs["action_screenshot_base64"]) == b"fake-screenshot-bytes"
    assert outputs["action_results"] == [
        {"index": 0, "type": "screenshot", "status": "success"},
        {
            "index": 1,
            "type": "totallyUnknownAction",
            "status": "ignored",
            "reason": "unknown_action",
        },
    ]


@pytest.mark.asyncio
async def test_apply_actions_supports_hover_doubleclick_and_select_option():
    service = ScrapingService()
    seen = {"hover": 0, "dblclick": 0, "select_option": 0}

    class DummyPage:
        async def hover(self, selector, timeout=None):
            _ = selector, timeout
            seen["hover"] += 1

        async def dblclick(self, selector, timeout=None):
            _ = selector, timeout
            seen["dblclick"] += 1

        async def select_option(self, selector, value, timeout=None):
            _ = selector, value, timeout
            seen["select_option"] += 1

    outputs = await service._apply_actions(
        DummyPage(),
        [
            ScrapeAction(type="hover", selector="#hero"),
            ScrapeAction(type="doubleClick", selector="#submit"),
            ScrapeAction(type="selectOption", selector="#plan", text="pro"),
        ],
        timeout_ms=1000,
    )
    assert outputs["action_results"] == [
        {"index": 0, "type": "hover", "status": "success"},
        {"index": 1, "type": "doubleClick", "status": "success"},
        {"index": 2, "type": "selectOption", "status": "success"},
    ]
    assert seen == {"hover": 1, "dblclick": 1, "select_option": 1}


@pytest.mark.asyncio
async def test_apply_actions_supports_key_events_and_wait_for_navigation():
    service = ScrapingService()
    seen = {"down": 0, "up": 0, "wait_nav": 0}

    class DummyKeyboard:
        async def down(self, key):
            _ = key
            seen["down"] += 1

        async def up(self, key):
            _ = key
            seen["up"] += 1

    class DummyPage:
        keyboard = DummyKeyboard()

        async def wait_for_load_state(self, state, timeout=None):
            _ = state, timeout
            seen["wait_nav"] += 1

    outputs = await service._apply_actions(
        DummyPage(),
        [
            ScrapeAction(type="keyDown", key="Shift"),
            ScrapeAction(type="keyUp", key="Shift"),
            ScrapeAction(type="waitForNavigation"),
        ],
        timeout_ms=1000,
    )
    assert outputs["action_results"] == [
        {"index": 0, "type": "keyDown", "status": "success"},
        {"index": 1, "type": "keyUp", "status": "success"},
        {"index": 2, "type": "waitForNavigation", "status": "success"},
    ]
    assert seen == {"down": 1, "up": 1, "wait_nav": 1}


@pytest.mark.asyncio
async def test_apply_actions_supports_focus_blur_check_uncheck_clear_drag_navigate_and_eval_output():
    service = ScrapingService()
    seen = {
        "focus": 0,
        "dispatch_event": 0,
        "check": 0,
        "uncheck": 0,
        "fill": 0,
        "drag_and_drop": 0,
        "goto": 0,
        "evaluate": 0,
    }

    class DummyPage:
        url = "https://example.com/start"

        async def focus(self, selector, timeout=None):
            _ = selector, timeout
            seen["focus"] += 1

        async def dispatch_event(self, selector, event, timeout=None):
            _ = selector, event, timeout
            seen["dispatch_event"] += 1

        async def check(self, selector, timeout=None):
            _ = selector, timeout
            seen["check"] += 1

        async def uncheck(self, selector, timeout=None):
            _ = selector, timeout
            seen["uncheck"] += 1

        async def fill(self, selector, value, timeout=None):
            _ = selector, value, timeout
            seen["fill"] += 1

        async def drag_and_drop(self, source, target, timeout=None):
            _ = source, target, timeout
            seen["drag_and_drop"] += 1

        async def goto(self, url, timeout=None, wait_until=None):
            _ = timeout, wait_until
            seen["goto"] += 1
            self.url = url

        async def evaluate(self, script):
            _ = script
            seen["evaluate"] += 1
            return {"ok": True}

    outputs = await service._apply_actions(
        DummyPage(),
        [
            ScrapeAction(type="focus", selector="#a"),
            ScrapeAction(type="blur", selector="#a"),
            ScrapeAction(type="check", selector="#agree"),
            ScrapeAction(type="uncheck", selector="#agree"),
            ScrapeAction(type="clear", selector="#q"),
            ScrapeAction(type="dragAndDrop", selector="#src", targetSelector="#dst"),
            ScrapeAction(type="navigate", url="https://example.com/next"),
            ScrapeAction(type="evaluate", script="1+1"),
        ],
        timeout_ms=1000,
    )
    assert [entry["status"] for entry in outputs["action_results"]] == ["success"] * 8
    assert outputs["action_results"][6]["output"] == {"url": "https://example.com/next"}
    assert outputs["action_results"][7]["output"] == {"ok": True}
    assert seen == {
        "focus": 1,
        "dispatch_event": 1,
        "check": 1,
        "uncheck": 1,
        "fill": 1,
        "drag_and_drop": 1,
        "goto": 1,
        "evaluate": 1,
    }


@pytest.mark.asyncio
async def test_apply_actions_reports_missing_parameters_as_ignored():
    service = ScrapingService()

    class DummyPage:
        async def wait_for_timeout(self, ms):
            _ = ms

    outputs = await service._apply_actions(
        DummyPage(),
        [
            ScrapeAction(type="click"),
            ScrapeAction(type="dragAndDrop", selector="#src"),
            ScrapeAction(type="navigate"),
            ScrapeAction(type="evaluate"),
        ],
        timeout_ms=1000,
    )
    assert outputs["action_results"] == [
        {"index": 0, "type": "click", "status": "ignored", "reason": "missing_selector"},
        {"index": 1, "type": "dragAndDrop", "status": "ignored", "reason": "missing_selector_or_target"},
        {"index": 2, "type": "navigate", "status": "ignored", "reason": "missing_url"},
        {"index": 3, "type": "evaluate", "status": "ignored", "reason": "missing_script"},
    ]


@pytest.mark.asyncio
async def test_apply_actions_screenshot_with_selector_uses_element_screenshot():
    service = ScrapingService()

    class DummyElement:
        async def screenshot(self):
            return b"element-shot"

    class DummyPage:
        async def query_selector(self, selector):
            _ = selector
            return DummyElement()

    outputs = await service._apply_actions(
        DummyPage(),
        [ScrapeAction(type="screenshot", selector="#capture-me")],
        timeout_ms=1000,
    )
    assert base64.b64decode(outputs["action_screenshot_base64"]) == b"element-shot"
    assert outputs["action_results"] == [
        {"index": 0, "type": "screenshot", "status": "success"}
    ]


@pytest.mark.asyncio
async def test_apply_actions_screenshot_with_missing_selector_marks_error() -> None:
    service = ScrapingService()

    class DummyPage:
        async def query_selector(self, selector):
            _ = selector
            return None

    outputs = await service._apply_actions(
        DummyPage(),
        [ScrapeAction(type="screenshot", selector="#missing")],
        timeout_ms=1000,
    )
    assert "action_screenshot_base64" not in outputs
    assert outputs["action_results"][0]["status"] == "error"
    assert outputs["action_results"][0]["type"] == "screenshot"
    assert "selector_not_found:#missing" in outputs["action_results"][0]["error"]


@pytest.mark.asyncio
async def test_apply_actions_wait_for_selector_timeout_marks_error() -> None:
    service = ScrapingService()

    class DummyPage:
        async def wait_for_selector(self, selector, timeout=None):
            _ = selector, timeout
            raise TimeoutError("Timeout 1000ms exceeded.")

    outputs = await service._apply_actions(
        DummyPage(),
        [ScrapeAction(type="waitForSelector", selector="#slow")],
        timeout_ms=1000,
    )
    assert outputs["action_results"][0]["status"] == "error"
    assert outputs["action_results"][0]["type"] == "waitForSelector"
    assert "Timeout" in outputs["action_results"][0]["error"]


@pytest.mark.asyncio
async def test_get_file_metadata_parses_pdf_to_markdown_when_enabled():
    service = ScrapingService()
    service._get_pdf_page_count_fast = AsyncMock(return_value=4)
    service._extract_pdf_markdown = AsyncMock(return_value=("## Page 1\n\nParsed body", 1))

    class DummyResponse:
        async def aread(self):
            return b"%PDF-1.4 fake bytes%"

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, *args, **kwargs):
            return DummyResponse()

    headers = httpx.Headers({"content-type": "application/pdf; charset=binary", "content-length": "19"})
    with patch("app.scraper.ResilientHttpClient", DummyClient):
        doc = await service._get_file_metadata(
            "https://example.com/doc.pdf",
            headers,
            ScrapeOptions(formats=["markdown"], parse_pdf=True, include_file_body=False),
        )

    assert doc.markdown == "## Page 1\n\nParsed body"
    assert doc.content_base64 is None
    assert doc.metadata is not None
    assert doc.metadata.file_metadata is not None
    assert doc.metadata.file_metadata["page_count"] == 4
    assert doc.metadata.file_metadata["pdf_parsed"] is True
    assert doc.metadata.file_metadata["pdf_parsed_pages"] == 1


@pytest.mark.asyncio
async def test_get_file_metadata_uses_ocr_fallback_for_low_text_pdf():
    service = ScrapingService()
    service._get_pdf_page_count_fast = AsyncMock(return_value=2)
    service._extract_pdf_markdown = AsyncMock(return_value=("tiny", 1))

    class DummyResponse:
        async def aread(self):
            return b"%PDF-1.4 fake bytes%"

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, *args, **kwargs):
            return DummyResponse()

    headers = httpx.Headers({"content-type": "application/pdf", "content-length": "19"})
    with (
        patch("app.scraper.ResilientHttpClient", DummyClient),
        patch(
            "app.scraper.document_ocr_client.extract_markdown",
            new_callable=AsyncMock,
            return_value=OCRExtractionResult(provider="mistral", markdown="OCR recovered text"),
        ),
    ):
        doc = await service._get_file_metadata(
            "https://example.com/scan.pdf",
            headers,
            ScrapeOptions(formats=["markdown"], parse_pdf=True, ocr=True, ocr_provider="mistral"),
        )

    assert doc.markdown == "OCR recovered text"
    assert doc.metadata is not None
    assert doc.metadata.file_metadata is not None
    assert doc.metadata.file_metadata["ocr_applied"] is True
    assert doc.metadata.file_metadata["ocr_provider"] == "mistral"


@pytest.mark.asyncio
async def test_get_file_metadata_runs_ocr_for_images_when_enabled():
    service = ScrapingService()
    service._get_image_dimensions_fast = AsyncMock(return_value=(640, 480))

    class DummyResponse:
        async def aread(self):
            return b"fake-image-bytes"

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, *args, **kwargs):
            return DummyResponse()

    headers = httpx.Headers({"content-type": "image/png", "content-length": "16"})
    with (
        patch("app.scraper.ResilientHttpClient", DummyClient),
        patch(
            "app.scraper.document_ocr_client.extract_markdown",
            new_callable=AsyncMock,
            return_value=OCRExtractionResult(provider="querylake_chandra", markdown="Image OCR markdown"),
        ),
    ):
        doc = await service._get_file_metadata(
            "https://example.com/figure.png",
            headers,
            ScrapeOptions(formats=["markdown"], ocr=True, ocr_provider="querylake_chandra"),
        )

    assert doc.markdown == "Image OCR markdown"
    assert doc.metadata is not None
    assert doc.metadata.file_metadata is not None
    assert doc.metadata.file_metadata["ocr_applied"] is True
    assert doc.metadata.file_metadata["ocr_provider"] == "querylake_chandra"
