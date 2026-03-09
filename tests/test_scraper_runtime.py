import pytest
from unittest.mock import patch

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
