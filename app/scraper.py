import asyncio
import base64
import io
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from markdownify import markdownify as md
from PIL import Image
from pypdf import PdfReader
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeoutError

from app.config import settings
from app.http_client import ResilientHttpClient
from app.models import FirecrawlDocument, DocumentMetadata, ScrapeOptions, ScrapeAction

class ScrapingService:
    def __init__(self):
        self.browser: Optional[Browser] = None
        
    async def initialize_browser(self):
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
    
    async def close_browser(self):
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                # Browser.close can fail during shutdown if the Playwright driver
                # is already gone (e.g., Ctrl-C / abrupt termination).
                pass
            self.browser = None
    
    async def scrape_url(self, url: str, options: ScrapeOptions = None) -> FirecrawlDocument:
        if options is None:
            options = ScrapeOptions()

        try:
            # 1. Make a HEAD request to check content type first
            async with httpx.AsyncClient(timeout=15) as client:
                headers = {"User-Agent": settings.user_agent, **(options.headers or {})}
                head_response = await client.head(url, headers=headers, follow_redirects=True)
                head_response.raise_for_status()
                content_type = head_response.headers.get('content-type', '').lower()

            # 2. Route to the correct handler based on content type
            if 'text/html' in content_type:
                return await self._scrape_html_page(url, options)
            else:
                logger.info(f"Detected file content-type '{content_type}'. Routing to file handler.")
                return await self._get_file_metadata(url, head_response.headers, options)

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            # Fallback for sites that don't support HEAD requests
            if isinstance(e, httpx.HTTPStatusError) and getattr(getattr(e, "response", None), "status_code", None) in {403, 405, 429}:
                logger.warning(
                    f"HEAD request blocked ({e.response.status_code}); escalating to browser/GET scrape for {url}"
                )
                return await self._scrape_html_page(url, options)
            if isinstance(e, httpx.UnsupportedProtocol) or "Method 'HEAD' not allowed" in str(e):
                logger.warning(f"HEAD request failed for {url}, falling back to GET-based HTML scraping.")
                return await self._scrape_html_page(url, options)
            return FirecrawlDocument(
                url=url,
                metadata=DocumentMetadata(source_url=url, status_code=getattr(e, 'response', None) and getattr(e.response, 'status_code', None))
            )

    def _needs_browser_rendering(self, options: ScrapeOptions) -> bool:
        return (
            options.wait_for is not None or
            "screenshot" in (options.formats or []) or
            options.mobile or
            (options.actions is not None and len(options.actions) > 0)
        )
    
    async def _scrape_with_http(self, url: str, options: ScrapeOptions, timeout_override: Optional[int] = None) -> FirecrawlDocument:
        headers = {
            "User-Agent": settings.user_agent,
            **(options.headers or {})
        }
        
        timeout = timeout_override or options.timeout or settings.default_timeout
        
        client = ResilientHttpClient(timeout=timeout, follow_redirects=True)
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        
        if "text/html" in content_type:
            # Also return raw_html to be used in the fallback check
            doc = await self._process_html_content(url, response.text, response.status_code, options)
            doc.raw_html = response.text
            return doc
        else:
            return FirecrawlDocument(
                url=url,
                metadata=DocumentMetadata(
                    source_url=url,
                    status_code=response.status_code,
                    content_type=content_type
                )
            )

    async def _scrape_with_browser(self, url: str, options: ScrapeOptions) -> FirecrawlDocument:
        await self.initialize_browser()
        use_actions = options.actions is not None and len(options.actions) > 0
        viewport = {"width": 1920, "height": 1080} if use_actions else {"width": 1920, "height": 10000}
        context = await self.browser.new_context(
            user_agent=settings.user_agent,
            viewport=viewport,
        )
        page = await context.new_page()

        try:
            if options.headers:
                await page.set_extra_http_headers(options.headers)
            
            timeout = (options.timeout or settings.default_timeout) * 1000
            await page.goto(url, timeout=timeout * 2, wait_until="domcontentloaded")

            # Best-effort: don't fail the scrape if the page never becomes "idle"
            # (common for pages with long-polling / websockets).
            try:
                logger.info(f"Waiting for network idle to ensure all content is loaded for {url}")
                await page.wait_for_load_state("networkidle", timeout=30000)  # 30-second timeout for idle
            except PlaywrightTimeoutError:
                logger.warning(f"Timed out waiting for network idle for {url}; continuing.")

            if options.wait_for:
                await page.wait_for_timeout(int(options.wait_for) * 1000)

            if use_actions:
                await self._apply_actions(page, options.actions, timeout_ms=int(timeout))

            # If we're not doing interactive actions, use the aggressive "zoom out"
            # heuristic to pull more lazy-loaded content into the viewport.
            if not use_actions:
                logger.info(f"Setting page zoom to 1% for {url}")
                await page.evaluate("document.body.style.zoom = '0.01'")
                # Give the page a moment to respond to the zoom/layout change.
                await page.wait_for_timeout(500)
                # Best-effort: trigger IntersectionObserver/in-view lazy loaders on long, JS-heavy pages
                # (e.g., componentized SPAs) by sweeping scroll positions after zooming out.
                try:
                    scroll_height = await page.evaluate("document.body.scrollHeight")
                    if isinstance(scroll_height, (int, float)) and scroll_height > 0:
                        steps = 12
                        for i in range(steps + 1):
                            y = int((scroll_height * i) / steps)
                            await page.evaluate("(y) => window.scrollTo(0, y)", y)
                            await page.wait_for_timeout(200)
                        await page.evaluate("window.scrollTo(0, 0)")
                        await page.wait_for_timeout(200)
                except Exception as e:
                    logger.warning(f"Scroll sweep failed for {url}: {e}")

            # Add a small delay for any post-action / post-idle rendering.
            await page.wait_for_timeout(1000 if use_actions else 3000)

            final_url = page.url or url
            html_content = await page.content()
            
            screenshot_data = None
            if "screenshot" in (options.formats or []):
                screenshot_bytes = await page.screenshot(full_page=True)
                screenshot_data = base64.b64encode(screenshot_bytes).decode()
            
            document = await self._process_html_content(final_url, html_content, 200, options)
            document.url = final_url
            
            if screenshot_data:
                document.screenshot = screenshot_data
            
            return document
            
        finally:
            await context.close()
    
    async def _scrape_html_page(self, url: str, options: ScrapeOptions) -> FirecrawlDocument:
        # This function contains the previous logic of trying httpx GET then falling back to browser
        if self._needs_browser_rendering(options):
            logger.info(f"Browser rendering required; scraping with browser for {url}")
            return await self._scrape_with_browser(url, options)
        try:
            document = await self._scrape_with_http(url, options, timeout_override=10)
            # Some JS-heavy sites return a "JavaScript required / unsupported browser" shell to non-browser fetches.
            # Detect that and escalate to a real browser render even if the response is not "short".
            raw = (getattr(document, "raw_html", None) or "") if document else ""
            md_text = (document.markdown or "") if document else ""
            js_stub_markers = [
                "JavaScript is required",
                "Please enable Javascript",
                "Please enable JavaScript",
                "we no longer support your browser",
                "Apologies, we no longer support your browser",
                "enable Javascript in order to use",
                "enable JavaScript in order to use",
            ]
            if any(m.lower() in raw.lower() for m in js_stub_markers) or any(
                m.lower() in md_text.lower() for m in js_stub_markers
            ):
                logger.info(f"Detected JS-required/unsupported-browser shell; escalating to browser for {url}")
                return await self._scrape_with_browser(url, options)
            # If HTTP fetch produced some content, avoid escalating to a headless browser
            # for short, simple pages (e.g., example.com) to keep lightweight behavior
            # and reduce dependency on Playwright in constrained environments.
            if not document.markdown or len(document.markdown) < 100:
                logger.info(f"HTML content too short, escalating to browser for {url}")
                return await self._scrape_with_browser(url, options)
            logger.info(f"Successfully scraped HTML with HTTP for {url}")
            return document
        except Exception:
            logger.info(f"Initial HTTP GET failed, escalating to browser for {url}")
            return await self._scrape_with_browser(url, options)

    async def _apply_actions(self, page: Page, actions: List[ScrapeAction], timeout_ms: int) -> None:
        for idx, action in enumerate(actions):
            if action is None:
                continue

            try:
                action_type = (action.type or "").strip().lower()

                if action_type in {"wait", "sleep"}:
                    ms = action.milliseconds if action.milliseconds is not None else 1000
                    await page.wait_for_timeout(int(ms))
                    continue

                if action_type in {"waitforselector", "wait_for_selector"}:
                    if isinstance(action.selector, str) and action.selector.strip():
                        await page.wait_for_selector(action.selector, timeout=timeout_ms)
                    continue

                if action_type == "click":
                    if isinstance(action.selector, str) and action.selector.strip():
                        await page.click(action.selector, timeout=timeout_ms)
                    continue

                if action_type in {"type", "fill", "write"}:
                    if isinstance(action.selector, str) and action.selector.strip():
                        text = action.text if isinstance(action.text, str) else ""
                        await page.fill(action.selector, text, timeout=timeout_ms)
                    continue

                if action_type in {"press", "keypress"}:
                    if isinstance(action.key, str) and action.key.strip():
                        if isinstance(action.selector, str) and action.selector.strip():
                            await page.press(action.selector, action.key, timeout=timeout_ms)
                        else:
                            await page.keyboard.press(action.key)
                    continue

                if action_type == "scroll":
                    await self._scroll_page(page, action.direction, action.milliseconds)
                    continue

                if action_type in {"evaluate", "script", "executejavascript", "execute_javascript"}:
                    script = action.script
                    if isinstance(script, str) and script.strip():
                        await page.evaluate(script)
                    continue

                if action_type in {"scrape", "screenshot", "generatepdf", "generate_pdf"}:
                    # These are Firecrawl v2 action types. This server always performs a final scrape
                    # at the end of /scrape, and supports screenshots via formats. PDF generation
                    # isn't implemented yet, so treat these as no-ops for now.
                    continue

                logger.warning(f"Unknown action type '{action.type}' at index {idx}; ignoring.")
            except Exception as e:
                logger.warning(f"Action {idx} failed ({getattr(action, 'type', None)}): {e}")

    async def _scroll_page(self, page: Page, direction: Optional[str], wait_ms: Optional[int]) -> None:
        d = (direction or "down").strip().lower()
        if d in {"top", "up"}:
            await page.evaluate("window.scrollTo(0, 0)")
        elif d in {"bottom", "down"}:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif d == "left":
            await page.evaluate("window.scrollTo(0, window.scrollY)")
        elif d == "right":
            await page.evaluate("window.scrollTo(document.body.scrollWidth, window.scrollY)")
        else:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        if wait_ms is not None:
            await page.wait_for_timeout(int(wait_ms))

    async def _get_file_metadata(self, url: str, headers: httpx.Headers, options: ScrapeOptions) -> FirecrawlDocument:
        content_type = headers.get('content-type', '').lower()
        content_length = headers.get('content-length')
        
        file_meta = {"content_type": content_type, "content_length": content_length}
        
        if content_type == 'application/pdf':
            page_count = await self._get_pdf_page_count_fast(url, options.headers)
            file_meta['page_count'] = page_count

        if content_type.startswith('image/'):
            dimensions = await self._get_image_dimensions_fast(url, options.headers)
            if dimensions:
                file_meta['width'], file_meta['height'] = dimensions

        doc_meta = DocumentMetadata(source_url=url, status_code=200, file_metadata=file_meta, content_type=content_type)
        document = FirecrawlDocument(url=url, metadata=doc_meta)

        if options.include_file_body:
            logger.info(f"Downloading full file body for {url}")
            http = ResilientHttpClient(timeout=options.timeout or settings.default_timeout)
            full_response = await http.get(url, headers={"User-Agent": settings.user_agent, **(options.headers or {})})
            document.content_base64 = base64.b64encode(await full_response.aread()).decode()

        return document

    async def _get_pdf_page_count_fast(self, url: str, req_headers: Optional[dict]) -> int:
        # First attempt: Fast ranged request
        try:
            logger.info(f"Attempting fast ranged request for PDF page count: {url}")
            # Increased range significantly for better compatibility with non-standard PDFs
            headers = {"User-Agent": settings.user_agent, **(req_headers or {}), "Range": "bytes=-4192"} # 4KB
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(url, headers=headers)
                if 200 <= response.status_code < 300:
                    pdf_stream = io.BytesIO(response.content)
                    # Run synchronous pypdf in a thread to avoid blocking
                    reader = await asyncio.to_thread(PdfReader, pdf_stream, strict=False)
                    return len(reader.pages)
                else:
                    logger.warning(f"Ranged request for PDF page count for {url} failed with status {response.status_code}. Returning -1.")
        except Exception as e:
            logger.warning(f"Ranged request for PDF page count failed for {url}: {e}. Returning -1.")
        
        return -1

    async def _get_image_dimensions_fast(self, url: str, req_headers: Optional[dict]) -> Optional[tuple[int, int]]:
        try:
            headers = {"User-Agent": settings.user_agent, **(req_headers or {}), "Range": "bytes=0-4096"}
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                if 200 <= response.status_code < 300:
                    image_stream = io.BytesIO(response.content)
                    with Image.open(image_stream) as img:
                        return img.size
        except Exception as e:
            logger.warning(f"Could not get image dimensions for {url} via ranged request: {e}")
        return None

    def _extract_metadata(self, soup: BeautifulSoup, url: str, status_code: Optional[int]) -> DocumentMetadata:
        title = None
        description = None
        language = None
        
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
        
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag:
            description = desc_tag.get("content", "").strip()
        
        html_tag = soup.find("html")
        if html_tag:
            language = html_tag.get("lang")
        
        return DocumentMetadata(
            title=title,
            description=description,
            language=language,
            source_url=url,
            status_code=status_code,
            content_type="text/html"
        )
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        # A more intelligent, additive approach to finding the main content
        
        # Heuristics for finding the main content container, with Wikipedia's specific one first
        main_content_selectors = ["#mw-content-text", "main", "article", "[role='main']", "#main", "#content"]
        
        main_container = None
        for selector in main_content_selectors:
            main_container = soup.select_one(selector)
            if main_container:
                logger.info(f"Found main content container with selector: {selector}")
                break
                
        if main_container:
            # If we found a main container, use it as the new soup
            # We create a new soup object to avoid modifying the original during iteration
            new_soup = BeautifulSoup(str(main_container), "html.parser")
            # Still remove script/style tags from the extracted content
            for tag in new_soup(["script", "style"]):
                tag.decompose()
            return new_soup
        else:
            # Fallback to the old subtractive method if no main container is found
            logger.warning("No main content container found, falling back to subtractive method.")
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            selectors_to_remove = [
                '[class*="nav"]', '[class*="sidebar"]', '[class*="menu"]',
                '[class*="footer"]', '[class*="header"]', '.skip-link', '[href="#content"]'
            ]
            
            for selector in selectors_to_remove:
                for element in soup.select(selector):
                    element.decompose()
            
            return soup

    async def _process_html_content(self, url: str, html: str, status_code: Optional[int], options: ScrapeOptions) -> FirecrawlDocument:
        soup = BeautifulSoup(html, "html.parser")
        
        # Store the original, unprocessed soup for raw_html
        raw_html_soup = html if "rawHtml" in (options.formats or []) else None
        
        # Content filtering happens FIRST
        if options.include_tags:
            logger.info(f"Using include_tags (whitelist mode): {options.include_tags}")
            new_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
            body = new_soup.body
            for selector in options.include_tags:
                for element in soup.select(selector):
                    body.append(element)
            soup = new_soup
        else:
            if options.only_main_content:
                 soup = self._extract_main_content(soup)
            
            if options.exclude_tags:
                logger.info(f"Using exclude_tags (blacklist mode): {options.exclude_tags}")
                for selector in options.exclude_tags:
                    for element in soup.select(selector):
                        element.decompose()

        if options.remove_base64_images:
            soup = self._remove_base64_images(soup)

        # Remove non-content tags that frequently pollute markdown output.
        # (We still preserve the original `rawHtml` via `raw_html_soup` when requested.)
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        # Absolutize links before final processing, unless instructed otherwise
        if not options.use_relative_links:
            soup = self._absolutize_links(soup, url)

        # Metadata extraction and document creation happens LAST, on the processed soup
        metadata = self._extract_metadata(soup, url, status_code)
        # When using include_tags, we should not return any metadata from the original page
        if options.include_tags:
            metadata.title = None
            metadata.description = None
            metadata.language = None

        document = FirecrawlDocument(url=url, metadata=metadata)
        formats = options.formats or ["markdown"]
        
        if "html" in formats:
            document.html = str(soup)
        
        if raw_html_soup:
            document.raw_html = raw_html_soup
        
        if "markdown" in formats or "content" in formats:
            markdown_content = self._html_to_markdown(str(soup))
            document.markdown = markdown_content
        
        if "links" in formats:
            # Links should probably be extracted from the processed soup to be relevant
            document.links = self._extract_links(soup, url)
        
        return document

    def _absolutize_links(self, soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
        """
        Converts all relative hrefs in <a> tags to absolute URLs.
        """
        for a_tag in soup.find_all("a", href=True):
            a_tag["href"] = urljoin(base_url, a_tag["href"])
        return soup

    def _remove_base64_images(self, soup: BeautifulSoup) -> BeautifulSoup:
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src.startswith("data:image"):
                img.decompose()
        return soup
    
    def _html_to_markdown(self, html: str) -> str:
        markdown_content = md(
            html,
            heading_style="ATX",
            bullets="-",
            strong_em_style="**",
            strip=["script", "style"]
        )
        
        # Post-processing - remove problematic content
        # Remove base64 encoded images that might have slipped through
        markdown_content = re.sub(r'!\[.*?\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}\)', 
                                  '![Image content removed - base64 encoded]', 
                                  markdown_content)
        
        # Remove skip to content links
        markdown_content = re.sub(r'\[Skip to Content\]\(#[^\)]*\)', "", markdown_content, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace
        markdown_content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
        markdown_content = markdown_content.strip()
        
        return markdown_content
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(base_url, href)
            if absolute_url not in links:
                links.append(absolute_url)
        return links

# Global scraping service instance
scraper = ScrapingService() 
