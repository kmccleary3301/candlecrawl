from __future__ import annotations

from typing import Optional, Tuple
from urllib.parse import quote

import httpx
from pydantic import BaseModel, HttpUrl, Field

from app.config import settings
from app.providers.base import ProviderError


class ScrapeDoRequest(BaseModel):
    url: HttpUrl
    render_js: bool = False
    timeout_ms: int = Field(default=60000, ge=5000, le=120000)
    geo_code: Optional[str] = None  # e.g., 'us'


class ScrapeDoResponse(BaseModel):
    content: str
    status_code: int
    content_type: Optional[str] = None
    # Cost tracking fields  
    credits_used: int = Field(default=1)
    estimated_cost: float = Field(default=0.000116)  # Hobby plan rate


class ScrapeDoClient:
    def __init__(self, *, api_key: Optional[str] = None, base_url: Optional[str] = None, client: Optional[httpx.AsyncClient] = None):
        self.api_key = api_key or settings.scrape_do_api_key
        self.base_url = (base_url or settings.scrapedo_base_url).rstrip("/")
        self._client = client

    async def fetch(self, req: ScrapeDoRequest) -> Tuple[ScrapeDoResponse, float]:
        if not self.api_key:
            raise ProviderError("Scrape.do API key not configured")

        # API mode: https://api.scrape.do/?token=KEY&url=ENCODED&render=true&timeout=ms&geoCode=us
        params = [
            ("token", self.api_key),
            ("url", quote(str(req.url), safe="")),
        ]
        if req.render_js:
            params.append(("render", "true"))
        if req.timeout_ms:
            params.append(("timeout", str(req.timeout_ms)))
        if req.geo_code:
            params.append(("geoCode", req.geo_code))

        query = "&".join([f"{k}={v}" for k, v in params])
        url = f"{self.base_url}/?{query}"

        created_client: Optional[httpx.AsyncClient] = None
        client = self._client
        if client is None:
            created_client = httpx.AsyncClient(timeout=req.timeout_ms / 1000.0)
            client = created_client
        try:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code >= 400:
                raise ProviderError("Scrape.do error", status_code=resp.status_code, payload=resp.text)
            response = ScrapeDoResponse(
                content=resp.text,
                status_code=resp.status_code,
                content_type=resp.headers.get("content-type"),
            )
            # Configurable fixed cost per successful scrape
            from app.config import settings as _settings
            cost = float(getattr(_settings, "cost_scrapedo_per_request_usd", 0.000116))
            return response, cost
        finally:
            if created_client:
                await created_client.aclose()

