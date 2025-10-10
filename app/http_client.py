import asyncio
import random
from typing import Optional
import httpx
from app.config import settings


class ResilientHttpClient:
    """HTTP client wrapper adding retries with exponential backoff and jitter.

    This is modular and can be swapped later with provider-backed clients or proxies.
    """

    def __init__(self, *, timeout: Optional[int] = None, follow_redirects: bool = True):
        self.timeout = timeout or settings.default_timeout
        self.follow_redirects = follow_redirects

    def _compute_backoff(self, attempt: int) -> float:
        base = settings.backoff_base_ms / 1000.0
        cap = settings.backoff_max_ms / 1000.0
        exp = min(cap, base * (2 ** (attempt - 1)))
        jitter = random.uniform(0, exp / 2)
        return exp + jitter

    def _should_retry(self, exc: Exception, resp: Optional[httpx.Response]) -> bool:
        if resp is not None and resp.status_code in (429, 500, 502, 503, 504):
            return True
        # networky errors
        return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteError, httpx.PoolTimeout))

    async def get(self, url: str, *, headers: Optional[dict] = None) -> httpx.Response:
        attempts = settings.retry_max_attempts
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=self.follow_redirects) as client:
            for attempt in range(1, attempts + 1):
                resp: Optional[httpx.Response] = None
                try:
                    resp = await client.get(url, headers=headers)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        # honor Retry-After if present
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                delay = float(ra)
                                await asyncio.sleep(delay)
                            except Exception:
                                pass
                        raise httpx.HTTPStatusError("retryable status", request=resp.request, response=resp)
                    return resp
                except Exception as e:
                    last_exc = e
                    if attempt >= attempts or not self._should_retry(e, resp):
                        raise
                    await asyncio.sleep(self._compute_backoff(attempt))

    async def head(self, url: str, *, headers: Optional[dict] = None) -> httpx.Response:
        attempts = settings.retry_max_attempts
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=self.follow_redirects) as client:
            for attempt in range(1, attempts + 1):
                resp: Optional[httpx.Response] = None
                try:
                    resp = await client.head(url, headers=headers)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                delay = float(ra)
                                await asyncio.sleep(delay)
                            except Exception:
                                pass
                        raise httpx.HTTPStatusError("retryable status", request=resp.request, response=resp)
                    return resp
                except Exception as e:
                    last_exc = e
                    if attempt >= attempts or not self._should_retry(e, resp):
                        raise
                    await asyncio.sleep(self._compute_backoff(attempt))




