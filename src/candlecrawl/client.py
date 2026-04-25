from __future__ import annotations

from typing import Any

import httpx

from candlecrawl.errors import (
    CandleCrawlAPIError,
    CandleCrawlAuthError,
    CandleCrawlNotFoundError,
    CandleCrawlRateLimitError,
    CandleCrawlTimeoutError,
    CandleCrawlTransportError,
)
from candlecrawl.schemas import HealthResponse
from candlecrawl.trace import TraceContext


class AsyncCandleCrawlClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:3010",
        api_key: str | None = None,
        timeout: float = 30.0,
        trace: TraceContext | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.trace = trace or TraceContext()
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "AsyncCandleCrawlClient":
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def health(self) -> HealthResponse:
        payload = await self.request_json("GET", "/health")
        return HealthResponse.model_validate(payload)

    async def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        close_after = self._client is None
        try:
            response = await client.request(
                method,
                self.base_url + "/" + path.lstrip("/"),
                headers=self._headers(kwargs.pop("headers", None)),
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise CandleCrawlTimeoutError(str(exc), retryable=True) from exc
        except httpx.HTTPError as exc:
            raise CandleCrawlTransportError(str(exc), retryable=True) from exc
        finally:
            if close_after:
                await client.aclose()

        if response.status_code >= 400:
            raise _api_error_from_response(response)
        return response.json()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        headers.update(self.trace.headers())
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers


def _api_error_from_response(response: httpx.Response) -> CandleCrawlAPIError:
    try:
        payload: Any = response.json()
    except ValueError:
        payload = response.text

    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("error")
    else:
        message = None
    message = message or f"CandleCrawl API error: HTTP {response.status_code}"
    kwargs = {
        "message": message,
        "status_code": response.status_code,
        "payload": payload,
        "trace_id": response.headers.get("X-Trace-Id"),
        "retryable": response.status_code in {429, 500, 502, 503, 504},
    }
    if response.status_code in {401, 403}:
        return CandleCrawlAuthError(**kwargs)
    if response.status_code == 404:
        return CandleCrawlNotFoundError(**kwargs)
    if response.status_code == 429:
        return CandleCrawlRateLimitError(**kwargs)
    return CandleCrawlAPIError(**kwargs)
