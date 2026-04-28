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
from candlecrawl.schemas import (
    ActionsCapabilitiesResponse,
    BatchScrapeRequest,
    CancelResponse,
    CrawlRequest,
    ExtractRequest,
    HealthResponse,
    JobCreateResponse,
    JobErrorsResponse,
    JobStatusResponse,
    MapRequest,
    MapResponse,
    ScrapeRequest,
    ScrapeResponse,
    SearchRequest,
    SearchResponse,
)
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

    async def actions_capabilities(self) -> ActionsCapabilitiesResponse:
        payload = await self.request_json("GET", "/v2/actions/capabilities")
        return ActionsCapabilitiesResponse.model_validate(payload)

    async def scrape(self, request: ScrapeRequest | dict[str, Any]) -> ScrapeResponse:
        payload = _payload(request)
        response = await self.request_json("POST", "/v2/scrape", json=payload)
        return ScrapeResponse.model_validate(response)

    async def map(self, request: MapRequest | dict[str, Any]) -> MapResponse:
        payload = _payload(request)
        response = await self.request_json("POST", "/v2/map", json=payload)
        return MapResponse.model_validate(response)

    async def search(self, request: SearchRequest | dict[str, Any]) -> SearchResponse:
        payload = _payload(request)
        response = await self.request_json("POST", "/v2/search", json=payload)
        return SearchResponse.model_validate(response)

    async def crawl(self, request: CrawlRequest | dict[str, Any]) -> JobCreateResponse:
        payload = _payload(request)
        response = await self.request_json("POST", "/v2/crawl", json=payload)
        return JobCreateResponse.model_validate(response)

    async def get_crawl(self, job_id: str, *, cursor: str | None = None) -> JobStatusResponse:
        params = {"cursor": cursor} if cursor else None
        response = await self.request_json("GET", f"/v2/crawl/{job_id}", params=params)
        return JobStatusResponse.model_validate(response)

    async def cancel_crawl(self, job_id: str) -> CancelResponse:
        response = await self.request_json("DELETE", f"/v2/crawl/{job_id}")
        return CancelResponse.model_validate(response)

    async def crawl_errors(self, job_id: str) -> JobErrorsResponse:
        response = await self.request_json("GET", f"/v2/crawl/{job_id}/errors")
        return JobErrorsResponse.model_validate(response)

    async def batch_scrape(self, request: BatchScrapeRequest | dict[str, Any]) -> JobCreateResponse:
        payload = _payload(request)
        response = await self.request_json("POST", "/v2/batch/scrape", json=payload)
        return JobCreateResponse.model_validate(response)

    async def get_batch(self, job_id: str, *, cursor: str | None = None) -> JobStatusResponse:
        params = {"cursor": cursor} if cursor else None
        response = await self.request_json("GET", f"/v2/batch/scrape/{job_id}", params=params)
        return JobStatusResponse.model_validate(response)

    async def cancel_batch(self, job_id: str) -> CancelResponse:
        response = await self.request_json("DELETE", f"/v2/batch/scrape/{job_id}")
        return CancelResponse.model_validate(response)

    async def batch_errors(self, job_id: str) -> JobErrorsResponse:
        response = await self.request_json("GET", f"/v2/batch/scrape/{job_id}/errors")
        return JobErrorsResponse.model_validate(response)

    async def extract(self, request: ExtractRequest | dict[str, Any]) -> JobCreateResponse:
        payload = _payload(request)
        response = await self.request_json("POST", "/v2/extract", json=payload)
        return JobCreateResponse.model_validate(response)

    async def get_extract(self, job_id: str) -> JobStatusResponse:
        response = await self.request_json("GET", f"/v2/extract/{job_id}")
        return JobStatusResponse.model_validate(response)

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


class CandleCrawlClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:3010",
        api_key: str | None = None,
        timeout: float = 30.0,
        trace: TraceContext | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.trace = trace or TraceContext()
        self._client = client or httpx.Client(timeout=self.timeout)
        self._owns_client = client is None

    def __enter__(self) -> "CandleCrawlClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self.request_json("GET", "/health"))

    def actions_capabilities(self) -> ActionsCapabilitiesResponse:
        return ActionsCapabilitiesResponse.model_validate(
            self.request_json("GET", "/v2/actions/capabilities")
        )

    def scrape(self, request: ScrapeRequest | dict[str, Any]) -> ScrapeResponse:
        return ScrapeResponse.model_validate(self.request_json("POST", "/v2/scrape", json=_payload(request)))

    def map(self, request: MapRequest | dict[str, Any]) -> MapResponse:
        return MapResponse.model_validate(self.request_json("POST", "/v2/map", json=_payload(request)))

    def search(self, request: SearchRequest | dict[str, Any]) -> SearchResponse:
        return SearchResponse.model_validate(self.request_json("POST", "/v2/search", json=_payload(request)))

    def crawl(self, request: CrawlRequest | dict[str, Any]) -> JobCreateResponse:
        return JobCreateResponse.model_validate(self.request_json("POST", "/v2/crawl", json=_payload(request)))

    def get_crawl(self, job_id: str, *, cursor: str | None = None) -> JobStatusResponse:
        params = {"cursor": cursor} if cursor else None
        return JobStatusResponse.model_validate(self.request_json("GET", f"/v2/crawl/{job_id}", params=params))

    def cancel_crawl(self, job_id: str) -> CancelResponse:
        return CancelResponse.model_validate(self.request_json("DELETE", f"/v2/crawl/{job_id}"))

    def crawl_errors(self, job_id: str) -> JobErrorsResponse:
        return JobErrorsResponse.model_validate(self.request_json("GET", f"/v2/crawl/{job_id}/errors"))

    def batch_scrape(self, request: BatchScrapeRequest | dict[str, Any]) -> JobCreateResponse:
        return JobCreateResponse.model_validate(
            self.request_json("POST", "/v2/batch/scrape", json=_payload(request))
        )

    def get_batch(self, job_id: str, *, cursor: str | None = None) -> JobStatusResponse:
        params = {"cursor": cursor} if cursor else None
        return JobStatusResponse.model_validate(
            self.request_json("GET", f"/v2/batch/scrape/{job_id}", params=params)
        )

    def cancel_batch(self, job_id: str) -> CancelResponse:
        return CancelResponse.model_validate(self.request_json("DELETE", f"/v2/batch/scrape/{job_id}"))

    def batch_errors(self, job_id: str) -> JobErrorsResponse:
        return JobErrorsResponse.model_validate(
            self.request_json("GET", f"/v2/batch/scrape/{job_id}/errors")
        )

    def extract(self, request: ExtractRequest | dict[str, Any]) -> JobCreateResponse:
        return JobCreateResponse.model_validate(self.request_json("POST", "/v2/extract", json=_payload(request)))

    def get_extract(self, job_id: str) -> JobStatusResponse:
        return JobStatusResponse.model_validate(self.request_json("GET", f"/v2/extract/{job_id}"))

    def request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                self.base_url + "/" + path.lstrip("/"),
                headers=self._headers(kwargs.pop("headers", None)),
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise CandleCrawlTimeoutError(str(exc), retryable=True) from exc
        except httpx.HTTPError as exc:
            raise CandleCrawlTransportError(str(exc), retryable=True) from exc

        if response.status_code >= 400:
            raise _api_error_from_response(response)
        return response.json()

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        headers.update(self.trace.headers())
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        return headers


def _payload(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    return dict(value)


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
