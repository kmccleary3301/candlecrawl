import pytest
import httpx
from unittest.mock import AsyncMock, patch
from app.http_client import ResilientHttpClient


@pytest.mark.asyncio
async def test_resilient_http_client_retries_on_429():
    client = ResilientHttpClient(timeout=2)
    mock_resp_429 = httpx.Response(429, request=httpx.Request("GET", "https://x"), headers={"Retry-After": "0.01"})
    mock_resp_ok = httpx.Response(200, request=httpx.Request("GET", "https://x"))

    async def fake_get(url, headers=None):
        fake_get.calls += 1
        if fake_get.calls == 1:
            return mock_resp_429
        return mock_resp_ok
    fake_get.calls = 0

    with patch("httpx.AsyncClient.get", new=AsyncMock(side_effect=fake_get)):
        resp = await client.get("https://x")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_resilient_http_client_passes_verify_flag():
    created_kwargs = []

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            created_kwargs.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, headers=None):
            return httpx.Response(200, request=httpx.Request("GET", url))

    with patch("app.http_client.httpx.AsyncClient", DummyAsyncClient):
        client = ResilientHttpClient(timeout=2, verify=False)
        resp = await client.get("https://x")
        assert resp.status_code == 200

    assert created_kwargs, "Expected AsyncClient to be instantiated"
    assert created_kwargs[-1]["verify"] is False

