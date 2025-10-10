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


