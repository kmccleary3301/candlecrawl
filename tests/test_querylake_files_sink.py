from __future__ import annotations

import json

import httpx
import pytest

from app.querylake_files_sink import QueryLakeFilesArtifactSink, QueryLakeFilesSinkConfig


@pytest.mark.asyncio
async def test_querylake_files_sink_upload_success_returns_artifact_ref() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"success": True, "document_id": "ql-doc-1"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sink = QueryLakeFilesArtifactSink(
            config=QueryLakeFilesSinkConfig(
                base_url="https://querylake.local",
                auth_token="service-token",
                retry_base_delay_ms=0,
            ),
            client=client,
        )
        ref = await sink.put_bytes(
            content=b"# hello",
            filename="doc.md",
            content_type="text/markdown",
            trace_headers={"X-Request-Id": "req-123", "X-Audit-Id": "aud-123", "X-Tenant-Id": "tenant-a"},
        )

    assert ref.mode == "querylake_files"
    assert ref.uri == "querylake://files/ql-doc-1"
    assert ref.content_type == "text/markdown"
    assert ref.size_bytes == len(b"# hello")
    assert captured["url"] == "https://querylake.local/api/files/ingest_markdown"
    assert captured["headers"]["x-service-token"] == "service-token"
    assert captured["headers"]["x-idempotency-key"].startswith("qlf:")
    assert captured["headers"]["x-request-id"] == "req-123"
    assert captured["headers"]["x-audit-id"] == "aud-123"
    assert captured["headers"]["x-tenant-id"] == "tenant-a"
    assert captured["payload"]["collection_id"] == "candlecrawl_artifacts"
    assert captured["payload"]["create_embeddings"] is False


@pytest.mark.asyncio
async def test_querylake_files_sink_retries_on_upstream_5xx() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(200, json={"success": True, "id": "ql-doc-retry"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sink = QueryLakeFilesArtifactSink(
            config=QueryLakeFilesSinkConfig(
                base_url="https://querylake.local",
                max_retries=2,
                retry_base_delay_ms=0,
            ),
            client=client,
        )
        ref = await sink.put_bytes(content=b"retry me")

    assert calls["count"] == 2
    assert ref.uri == "querylake://files/ql-doc-retry"


@pytest.mark.asyncio
async def test_querylake_files_sink_retries_on_transport_error() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"success": True, "document_id": "ql-doc-transport"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sink = QueryLakeFilesArtifactSink(
            config=QueryLakeFilesSinkConfig(
                base_url="https://querylake.local",
                max_retries=1,
                retry_base_delay_ms=0,
            ),
            client=client,
        )
        ref = await sink.put_bytes(content=b"transport")

    assert calls["count"] == 2
    assert ref.uri == "querylake://files/ql-doc-transport"


@pytest.mark.asyncio
async def test_querylake_files_sink_does_not_retry_4xx() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sink = QueryLakeFilesArtifactSink(
            config=QueryLakeFilesSinkConfig(
                base_url="https://querylake.local",
                max_retries=3,
                retry_base_delay_ms=0,
            ),
            client=client,
        )
        with pytest.raises(RuntimeError):
            await sink.put_bytes(content=b"bad")

    assert calls["count"] == 1
