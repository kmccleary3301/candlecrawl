import pytest

from app.providers.base import ProviderError
from app.providers.document_ocr import DocumentOCRClient, OCRExtractionResult, _extract_markdown_from_payload


def test_extract_markdown_from_payload_handles_pages_and_nested_shapes():
    payload = {
        "pages": [
            {"markdown": "First page"},
            {"data": {"text": "Second page"}},
        ]
    }
    text = _extract_markdown_from_payload(payload)
    assert text is not None
    assert "First page" in text
    assert "Second page" in text


def test_resolve_provider_auto_prefers_configured_backends(monkeypatch):
    client = DocumentOCRClient()
    monkeypatch.setattr("app.providers.document_ocr.settings.document_ocr_provider_default", "auto")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", "http://localhost:9000")
    monkeypatch.setattr("app.providers.document_ocr.settings.mistral_api_key", None)
    assert client.resolve_provider() == "querylake_chandra"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", None)
    monkeypatch.setattr("app.providers.document_ocr.settings.mistral_api_key", "mistral-key")
    assert client.resolve_provider() == "mistral"

    monkeypatch.setattr("app.providers.document_ocr.settings.mistral_api_key", None)
    assert client.resolve_provider() == "none"


@pytest.mark.asyncio
async def test_extract_markdown_rejects_when_provider_disabled(monkeypatch):
    client = DocumentOCRClient()
    monkeypatch.setattr("app.providers.document_ocr.settings.document_ocr_provider_default", "none")
    with pytest.raises(ProviderError):
        await client.extract_markdown(
            file_bytes=b"abc",
            content_type="application/pdf",
            source_url="https://example.com/a.pdf",
        )


class _StubResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("No JSON payload")
        return self._payload


def _stub_async_client(scripted_calls):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            assert scripted_calls, f"Unexpected POST call to {url}"
            expected = scripted_calls.pop(0)
            if "url" in expected:
                assert url == expected["url"]
            return _StubResponse(
                status_code=expected.get("status_code", 200),
                payload=expected.get("json"),
                text=expected.get("text", ""),
            )

    return _Client


@pytest.mark.asyncio
async def test_querylake_auto_falls_back_to_kernel_files(monkeypatch):
    client = DocumentOCRClient()
    base = "http://querylake.local"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", base)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_mode", "auto")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_path", "/v1/chandra/ocr")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_auth_token", "test-oauth-token")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_search_limit", 50)

    scripted = [
        {"url": f"{base}/v1/chandra/ocr", "status_code": 404, "text": "not found"},
        {"url": f"{base}/v2/kernel/files", "json": {"success": True, "file_id": "f1", "version_id": "v1"}},
        {"url": f"{base}/files/f1/versions/v1/process", "json": {"success": True, "status": "COMPLETED"}},
        {
            "url": f"{base}/v2/kernel/api/search_file_chunks",
            "json": {
                "success": True,
                "result": {
                    "results": [
                        {"file_version_id": "v1", "text": "OCR chunk text", "created_at": 10.0},
                        {"file_version_id": "other", "text": "ignore", "created_at": 11.0},
                    ]
                },
            },
        },
    ]
    monkeypatch.setattr("app.providers.document_ocr.httpx.AsyncClient", _stub_async_client(scripted))

    result = await client.extract_markdown(
        file_bytes=b"%PDF-1.4 demo",
        content_type="application/pdf",
        requested_provider="querylake_chandra",
    )
    assert result.provider == "querylake_chandra"
    assert result.markdown == "OCR chunk text"
    assert result.details is not None
    assert result.details.get("mode") == "querylake_files"


@pytest.mark.asyncio
async def test_querylake_kernel_files_bootstraps_user_when_login_fails(monkeypatch):
    client = DocumentOCRClient()
    base = "http://querylake.local"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", base)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_mode", "querylake_files")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_auth_token", None)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_username", "ocr_user")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_password", "ocr_password")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_auto_create_user", True)

    scripted = [
        {"url": f"{base}/v2/kernel/api/login", "json": {"success": False, "error": "User Not Found"}},
        {"url": f"{base}/v2/kernel/api/add_user", "json": {"success": True, "result": {"auth": "oauth-created"}}},
        {"url": f"{base}/v2/kernel/files", "json": {"success": True, "file_id": "f2", "version_id": "v2"}},
        {"url": f"{base}/files/f2/versions/v2/process", "json": {"success": True, "status": "COMPLETED"}},
        {
            "url": f"{base}/v2/kernel/api/search_file_chunks",
            "json": {"success": True, "result": {"results": [{"file_version_id": "v2", "text": "text from ql"}]}},
        },
    ]
    monkeypatch.setattr("app.providers.document_ocr.httpx.AsyncClient", _stub_async_client(scripted))

    result = await client.extract_markdown(
        file_bytes=b"%PDF-1.4 demo",
        content_type="application/pdf",
        requested_provider="querylake_chandra",
    )
    assert result.markdown == "text from ql"
    assert getattr(client, "_querylake_cached_oauth2") == "oauth-created"


@pytest.mark.asyncio
async def test_querylake_kernel_files_requires_auth_material(monkeypatch):
    client = DocumentOCRClient()
    base = "http://querylake.local"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", base)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_mode", "querylake_files")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_auth_token", None)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_username", None)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_password", None)
    monkeypatch.setattr("app.providers.document_ocr.httpx.AsyncClient", _stub_async_client([]))

    with pytest.raises(ProviderError, match="credentials missing"):
        await client.extract_markdown(
            file_bytes=b"%PDF-1.4 demo",
            content_type="application/pdf",
            requested_provider="querylake_chandra",
        )


@pytest.mark.asyncio
async def test_querylake_kernel_files_process_path_falls_back_on_404(monkeypatch):
    client = DocumentOCRClient()
    base = "http://querylake.local"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", base)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_mode", "querylake_files")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_auth_token", "oauth-token")
    monkeypatch.setattr(
        "app.providers.document_ocr.settings.querylake_chandra_ocr_files_process_path_template",
        "/v2/kernel/files/{file_id}/versions/{version_id}/process",
    )

    scripted = [
        {"url": f"{base}/v2/kernel/files", "json": {"success": True, "file_id": "f3", "version_id": "v3"}},
        {"url": f"{base}/v2/kernel/files/f3/versions/v3/process", "status_code": 404, "text": "not found"},
        {"url": f"{base}/files/f3/versions/v3/process", "json": {"success": True, "status": "COMPLETED"}},
        {
            "url": f"{base}/v2/kernel/api/search_file_chunks",
            "json": {"success": True, "result": {"results": [{"file_version_id": "v3", "text": "fallback chunk"}]}},
        },
    ]
    monkeypatch.setattr("app.providers.document_ocr.httpx.AsyncClient", _stub_async_client(scripted))

    result = await client.extract_markdown(
        file_bytes=b"%PDF-1.4 demo",
        content_type="application/pdf",
        requested_provider="querylake_chandra",
    )

    assert result.markdown == "fallback chunk"
    assert result.details is not None
    assert result.details.get("process_path") == "/files/f3/versions/v3/process"


@pytest.mark.asyncio
async def test_querylake_falls_back_to_mistral_when_enabled(monkeypatch):
    client = DocumentOCRClient()
    base = "http://querylake.local"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", base)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_mode", "direct")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_path", "/v1/chandra/ocr")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_enable_mistral_fallback", True)
    monkeypatch.setattr("app.providers.document_ocr.settings.mistral_api_key", "mistral-key")

    scripted = [
        {"url": f"{base}/v1/chandra/ocr", "status_code": 503, "text": "upstream unavailable"},
    ]
    monkeypatch.setattr("app.providers.document_ocr.httpx.AsyncClient", _stub_async_client(scripted))

    async def _fake_mistral(**kwargs):
        return OCRExtractionResult(
            provider="mistral",
            markdown="fallback markdown",
            details={"model": "mistral-ocr-latest"},
        )

    monkeypatch.setattr(client, "_extract_with_mistral", _fake_mistral)

    result = await client.extract_markdown(
        file_bytes=b"%PDF-1.4 demo",
        content_type="application/pdf",
        requested_provider="querylake_chandra",
    )

    assert result.provider == "mistral"
    assert result.markdown == "fallback markdown"
    assert result.details is not None
    assert result.details.get("fallback_from") == "querylake_chandra"
    assert "querylake_error" in result.details


@pytest.mark.asyncio
async def test_querylake_mistral_fallback_reports_combined_failure(monkeypatch):
    client = DocumentOCRClient()
    base = "http://querylake.local"

    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_base_url", base)
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_mode", "direct")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_path", "/v1/chandra/ocr")
    monkeypatch.setattr("app.providers.document_ocr.settings.querylake_chandra_ocr_enable_mistral_fallback", True)
    monkeypatch.setattr("app.providers.document_ocr.settings.mistral_api_key", "mistral-key")

    scripted = [
        {"url": f"{base}/v1/chandra/ocr", "status_code": 502, "text": "bad gateway"},
    ]
    monkeypatch.setattr("app.providers.document_ocr.httpx.AsyncClient", _stub_async_client(scripted))

    async def _fake_mistral_fail(**kwargs):
        raise ProviderError("Mistral OCR error", status_code=500, payload="provider down")

    monkeypatch.setattr(client, "_extract_with_mistral", _fake_mistral_fail)

    with pytest.raises(ProviderError, match="Mistral fallback failed"):
        await client.extract_markdown(
            file_bytes=b"%PDF-1.4 demo",
            content_type="application/pdf",
            requested_provider="querylake_chandra",
        )
