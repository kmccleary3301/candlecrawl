from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any

import httpx

from candlecrawl._server.artifacts import ArtifactRef


@dataclass(frozen=True)
class QueryLakeFilesSinkConfig:
    base_url: str
    ingestion_path: str = "/api/files/ingest_markdown"
    auth_header: str = "X-Service-Token"
    auth_token: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_base_delay_ms: int = 200
    tenant_id: str = "default"
    collection_id: str = "candlecrawl_artifacts"
    source: str = "candlecrawl"


class QueryLakeFilesArtifactSink:
    mode = "querylake_files"

    def __init__(
        self,
        *,
        config: QueryLakeFilesSinkConfig,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self._client = client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _build_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/{self.config.ingestion_path.lstrip('/')}"

    def _build_headers(self, idempotency_key: str, trace_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "X-Idempotency-Key": idempotency_key}
        if self.config.auth_token:
            headers[self.config.auth_header] = self.config.auth_token
        if trace_headers:
            for header_name in ("X-Request-Id", "X-Audit-Id", "X-Tenant-Id"):
                value = trace_headers.get(header_name) or trace_headers.get(header_name.lower())
                if isinstance(value, str) and value.strip():
                    headers[header_name] = value.strip()
        return headers

    @staticmethod
    def _backoff_seconds(attempt: int, base_delay_ms: int) -> float:
        return max(0.0, (base_delay_ms / 1000.0) * (2**attempt))

    async def put_bytes(
        self,
        *,
        content: bytes,
        filename: str | None = None,
        content_type: str | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        doc_id = f"cc-{digest[:32]}"
        idempotency_key = f"qlf:{self.config.tenant_id}:{self.config.collection_id}:{doc_id}"
        payload: dict[str, Any] = {
            "tenant_id": self.config.tenant_id,
            "collection_id": self.config.collection_id,
            "document_id": doc_id,
            "source_url": None,
            "title": filename or doc_id,
            "content_markdown": content.decode("utf-8", errors="replace"),
            "metadata": {
                "artifact_sha256": digest,
                "artifact_content_type": content_type,
                "artifact_filename": filename,
                "artifact_source": self.config.source,
                "artifact_mode": self.mode,
            },
            "provenance": {"source": self.config.source},
            "create_embeddings": False,
            "idempotency_key": idempotency_key,
        }
        headers = self._build_headers(idempotency_key, trace_headers=trace_headers)
        url = self._build_url()

        created_client: httpx.AsyncClient | None = None
        client = self._client
        if client is None:
            created_client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
            client = created_client
        try:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    if attempt >= self.config.max_retries:
                        raise RuntimeError(f"QueryLake upload transport error: {exc}") from exc
                    await asyncio.sleep(self._backoff_seconds(attempt, self.config.retry_base_delay_ms))
                    continue

                if response.status_code >= 500 and attempt < self.config.max_retries:
                    await asyncio.sleep(self._backoff_seconds(attempt, self.config.retry_base_delay_ms))
                    continue

                if response.status_code >= 400:
                    detail = response.text[:400]
                    raise RuntimeError(f"QueryLake upload failed with status {response.status_code}: {detail}")

                data = response.json() if response.content else {}
                success = bool(data.get("success", True))
                upstream_id = data.get("document_id") or data.get("id") or doc_id
                if not success:
                    raise RuntimeError(str(data.get("error") or "QueryLake upload failed"))
                if not isinstance(upstream_id, str) or not upstream_id.strip():
                    raise RuntimeError("QueryLake upload returned invalid document identifier")

                return ArtifactRef(
                    mode=self.mode,
                    uri=f"querylake://files/{upstream_id.strip()}",
                    content_type=content_type,
                    size_bytes=len(content),
                )

            raise RuntimeError("QueryLake upload failed after retries")
        finally:
            if created_client is not None:
                await created_client.aclose()
