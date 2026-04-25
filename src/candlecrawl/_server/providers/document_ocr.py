from __future__ import annotations

import base64
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from PIL import Image

from candlecrawl._server.config import settings
from candlecrawl._server.providers.base import ProviderError


def _extract_markdown_from_payload(payload: Any) -> str | None:
    if isinstance(payload, str):
        text = payload.strip()
        return text if text else None

    if isinstance(payload, dict):
        for key in ("markdown", "text_markdown", "text", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        pages = payload.get("pages")
        if isinstance(pages, list):
            rendered_pages: list[str] = []
            for idx, page in enumerate(pages, start=1):
                page_md = _extract_markdown_from_payload(page)
                if not page_md:
                    continue
                rendered_pages.append(f"## Page {idx}\n\n{page_md}")
            joined = "\n\n".join(rendered_pages).strip()
            if joined:
                return joined

        for nested_key in ("data", "result", "output", "document"):
            nested = payload.get(nested_key)
            if nested is None:
                continue
            nested_text = _extract_markdown_from_payload(nested)
            if nested_text:
                return nested_text
        return None

    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            item_text = _extract_markdown_from_payload(item)
            if item_text:
                parts.append(item_text)
        joined = "\n\n".join(parts).strip()
        return joined if joined else None

    return None


def _normalize_provider(raw: Optional[str]) -> str:
    provider = (raw or "").strip().lower()
    aliases = {
        "querylake": "querylake_chandra",
        "querylake-chandra": "querylake_chandra",
        "chandra": "querylake_chandra",
        "mistral_ocr": "mistral",
        "mistral-ocr": "mistral",
        "off": "none",
        "disabled": "none",
    }
    return aliases.get(provider, provider or "none")


def _normalize_querylake_mode(raw: Optional[str]) -> str:
    mode = (raw or "").strip().lower()
    aliases = {
        "legacy": "direct",
        "files": "querylake_files",
        "kernel_files": "querylake_files",
        "querylake-kernel-files": "querylake_files",
    }
    normalized = aliases.get(mode, mode or "auto")
    if normalized not in {"auto", "direct", "querylake_files"}:
        return "auto"
    return normalized


def _join_url(base_url: str, path: str) -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{base_url.rstrip('/')}{normalized_path}"


def _extract_querylake_token(raw: Optional[str]) -> Optional[str]:
    token = (raw or "").strip()
    if not token:
        return None
    if token.lower().startswith("bearer "):
        return token.split(" ", 1)[1].strip()
    return token


@dataclass(frozen=True)
class OCRExtractionResult:
    provider: str
    markdown: str
    details: dict[str, Any] | None = None


class DocumentOCRClient:
    def __init__(self) -> None:
        self._querylake_cached_oauth2: Optional[str] = None

    def resolve_provider(self, requested_provider: Optional[str] = None) -> str:
        provider = _normalize_provider(requested_provider or settings.document_ocr_provider_default)
        if provider != "auto":
            return provider
        if settings.querylake_chandra_ocr_base_url:
            return "querylake_chandra"
        if settings.mistral_api_key:
            return "mistral"
        return "none"

    async def extract_markdown(
        self,
        *,
        file_bytes: bytes,
        content_type: str,
        source_url: Optional[str] = None,
        requested_provider: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> OCRExtractionResult:
        provider = self.resolve_provider(requested_provider=requested_provider)
        if provider == "none":
            raise ProviderError("OCR provider disabled or not configured")
        if provider == "querylake_chandra":
            return await self._extract_with_querylake_chandra(
                file_bytes=file_bytes,
                content_type=content_type,
                source_url=source_url,
                prompt=prompt,
            )
        if provider == "mistral":
            return await self._extract_with_mistral(
                file_bytes=file_bytes,
                content_type=content_type,
                prompt=prompt,
            )
        raise ProviderError(f"Unsupported OCR provider '{provider}'")

    async def _extract_with_querylake_chandra(
        self,
        *,
        file_bytes: bytes,
        content_type: str,
        source_url: Optional[str],
        prompt: Optional[str],
    ) -> OCRExtractionResult:
        if not settings.querylake_chandra_ocr_base_url:
            raise ProviderError("QueryLake Chandra OCR base URL not configured")
        mode = _normalize_querylake_mode(settings.querylake_chandra_ocr_mode)

        if mode == "direct":
            try:
                return await self._extract_with_querylake_chandra_direct(
                    file_bytes=file_bytes,
                    content_type=content_type,
                    source_url=source_url,
                    prompt=prompt,
                )
            except ProviderError as exc:
                return await self._maybe_fallback_to_mistral(
                    file_bytes=file_bytes,
                    content_type=content_type,
                    prompt=prompt,
                    querylake_error=exc,
                )

        if mode == "querylake_files":
            try:
                return await self._extract_with_querylake_kernel_files(
                    file_bytes=file_bytes,
                    content_type=content_type,
                )
            except ProviderError as exc:
                return await self._maybe_fallback_to_mistral(
                    file_bytes=file_bytes,
                    content_type=content_type,
                    prompt=prompt,
                    querylake_error=exc,
                )

        direct_error: Optional[ProviderError] = None
        try:
            return await self._extract_with_querylake_chandra_direct(
                file_bytes=file_bytes,
                content_type=content_type,
                source_url=source_url,
                prompt=prompt,
            )
        except ProviderError as exc:
            direct_error = exc

        try:
            return await self._extract_with_querylake_kernel_files(
                file_bytes=file_bytes,
                content_type=content_type,
            )
        except ProviderError as files_exc:
            fallback_error = ProviderError(
                "QueryLake OCR failed in both direct and kernel/files modes",
                payload={
                    "direct_error": self._provider_error_summary(direct_error),
                    "files_error": self._provider_error_summary(files_exc),
                },
            )
            try:
                return await self._maybe_fallback_to_mistral(
                    file_bytes=file_bytes,
                    content_type=content_type,
                    prompt=prompt,
                    querylake_error=fallback_error,
                )
            except ProviderError:
                pass
            raise ProviderError(
                "QueryLake OCR failed in both direct and kernel/files modes",
                payload={
                    "direct_error": self._provider_error_summary(direct_error),
                    "files_error": self._provider_error_summary(files_exc),
                },
            ) from files_exc

    async def _maybe_fallback_to_mistral(
        self,
        *,
        file_bytes: bytes,
        content_type: str,
        prompt: Optional[str],
        querylake_error: ProviderError,
    ) -> OCRExtractionResult:
        if not settings.querylake_chandra_ocr_enable_mistral_fallback:
            raise querylake_error
        if not settings.mistral_api_key:
            raise querylake_error

        try:
            mistral_result = await self._extract_with_mistral(
                file_bytes=file_bytes,
                content_type=content_type,
                prompt=prompt,
            )
        except ProviderError as mistral_exc:
            raise ProviderError(
                "QueryLake OCR failed and Mistral fallback failed",
                payload={
                    "querylake_error": self._provider_error_summary(querylake_error),
                    "mistral_error": self._provider_error_summary(mistral_exc),
                },
            ) from mistral_exc

        details = dict(mistral_result.details or {})
        details["fallback_from"] = "querylake_chandra"
        details["querylake_error"] = self._provider_error_summary(querylake_error)
        return OCRExtractionResult(
            provider=mistral_result.provider,
            markdown=mistral_result.markdown,
            details=details,
        )

    @staticmethod
    def _provider_error_summary(error: Optional[ProviderError]) -> Optional[dict[str, Any]]:
        if error is None:
            return None
        return {
            "message": str(error),
            "status_code": error.status_code,
            "payload": error.payload,
        }

    @staticmethod
    def _candidate_paths(primary_path: str, *fallback_paths: str) -> list[str]:
        candidates: list[str] = []
        for raw_path in (primary_path, *fallback_paths):
            path = (raw_path or "").strip()
            if not path:
                continue
            normalized = path if path.startswith("/") else f"/{path}"
            if normalized not in candidates:
                candidates.append(normalized)
        return candidates

    @staticmethod
    def _is_retryable_path_error(error: ProviderError) -> bool:
        if error.status_code == 404:
            return True
        payload = error.payload
        if isinstance(payload, str):
            return "not found" in payload.lower()
        if isinstance(payload, dict):
            return "not found" in str(payload).lower()
        return False

    async def _post_querylake_with_fallback(
        self,
        *,
        client: httpx.AsyncClient,
        base_url: str,
        paths: list[str],
        context: str,
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
        retry_on_any_error: bool = False,
    ) -> tuple[Any, str, httpx.Response]:
        if not paths:
            raise ProviderError(f"QueryLake {context} has no candidate paths configured")

        attempt_summaries: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            url = _join_url(base_url, path)
            response = await client.post(
                url,
                headers=headers,
                params=params,
                json=json,
                files=files,
            )
            try:
                data = self._unwrap_querylake_response(response, context=context)
                return data, path, response
            except ProviderError as exc:
                attempt_summaries.append(
                    {
                        "path": path,
                        "status_code": exc.status_code,
                        "error": str(exc),
                        "payload": exc.payload,
                    }
                )
                has_more = index < len(paths) - 1
                if has_more and (retry_on_any_error or self._is_retryable_path_error(exc)):
                    continue
                raise

        raise ProviderError(
            f"QueryLake {context} failed across candidate paths",
            payload={"attempts": attempt_summaries},
        )

    @staticmethod
    def _unwrap_querylake_response(resp: httpx.Response, *, context: str) -> Any:
        if resp.status_code >= 400:
            raise ProviderError(
                f"QueryLake {context} error",
                status_code=resp.status_code,
                payload=resp.text[:2000],
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(
                f"QueryLake {context} returned non-JSON response",
                status_code=resp.status_code,
                payload=resp.text[:2000],
            ) from exc

        if isinstance(data, dict) and "success" in data:
            if not bool(data.get("success")):
                error_detail = data.get("error")
                message = f"QueryLake {context} error"
                if isinstance(error_detail, str) and error_detail.strip():
                    message = f"{message}: {error_detail}"
                raise ProviderError(
                    message,
                    status_code=resp.status_code,
                    payload=error_detail or data,
                )
            if "result" in data and data.get("result") is not None:
                return data.get("result")
        return data

    @staticmethod
    def _extract_oauth2_from_payload(payload: Any) -> Optional[str]:
        if isinstance(payload, str):
            return _extract_querylake_token(payload)
        if not isinstance(payload, dict):
            return None

        for key in ("oauth2", "auth", "token", "access_token"):
            value = payload.get(key)
            if isinstance(value, str):
                token = _extract_querylake_token(value)
                if token:
                    return token

        for nested_key in ("result", "data"):
            nested = payload.get(nested_key)
            token = DocumentOCRClient._extract_oauth2_from_payload(nested)
            if token:
                return token
        return None

    async def _resolve_querylake_oauth2(self, client: httpx.AsyncClient) -> str:
        if self._querylake_cached_oauth2:
            return self._querylake_cached_oauth2

        static_token = _extract_querylake_token(settings.querylake_chandra_ocr_auth_token)
        if static_token:
            self._querylake_cached_oauth2 = static_token
            return static_token

        username = (settings.querylake_chandra_ocr_username or "").strip()
        password = (settings.querylake_chandra_ocr_password or "").strip()
        if not username or not password:
            raise ProviderError(
                "QueryLake OCR credentials missing. Set querylake_chandra_ocr_auth_token or querylake_chandra_ocr_username/password."
            )

        base = settings.querylake_chandra_ocr_base_url.rstrip("/")
        login_url = _join_url(base, settings.querylake_chandra_ocr_login_path)
        add_user_url = _join_url(base, settings.querylake_chandra_ocr_add_user_path)

        login_payload = {"auth": {"username": username, "password": password}}
        login_error: Optional[ProviderError] = None
        try:
            login_resp = await client.post(login_url, json=login_payload, headers={"Content-Type": "application/json"})
            login_data = self._unwrap_querylake_response(login_resp, context="login")
            token = self._extract_oauth2_from_payload(login_data)
            if token:
                self._querylake_cached_oauth2 = token
                return token
        except ProviderError as exc:
            login_error = exc

        if settings.querylake_chandra_ocr_auto_create_user:
            add_payload = {"username": username, "password": password}
            try:
                add_resp = await client.post(add_user_url, json=add_payload, headers={"Content-Type": "application/json"})
                add_data = self._unwrap_querylake_response(add_resp, context="add_user")
                token = self._extract_oauth2_from_payload(add_data)
                if token:
                    self._querylake_cached_oauth2 = token
                    return token
            except ProviderError:
                # Existing-user races are acceptable; retry login below.
                pass

            login_resp = await client.post(login_url, json=login_payload, headers={"Content-Type": "application/json"})
            login_data = self._unwrap_querylake_response(login_resp, context="login_retry")
            token = self._extract_oauth2_from_payload(login_data)
            if token:
                self._querylake_cached_oauth2 = token
                return token

        raise ProviderError(
            "Unable to resolve QueryLake OAuth token for OCR",
            payload=self._provider_error_summary(login_error),
        )

    async def _extract_with_querylake_chandra_direct(
        self,
        *,
        file_bytes: bytes,
        content_type: str,
        source_url: Optional[str],
        prompt: Optional[str],
    ) -> OCRExtractionResult:
        base = settings.querylake_chandra_ocr_base_url.rstrip("/")
        url = _join_url(base, settings.querylake_chandra_ocr_path)

        payload: dict[str, Any] = {
            "contentBase64": base64.b64encode(file_bytes).decode("utf-8"),
            "contentType": content_type or "application/octet-stream",
        }
        if source_url:
            payload["sourceUrl"] = source_url
        effective_prompt = (prompt or settings.document_ocr_default_prompt).strip()
        if effective_prompt:
            payload["prompt"] = effective_prompt

        headers: dict[str, str] = {"Content-Type": "application/json"}
        token = settings.querylake_chandra_ocr_auth_token
        if token:
            headers[settings.querylake_chandra_ocr_auth_header] = token

        async with httpx.AsyncClient(timeout=settings.querylake_chandra_ocr_timeout_seconds) as client:
            resp = await client.post(url, headers=headers, json=payload)
            data = self._unwrap_querylake_response(resp, context="direct_ocr")

        markdown = _extract_markdown_from_payload(data)
        if not markdown:
            raise ProviderError("QueryLake Chandra OCR response missing markdown text", payload=data)
        return OCRExtractionResult(
            provider="querylake_chandra",
            markdown=markdown,
            details={"mode": "direct", "status_code": resp.status_code},
        )

    @staticmethod
    def _convert_image_to_pdf(file_bytes: bytes) -> bytes:
        try:
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
        except Exception as exc:
            raise ProviderError("Failed to decode image for QueryLake OCR conversion") from exc
        output = BytesIO()
        image.save(output, format="PDF")
        return output.getvalue()

    async def _extract_with_querylake_kernel_files(
        self,
        *,
        file_bytes: bytes,
        content_type: str,
    ) -> OCRExtractionResult:
        base = settings.querylake_chandra_ocr_base_url.rstrip("/")

        upload_bytes = file_bytes
        upload_content_type = (content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
        converted_from_image = False
        if upload_content_type.startswith("image/"):
            upload_bytes = self._convert_image_to_pdf(file_bytes)
            upload_content_type = "application/pdf"
            converted_from_image = True

        async with httpx.AsyncClient(timeout=settings.querylake_chandra_ocr_timeout_seconds) as client:
            oauth2_token = await self._resolve_querylake_oauth2(client)
            auth_headers = {"Authorization": f"Bearer {oauth2_token}"}

            upload_paths = self._candidate_paths(
                settings.querylake_chandra_ocr_files_upload_path,
                "/v2/kernel/files",
                "/files",
            )
            upload_data, upload_path, _ = await self._post_querylake_with_fallback(
                client=client,
                base_url=base,
                paths=upload_paths,
                context="files_upload",
                headers=auth_headers,
                files={"file": ("ocr_input.pdf", upload_bytes, upload_content_type)},
            )
            if not isinstance(upload_data, dict):
                raise ProviderError("Invalid QueryLake upload response", payload=upload_data)
            file_id = upload_data.get("file_id")
            version_id = upload_data.get("version_id")
            if not file_id or not version_id:
                raise ProviderError("QueryLake upload missing file_id/version_id", payload=upload_data)

            process_template = settings.querylake_chandra_ocr_files_process_path_template
            try:
                configured_process_path = process_template.format(file_id=file_id, version_id=version_id)
            except Exception as exc:
                raise ProviderError("Invalid QueryLake files process path template", payload=process_template) from exc
            process_paths = self._candidate_paths(
                configured_process_path,
                f"/files/{file_id}/versions/{version_id}/process",
                f"/v2/kernel/files/{file_id}/versions/{version_id}/process",
            )
            process_params = {}
            profile = (settings.querylake_chandra_ocr_profile or "").strip()
            if profile:
                process_params["ocr_profile"] = profile
            _, process_path, _ = await self._post_querylake_with_fallback(
                client=client,
                base_url=base,
                paths=process_paths,
                context="files_process",
                headers=auth_headers,
                params=process_params or None,
            )

            search_payload = {
                "auth": {"oauth2": oauth2_token},
                "query": "*",
                "limit": max(20, int(settings.querylake_chandra_ocr_search_limit)),
                "offset": 0,
                "sort_by": "created_at",
                "sort_dir": "DESC",
                "filters": {
                    "file_ids": [file_id],
                    "file_version_ids": [version_id],
                },
            }
            search_paths = self._candidate_paths(
                settings.querylake_chandra_ocr_search_chunks_path,
                "/v2/kernel/api/search_file_chunks",
                "/api/search_file_chunks",
            )
            search_data, search_path, _ = await self._post_querylake_with_fallback(
                client=client,
                base_url=base,
                paths=search_paths,
                context="search_file_chunks",
                headers={"Content-Type": "application/json"},
                json=search_payload,
                retry_on_any_error=True,
            )

        if isinstance(search_data, dict):
            rows = search_data.get("results")
            if rows is None and isinstance(search_data.get("data"), dict):
                rows = search_data["data"].get("results")
            if rows is None and isinstance(search_data.get("rows"), list):
                rows = search_data.get("rows")
        elif isinstance(search_data, list):
            rows = search_data
        else:
            rows = None
        if not isinstance(rows, list):
            raise ProviderError("QueryLake search_file_chunks missing results list", payload=search_data)

        matching_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text_value = row.get("text")
            if not isinstance(text_value, str) or not text_value.strip():
                continue
            row_version_id = row.get("file_version_id") or row.get("version_id") or row.get("fileVersionId")
            row_file_id = row.get("file_id") or row.get("fileId")
            if row_version_id and row_version_id != version_id:
                continue
            if row_file_id and row_file_id != file_id:
                continue
            matching_rows.append(row)

        if not matching_rows:
            raise ProviderError(
                "QueryLake files OCR produced no text chunks for uploaded version",
                payload={
                    "file_id": file_id,
                    "version_id": version_id,
                    "results_count": len(rows),
                    "search_path": search_path,
                },
            )

        matching_rows.sort(key=lambda row: float(row.get("created_at", 0.0)))
        markdown = "\n\n".join(row["text"].strip() for row in matching_rows if row.get("text"))
        if not markdown:
            raise ProviderError("QueryLake files OCR returned empty text chunks", payload={"version_id": version_id})
        if markdown.startswith("CAS:"):
            raise ProviderError(
                "QueryLake files OCR returned placeholder chunk text (OCR engine likely unavailable)",
                payload={"version_id": version_id, "preview": markdown[:240]},
            )

        return OCRExtractionResult(
            provider="querylake_chandra",
            markdown=markdown,
            details={
                "mode": "querylake_files",
                "file_id": file_id,
                "version_id": version_id,
                "converted_from_image": converted_from_image,
                "upload_path": upload_path,
                "process_path": process_path,
                "search_path": search_path,
            },
        )

    async def _extract_with_mistral(
        self,
        *,
        file_bytes: bytes,
        content_type: str,
        prompt: Optional[str],
    ) -> OCRExtractionResult:
        if not settings.mistral_api_key:
            raise ProviderError("Mistral OCR API key not configured")
        base = settings.mistral_ocr_base_url.rstrip("/")
        url = base if base.endswith("/ocr") else f"{base}/ocr"

        mime = (content_type or "application/octet-stream").split(";")[0].strip().lower()
        data_url = f"data:{mime};base64,{base64.b64encode(file_bytes).decode('utf-8')}"
        document_field = "image_url" if mime.startswith("image/") else "document_url"
        payload: dict[str, Any] = {
            "model": settings.mistral_ocr_model,
            "document": {
                "type": document_field,
                document_field: data_url,
            },
        }
        effective_prompt = (prompt or settings.document_ocr_default_prompt).strip()
        if effective_prompt:
            payload["prompt"] = effective_prompt

        headers = {
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=settings.mistral_ocr_timeout_seconds) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                raise ProviderError(
                    "Mistral OCR error",
                    status_code=resp.status_code,
                    payload=resp.text[:2000],
                )
            data = resp.json()

        markdown = _extract_markdown_from_payload(data)
        if not markdown:
            raise ProviderError("Mistral OCR response missing markdown text", payload=data)
        return OCRExtractionResult(
            provider="mistral",
            markdown=markdown,
            details={"status_code": resp.status_code, "model": settings.mistral_ocr_model},
        )


document_ocr_client = DocumentOCRClient()
