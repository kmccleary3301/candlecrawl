from __future__ import annotations

from typing import Any, Dict

from fastapi import Response
from fastapi.responses import JSONResponse

from app.config import settings


def is_strict_v2_mode() -> bool:
    """Return whether strict Firecrawl v2 shaping is enabled."""
    return bool(settings.strict_firecrawl_v2)


def apply_v2_contract_headers(response: Response) -> None:
    """Attach stable contract headers for all /v2 responses."""
    response.headers["X-Contract-Version"] = settings.v2_contract_version
    response.headers["X-CandleCrawl-Profile"] = "strict" if is_strict_v2_mode() else "compat"


def success_payload(data: Dict[str, Any] | None = None, **extras: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"success": True}
    if data is not None:
        payload["data"] = data
    payload.update(extras)
    return payload


def error_payload(error: str, *, code: str = "INVALID_REQUEST", **extras: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"success": False, "error": error, "code": code}
    payload.update(extras)
    return payload


def json_error(
    error: str,
    *,
    status_code: int = 400,
    code: str = "INVALID_REQUEST",
    **extras: Any,
) -> JSONResponse:
    """Return an error response envelope used by strict /v2 handlers."""
    return JSONResponse(status_code=status_code, content=error_payload(error, code=code, **extras))

