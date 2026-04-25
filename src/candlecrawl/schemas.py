from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CandleCrawlModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class HealthResponse(CandleCrawlModel):
    status: str = "unknown"
    version: str | None = None
    browser_ready: bool | None = Field(default=None, alias="browserReady")
    browser_error: str | None = Field(default=None, alias="browserError")


class ErrorResponse(CandleCrawlModel):
    error: str | None = None
    message: str | None = None
    code: str | None = None
    detail: Any = None
