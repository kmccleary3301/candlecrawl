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


class ScrapeRequest(CandleCrawlModel):
    url: str
    formats: list[str] | None = None
    only_main_content: bool | None = Field(default=None, alias="onlyMainContent")
    actions: list[dict[str, Any]] | None = None
    options: dict[str, Any] | None = None


class ScrapeResponse(CandleCrawlModel):
    success: bool = True
    data: dict[str, Any] | None = None
    warning: str | None = None
    id: str | None = None
    credits_used: int | None = Field(default=None, alias="creditsUsed")


class MapRequest(CandleCrawlModel):
    url: str
    search: str | None = None
    limit: int | None = None
    include_subdomains: bool | None = Field(default=None, alias="includeSubdomains")


class MapResponse(CandleCrawlModel):
    success: bool = True
    links: list[Any] = Field(default_factory=list)
    warning: str | None = None
    id: str | None = None
    credits_used: int | None = Field(default=None, alias="creditsUsed")


class SearchRequest(CandleCrawlModel):
    query: str
    sources: list[Any] | None = None
    categories: list[str] | None = None
    limit: int | None = None
    country: str | None = None
    lang: str | None = None
    location: dict[str, Any] | None = None
    scrape_options: dict[str, Any] | None = Field(default=None, alias="scrapeOptions")


class SearchResponse(CandleCrawlModel):
    success: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    warning: str | None = None
    id: str | None = None
    credits_used: int | None = Field(default=None, alias="creditsUsed")


class CrawlRequest(CandleCrawlModel):
    url: str
    limit: int | None = None
    max_depth: int | None = Field(default=None, alias="maxDepth")
    scrape_options: dict[str, Any] | None = Field(default=None, alias="scrapeOptions")
    webhook: Any = None


class BatchScrapeRequest(CandleCrawlModel):
    urls: list[str]
    scrape_options: dict[str, Any] | None = Field(default=None, alias="scrapeOptions")
    webhook: Any = None


class ExtractRequest(CandleCrawlModel):
    urls: list[str]
    prompt: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    enable_web_search: bool | None = Field(default=None, alias="enableWebSearch")
    ignore_sitemap: bool | None = Field(default=None, alias="ignoreSitemap")
    show_sources: bool | None = Field(default=None, alias="showSources")
    webhook: Any = None


class JobCreateResponse(CandleCrawlModel):
    success: bool = True
    id: str
    url: str | None = None
    warning: str | None = None


class JobStatusResponse(CandleCrawlModel):
    status: str
    data: list[Any] | dict[str, Any] | None = None
    total: int | None = None
    completed: int | None = None
    credits_used: int | None = Field(default=None, alias="creditsUsed")
    next: str | None = None


class JobErrorsResponse(CandleCrawlModel):
    errors: list[Any] = Field(default_factory=list)
    robots_blocked: list[str] = Field(default_factory=list, alias="robotsBlocked")


class CancelResponse(CandleCrawlModel):
    success: bool = True
    message: str | None = None


class ActionsCapabilitiesResponse(CandleCrawlModel):
    actions: list[dict[str, Any]] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    output_fields: list[str] = Field(default_factory=list, alias="outputFields")
