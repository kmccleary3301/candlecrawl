from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CandleCrawlError(Exception):
    message: str
    status_code: int | None = None
    payload: Any = None
    trace_id: str | None = None
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


class CandleCrawlTransportError(CandleCrawlError):
    pass


class CandleCrawlTimeoutError(CandleCrawlTransportError):
    pass


class CandleCrawlAPIError(CandleCrawlError):
    pass


class CandleCrawlAuthError(CandleCrawlAPIError):
    pass


class CandleCrawlRateLimitError(CandleCrawlAPIError):
    pass


class CandleCrawlNotFoundError(CandleCrawlAPIError):
    pass


class CandleCrawlContractError(CandleCrawlError):
    pass


class CandleCrawlCompatibilityError(CandleCrawlError):
    pass
