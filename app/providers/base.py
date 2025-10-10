from __future__ import annotations

from typing import Any, Optional


class ProviderError(Exception):
    def __init__(self, message: str, *, status_code: Optional[int] = None, payload: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


