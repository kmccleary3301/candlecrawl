from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceContext:
    trace_id: str | None = None
    request_id: str | None = None

    def headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.trace_id:
            headers["X-Trace-Id"] = self.trace_id
        if self.request_id:
            headers["X-Request-Id"] = self.request_id
        return headers
