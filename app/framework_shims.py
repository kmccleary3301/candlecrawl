from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable


class HTTPException(Exception):
    def __init__(self, *, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


class Depends:
    def __init__(self, dependency: Callable[..., Any] | None = None):
        self.dependency = dependency


class BackgroundTasks:
    def add_task(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)


class _DummyState:
    pass


class FastAPI:
    def __init__(self, *args: Any, **kwargs: Any):
        self.state = _DummyState()

    def add_middleware(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_exception_handler(self, *args: Any, **kwargs: Any) -> None:
        return None

    def include_router(self, *args: Any, **kwargs: Any) -> None:
        return None

    def _route_decorator(self, *args: Any, **kwargs: Any):
        def decorator(fn: Callable[..., Any]):
            return fn

        return decorator

    def get(self, *args: Any, **kwargs: Any):
        return self._route_decorator(*args, **kwargs)

    def post(self, *args: Any, **kwargs: Any):
        return self._route_decorator(*args, **kwargs)

    def put(self, *args: Any, **kwargs: Any):
        return self._route_decorator(*args, **kwargs)

    def patch(self, *args: Any, **kwargs: Any):
        return self._route_decorator(*args, **kwargs)

    def delete(self, *args: Any, **kwargs: Any):
        return self._route_decorator(*args, **kwargs)


class APIRouter(FastAPI):
    pass


def Query(*, default: Any = None, **kwargs: Any) -> Any:
    return default


class Request:
    def __init__(self, headers: dict[str, str] | None = None):
        self.headers = headers or {}


class CORSMiddleware:
    pass


class RateLimitExceeded(Exception):
    pass


class SlowAPIMiddleware:
    pass


def _rate_limit_exceeded_handler(*args: Any, **kwargs: Any):
    return {"detail": "Rate limit exceeded"}


def get_remote_address(request: Any) -> str:
    return getattr(request, "ip_addr", None) or "unknown"


class Limiter:
    def __init__(self, key_func: Callable[..., str] | None = None):
        self.key_func = key_func

    def limit(self, _rule: str):
        def decorator(fn: Callable[..., Any]):
            return fn

        return decorator


@dataclass
class RawResponse:
    content: str
    media_type: str
    status_code: int = 200
