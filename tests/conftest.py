from __future__ import annotations

import socket
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.main import app


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def base_url() -> str:
    port = _pick_free_port()
    thread = threading.Thread(
        target=lambda: app.start(host="127.0.0.1", port=port, _check_port=False),
        daemon=True,
    )
    thread.start()

    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(base_url=url, timeout=0.5) as c:
                r = c.get("/health")
                if r.status_code == 200:
                    return url
        except Exception as exc:  # pragma: no cover - startup poll
            last_error = exc
            time.sleep(0.1)

    raise RuntimeError(f"Robyn server failed to start. Last error: {last_error}")


@pytest.fixture(scope="session")
def client(base_url: str):
    with httpx.Client(base_url=base_url, timeout=30) as c:
        yield c


@pytest.fixture
def mocker(request):
    """Lightweight pytest-mock substitute to provide mocker.patch and AsyncMock.

    Ensures patches are stopped after each test.
    """
    active_patches = []

    class _Mocker:
        AsyncMock = AsyncMock
        MagicMock = MagicMock

        def patch(self, target, *args, **kwargs):
            p = patch(target, *args, **kwargs)
            m = p.start()
            active_patches.append(p)
            return m

    def fin():
        for p in reversed(active_patches):
            try:
                p.stop()
            except Exception:
                pass

    request.addfinalizer(fin)
    return _Mocker()
