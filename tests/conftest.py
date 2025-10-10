import pytest
from unittest.mock import patch, AsyncMock, MagicMock


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

