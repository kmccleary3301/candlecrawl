import pytest
import asyncio
from app.frontier import MemoryFrontier


@pytest.mark.asyncio
async def test_memory_frontier_basic():
    f = MemoryFrontier()
    await f.enqueue("https://a.com", 0)
    await f.enqueue("https://a.com", 0)  # dedupe
    assert f.size() == 1
    url, depth = await f.dequeue()
    assert url == "https://a.com" and depth == 0
    await f.mark_done(url)
    assert f.size() == 0




