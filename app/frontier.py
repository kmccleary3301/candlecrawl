"""Pluggable frontier interfaces (in-memory now; Redis later).

This module provides abstractions to schedule and deduplicate crawl URLs.
It is intentionally small and forward-compatible with Redis-backed impls.
"""
from __future__ import annotations
import asyncio
from typing import Optional, Tuple


class BaseFrontier:
    async def enqueue(self, url: str, depth: int) -> None:
        raise NotImplementedError

    async def dequeue(self, timeout: Optional[float] = None) -> Tuple[str, int]:
        raise NotImplementedError

    async def mark_inflight(self, url: str) -> None:
        raise NotImplementedError

    async def mark_done(self, url: str) -> None:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def seen(self, url: str) -> bool:
        raise NotImplementedError

    def inflight_size(self) -> int:
        raise NotImplementedError


class MemoryFrontier(BaseFrontier):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        self._seen: set[str] = set()
        self._inflight: set[str] = set()

    async def enqueue(self, url: str, depth: int) -> None:
        if url in self._seen or url in self._inflight:
            return
        await self._queue.put((url, depth))
        self._seen.add(url)

    async def dequeue(self, timeout: Optional[float] = None) -> Tuple[str, int]:
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        url, depth = item
        self._inflight.add(url)
        return url, depth

    async def mark_inflight(self, url: str) -> None:
        self._inflight.add(url)

    async def mark_done(self, url: str) -> None:
        if url in self._inflight:
            self._inflight.remove(url)
        self._queue.task_done()

    def size(self) -> int:
        return self._queue.qsize()

    def seen(self, url: str) -> bool:
        return url in self._seen or url in self._inflight

    def inflight_size(self) -> int:
        return len(self._inflight)


