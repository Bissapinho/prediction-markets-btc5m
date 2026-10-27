"""Async HTTP client: explicit timeout, retries with exponential backoff + jitter, rate limit."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from pipelines.common.log import get_logger, log

logger = get_logger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class HttpError(Exception):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} {url}: {body[:200]}")
        self.status = status
        self.url = url
        self.body = body


class RateLimiter:
    """Spaces requests by at least 1/rate seconds."""

    def __init__(self, rate_per_s: float):
        self._interval = 1.0 / rate_per_s if rate_per_s > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next = now + self._interval


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Exponential backoff with jitter: base * 2^attempt, capped, multiplied by U(0.5, 1.5)."""
    return min(cap, base * 2**attempt) * random.uniform(0.5, 1.5)


class HttpClient:
    def __init__(self, base_url: str, rate_per_s: float, timeout_s: float, max_retries: int):
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout_s))
        self._limiter = RateLimiter(rate_per_s)
        self._max_retries = max_retries

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries. Raises HttpError on a non-retryable status or once the retries are exhausted."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                resp = await self._client.get(path, params=params)
            except httpx.TransportError as e:
                last_exc = e
                reason = type(e).__name__
            else:
                if resp.status_code < 400:
                    return resp
                if resp.status_code not in RETRYABLE_STATUS:
                    raise HttpError(resp.status_code, str(resp.url), resp.text)
                last_exc = HttpError(resp.status_code, str(resp.url), resp.text)
                reason = f"http_{resp.status_code}"
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    await asyncio.sleep(float(retry_after))
            if attempt < self._max_retries:
                delay = backoff_delay(attempt)
                log(logger, "retry", logging.WARNING, path=path, attempt=attempt + 1, reason=reason, delay_s=round(delay, 2))
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    async def get_json(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, bytes]:
        """Returns (decoded JSON, raw bytes). The raw bytes go to the raw layer."""
        resp = await self.get(path, params)
        return resp.json(), resp.content
