from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import aiohttp

from .errors import GDRequestError


BASE_URL = "https://www.boomlings.com/database"
COMMON_SECRET = "Wmfd2893gb7"
GAME_VERSION = "22"
BINARY_VERSION = "47"

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


class AsyncRateLimiter:
    def __init__(self, rate_per_second: float) -> None:
        self.interval = 0.0 if rate_per_second <= 0 else 1.0 / rate_per_second
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return

        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            if now < self._next_at:
                await asyncio.sleep(self._next_at - now)
                now = loop.time()
            self._next_at = max(now, self._next_at) + self.interval


@dataclass(frozen=True)
class GDClientConfig:
    timeout_seconds: float = 20.0
    retries: int = 3
    backoff_seconds: float = 1.0
    search_rate_per_second: float = 2.0
    download_rate_per_second: float = 0.33
    comment_rate_per_second: float = 0.33


class GDClient:
    def __init__(self, config: GDClientConfig) -> None:
        self.config = config
        self.search_limiter = AsyncRateLimiter(config.search_rate_per_second)
        self.download_limiter = AsyncRateLimiter(config.download_rate_per_second)
        self.comment_limiter = AsyncRateLimiter(config.comment_rate_per_second)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "GDClient":
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        headers = {
            "User-Agent": "",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._session is not None:
            await self._session.close()

    async def search_levels(self, params: dict[str, Any]) -> str:
        return await self._post("getGJLevels21.php", params, self.search_limiter)

    async def download_level(self, level_id: int) -> str:
        params = {
            "levelID": str(level_id),
            "inc": "0",
            "extras": "0",
        }
        return await self._post("downloadGJLevel22.php", params, self.download_limiter)

    async def get_comments(
        self,
        level_id: int,
        *,
        page: int,
        mode: int = 0,
        count: int = 10,
    ) -> str:
        params = {
            "levelID": str(level_id),
            "page": str(page),
            "mode": str(mode),
            "total": "0",
            "count": str(count),
        }
        return await self._post("getGJComments21.php", params, self.comment_limiter)

    async def _post(
        self,
        endpoint: str,
        params: dict[str, Any],
        limiter: AsyncRateLimiter,
    ) -> str:
        if self._session is None:
            raise RuntimeError("GDClient must be used as an async context manager")

        payload = {
            "gameVersion": GAME_VERSION,
            "binaryVersion": BINARY_VERSION,
            "secret": COMMON_SECRET,
        }
        payload.update({key: value for key, value in params.items() if value is not None})
        url = f"{BASE_URL}/{endpoint}"
        last_error: GDRequestError | None = None

        for attempt in range(1, self.config.retries + 1):
            await limiter.wait()
            try:
                async with self._session.post(url, data=payload) as response:
                    text = await response.text()
                    if response.status == 200 and text:
                        return text
                    message = f"{endpoint} returned HTTP {response.status}"
                    if response.status == 200 and not text:
                        message = f"{endpoint} returned an empty response"
                    last_error = GDRequestError(
                        message,
                        endpoint=endpoint,
                        payload=payload,
                        status=response.status,
                        response_text=text[:500],
                    )
                    if response.status not in RETRYABLE_STATUS_CODES:
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = GDRequestError(
                    f"{endpoint} request failed: {exc}",
                    endpoint=endpoint,
                    payload=payload,
                )

            if attempt < self.config.retries:
                delay = self.config.backoff_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0, self.config.backoff_seconds)
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise GDRequestError(
            f"{endpoint} request failed",
            endpoint=endpoint,
            payload=payload,
        )
