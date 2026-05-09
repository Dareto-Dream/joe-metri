from __future__ import annotations

import asyncio
import unittest

from gd_scraper.api import AsyncRateLimiter
from gd_scraper.errors import ShutdownRequested


class ApiShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_rate_limiter_wait_wakes_on_shutdown(self) -> None:
        limiter = AsyncRateLimiter(rate_per_second=1.0)
        limiter._next_at = asyncio.get_running_loop().time() + 30.0
        shutdown_event = asyncio.Event()

        wait_task = asyncio.create_task(limiter.wait(shutdown_event))
        await asyncio.sleep(0)
        shutdown_event.set()

        with self.assertRaises(ShutdownRequested):
            await asyncio.wait_for(wait_task, timeout=1)


if __name__ == "__main__":
    unittest.main()
