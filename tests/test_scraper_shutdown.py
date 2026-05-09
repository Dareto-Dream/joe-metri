from __future__ import annotations

import asyncio
import base64
import gzip
import tempfile
import unittest
from pathlib import Path

from gd_scraper.scraper import GeometryDashScraper
from gd_scraper.sources import DiscoverySource


def encoded_level(raw: str) -> str:
    return base64.urlsafe_b64encode(gzip.compress(raw.encode("utf-8"))).decode("ascii").rstrip("=")


class BlockingClient:
    def __init__(self) -> None:
        self.download_started = asyncio.Event()
        self.release_download = asyncio.Event()
        self.comments_requested = 0

    async def search_levels(self, _params: dict[str, object]) -> str:
        return (
            "1:123:2:Example:5:1:6:42:8:10:9:50:10:1000:12:0:14:55:15:3:17:0:18:6:25::35:999"
            "#42:Player:7"
            "#1~|~999~|~2~|~Song Name~|~4~|~Artist"
            "#1:0:10#hash"
        )

    async def download_level(self, _level_id: int) -> str:
        self.download_started.set()
        await self.release_download.wait()
        level_data = encoded_level("kS1,0;k1,1,2,15;k1,2,2,30")
        return f"1:123:2:Example:4:{level_data}:6:42:8:10:9:30:10:10:12:0:14:5#hash1#hash2##"

    async def get_comments(self, *_args: object, **_kwargs: object) -> str:
        self.comments_requested += 1
        return "2~Comment~3~42#1:0:10"


class ScraperShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_finishes_active_download_and_skips_new_comment_requests(self) -> None:
        client = BlockingClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = GeometryDashScraper(
                client=client,
                data_dir=Path(tmp_dir),
                sources=[DiscoverySource("test", {"type": 1})],
                pages_per_source=1,
                target_count=0,
                concurrency=1,
                include_comments=True,
                comments_pages=1,
                metrics_interval=0,
            )

            run_task = asyncio.create_task(scraper.run())
            await asyncio.wait_for(client.download_started.wait(), timeout=1)

            with self.assertLogs("gd_scraper.scraper", level="WARNING"):
                scraper.request_shutdown("test")
            client.release_download.set()
            stats = await asyncio.wait_for(run_task, timeout=1)

            self.assertEqual(stats.saved, 1)
            self.assertEqual(client.comments_requested, 0)
            self.assertIn("123", (Path(tmp_dir) / "checkpoints" / "downloaded_level_ids.txt").read_text())


if __name__ == "__main__":
    unittest.main()
