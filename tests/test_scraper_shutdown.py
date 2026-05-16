from __future__ import annotations

import asyncio
import base64
import gzip
import tempfile
import unittest
from pathlib import Path

from gd_scraper.errors import ShutdownRequested
from gd_scraper.scraper import GeometryDashScraper, source_name_matches
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


class ShutdownAwareWaitingClient:
    def __init__(self) -> None:
        self.download_started = asyncio.Event()
        self.shutdown_event: asyncio.Event | None = None

    def set_shutdown_event(self, shutdown_event: asyncio.Event) -> None:
        self.shutdown_event = shutdown_event

    async def search_levels(self, _params: dict[str, object]) -> str:
        return (
            "1:123:2:Example:5:1:6:42:8:10:9:50:10:1000:12:0:14:55:15:3:17:0:18:6:25::35:999"
            "#42:Player:7"
            "#1~|~999~|~2~|~Song Name~|~4~|~Artist"
            "#1:0:10#hash"
        )

    async def download_level(self, _level_id: int) -> str:
        if self.shutdown_event is None:
            raise AssertionError("shutdown event was not installed")
        self.download_started.set()
        await self.shutdown_event.wait()
        raise ShutdownRequested("shutdown requested")


class HangingDownloadClient:
    def __init__(self) -> None:
        self.download_started = asyncio.Event()

    async def search_levels(self, _params: dict[str, object]) -> str:
        return (
            "1:123:2:Example:5:1:6:42:8:10:9:50:10:1000:12:0:14:55:15:3:17:0:18:6:25::35:999"
            "#42:Player:7"
            "#1~|~999~|~2~|~Song Name~|~4~|~Artist"
            "#1:0:10#hash"
        )

    async def download_level(self, _level_id: int) -> str:
        self.download_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class ScraperShutdownTests(unittest.IsolatedAsyncioTestCase):
    def test_source_name_filter_matches_layout_levels_only(self) -> None:
        source = DiscoverySource("layout", {"type": 0, "str": "layout"}, ("layout",))

        self.assertTrue(source_name_matches("Clean Layout Preview", source))
        self.assertTrue(source_name_matches("LAYOUT challenge", source))
        self.assertFalse(source_name_matches("Nine Circles Remake", source))

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

    async def test_shutdown_aborts_downloads_waiting_for_start(self) -> None:
        client = ShutdownAwareWaitingClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = GeometryDashScraper(
                client=client,
                data_dir=Path(tmp_dir),
                sources=[DiscoverySource("test", {"type": 1})],
                pages_per_source=1,
                target_count=0,
                concurrency=1,
                metrics_interval=0,
            )

            run_task = asyncio.create_task(scraper.run())
            await asyncio.wait_for(client.download_started.wait(), timeout=1)

            with self.assertLogs("gd_scraper.scraper", level="WARNING"):
                scraper.request_shutdown("test")
            stats = await asyncio.wait_for(run_task, timeout=1)

            self.assertEqual(stats.saved, 0)
            self.assertEqual(stats.failed, 0)

    async def test_force_shutdown_cancels_hanging_download(self) -> None:
        client = HangingDownloadClient()
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = GeometryDashScraper(
                client=client,
                data_dir=Path(tmp_dir),
                sources=[DiscoverySource("test", {"type": 1})],
                pages_per_source=1,
                target_count=0,
                concurrency=1,
                metrics_interval=0,
            )

            run_task = asyncio.create_task(scraper.run())
            await asyncio.wait_for(client.download_started.wait(), timeout=1)

            with self.assertLogs("gd_scraper.scraper", level="WARNING"):
                scraper.request_shutdown("test")
            await asyncio.sleep(0)
            self.assertFalse(run_task.done())

            with self.assertLogs("gd_scraper.scraper", level="WARNING"):
                scraper.request_shutdown("test", force=True)
            run_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(run_task, timeout=1)


if __name__ == "__main__":
    unittest.main()
