from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from .sources import DEFAULT_SOURCES, resolve_sources


DEFAULT_SOURCE_NAMES = [
    "featured",
    "rated",
    "trending",
    "popular",
    "most_liked",
    "demons",
    "easy_demons",
    "medium_demons",
    "hard_demons",
    "insane_demons",
    "extreme_demons",
]


class ShutdownSignalHandler:
    def __init__(self, scraper: Any) -> None:
        self.scraper = scraper
        self.logger = logging.getLogger(__name__)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.main_task: asyncio.Task[Any] | None = None
        self.exit_code = 0
        self._signals = [signal.SIGINT]
        if hasattr(signal, "SIGTERM"):
            self._signals.append(signal.SIGTERM)
        self._previous_handlers: dict[signal.Signals, Any] = {}
        self._loop_handlers: set[signal.Signals] = set()
        self._requests = 0

    def __enter__(self) -> "ShutdownSignalHandler":
        self.loop = asyncio.get_running_loop()
        self.main_task = asyncio.current_task()
        for sig in self._signals:
            self._previous_handlers[sig] = signal.getsignal(sig)
            try:
                self.loop.add_signal_handler(sig, self._handle_signal, sig)
                self._loop_handlers.add(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                signal.signal(sig, self._sync_handle_signal)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for sig in self._signals:
            if sig in self._loop_handlers and self.loop is not None:
                self.loop.remove_signal_handler(sig)
            previous = self._previous_handlers.get(sig)
            if previous is not None:
                signal.signal(sig, previous)

    def _sync_handle_signal(self, signum: int, _frame: object | None) -> None:
        self._handle_signal(signal.Signals(signum))

    def _handle_signal(self, sig: signal.Signals) -> None:
        self._requests += 1
        self.exit_code = 128 + sig.value
        if self._requests == 1:
            self.logger.warning("received %s; stopping scraper gracefully", sig.name)
            self.scraper.request_shutdown(sig.name)
            return

        self.logger.warning("received %s again; cancelling immediately", sig.name)
        if self.main_task is not None:
            self.main_task.cancel()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-scrape",
        description="Download raw Geometry Dash level data into JSONL files.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--target-count", type=int, default=10_000, help="New valid levels to save. Use 0 for no cap.")
    parser.add_argument("--pages-per-source", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--search-rate", type=float, default=2.0, help="Search requests per second.")
    parser.add_argument(
        "--download-rate",
        type=float,
        default=0.33,
        help="Download requests per second. Default stays near the public GD docs limit.",
    )
    parser.add_argument("--comment-rate", type=float, default=0.33, help="Comment requests per second.")
    parser.add_argument("--include-comments", action="store_true", help="Fetch raw comment pages for saved levels.")
    parser.add_argument("--comments-pages", type=int, default=1, help="Comment pages per saved level when enabled.")
    parser.add_argument("--comments-mode", type=int, default=0, help="GD comments mode parameter.")
    parser.add_argument("--comments-count", type=int, default=10, help="Comments per page request.")
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=60.0,
        help="Seconds between metrics snapshots. Use 0 to write on each checkpoint event.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=DEFAULT_SOURCE_NAMES,
        help=f"Comma-separated or space-separated source names. Valid: {', '.join(sorted(DEFAULT_SOURCES))}",
    )
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


async def async_main(args: argparse.Namespace) -> int:
    from .api import GDClient, GDClientConfig
    from .scraper import GeometryDashScraper
    from .storage import coerce_source_names

    log_dir = args.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "scraper.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    source_names = coerce_source_names(args.sources)
    sources = resolve_sources(source_names)
    config = GDClientConfig(
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
        search_rate_per_second=args.search_rate,
        download_rate_per_second=args.download_rate,
        comment_rate_per_second=args.comment_rate,
    )

    async with GDClient(config) as client:
        scraper = GeometryDashScraper(
            client=client,
            data_dir=args.data_dir,
            sources=sources,
            pages_per_source=args.pages_per_source,
            target_count=args.target_count,
            concurrency=args.concurrency,
            retry_rejected=args.retry_rejected,
            include_comments=args.include_comments,
            comments_pages=args.comments_pages,
            comments_mode=args.comments_mode,
            comments_count=args.comments_count,
            metrics_interval=args.metrics_interval,
        )
        with ShutdownSignalHandler(scraper) as shutdown:
            stats = await scraper.run()

    logging.getLogger(__name__).info("run summary: %s", stats.to_json())
    return shutdown.exit_code


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except ModuleNotFoundError as exc:
        if exc.name in {"aiohttp", "aiofiles", "orjson"}:
            print(f"Missing dependency: {exc.name}. Run `pip install -r requirements.txt`.", file=sys.stderr)
            return 1
        raise
    except KeyboardInterrupt:
        return 130
    except asyncio.CancelledError:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
