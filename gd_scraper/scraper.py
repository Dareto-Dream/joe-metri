from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from .errors import GDRequestError, ShutdownRequested
from .models import DATASET_VERSION, SCRAPER_VERSION, Candidate, RunStats, Song, epoch_seconds
from .parser import (
    ParseError,
    ValidationError,
    candidate_from_search_level,
    comment_page_record,
    is_auto_level,
    level_audio_record,
    level_record,
    parse_comment_response,
    parse_download_response,
    parse_search_response,
    validate_level_data,
)
from .storage import (
    CheckpointStore,
    JsonlWriter,
    append_id_async,
    load_candidates,
    load_ids_from_jsonl,
    load_ids_from_text,
    load_rejected_ids,
)
from .sources import DiscoverySource


LOGGER = logging.getLogger(__name__)


def source_name_matches(level_name: str, source: DiscoverySource) -> bool:
    required = source.name_contains
    if not required:
        return True
    normalized = level_name.casefold()
    return all(term.casefold() in normalized for term in required)


class GeometryDashScraper:
    def __init__(
        self,
        *,
        client: Any,
        data_dir: Path,
        sources: list[DiscoverySource],
        pages_per_source: int,
        target_count: int,
        concurrency: int,
        retry_rejected: bool = False,
        include_comments: bool = False,
        comments_pages: int = 0,
        comments_mode: int = 0,
        comments_count: int = 10,
        download_audio: bool = False,
        metrics_interval: float = 60.0,
    ) -> None:
        self.client = client
        self.store = CheckpointStore(data_dir)
        self.sources = sources
        self.pages_per_source = pages_per_source
        self.target_count = target_count
        self.concurrency = concurrency
        self.retry_rejected = retry_rejected
        self.include_comments = include_comments
        self.comments_pages = max(comments_pages, 0)
        self.comments_mode = comments_mode
        self.comments_count = comments_count
        self.download_audio = download_audio
        self.metrics_interval = metrics_interval

        self.stats = RunStats()
        self.stop_event = asyncio.Event()
        self.id_lock = asyncio.Lock()
        self.song_lock = asyncio.Lock()
        self.state_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        self.download_semaphore = asyncio.Semaphore(max(concurrency, 1))
        self.comment_semaphore = asyncio.Semaphore(max(min(concurrency, 5), 1))

        self.downloaded_ids: set[int] = set()
        self.rejected_ids: set[int] = set()
        self.seen_candidate_ids: set[int] = set()
        self.song_ids: set[int] = set()
        self.state: dict[str, Any] = {"sources": {}}
        self.next_sequence = 0
        self.reserved_saves = 0
        self.metrics_writer: JsonlWriter | None = None
        self.last_metrics_at = 0.0
        self.shutdown_requested = False
        self.force_shutdown_requested = False
        self.shutdown_reason: str | None = None

        set_shutdown_event = getattr(self.client, "set_shutdown_event", None)
        if callable(set_shutdown_event):
            set_shutdown_event(self.stop_event)

    def request_shutdown(self, reason: str = "requested", *, force: bool = False) -> None:
        was_shutdown_requested = self.shutdown_requested
        was_force_requested = self.force_shutdown_requested
        if force:
            self.force_shutdown_requested = True
        if was_shutdown_requested and (not force or was_force_requested):
            return
        self.shutdown_requested = True
        self.shutdown_reason = reason
        self.stop_event.set()
        if force:
            LOGGER.warning("forced shutdown requested: %s", reason)
        else:
            LOGGER.warning("shutdown requested: %s", reason)

    async def run(self) -> RunStats:
        self.store.ensure()
        self.state = self.store.load_state()
        self.state.setdefault("sources", {})

        self.downloaded_ids = load_ids_from_jsonl(self.store.levels_path, "level_id")
        self.downloaded_ids.update(load_ids_from_text(self.store.downloaded_ids_path))
        self.rejected_ids = load_rejected_ids(
            self.store.rejected_levels_path,
            self.store.legacy_rejected_ids_path,
        )
        self.song_ids = load_ids_from_jsonl(self.store.songs_path, "song_id")
        candidates = load_candidates(self.store.discovered_path)
        self.next_sequence = max((candidate.sequence for candidate in candidates.values()), default=-1) + 1

        self.stats.already_downloaded = len(self.downloaded_ids)
        if self.retry_rejected:
            self.rejected_ids.clear()

        queue: asyncio.Queue[Candidate | None] = asyncio.Queue(maxsize=max(self.concurrency * 5, 1))

        async with (
            JsonlWriter(self.store.levels_path) as level_writer,
            JsonlWriter(self.store.songs_path) as song_writer,
            JsonlWriter(self.store.level_audio_path) as level_audio_writer,
            JsonlWriter(self.store.comments_path) as comment_writer,
            JsonlWriter(self.store.discovered_path) as discovered_writer,
            JsonlWriter(self.store.failures_path) as failure_writer,
            JsonlWriter(self.store.rejected_levels_path) as rejected_writer,
            JsonlWriter(self.store.metrics_path) as metrics_writer,
        ):
            self.metrics_writer = metrics_writer
            self.last_metrics_at = time.monotonic()
            worker_count = max(self.concurrency, 1)
            workers = [
                asyncio.create_task(
                    self._download_worker(
                        index,
                        queue,
                        level_writer,
                        song_writer,
                        level_audio_writer,
                        comment_writer,
                        rejected_writer,
                        failure_writer,
                    )
                )
                for index in range(worker_count)
            ]

            try:
                await self._enqueue_checkpoint_candidates(candidates, queue)
                for source in self.sources:
                    await self._discover_source(source, queue, discovered_writer, song_writer, failure_writer)
                await queue.join()
                await self._emit_metrics(force=True)
            except asyncio.CancelledError:
                self.request_shutdown("cancelled", force=True)
                raise
            finally:
                try:
                    if self.force_shutdown_requested:
                        await self._cancel_workers(workers)
                    else:
                        await self._stop_workers(queue, workers)
                except asyncio.CancelledError:
                    self.request_shutdown("cancelled", force=True)
                    await self._cancel_workers(workers)
                    raise
                finally:
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._emit_metrics(force=True)

        return self.stats

    async def _enqueue_checkpoint_candidates(
        self,
        candidates: dict[int, Candidate],
        queue: asyncio.Queue[Candidate | None],
    ) -> None:
        ordered_candidates = sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.sequence,
                candidate.source,
                candidate.page,
                candidate.level_id,
            ),
        )
        for candidate in ordered_candidates:
            if self._should_stop():
                return
            if candidate.level_id in self.downloaded_ids or candidate.level_id in self.rejected_ids:
                continue
            async with self.id_lock:
                if candidate.level_id in self.seen_candidate_ids:
                    continue
                self.seen_candidate_ids.add(candidate.level_id)
            await queue.put(candidate)
            self.stats.queued_from_checkpoint += 1

    async def _discover_source(
        self,
        source: DiscoverySource,
        queue: asyncio.Queue[Candidate | None],
        discovered_writer: JsonlWriter,
        song_writer: JsonlWriter,
        failure_writer: JsonlWriter,
    ) -> None:
        source_state = self.state["sources"].setdefault(source.name, {})
        if source_state.get("done"):
            LOGGER.info("source %s already marked done", source.name)
            return

        start_page = int(source_state.get("next_page", 0))
        end_page = start_page + self.pages_per_source

        for page in range(start_page, end_page):
            if self._should_stop():
                return

            params = dict(source.params)
            params.update({"page": page, "total": 0})

            try:
                raw = await self._timed_request(self.client.search_levels(params))
                parsed = parse_search_response(raw)
            except ShutdownRequested:
                await self._emit_metrics()
                return
            except GDRequestError as exc:
                self.stats.failed += 1
                await self._log_failure(
                    failure_writer,
                    stage="discovery",
                    source=source.name,
                    page=page,
                    error=str(exc),
                    endpoint=exc.endpoint,
                    payload=exc.payload,
                    status=exc.status,
                    response_text=exc.response_text,
                )
                await self._emit_metrics()
                return
            except Exception as exc:  # noqa: BLE001
                self.stats.failed += 1
                await self._log_failure(
                    failure_writer,
                    stage="discovery",
                    source=source.name,
                    page=page,
                    error=str(exc),
                )
                await self._emit_metrics()
                return

            if self._should_stop():
                await self._emit_metrics()
                return

            if not parsed.levels:
                await self._mark_source(source.name, next_page=page, done=True)
                LOGGER.info("source %s ended at page %s", source.name, page)
                return

            for song in parsed.songs:
                if self._should_stop():
                    await self._emit_metrics()
                    return
                await self._write_song(song_writer, song, source=source.name)

            for raw_level in parsed.levels:
                if self._should_stop():
                    await self._emit_metrics()
                    return
                if is_auto_level(raw_level):
                    self.stats.auto_skipped += 1
                    self.stats.skipped += 1
                    continue
                candidate = candidate_from_search_level(
                    raw_level,
                    parsed.creators,
                    source=source.name,
                    page=page,
                    raw_search=raw_level.get("_raw", ""),
                )
                if candidate is None:
                    self.stats.skipped += 1
                    continue
                if not source_name_matches(candidate.name, source):
                    self.stats.skipped += 1
                    continue

                async with self.id_lock:
                    if (
                        candidate.level_id in self.downloaded_ids
                        or candidate.level_id in self.rejected_ids
                        or candidate.level_id in self.seen_candidate_ids
                    ):
                        self.stats.duplicates_skipped += 1
                        self.stats.skipped += 1
                        continue
                    candidate.sequence = self.next_sequence
                    self.next_sequence += 1
                    self.seen_candidate_ids.add(candidate.level_id)

                if self._should_stop():
                    await self._emit_metrics()
                    return
                await discovered_writer.write(candidate.to_json())
                self.stats.discovered += 1
                if self._should_stop():
                    await self._emit_metrics()
                    return
                await queue.put(candidate)

            if self._should_stop():
                await self._emit_metrics()
                return
            await self._mark_source(source.name, next_page=page + 1, done=False)
            if self._should_stop():
                await self._emit_metrics()
                return
            await self._emit_metrics()
            LOGGER.info("source %s page %s discovered %s levels", source.name, page, len(parsed.levels))

    async def _download_worker(
        self,
        index: int,
        queue: asyncio.Queue[Candidate | None],
        level_writer: JsonlWriter,
        song_writer: JsonlWriter,
        level_audio_writer: JsonlWriter,
        comment_writer: JsonlWriter,
        rejected_writer: JsonlWriter,
        failure_writer: JsonlWriter,
    ) -> None:
        while True:
            candidate = await queue.get()
            try:
                if candidate is None:
                    return
                if self._should_stop():
                    continue
                async with self.download_semaphore:
                    if self._should_stop():
                        continue
                    await self._download_candidate(
                        candidate,
                        level_writer,
                        song_writer,
                        level_audio_writer,
                        comment_writer,
                        rejected_writer,
                        failure_writer,
                    )
            finally:
                queue.task_done()

    async def _download_candidate(
        self,
        candidate: Candidate,
        level_writer: JsonlWriter,
        song_writer: JsonlWriter,
        level_audio_writer: JsonlWriter,
        comment_writer: JsonlWriter,
        rejected_writer: JsonlWriter,
        failure_writer: JsonlWriter,
    ) -> None:
        if candidate.level_id in self.downloaded_ids or candidate.level_id in self.rejected_ids:
            self.stats.duplicates_skipped += 1
            self.stats.skipped += 1
            return
        if not await self._reserve_save_slot():
            self.stats.skipped += 1
            return

        try:
            raw = await self._timed_request(self.client.download_level(candidate.level_id))
            download = parse_download_response(raw)
            validation = validate_level_data(download.level.get("4", ""))
        except ShutdownRequested:
            await self._release_save_slot()
            await self._emit_metrics()
            return
        except GDRequestError as exc:
            await self._release_save_slot()
            self.stats.failed += 1
            await self._log_failure(
                failure_writer,
                stage="download",
                level_id=candidate.level_id,
                source=candidate.source,
                error=str(exc),
                endpoint=exc.endpoint,
                payload=exc.payload,
                status=exc.status,
                response_text=exc.response_text,
            )
            await self._emit_metrics()
            return
        except (ParseError, ValidationError) as exc:
            await self._release_save_slot()
            self.stats.rejected += 1
            self.rejected_ids.add(candidate.level_id)
            await rejected_writer.write(self._rejected_record(candidate, exc))
            await self._log_failure(
                failure_writer,
                stage="validation",
                level_id=candidate.level_id,
                source=candidate.source,
                reason=exc.reason,
                error=exc.detail,
            )
            await self._emit_metrics()
            return
        except Exception as exc:  # noqa: BLE001
            await self._release_save_slot()
            self.stats.failed += 1
            await self._log_failure(
                failure_writer,
                stage="download",
                level_id=candidate.level_id,
                source=candidate.source,
                error=str(exc),
            )
            await self._emit_metrics()
            return

        record = level_record(download, candidate, validation)
        audio_record = level_audio_record(download, candidate)
        if self.download_audio:
            await self._cache_level_audio(audio_record, failure_writer)
        async with self.id_lock:
            if self._target_reached():
                self.reserved_saves = max(self.reserved_saves - 1, 0)
                self.stats.skipped += 1
                return
            await level_writer.write(record)
            await level_audio_writer.write(audio_record)
            await append_id_async(self.store.downloaded_ids_path, candidate.level_id)
            self.downloaded_ids.add(candidate.level_id)
            self.stats.saved += 1
            self.reserved_saves = max(self.reserved_saves - 1, 0)
            if self._should_stop():
                self.stop_event.set()

        for song in download.songs:
            await self._write_song(song_writer, song, source=candidate.source)

        if self.include_comments and self.comments_pages > 0 and not self.shutdown_requested:
            await self._download_comments(candidate, comment_writer, failure_writer)

        await self._emit_metrics()
        LOGGER.info("saved level %s from worker", candidate.level_id)

    async def _write_song(self, writer: JsonlWriter, song: Song, *, source: str) -> None:
        async with self.song_lock:
            if song.song_id in self.song_ids:
                return
            await writer.write(
                {
                    "dataset_version": DATASET_VERSION,
                    "scraper_version": SCRAPER_VERSION,
                    "song_id": song.song_id,
                    "name": song.name,
                    "artist": song.artist,
                    "size": song.size,
                    "download_url": song.download_url,
                    "raw": song.raw,
                    "parsed": song.data,
                    "source": source,
                    "seen_at": epoch_seconds(),
                }
            )
            self.song_ids.add(song.song_id)
            self.stats.songs_saved += 1

    async def _cache_level_audio(self, audio_record: dict[str, Any], failure_writer: JsonlWriter) -> None:
        audio = dict(audio_record.get("audio") or {})
        url = str(audio.get("download_url") or "")
        if not url:
            audio["cached"] = False
            audio_record["audio"] = audio
            return

        song_id = int(audio_record.get("song_id") or 0)
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in {".mp3", ".ogg", ".wav"}:
            suffix = ".mp3"
        audio_path = self.store.audio_dir / f"{song_id}{suffix}"
        audio["local_path"] = str(audio_path)
        if audio_path.exists() and audio_path.stat().st_size > 0:
            audio["cached"] = True
            audio["bytes"] = audio_path.stat().st_size
            audio_record["audio"] = audio
            return

        download_audio = getattr(self.client, "download_audio", None)
        if not callable(download_audio):
            audio["cached"] = False
            audio["cache_error"] = "client_audio_download_unavailable"
            audio_record["audio"] = audio
            return

        try:
            payload = await download_audio(url)
        except Exception as exc:  # noqa: BLE001
            audio["cached"] = False
            audio["cache_error"] = str(exc)
            audio_record["audio"] = audio
            await self._log_failure(
                failure_writer,
                stage="audio",
                level_id=int(audio_record.get("level_id") or 0),
                source=str(audio_record.get("source") or ""),
                error=str(exc),
            )
            return

        audio_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = audio_path.with_suffix(audio_path.suffix + ".tmp")
        tmp_path.write_bytes(payload)
        tmp_path.replace(audio_path)
        audio["cached"] = True
        audio["bytes"] = len(payload)
        audio_record["audio"] = audio

    async def _download_comments(
        self,
        candidate: Candidate,
        comment_writer: JsonlWriter,
        failure_writer: JsonlWriter,
    ) -> None:
        async with self.comment_semaphore:
            for page in range(self.comments_pages):
                if self.shutdown_requested:
                    return
                try:
                    raw = await self._timed_request(
                        self.client.get_comments(
                            candidate.level_id,
                            page=page,
                            mode=self.comments_mode,
                            count=self.comments_count,
                        )
                    )
                except GDRequestError as exc:
                    self.stats.failed += 1
                    await self._log_failure(
                        failure_writer,
                        stage="comments",
                        level_id=candidate.level_id,
                        source=candidate.source,
                        page=page,
                        error=str(exc),
                        endpoint=exc.endpoint,
                        payload=exc.payload,
                        status=exc.status,
                        response_text=exc.response_text,
                    )
                    await self._emit_metrics()
                    return

                comments, _page_info = parse_comment_response(raw)
                if not comments:
                    return

                await comment_writer.write(
                    comment_page_record(
                        level_id=candidate.level_id,
                        source=candidate.source,
                        source_page=candidate.page,
                        page=page,
                        mode=self.comments_mode,
                        count=self.comments_count,
                        raw=raw,
                    )
                )
                self.stats.comments_saved += len(comments)

    async def _mark_source(self, source_name: str, *, next_page: int, done: bool) -> None:
        async with self.state_lock:
            self.state.setdefault("sources", {})[source_name] = {
                "next_page": next_page,
                "done": done,
                "updated_at": epoch_seconds(),
            }
            await self.store.save_state(self.state)

    async def _log_failure(self, writer: JsonlWriter, **value: Any) -> None:
        value.setdefault("dataset_version", DATASET_VERSION)
        value.setdefault("scraper_version", SCRAPER_VERSION)
        value.setdefault("timestamp", epoch_seconds())
        await writer.write(value)

    async def _reserve_save_slot(self) -> bool:
        if self.target_count <= 0:
            return True
        async with self.id_lock:
            if self.stats.saved + self.reserved_saves >= self.target_count:
                return False
            self.reserved_saves += 1
            return True

    async def _release_save_slot(self) -> None:
        if self.target_count <= 0:
            return
        async with self.id_lock:
            self.reserved_saves = max(self.reserved_saves - 1, 0)

    def _rejected_record(self, candidate: Candidate, exc: ParseError | ValidationError) -> dict[str, Any]:
        return {
            "dataset_version": DATASET_VERSION,
            "scraper_version": SCRAPER_VERSION,
            "level_id": candidate.level_id,
            "reason": exc.reason,
            "detail": exc.detail,
            "timestamp": epoch_seconds(),
            "candidate_sequence": candidate.sequence,
            "source": candidate.source,
            "source_page": candidate.page,
        }

    async def _timed_request(self, awaitable: Any) -> str:
        started = time.monotonic()
        record_request = True
        try:
            return await awaitable
        except ShutdownRequested:
            record_request = False
            raise
        finally:
            if record_request:
                elapsed = time.monotonic() - started
                async with self.stats_lock:
                    self.stats.record_request(elapsed)

    async def _stop_workers(
        self,
        queue: asyncio.Queue[Candidate | None],
        workers: list[asyncio.Task[None]],
    ) -> None:
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    async def _cancel_workers(self, workers: list[asyncio.Task[None]]) -> None:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    async def _emit_metrics(self, *, force: bool = False) -> None:
        if self.metrics_writer is None:
            return
        if self.metrics_interval < 0 and not force:
            return

        now = time.monotonic()
        if not force and self.metrics_interval > 0 and now - self.last_metrics_at < self.metrics_interval:
            return
        self.last_metrics_at = now
        await self.metrics_writer.write(self.stats.to_json())

    def _should_stop(self) -> bool:
        return self.stop_event.is_set() or self._target_reached()

    def _target_reached(self) -> bool:
        return self.target_count > 0 and self.stats.saved >= self.target_count
