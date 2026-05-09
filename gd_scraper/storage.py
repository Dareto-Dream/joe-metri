from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Iterable

import aiofiles
import orjson

from .models import Candidate


def dumps_jsonl(value: dict[str, Any]) -> bytes:
    return orjson.dumps(value) + b"\n"


def dumps_pretty(value: dict[str, Any]) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)


def loads_json(value: str | bytes) -> Any:
    return orjson.loads(value)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self._file: Any = None

    async def __aenter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = await aiofiles.open(self.path, "ab")
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._file is not None:
            close_task = asyncio.create_task(self._file.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                await close_task
                raise

    async def write(self, value: dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("JsonlWriter must be used as an async context manager")
        line = dumps_jsonl(value)
        async with self._lock:
            await self._file.write(line)
            await self._file.flush()


class CheckpointStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.tokenized_dir = data_dir / "tokenized"
        self.checkpoint_dir = data_dir / "checkpoints"
        self.logs_dir = data_dir / "logs"
        self.levels_path = self.raw_dir / "levels.jsonl"
        self.comments_path = self.raw_dir / "comments.jsonl"
        self.songs_path = self.raw_dir / "songs.jsonl"
        self.parsed_levels_path = self.processed_dir / "parsed_levels.jsonl"
        self.mechanics_tokens_path = self.tokenized_dir / "mechanics_tokens.jsonl"
        self.state_path = self.checkpoint_dir / "state.json"
        self.discovered_path = self.checkpoint_dir / "discovered_levels.jsonl"
        self.downloaded_ids_path = self.checkpoint_dir / "downloaded_level_ids.txt"
        self.rejected_levels_path = self.checkpoint_dir / "rejected_levels.jsonl"
        self.legacy_rejected_ids_path = self.checkpoint_dir / "rejected_level_ids.txt"
        self.failures_path = self.checkpoint_dir / "failed_requests.jsonl"
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.scraper_log_path = self.logs_dir / "scraper.log"

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.levels_path.touch(exist_ok=True)
        self.comments_path.touch(exist_ok=True)
        self.songs_path.touch(exist_ok=True)
        self.parsed_levels_path.touch(exist_ok=True)
        self.mechanics_tokens_path.touch(exist_ok=True)
        self.discovered_path.touch(exist_ok=True)
        self.downloaded_ids_path.touch(exist_ok=True)
        self.rejected_levels_path.touch(exist_ok=True)
        self.failures_path.touch(exist_ok=True)
        self.metrics_path.touch(exist_ok=True)

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"sources": {}}
        try:
            return loads_json(self.state_path.read_bytes())
        except orjson.JSONDecodeError:
            return {"sources": {}}

    async def save_state(self, state: dict[str, Any]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_bytes(dumps_pretty(state))
        tmp_path.replace(self.state_path)


def load_ids_from_jsonl(path: Path, key: str) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = loads_json(line)
            except orjson.JSONDecodeError:
                continue
            try:
                ids.add(int(value[key]))
            except (KeyError, TypeError, ValueError):
                continue
    return ids


def load_ids_from_text(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                ids.add(int(line.strip()))
            except ValueError:
                continue
    return ids


async def append_id_async(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "a", encoding="utf-8", newline="\n") as handle:
        await handle.write(f"{value}\n")
        await handle.flush()


def load_candidates(path: Path) -> dict[int, Candidate]:
    candidates: dict[int, Candidate] = {}
    if not path.exists():
        return candidates

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = Candidate.from_json(loads_json(line))
            except (orjson.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            candidates[candidate.level_id] = candidate
    return candidates


def load_rejected_ids(path: Path, legacy_path: Path | None = None) -> set[int]:
    ids = load_ids_from_jsonl(path, "level_id")
    if legacy_path is not None:
        ids.update(load_ids_from_text(legacy_path))
    return ids


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def coerce_source_names(raw_sources: Iterable[str]) -> list[str]:
    names: list[str] = []
    for raw_source in raw_sources:
        for part in raw_source.split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names
