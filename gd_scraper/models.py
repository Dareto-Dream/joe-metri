from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from . import __version__


DATASET_VERSION = 1
SCRAPER_VERSION = __version__


def epoch_seconds() -> int:
    return int(time.time())


@dataclass(frozen=True)
class Creator:
    player_id: int
    username: str
    account_id: int


@dataclass(frozen=True)
class Song:
    song_id: int
    raw: str
    data: dict[str, str]

    @property
    def name(self) -> str:
        return self.data.get("2", "")

    @property
    def artist(self) -> str:
        return self.data.get("4", "")


@dataclass
class Candidate:
    level_id: int
    source: str
    page: int
    sequence: int = 0
    name: str = ""
    author: str = "-"
    player_id: int = 0
    account_id: int = 0
    raw_search: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "source": self.source,
            "page": self.page,
            "sequence": self.sequence,
            "name": self.name,
            "author": self.author,
            "player_id": self.player_id,
            "account_id": self.account_id,
            "raw_search": self.raw_search,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Candidate":
        return cls(
            level_id=int(value["level_id"]),
            source=str(value.get("source", "checkpoint")),
            page=int(value.get("page", 0)),
            sequence=int(value.get("sequence") or 0),
            name=str(value.get("name", "")),
            author=str(value.get("author", "-")),
            player_id=int(value.get("player_id") or 0),
            account_id=int(value.get("account_id") or 0),
            raw_search=str(value.get("raw_search", "")),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SearchResponse:
    raw: str
    levels: list[dict[str, str]]
    creators: dict[int, Creator]
    songs: list[Song]
    page_info: dict[str, int] | None


@dataclass(frozen=True)
class DownloadResponse:
    raw: str
    level: dict[str, str]
    hash1: str = ""
    hash2: str = ""
    user: str = ""
    songs: list[Song] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    object_count: int
    decoded_length: int


@dataclass
class RunStats:
    started_at: float = field(default_factory=time.time)
    already_downloaded: int = 0
    queued_from_checkpoint: int = 0
    discovered: int = 0
    saved: int = 0
    rejected: int = 0
    failed: int = 0
    skipped: int = 0
    duplicates_skipped: int = 0
    auto_skipped: int = 0
    songs_saved: int = 0
    comments_saved: int = 0
    requests: int = 0
    response_time_seconds: float = 0.0

    def record_request(self, elapsed_seconds: float) -> None:
        self.requests += 1
        self.response_time_seconds += elapsed_seconds

    @property
    def elapsed_minutes(self) -> float:
        elapsed_seconds = max(time.time() - self.started_at, 1e-9)
        return elapsed_seconds / 60.0

    @property
    def avg_response_time(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.response_time_seconds / self.requests

    def to_json(self) -> dict[str, int | float]:
        return {
            "dataset_version": DATASET_VERSION,
            "scraper_version": SCRAPER_VERSION,
            "timestamp": epoch_seconds(),
            "already_downloaded": self.already_downloaded,
            "queued_from_checkpoint": self.queued_from_checkpoint,
            "levels_discovered": self.discovered,
            "levels_saved": self.saved,
            "levels_rejected": self.rejected,
            "levels_failed": self.failed,
            "duplicates_skipped": self.duplicates_skipped,
            "avg_response_time": round(self.avg_response_time, 6),
            "requests_per_minute": round(self.requests / self.elapsed_minutes, 6),
            "levels_per_minute": round(self.saved / self.elapsed_minutes, 6),
            "requests": self.requests,
            "discovered": self.discovered,
            "saved": self.saved,
            "rejected": self.rejected,
            "failed": self.failed,
            "skipped": self.skipped,
            "auto_skipped": self.auto_skipped,
            "songs_saved": self.songs_saved,
            "comments_saved": self.comments_saved,
        }
