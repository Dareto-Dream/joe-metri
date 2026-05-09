from __future__ import annotations

import base64
import binascii
import re
import time
import zlib
from typing import Any

from .models import (
    DATASET_VERSION,
    SCRAPER_VERSION,
    Candidate,
    Creator,
    DownloadResponse,
    SearchResponse,
    Song,
    ValidationResult,
)


DIFFICULTY_BY_NUMERATOR = {
    0: "NA",
    10: "EASY",
    20: "NORMAL",
    30: "HARD",
    40: "HARDER",
    50: "INSANE",
}

DEMON_DIFFICULTY_BY_TYPE = {
    3: "EASY_DEMON",
    4: "MEDIUM_DEMON",
    5: "INSANE_DEMON",
    6: "EXTREME_DEMON",
}

LENGTH_BY_ID = {
    0: "TINY",
    1: "SHORT",
    2: "MEDIUM",
    3: "LONG",
    4: "XL",
    5: "PLATFORMER",
}


class ParseError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


class ValidationError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


def epoch_now() -> int:
    return int(time.time())


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_key_value_pairs(text: str, delimiter: str = ":") -> dict[str, str]:
    parts = text.split(delimiter)
    result: dict[str, str] = {}
    for index in range(0, len(parts) - 1, 2):
        result[parts[index]] = parts[index + 1]
    return result


def parse_creators(section: str) -> dict[int, Creator]:
    creators: dict[int, Creator] = {}
    if not section:
        return creators

    for raw_creator in section.split("|"):
        if not raw_creator or raw_creator.startswith("~"):
            continue
        parts = raw_creator.split(":")
        if len(parts) < 3:
            continue
        player_id = to_int(parts[0])
        if player_id <= 0:
            continue
        creators[player_id] = Creator(
            player_id=player_id,
            username=parts[1],
            account_id=to_int(parts[2]),
        )
    return creators


def split_song_section(section: str) -> list[str]:
    section = section.strip()
    if not section or section == "~":
        return []
    return [part for part in re.split(r":(?=~?1~\|~)", section) if part]


def parse_song(raw_song: str) -> Song | None:
    cleaned = raw_song.strip()
    if cleaned.startswith("~"):
        cleaned = cleaned[1:]
    if cleaned.endswith("~"):
        cleaned = cleaned[:-1]
    data = parse_key_value_pairs(cleaned, "~|~")
    song_id = to_int(data.get("1"))
    if song_id <= 0:
        return None
    return Song(song_id=song_id, raw=raw_song, data=data)


def parse_songs(section: str) -> list[Song]:
    songs: list[Song] = []
    for raw_song in split_song_section(section):
        song = parse_song(raw_song)
        if song is not None:
            songs.append(song)
    return songs


def parse_page_info(section: str) -> dict[str, int] | None:
    if not section:
        return None
    parts = section.split(":")
    if len(parts) < 3:
        return None
    return {
        "total": to_int(parts[0]),
        "offset": to_int(parts[1]),
        "amount": to_int(parts[2]),
    }


def parse_search_response(raw: str) -> SearchResponse:
    if not raw or raw.strip() == "-1":
        return SearchResponse(raw=raw, levels=[], creators={}, songs=[], page_info=None)

    sections = raw.split("#")
    levels_section = sections[0] if len(sections) > 0 else ""
    creators_section = sections[1] if len(sections) > 1 else ""
    songs_section = sections[2] if len(sections) > 2 else ""
    page_info_section = sections[3] if len(sections) > 3 else ""

    levels: list[dict[str, str]] = []
    for raw_level in levels_section.split("|"):
        if not raw_level:
            continue
        parsed_level = parse_key_value_pairs(raw_level)
        parsed_level["_raw"] = raw_level
        levels.append(parsed_level)

    return SearchResponse(
        raw=raw,
        levels=levels,
        creators=parse_creators(creators_section),
        songs=parse_songs(songs_section),
        page_info=parse_page_info(page_info_section),
    )


def parse_download_response(raw: str) -> DownloadResponse:
    if not raw:
        raise ParseError("empty_response", "download response was empty")
    if raw.strip() == "-1":
        raise ParseError("not_found_response", "download response was -1")

    sections = raw.split("#")
    if not sections or not sections[0]:
        raise ParseError("missing_level_object", "download response did not contain a level object")

    level = parse_key_value_pairs(sections[0])
    if not level.get("1"):
        raise ParseError("missing_level_id", "download response level object is missing level id")

    return DownloadResponse(
        raw=raw,
        level=level,
        hash1=sections[1] if len(sections) > 1 else "",
        hash2=sections[2] if len(sections) > 2 else "",
        user=sections[3] if len(sections) > 3 else "",
        songs=parse_songs(sections[4] if len(sections) > 4 else ""),
    )


def parse_comment_response(raw: str) -> tuple[list[str], dict[str, int] | None]:
    if not raw or raw.strip() == "-1":
        return [], None

    comments_section, _, page_info_section = raw.partition("#")
    comments = [comment for comment in comments_section.split("|") if comment]
    return comments, parse_page_info(page_info_section)


def comment_page_record(
    *,
    level_id: int,
    source: str,
    source_page: int,
    page: int,
    mode: int,
    count: int,
    raw: str,
) -> dict[str, Any]:
    comments, page_info = parse_comment_response(raw)
    return {
        "dataset_version": DATASET_VERSION,
        "scraper_version": SCRAPER_VERSION,
        "level_id": level_id,
        "source": source,
        "source_page": source_page,
        "page": page,
        "mode": mode,
        "count": count,
        "comment_count": len(comments),
        "page_info": page_info,
        "fetched_at": epoch_now(),
        "raw": raw,
    }


def difficulty_label(level: dict[str, str]) -> str:
    if to_int(level.get("25")) > 0:
        return "AUTO"
    if to_int(level.get("17")) > 0:
        return DEMON_DIFFICULTY_BY_TYPE.get(to_int(level.get("43")), "HARD_DEMON")
    if to_int(level.get("8")) == 0:
        return "NA"
    return DIFFICULTY_BY_NUMERATOR.get(to_int(level.get("9")), "NA")


def song_id_and_type(level: dict[str, str]) -> tuple[int, str]:
    custom_song = to_int(level.get("35"))
    if custom_song > 0:
        return custom_song, "custom"
    return to_int(level.get("12")) + 1, "official"


def is_auto_level(level: dict[str, str]) -> bool:
    return to_int(level.get("25")) > 0


def candidate_from_search_level(
    level: dict[str, str],
    creators: dict[int, Creator],
    source: str,
    page: int,
    raw_search: str,
    sequence: int = 0,
) -> Candidate | None:
    level_id = to_int(level.get("1"))
    if level_id <= 0:
        return None

    player_id = to_int(level.get("6"))
    creator = creators.get(player_id)
    song_id, song_type = song_id_and_type(level)
    metadata: dict[str, Any] = {
        "difficulty": difficulty_label(level),
        "downloads": to_int(level.get("10")),
        "likes": to_int(level.get("14")),
        "stars": to_int(level.get("18")),
        "length": LENGTH_BY_ID.get(to_int(level.get("15")), "UNKNOWN"),
        "version": to_int(level.get("5")),
        "game_version": to_int(level.get("13")),
        "object_count_hint": to_int(level.get("45")),
        "song_id": song_id,
        "song_type": song_type,
        "featured_score": to_int(level.get("19")),
        "epic_score": to_int(level.get("42")),
    }

    return Candidate(
        level_id=level_id,
        source=source,
        page=page,
        sequence=sequence,
        name=level.get("2", ""),
        author=creator.username if creator else "-",
        player_id=player_id,
        account_id=creator.account_id if creator else 0,
        raw_search=raw_search,
        metadata=metadata,
    )


def _padded_base64(value: str) -> str:
    padded = value + ("=" * (-len(value) % 4))
    return padded


def decode_level_data(level_data: str) -> str:
    try:
        normalized = _padded_base64(level_data).replace("-", "+").replace("_", "/")
        decoded = base64.b64decode(normalized.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ValidationError("base64_decode_failed", str(exc)) from exc

    try:
        decompressed = zlib.decompress(decoded, 15 | 32)
    except zlib.error as exc:
        raise ValidationError("zlib_decode_failed", str(exc)) from exc

    try:
        return decompressed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("unicode_decode_failed", str(exc)) from exc


def validate_level_data(level_data: str) -> ValidationResult:
    if not level_data:
        raise ValidationError("level_data_missing", "level_data is missing")

    decoded = decode_level_data(level_data)

    object_count = sum(1 for part in decoded.split(";")[1:] if part.strip())
    if object_count <= 0:
        raise ValidationError("zero_objects", "decoded level has zero objects")

    return ValidationResult(object_count=object_count, decoded_length=len(decoded))


def level_record(
    download: DownloadResponse,
    candidate: Candidate,
    validation: ValidationResult,
) -> dict[str, Any]:
    level = download.level
    level_id = to_int(level.get("1"))
    song_id, song_type = song_id_and_type(level)

    user_author = ""
    if download.user:
        user_parts = download.user.split(":")
        if len(user_parts) >= 2:
            user_author = user_parts[1]

    return {
        "dataset_version": DATASET_VERSION,
        "scraper_version": SCRAPER_VERSION,
        "level_id": level_id,
        "name": level.get("2") or candidate.name,
        "author": user_author or candidate.author or "-",
        "difficulty": difficulty_label(level),
        "downloads": to_int(level.get("10")),
        "likes": to_int(level.get("14")),
        "song_id": song_id,
        "song_type": song_type,
        "source": candidate.source,
        "source_page": candidate.page,
        "object_count": validation.object_count,
        "level_hash": download.hash1,
        "fetched_at": epoch_now(),
        "level_data": level.get("4", ""),
        "raw": download.raw,
        "metadata": {
            "download_hash2": download.hash2,
            "candidate_sequence": candidate.sequence,
            "player_id": to_int(level.get("6")) or candidate.player_id,
            "account_id": candidate.account_id,
            "description_b64": level.get("3", ""),
            "version": to_int(level.get("5")),
            "game_version": to_int(level.get("13")),
            "stars": to_int(level.get("18")),
            "length": LENGTH_BY_ID.get(to_int(level.get("15")), "UNKNOWN"),
            "coins": to_int(level.get("37")),
            "verified_coins": to_int(level.get("38")) > 0,
            "featured_score": to_int(level.get("19")),
            "epic_score": to_int(level.get("42")),
            "object_count_hint": to_int(level.get("45")),
            "decoded_length": validation.decoded_length,
            "search": candidate.metadata,
        },
    }
