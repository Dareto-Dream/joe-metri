from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .gd_objects import (
    BASE_VOCAB_TOKENS,
    MAX_WIDTH,
    PRIMARY_MECHANIC_TOKENS,
    SOLID_TOKENS,
    TOKENIZER_VERSION,
    X_STEP_RESOLUTION,
    Y_LANES,
    difficulty_token,
    is_ignored_object_id,
    map_object_id,
    ordered_unique,
    width_token,
    y_token,
)
from .models import DATASET_VERSION, epoch_seconds
from .parser import ValidationError, decode_level_data, to_int
from .reconstruction import validate_token_grammar
from .storage import CheckpointStore, dumps_jsonl, dumps_pretty, loads_json


PARSER_VERSION = "0.1.0"


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class GDObject:
    object_id: int
    x: float
    y: float
    rotation: float = 0.0
    scale: float = 1.0

    def to_json(self) -> dict[str, int | float]:
        return {
            "object_id": self.object_id,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "rotation": round(self.rotation, 6),
            "scale": round(self.scale, 6),
        }


@dataclass(frozen=True)
class GameplayObject:
    object_id: int
    token: str
    x: float
    y: float
    x_step: int
    y_lane: int
    width: int = 1

    def to_json(self) -> dict[str, int | float | str]:
        value: dict[str, int | float | str] = {
            "object_id": self.object_id,
            "token": self.token,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "x_step": self.x_step,
            "y_lane": self.y_lane,
        }
        if self.token in SOLID_TOKENS:
            value["width"] = self.width
        return value

    def event_tokens(self) -> list[str]:
        tokens = [self.token, y_token(self.y_lane)]
        if self.token in SOLID_TOKENS:
            tokens.append(width_token(self.width))
        return tokens


@dataclass(frozen=True)
class TokenizerConfig:
    x_step_resolution: int = X_STEP_RESOLUTION
    y_lanes: int = Y_LANES
    min_gameplay_objects: int = 20
    min_tokens: int = 50
    max_unknown_ratio: float = 0.995
    max_token_length: int = 12_000


@dataclass
class TokenizationArtifacts:
    parsed_record: dict[str, Any] | None
    gameplay_record: dict[str, Any] | None
    token_record: dict[str, Any] | None
    stats_record: dict[str, Any]


def parse_gd_object(raw_object: str) -> GDObject | None:
    raw_parts = raw_object.split(",")
    if len(raw_parts) < 2:
        return None

    properties: dict[str, str] = {}
    for index in range(0, len(raw_parts) - 1, 2):
        key = raw_parts[index]
        value = raw_parts[index + 1]
        if key:
            properties[key] = value

    object_id = to_int(properties.get("1"))
    if object_id <= 0:
        return None

    return GDObject(
        object_id=object_id,
        x=to_float(properties.get("2")),
        y=to_float(properties.get("3")),
        rotation=to_float(properties.get("6")),
        scale=to_float(properties.get("32"), 1.0),
    )


def parse_level_objects(decoded_level_data: str) -> tuple[str, list[GDObject]]:
    sections = decoded_level_data.split(";")
    header = sections[0] if sections else ""
    objects: list[GDObject] = []
    for raw_object in sections[1:]:
        raw_object = raw_object.strip()
        if not raw_object:
            continue
        parsed = parse_gd_object(raw_object)
        if parsed is not None:
            objects.append(parsed)
    return header, objects


def quantize_step(value: float, origin: float, resolution: int) -> int:
    return max(0, int(round((value - origin) / resolution)))


def quantize_lane(value: float, origin: float, resolution: int, lane_count: int) -> int:
    lane = int(round((value - origin) / resolution))
    return max(0, min(lane_count - 1, lane))


def normalize_gameplay_objects(objects: list[GDObject], config: TokenizerConfig) -> list[GameplayObject]:
    mapped: list[tuple[GDObject, str]] = []
    for gd_object in objects:
        token = map_object_id(gd_object.object_id)
        if token is not None:
            mapped.append((gd_object, token))

    if not mapped:
        return []

    x_origin = min(gd_object.x for gd_object, _token in mapped)
    y_origin = percentile([gd_object.y for gd_object, _token in mapped], 0.05)
    normalized: list[GameplayObject] = []
    for gd_object, token in mapped:
        normalized.append(
            GameplayObject(
                object_id=gd_object.object_id,
                token=token,
                x=gd_object.x,
                y=gd_object.y,
                x_step=quantize_step(gd_object.x, x_origin, config.x_step_resolution),
                y_lane=quantize_lane(gd_object.y, y_origin, config.x_step_resolution, config.y_lanes),
            )
        )
    return merge_solid_widths(normalized)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * max(0.0, min(1.0, fraction))))
    return ordered[index]


def merge_solid_widths(objects: list[GameplayObject]) -> list[GameplayObject]:
    solids_by_key: dict[tuple[str, int], set[int]] = defaultdict(set)
    solid_example_by_key: dict[tuple[str, int, int], GameplayObject] = {}
    passthrough: dict[tuple[str, int, int], GameplayObject] = {}

    for item in objects:
        if item.token in SOLID_TOKENS:
            key = (item.token, item.y_lane)
            solids_by_key[key].add(item.x_step)
            solid_example_by_key.setdefault((item.token, item.y_lane, item.x_step), item)
            continue
        passthrough.setdefault((item.token, item.x_step, item.y_lane), item)

    merged: list[GameplayObject] = list(passthrough.values())
    for (token, lane), steps in solids_by_key.items():
        sorted_steps = sorted(steps)
        if not sorted_steps:
            continue
        run_start = sorted_steps[0]
        previous = sorted_steps[0]
        for step in sorted_steps[1:]:
            if step == previous + 1:
                previous = step
                continue
            merged.append(solid_run_object(solid_example_by_key, token, lane, run_start, previous))
            run_start = step
            previous = step
        merged.append(solid_run_object(solid_example_by_key, token, lane, run_start, previous))

    return sorted(merged, key=lambda item: (item.x_step, token_sort_key(item.token), item.y_lane, item.object_id))


def solid_run_object(
    examples: dict[tuple[str, int, int], GameplayObject],
    token: str,
    lane: int,
    start_step: int,
    end_step: int,
) -> GameplayObject:
    example = examples[(token, lane, start_step)]
    return GameplayObject(
        object_id=example.object_id,
        token=token,
        x=example.x,
        y=example.y,
        x_step=start_step,
        y_lane=lane,
        width=max(1, min(MAX_WIDTH, end_step - start_step + 1)),
    )


def token_sort_key(token: str) -> int:
    try:
        return PRIMARY_MECHANIC_TOKENS.index(token)
    except ValueError:
        return len(PRIMARY_MECHANIC_TOKENS)


def emit_tokens(
    *,
    gameplay_objects: list[GameplayObject],
    difficulty: str,
    config: TokenizerConfig,
) -> tuple[list[str], str | None]:
    tokens = ["START", difficulty_token(difficulty), "ALIGN_UNKNOWN"]
    by_step: dict[int, list[GameplayObject]] = defaultdict(list)
    for item in gameplay_objects:
        by_step[item.x_step].append(item)

    current_step = 0
    for step in sorted(by_step):
        gap = max(0, step - current_step)
        if len(tokens) + gap > config.max_token_length:
            return tokens, "absurd_token_length"
        tokens.extend(["STEP"] * gap)
        for item in sorted(by_step[step], key=lambda event: (token_sort_key(event.token), event.y_lane, event.object_id)):
            tokens.extend(item.event_tokens())
            if len(tokens) > config.max_token_length:
                return tokens, "absurd_token_length"
        current_step = step

    if tokens[-1] != "STEP":
        tokens.append("STEP")
    tokens.append("END")
    return tokens, None


def object_count_stats(objects: list[GDObject]) -> dict[str, Any]:
    raw_count = len(objects)
    gameplay_source_count = 0
    ignored_count = 0
    unknown_count = 0
    unknown_ids: Counter[int] = Counter()
    for gd_object in objects:
        if map_object_id(gd_object.object_id) is not None:
            gameplay_source_count += 1
        elif is_ignored_object_id(gd_object.object_id):
            ignored_count += 1
        else:
            unknown_count += 1
            unknown_ids[gd_object.object_id] += 1
    return {
        "object_count_raw": raw_count,
        "gameplay_source_object_count": gameplay_source_count,
        "ignored_object_count": ignored_count,
        "unknown_object_count": unknown_count,
        "unknown_object_ratio": round(unknown_count / raw_count, 6) if raw_count else 1.0,
        "top_unknown_object_ids": [
            {"object_id": object_id, "count": count}
            for object_id, count in unknown_ids.most_common(20)
        ],
    }


def rejection_reason(
    *,
    objects: list[GDObject],
    gameplay_objects: list[GameplayObject],
    tokens: list[str],
    emit_error: str | None,
    stats: dict[str, Any],
    config: TokenizerConfig,
) -> str | None:
    if not objects:
        return "decoded_level_object_count_zero"
    if len(gameplay_objects) < config.min_gameplay_objects:
        return "too_few_gameplay_objects"
    if len(tokens) < config.min_tokens:
        return "too_few_tokens"
    if "START" not in tokens or "END" not in tokens:
        return "missing_start_or_end"
    if validate_token_grammar(tokens):
        return "invalid_token_grammar"
    if float(stats["unknown_object_ratio"]) > config.max_unknown_ratio:
        return "unknown_object_ratio_too_high"
    if emit_error is not None:
        return emit_error
    if len(tokens) > config.max_token_length:
        return "absurd_token_length"
    if len({item.x_step for item in gameplay_objects}) <= 1:
        return "all_objects_one_timestep"
    return None


def tokenized_record_from_level(
    level: dict[str, Any],
    config: TokenizerConfig,
    *,
    include_parsed_objects: bool = False,
) -> TokenizationArtifacts:
    base_stats: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "record_type": "level",
        "level_id": to_int(level.get("level_id")),
        "difficulty": str(level.get("difficulty", "NA")),
        "source": str(level.get("source", "")),
        "accepted": False,
    }

    try:
        decoded = decode_level_data(str(level.get("level_data", "")))
        header, objects = parse_level_objects(decoded)
    except ValidationError as exc:
        stats_record = {
            **base_stats,
            "reason": exc.reason,
            "detail": exc.detail,
            "object_count_raw": 0,
            "object_count_gameplay": 0,
        }
        return TokenizationArtifacts(None, None, None, stats_record)

    gameplay_objects = normalize_gameplay_objects(objects, config)
    stats = object_count_stats(objects)
    tokens, emit_error = emit_tokens(
        gameplay_objects=gameplay_objects,
        difficulty=str(level.get("difficulty", "NA")),
        config=config,
    )
    reason = rejection_reason(
        objects=objects,
        gameplay_objects=gameplay_objects,
        tokens=tokens,
        emit_error=emit_error,
        stats=stats,
        config=config,
    )

    common = {
        "dataset_version": DATASET_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "level_id": to_int(level.get("level_id")),
        "difficulty": str(level.get("difficulty", "NA")),
        "song_id": to_int(level.get("song_id")),
        "source": str(level.get("source", "")),
        "object_count_raw": len(objects),
        "object_count_gameplay": len(gameplay_objects),
        "x_step_resolution": config.x_step_resolution,
        "y_lanes": config.y_lanes,
    }
    parsed_record: dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "parser_version": PARSER_VERSION,
        "level_id": common["level_id"],
        "difficulty": common["difficulty"],
        "song_id": common["song_id"],
        "source": common["source"],
        "decoded_length": len(decoded),
        "header": header,
        "object_count_raw": len(objects),
        "x_min": round(min((item.x for item in objects), default=0.0), 6),
        "x_max": round(max((item.x for item in objects), default=0.0), 6),
        "y_min": round(min((item.y for item in objects), default=0.0), 6),
        "y_max": round(max((item.y for item in objects), default=0.0), 6),
        "object_id_counts": [
            {"object_id": object_id, "count": count}
            for object_id, count in Counter(item.object_id for item in objects).most_common()
        ],
    }
    if include_parsed_objects:
        parsed_record["objects"] = [item.to_json() for item in objects]
    gameplay_record = {
        **common,
        "objects": [item.to_json() for item in gameplay_objects],
    }
    token_record = None
    if reason is None:
        token_record = {
            **common,
            "tokens": tokens,
        }

    stats_record = {
        **base_stats,
        **stats,
        "accepted": reason is None,
        "reason": reason or "",
        "object_count_gameplay": len(gameplay_objects),
        "token_count": len(tokens),
        "unique_x_steps": len({item.x_step for item in gameplay_objects}),
    }
    return TokenizationArtifacts(parsed_record, gameplay_record, token_record, stats_record)


def token_counts_from_records(token_records: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in token_records:
        counter.update(str(token) for token in record.get("tokens", []) if isinstance(token, str))
    return counter


def build_vocab(
    token_records: list[dict[str, Any]] | None = None,
    *,
    token_counts: Counter[str] | None = None,
    unknown_object_count: int = 0,
    top_unknown_object_ids: Counter[int] | None = None,
) -> dict[str, Any]:
    counts = token_counts if token_counts is not None else token_counts_from_records(token_records or [])
    observed = ordered_unique(counts.keys())
    tokens = ordered_unique([*BASE_VOCAB_TOKENS, *observed])
    token_to_id = {token: index for index, token in enumerate(tokens)}
    return {
        "dataset_version": DATASET_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "vocab_size": len(tokens),
        "tokens": tokens,
        "token_to_id": token_to_id,
        "id_to_token": tokens,
        "token_counts": {token: int(counts.get(token, 0)) for token in tokens},
        "unknown_token_count": int(counts.get("<UNK>", 0)),
        "unknown_object_count": int(unknown_object_count),
        "top_unknown_object_ids": [
            {"object_id": object_id, "count": count}
            for object_id, count in (top_unknown_object_ids or Counter()).most_common(50)
        ],
    }


def run_tokenizer(args: argparse.Namespace) -> int:
    store = CheckpointStore(args.data_dir)
    store.ensure()
    config = TokenizerConfig(
        x_step_resolution=args.x_step_resolution,
        y_lanes=args.y_lanes,
        min_gameplay_objects=args.min_gameplay_objects,
        min_tokens=args.min_tokens,
        max_unknown_ratio=args.max_unknown_ratio,
        max_token_length=args.max_token_length,
    )

    token_counts: Counter[str] = Counter()
    inspected_records: list[dict[str, Any]] = []
    unknown_object_ids: Counter[int] = Counter()
    levels_seen = 0
    levels_accepted = 0
    levels_rejected = 0
    tokens_total = 0
    token_length_max = 0
    step_density_total = 0.0
    raw_object_total = 0
    unknown_object_total = 0
    rejection_reasons: Counter[str] = Counter()
    mode = "wb" if args.overwrite else "ab"

    for path in (
        store.parsed_levels_path,
        store.gameplay_objects_path,
        store.mechanics_tokens_path,
        store.tokenizer_stats_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    with (
        store.levels_path.open("rb") as input_handle,
        store.parsed_levels_path.open(mode) as parsed_handle,
        store.gameplay_objects_path.open(mode) as gameplay_handle,
        store.mechanics_tokens_path.open(mode) as token_handle,
        store.tokenizer_stats_path.open(mode) as stats_handle,
    ):
        for raw_line in input_handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            level = loads_json(raw_line)
            artifacts = tokenized_record_from_level(
                level,
                config,
                include_parsed_objects=args.write_parsed_objects,
            )
            if artifacts.parsed_record is not None:
                parsed_handle.write(dumps_jsonl(artifacts.parsed_record))
            if artifacts.gameplay_record is not None:
                gameplay_handle.write(dumps_jsonl(artifacts.gameplay_record))
            if artifacts.token_record is not None:
                token_values = [str(token) for token in artifacts.token_record["tokens"]]
                token_counts.update(token_values)
                token_handle.write(dumps_jsonl(artifacts.token_record))
                levels_accepted += 1
                token_length = len(token_values)
                tokens_total += token_length
                token_length_max = max(token_length_max, token_length)
                step_density_total += token_values.count("STEP") / max(token_length, 1)
                if len(inspected_records) < args.inspect:
                    inspected_records.append(artifacts.token_record)
            else:
                levels_rejected += 1
                rejection_reasons[str(artifacts.stats_record.get("reason", "unknown") or "unknown")] += 1
            raw_object_total += int(artifacts.stats_record.get("object_count_raw", 0) or 0)
            unknown_object_total += int(artifacts.stats_record.get("unknown_object_count", 0) or 0)
            for item in artifacts.stats_record.get("top_unknown_object_ids", []) or []:
                try:
                    unknown_object_ids[int(item["object_id"])] += int(item["count"])
                except (KeyError, TypeError, ValueError):
                    continue
            stats_handle.write(dumps_jsonl(artifacts.stats_record))
            levels_seen += 1
            if args.limit > 0 and levels_seen >= args.limit:
                break

        summary = {
            "dataset_version": DATASET_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "record_type": "summary",
            "timestamp": epoch_seconds(),
            "levels_seen": levels_seen,
            "levels_accepted": levels_accepted,
            "levels_rejected": levels_rejected,
            "tokens_total": tokens_total,
            "avg_token_length": round(tokens_total / max(levels_accepted, 1), 6),
            "max_token_length": token_length_max,
            "avg_step_density": round(step_density_total / max(levels_accepted, 1), 6),
            "unknown_object_rate": round(unknown_object_total / max(raw_object_total, 1), 6),
            "rejection_reasons": dict(rejection_reasons),
            "x_step_resolution": config.x_step_resolution,
            "y_lanes": config.y_lanes,
        }
        stats_handle.write(dumps_jsonl(summary))

    vocab = build_vocab(
        token_counts=token_counts,
        unknown_object_count=unknown_object_total,
        top_unknown_object_ids=unknown_object_ids,
    )
    store.vocab_path.write_bytes(dumps_pretty(vocab))

    if args.inspect > 0:
        for record in inspected_records:
            preview = " ".join(record["tokens"][: args.inspect_tokens])
            print(
                f"level_id={record['level_id']} difficulty={record['difficulty']} "
                f"gameplay={record['object_count_gameplay']} tokens={len(record['tokens'])}"
            )
            print(preview)
            print()

    if not args.skip_analytics:
        from .analytics import analyze_dataset

        analytics = analyze_dataset(args.data_dir)
        store.tokenizer_analytics_path.write_bytes(dumps_pretty(analytics))

    print(
        f"tokenized {levels_accepted}/{levels_seen} levels; "
        f"wrote {store.mechanics_tokens_path}, {store.vocab_path}, and {store.tokenizer_stats_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-tokenize",
        description="Parse raw GD levels and emit Tokenizer v1 mechanics tokens.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=100, help="Raw levels to process. Use 0 for all.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output JSONL files instead of appending.")
    parser.add_argument("--x-step-resolution", type=int, default=X_STEP_RESOLUTION)
    parser.add_argument("--y-lanes", type=int, default=Y_LANES)
    parser.add_argument("--min-gameplay-objects", type=int, default=20)
    parser.add_argument("--min-tokens", type=int, default=50)
    parser.add_argument("--max-unknown-ratio", type=float, default=0.995)
    parser.add_argument("--max-token-length", type=int, default=12_000)
    parser.add_argument("--inspect", type=int, default=0)
    parser.add_argument("--inspect-tokens", type=int, default=120)
    parser.add_argument("--skip-analytics", action="store_true", help="Do not write tokenizer_analytics.json.")
    parser.add_argument(
        "--write-parsed-objects",
        action="store_true",
        help="Include every parsed raw object in parsed_levels.jsonl. Defaults to a compact summary.",
    )
    return parser


def main() -> int:
    return run_tokenizer(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
