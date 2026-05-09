from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Any

from .gd_objects import (
    ORB_TOKENS,
    PORTAL_TOKENS,
    PRIMARY_MECHANIC_TOKENS,
    TOKENIZER_VERSION,
)
from .models import DATASET_VERSION, epoch_seconds
from .storage import CheckpointStore, dumps_pretty, loads_json


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        probability = count / total
        value -= probability * math.log2(probability)
    return value


def counter_records(counter: Counter[str], *, most_common: bool = True, limit: int = 25) -> list[dict[str, int | str]]:
    items = counter.most_common() if most_common else sorted(counter.items(), key=lambda item: (item[1], item[0]))
    return [{"token": token, "count": count} for token, count in items[:limit]]


def object_id_records(counter: Counter[int], limit: int = 25) -> list[dict[str, int]]:
    return [{"object_id": object_id, "count": count} for object_id, count in counter.most_common(limit)]


def analyze_dataset(data_dir: Path) -> dict[str, Any]:
    store = CheckpointStore(data_dir)
    token_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    portal_counts: Counter[str] = Counter()
    orb_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    token_lengths: list[int] = []
    step_densities: list[float] = []
    levels_accepted = 0

    with store.mechanics_tokens_path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = loads_json(raw_line)
            tokens = [str(token) for token in record.get("tokens", [])]
            if not tokens:
                continue
            levels_accepted += 1
            token_counts.update(tokens)
            token_lengths.append(len(tokens))
            step_densities.append(tokens.count("STEP") / max(len(tokens), 1))
            difficulty_counts[str(record.get("difficulty", "NA"))] += 1
            for token in tokens:
                if token in PRIMARY_MECHANIC_TOKENS:
                    object_counts[token] += 1
                if token in PORTAL_TOKENS:
                    portal_counts[token] += 1
                if token in ORB_TOKENS:
                    orb_counts[token] += 1

    levels_processed = 0
    levels_rejected = 0
    raw_objects = 0
    unknown_objects = 0
    unknown_object_ids: Counter[int] = Counter()
    rejection_reasons: Counter[str] = Counter()
    with store.tokenizer_stats_path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = loads_json(raw_line)
            if record.get("record_type") == "summary":
                continue
            levels_processed += 1
            raw_objects += int(record.get("object_count_raw", 0) or 0)
            unknown_objects += int(record.get("unknown_object_count", 0) or 0)
            if not bool(record.get("accepted")):
                levels_rejected += 1
                rejection_reasons[str(record.get("reason", "unknown") or "unknown")] += 1
            for item in record.get("top_unknown_object_ids", []) or []:
                try:
                    unknown_object_ids[int(item["object_id"])] += int(item["count"])
                except (KeyError, TypeError, ValueError):
                    continue

    total_tokens = sum(token_counts.values())
    max_token_length = max(token_lengths, default=0)
    analytics = {
        "dataset_version": DATASET_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "generated_at": epoch_seconds(),
        "dataset_metrics": {
            "levels_processed": levels_processed,
            "levels_accepted": levels_accepted,
            "levels_rejected": levels_rejected,
            "avg_token_length": round(sum(token_lengths) / max(len(token_lengths), 1), 6),
            "max_token_length": max_token_length,
            "avg_step_density": round(sum(step_densities) / max(len(step_densities), 1), 6),
            "unknown_object_rate": round(unknown_objects / max(raw_objects, 1), 6),
            "total_tokens": total_tokens,
            "total_raw_objects": raw_objects,
            "total_unknown_objects": unknown_objects,
            "rejection_reasons": dict(rejection_reasons),
            "difficulty_counts": dict(difficulty_counts),
        },
        "vocabulary_metrics": {
            "vocab_observed": len(token_counts),
            "top_tokens": counter_records(token_counts, most_common=True),
            "rarest_tokens": counter_records(token_counts, most_common=False),
            "token_entropy": round(entropy(token_counts), 6),
            "object_frequency": counter_records(object_counts, most_common=True),
            "portal_frequency": counter_records(portal_counts, most_common=True),
            "orb_frequency": counter_records(orb_counts, most_common=True),
            "unknown_token_count": token_counts.get("<UNK>", 0),
            "top_unknown_object_ids": object_id_records(unknown_object_ids),
        },
    }
    return analytics


def run_analytics(args: argparse.Namespace) -> int:
    store = CheckpointStore(args.data_dir)
    analytics = analyze_dataset(args.data_dir)
    store.tokenizer_analytics_path.write_bytes(dumps_pretty(analytics))
    metrics = analytics["dataset_metrics"]
    print(
        f"analyzed {metrics['levels_accepted']}/{metrics['levels_processed']} accepted levels; "
        f"wrote {store.tokenizer_analytics_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-analyze-mechanics",
        description="Generate tokenizer dataset and vocabulary analytics.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def main() -> int:
    return run_analytics(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
