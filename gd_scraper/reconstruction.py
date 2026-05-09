from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gd_objects import (
    DIFFICULTY_TOKENS,
    PRIMARY_MECHANIC_TOKENS,
    SOLID_TOKENS,
    TOKENIZER_VERSION,
    WIDTH_TOKENS,
    Y_TOKENS,
)
from .models import DATASET_VERSION, epoch_seconds
from .storage import CheckpointStore, dumps_jsonl, dumps_pretty, loads_json


@dataclass(frozen=True)
class ReconstructedObject:
    token: str
    x_step: int
    y_lane: int
    width: int = 1
    sequence: int = 0

    def to_json(self) -> dict[str, int | str]:
        value: dict[str, int | str] = {
            "token": self.token,
            "x_step": self.x_step,
            "y_lane": self.y_lane,
            "sequence": self.sequence,
        }
        if self.token in SOLID_TOKENS:
            value["width"] = self.width
        return value


def parse_y_lane(token: str) -> int | None:
    if token not in Y_TOKENS:
        return None
    try:
        return int(token[1:])
    except ValueError:
        return None


def parse_width(token: str) -> int | None:
    if token not in WIDTH_TOKENS:
        return None
    try:
        return int(token.removeprefix("WIDTH_"))
    except ValueError:
        return None


def grammar_prefix_length(tokens: list[str]) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not tokens:
        return 0, ["empty_sequence"]
    if tokens[0] != "START":
        errors.append("missing_start")

    index = 1
    if index < len(tokens) and tokens[index] in DIFFICULTY_TOKENS:
        index += 1
    if index < len(tokens) and tokens[index] == "ALIGN_UNKNOWN":
        index += 1
    return index, errors


def reconstruct_tokens(tokens: list[str]) -> tuple[list[ReconstructedObject], list[str]]:
    prefix_length, errors = grammar_prefix_length(tokens)
    if not tokens:
        return [], errors
    if tokens[-1] != "END":
        errors.append("missing_end")

    objects: list[ReconstructedObject] = []
    current_step = 0
    index = prefix_length
    sequence = 0

    while index < len(tokens):
        token = tokens[index]
        if token == "END":
            if index != len(tokens) - 1:
                errors.append("tokens_after_end")
            return objects, errors
        if token == "START":
            errors.append("unexpected_start")
            index += 1
            continue
        if token in DIFFICULTY_TOKENS or token == "ALIGN_UNKNOWN":
            errors.append(f"unexpected_prefix_token:{token}")
            index += 1
            continue
        if token == "STEP":
            current_step += 1
            index += 1
            continue
        if token in Y_TOKENS:
            errors.append(f"orphan_y_token:{token}")
            index += 1
            continue
        if token in WIDTH_TOKENS:
            errors.append(f"orphan_width_token:{token}")
            index += 1
            continue
        if token not in PRIMARY_MECHANIC_TOKENS:
            errors.append(f"unknown_token:{token}")
            index += 1
            continue

        if index + 1 >= len(tokens):
            errors.append(f"missing_y_after:{token}")
            return objects, errors
        y_lane = parse_y_lane(tokens[index + 1])
        if y_lane is None:
            errors.append(f"missing_y_after:{token}")
            index += 1
            continue
        index += 2

        width = 1
        if token in SOLID_TOKENS:
            if index >= len(tokens):
                errors.append(f"missing_width_after:{token}")
                return objects, errors
            parsed_width = parse_width(tokens[index])
            if parsed_width is None:
                errors.append(f"missing_width_after:{token}")
                continue
            width = parsed_width
            index += 1
        elif index < len(tokens) and tokens[index] in WIDTH_TOKENS:
            errors.append(f"unexpected_width_after:{token}")
            index += 1

        objects.append(
            ReconstructedObject(
                token=token,
                x_step=current_step,
                y_lane=y_lane,
                width=width,
                sequence=sequence,
            )
        )
        sequence += 1

    return objects, errors


def validate_token_grammar(tokens: list[str]) -> list[str]:
    _objects, errors = reconstruct_tokens(tokens)
    return errors


def gameplay_signature(gameplay_record: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    result: list[tuple[str, int, int, int]] = []
    for item in gameplay_record.get("objects", []):
        token = str(item.get("token", ""))
        width = int(item.get("width", 1)) if token in SOLID_TOKENS else 1
        result.append(
            (
                token,
                int(item.get("x_step", 0)),
                int(item.get("y_lane", 0)),
                width,
            )
        )
    return result


def reconstructed_signature(objects: list[ReconstructedObject]) -> list[tuple[str, int, int, int]]:
    return [(item.token, item.x_step, item.y_lane, item.width) for item in objects]


def validate_reconstruction_record(
    token_record: dict[str, Any],
    gameplay_record: dict[str, Any] | None,
) -> dict[str, Any]:
    tokens = [str(token) for token in token_record.get("tokens", [])]
    reconstructed, grammar_errors = reconstruct_tokens(tokens)
    expected = gameplay_signature(gameplay_record or {})
    actual = reconstructed_signature(reconstructed)

    reason = ""
    mismatch_index = -1
    if grammar_errors:
        reason = "invalid_grammar"
    elif gameplay_record is None:
        reason = "missing_gameplay_record"
    elif len(actual) != len(expected):
        reason = "object_count_mismatch"
    else:
        for index, (left, right) in enumerate(zip(actual, expected)):
            if left != right:
                reason = "object_mismatch"
                mismatch_index = index
                break

    return {
        "dataset_version": DATASET_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "record_type": "level",
        "level_id": int(token_record.get("level_id", 0) or 0),
        "valid": reason == "",
        "reason": reason,
        "grammar_errors": grammar_errors[:20],
        "mismatch_index": mismatch_index,
        "reconstructed_object_count": len(actual),
        "expected_object_count": len(expected),
        "max_x_step": max((item.x_step for item in reconstructed), default=0),
    }


def load_gameplay_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = loads_json(raw_line)
            try:
                records[int(record["level_id"])] = record
            except (KeyError, TypeError, ValueError):
                continue
    return records


def run_reconstruction_validation(args: argparse.Namespace) -> int:
    store = CheckpointStore(args.data_dir)
    gameplay_records = load_gameplay_records(store.gameplay_objects_path)
    output_path = store.reconstruction_validation_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if args.overwrite else "ab"

    processed = 0
    valid = 0
    rejected_reasons: dict[str, int] = {}
    with (
        store.mechanics_tokens_path.open("rb") as token_handle,
        output_path.open(mode) as output_handle,
    ):
        for raw_line in token_handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            token_record = loads_json(raw_line)
            level_id = int(token_record.get("level_id", 0) or 0)
            validation = validate_reconstruction_record(token_record, gameplay_records.get(level_id))
            output_handle.write(dumps_jsonl(validation))
            processed += 1
            if validation["valid"]:
                valid += 1
            else:
                reason = str(validation["reason"])
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            if args.limit > 0 and processed >= args.limit:
                break

        summary = {
            "dataset_version": DATASET_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "record_type": "summary",
            "timestamp": epoch_seconds(),
            "levels_validated": processed,
            "levels_valid": valid,
            "levels_invalid": processed - valid,
            "invalid_reasons": rejected_reasons,
        }
        output_handle.write(dumps_jsonl(summary))

    summary_path = store.reconstruction_summary_path
    summary_path.write_bytes(dumps_pretty(summary))
    print(
        f"validated reconstruction for {valid}/{processed} levels; "
        f"wrote {output_path} and {summary_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-validate-reconstruction",
        description="Validate mechanics token reconstruction against gameplay objects.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=0, help="Tokenized records to validate. Use 0 for all.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    return run_reconstruction_validation(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
