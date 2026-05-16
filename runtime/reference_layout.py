from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from gd_scraper.gd_objects import map_object_id

from .conditioning import ConditioningProfile
from .reconstructor import LEVEL_STRING_HEADER, RuntimeLayout, RuntimeObject
from .save_codec import decode_level_string_k4, find_value_after_key, parse_save_xml


CHUNK_STEPS = 32
STEP_UNITS = 30.0


@dataclass(frozen=True)
class ReferenceObject:
    object_id: int
    x: float
    y: float
    raw: str
    source: str

    @property
    def x_step(self) -> int:
        return int(round(self.x / STEP_UNITS))

    @property
    def y_lane(self) -> int:
        return int(round(self.y / STEP_UNITS))


@dataclass(frozen=True)
class ReferenceLevel:
    path: Path
    objects: list[ReferenceObject]

    @property
    def width_steps(self) -> int:
        if not self.objects:
            return 0
        min_x = min(item.x for item in self.objects)
        max_x = max(item.x for item in self.objects)
        return max(1, int(round((max_x - min_x) / STEP_UNITS)))


def build_reference_layout(
    conditioning: ConditioningProfile,
    *,
    seed: int,
    examples_dir: Path = Path("examples"),
) -> RuntimeLayout | None:
    levels = load_reference_levels(examples_dir)
    if not levels:
        return None

    rng = random.Random(seed)
    target_steps = _target_steps(conditioning)
    chunks = _reference_chunks(levels, chunk_steps=CHUNK_STEPS)
    if not chunks:
        return None

    selected = _select_chunks(chunks, target_steps=target_steps, rng=rng)
    gd_objects: list[str] = []
    runtime_objects: list[RuntimeObject] = []
    cursor_x = 15.0
    sequence = 0

    for chunk in selected:
        min_x = min(item.x for item in chunk)
        for item in sorted(chunk, key=lambda value: (value.x, value.y, value.object_id)):
            x = cursor_x + (item.x - min_x)
            raw = rewrite_object_x(item.raw, x)
            gd_objects.append(raw)
            runtime_objects.append(
                RuntimeObject(
                    token=map_object_id(item.object_id) or f"OBJ_{item.object_id}",
                    object_id=item.object_id,
                    x_step=max(0, int(round(x / STEP_UNITS))),
                    y_lane=int(round(item.y / STEP_UNITS)),
                    x=int(round(x)),
                    y=int(round(item.y)),
                    sequence=sequence,
                )
            )
            sequence += 1
        cursor_x += CHUNK_STEPS * STEP_UNITS

    gd_objects = ordered_unique(gd_objects)
    width_steps = max((item.x_step for item in runtime_objects), default=0) + 1
    height_lanes = max((item.y_lane for item in runtime_objects), default=0) + 1
    return RuntimeLayout(
        objects=runtime_objects,
        gd_object_strings=gd_objects,
        level_string=LEVEL_STRING_HEADER + ";" + ";".join(gd_objects) + (";" if gd_objects else ""),
        width_steps=width_steps,
        height_lanes=height_lanes,
        errors=[],
    )


def load_reference_levels(examples_dir: Path) -> list[ReferenceLevel]:
    if not examples_dir.exists():
        return []

    levels: list[ReferenceLevel] = []
    for path in sorted(examples_dir.glob("*.gmd")):
        objects = parse_reference_objects(path)
        if objects:
            levels.append(ReferenceLevel(path=path, objects=objects))
    return levels


def parse_reference_objects(path: Path) -> list[ReferenceObject]:
    text = path.read_text(encoding="utf-8", errors="replace")
    level_string = text
    try:
        root = parse_save_xml(text)
        k4 = find_value_after_key(root, "k4")
        if k4 is not None and (k4.text or ""):
            level_string = decode_level_string_k4(k4.text or "")
    except Exception:
        level_string = text

    objects: list[ReferenceObject] = []
    for raw in level_string.split(";"):
        parsed = parse_object(raw)
        if parsed is None:
            continue
        object_id, x, y = parsed
        objects.append(ReferenceObject(object_id=object_id, x=x, y=y, raw=raw, source=path.name))
    return objects


def parse_object(raw: str) -> tuple[int, float, float] | None:
    if not raw.startswith("1,"):
        return None
    parts = raw.split(",")
    if len(parts) < 6:
        return None
    values = {parts[index]: parts[index + 1] for index in range(0, len(parts) - 1, 2)}
    try:
        return int(float(values["1"])), float(values["2"]), float(values["3"])
    except (KeyError, ValueError):
        return None


def rewrite_object_x(raw: str, x: float) -> str:
    parts = raw.split(",")
    for index in range(0, len(parts) - 1, 2):
        if parts[index] == "2":
            parts[index + 1] = format_number(x)
            break
    return ",".join(parts)


def _reference_chunks(levels: list[ReferenceLevel], *, chunk_steps: int) -> list[list[ReferenceObject]]:
    chunks: list[list[ReferenceObject]] = []
    width = chunk_steps * STEP_UNITS
    for level in levels:
        if not level.objects:
            continue
        min_x = min(item.x for item in level.objects)
        max_x = max(item.x for item in level.objects)
        start = min_x
        while start <= max_x:
            end = start + width
            chunk = [item for item in level.objects if start <= item.x < end]
            if _chunk_is_useful(chunk):
                chunks.append(chunk)
            start += width
    return chunks


def _chunk_is_useful(chunk: list[ReferenceObject]) -> bool:
    if len(chunk) < 24:
        return False
    if _interactive_count(chunk) >= 4:
        return True
    return len({item.object_id for item in chunk}) >= 4


def _select_chunks(
    chunks: list[list[ReferenceObject]],
    *,
    target_steps: int,
    rng: random.Random,
) -> list[list[ReferenceObject]]:
    count = max(1, min(12, (target_steps + CHUNK_STEPS - 1) // CHUNK_STEPS))
    scored = sorted(chunks, key=lambda chunk: (_chunk_score(chunk), rng.random()), reverse=True)
    if len(scored) >= count:
        return scored[:count]
    result = scored[:]
    while result and len(result) < count:
        result.append(rng.choice(scored))
    return result


def _chunk_score(chunk: list[ReferenceObject]) -> float:
    ids = {item.object_id for item in chunk}
    density = min(len(chunk) / CHUNK_STEPS, 4.5)
    interactive = _interactive_count(chunk)
    known = sum(1 for item in chunk if map_object_id(item.object_id) is not None)
    ys = [item.y for item in chunk]
    y_span = max(ys, default=0.0) - min(ys, default=0.0)
    y_penalty = max(0.0, (y_span - 720.0) / 30.0)
    high_penalty = max(0.0, (max(ys, default=0.0) - 960.0) / 60.0)
    return interactive * 2.6 + known * 0.18 + density * 8.0 + len(ids) * 1.4 - y_penalty - high_penalty


def _interactive_count(chunk: list[ReferenceObject]) -> int:
    interactive_ids = {8, 9, 10, 11, 12, 13, 35, 36, 47, 67, 84, 111, 141, 660, 1333, 1334}
    return sum(1 for item in chunk if item.object_id in interactive_ids)


def _target_steps(conditioning: ConditioningProfile) -> int:
    audio_steps = max(conditioning.energy_by_step, default=96)
    budget_steps = max(48, int(conditioning.target_tokens * 0.9))
    return max(48, min(max(audio_steps, budget_steps), 640))


def format_number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.001:
        return str(int(rounded))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
