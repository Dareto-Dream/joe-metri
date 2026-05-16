from __future__ import annotations

from dataclasses import dataclass, replace

from gd_scraper.gd_objects import ORB_TOKENS, PORTAL_TOKENS, SOLID_TOKENS
from gd_scraper.reconstruction import reconstruct_tokens

from .mappings import STEP_UNITS, TOKEN_TO_GD_ID, X_ORIGIN, Y_ORIGIN, Y_UNITS


DEFAULT_COLOR_CHANNELS = (
    "1_255_2_255_3_255_11_255_12_255_13_255_4_-1_6_1000_7_1_8_1_15_1_18_1|"
    "1_0_2_0_3_0_11_255_12_255_13_255_4_-1_6_1001_7_1_8_1_15_1_18_1|"
    "1_0_2_0_3_0_11_255_12_255_13_255_4_-1_6_1009_7_1_8_1_15_1_18_1|"
    "1_0_2_0_3_0_11_255_12_255_13_255_4_-1_6_1013_7_1_8_1_15_1_18_1|"
    "1_0_2_0_3_0_11_255_12_255_13_255_4_-1_6_1014_7_1_8_1_15_1_18_1|"
    "1_200_2_200_3_200_11_255_12_255_13_255_4_-1_6_1004_7_1_8_1_15_1_18_1|"
    "1_255_2_255_3_255_11_255_12_255_13_255_4_-1_6_1005_7_1_8_1_15_1_18_1|"
    "1_255_2_255_3_255_11_255_12_255_13_255_4_-1_6_1006_7_1_8_1_15_1_18_1|"
)
LEVEL_STRING_HEADER = f"kS38,{DEFAULT_COLOR_CHANNELS},"
GROUND_SEGMENT_WIDTH = 8
MAX_INTERACTIVE_LANE = 9
MAX_SOLID_LANE = 5
MAX_OBJECTS_PER_STEP = 2
CONTROL_SPACING_STEPS = 8
GRAVITY_TOKENS = {"GRAVITY_UP", "GRAVITY_DOWN"}
SPEED_TOKENS = {"SPEED_SLOW", "SPEED_NORMAL", "SPEED_FAST"}
CONTROL_TOKENS = PORTAL_TOKENS | GRAVITY_TOKENS | SPEED_TOKENS
PAD_TOKENS = {"PAD_YELLOW", "PAD_BLUE"}
HAZARD_TOKENS = {"SPIKE", "SAW"}


@dataclass(frozen=True)
class RuntimeObject:
    token: str
    object_id: int
    x_step: int
    y_lane: int
    x: int
    y: int
    width: int = 1
    sequence: int = 0

    def to_json(self) -> dict[str, int | str]:
        value: dict[str, int | str] = {
            "token": self.token,
            "object_id": self.object_id,
            "x_step": self.x_step,
            "y_lane": self.y_lane,
            "x": self.x,
            "y": self.y,
            "sequence": self.sequence,
        }
        if self.token in SOLID_TOKENS:
            value["width"] = self.width
        return value

    def gd_object_strings(self) -> list[str]:
        if self.token in SOLID_TOKENS:
            return [
                gd_object_string(self.object_id, self.x + offset * STEP_UNITS, self.y)
                for offset in range(max(1, self.width))
            ]
        return [gd_object_string(self.object_id, self.x, self.y)]


@dataclass(frozen=True)
class RuntimeLayout:
    objects: list[RuntimeObject]
    gd_object_strings: list[str]
    level_string: str
    width_steps: int
    height_lanes: int
    errors: list[str]

    def to_json(self) -> dict[str, object]:
        return {
            "objects": [item.to_json() for item in self.objects],
            "gd_object_strings": self.gd_object_strings,
            "level_string": self.level_string,
            "width_steps": self.width_steps,
            "height_lanes": self.height_lanes,
            "errors": self.errors,
        }


def reconstruct_layout(tokens: list[str]) -> RuntimeLayout:
    reconstructed, errors = reconstruct_tokens(tokens)
    runtime_objects: list[RuntimeObject] = []
    for item in reconstructed:
        object_id = TOKEN_TO_GD_ID.get(item.token, 1)
        runtime_objects.append(
            RuntimeObject(
                token=item.token,
                object_id=object_id,
                x_step=item.x_step,
                y_lane=item.y_lane,
                x=X_ORIGIN + item.x_step * STEP_UNITS,
                y=runtime_y(item.token, item.y_lane),
                width=item.width,
                sequence=item.sequence,
            )
        )

    runtime_objects = polish_runtime_objects(runtime_objects)
    object_strings = [raw for item in runtime_objects for raw in item.gd_object_strings()]
    object_strings = ordered_unique(object_strings)
    level_string = LEVEL_STRING_HEADER + ";" + ";".join(object_strings) + (";" if object_strings else "")
    width_steps = max((item.x_step + max(1, item.width) for item in runtime_objects), default=0)
    height_lanes = max((item.y_lane for item in runtime_objects), default=0) + 1
    return RuntimeLayout(
        objects=runtime_objects,
        gd_object_strings=object_strings,
        level_string=level_string,
        width_steps=width_steps,
        height_lanes=height_lanes,
        errors=errors,
    )


def polish_runtime_objects(objects: list[RuntimeObject]) -> list[RuntimeObject]:
    if not objects:
        return []

    polished: list[RuntimeObject] = []
    occupied_cells: set[tuple[int, int]] = set()
    step_counts: dict[int, int] = {}

    last_control_step: dict[str, int] = {}
    seen_interactives: set[tuple[str, int, int]] = set()
    solid_columns: dict[int, int] = {}

    for item in sorted(objects, key=lambda value: (value.x_step, value.sequence)):
        candidate = normalize_runtime_object(item)
        if candidate is None:
            continue

        if candidate.token in SOLID_TOKENS:
            if solid_columns.get(candidate.x_step, 0) >= 1:
                continue
            free_offsets = [
                offset
                for offset in range(max(1, candidate.width))
                if (candidate.x_step + offset, candidate.y_lane) not in occupied_cells
            ]
            if not free_offsets:
                continue
            candidate = replace(candidate, width=max(1, min(candidate.width, len(free_offsets))))
            for offset in range(candidate.width):
                occupied_cells.add((candidate.x_step + offset, candidate.y_lane))
            solid_columns[candidate.x_step] = solid_columns.get(candidate.x_step, 0) + 1
            polished.append(candidate)
            continue

        if step_counts.get(candidate.x_step, 0) >= MAX_OBJECTS_PER_STEP:
            continue
        if candidate.token in CONTROL_TOKENS and not control_is_allowed(candidate, last_control_step):
            continue
        if candidate.token not in CONTROL_TOKENS and (candidate.x_step, candidate.y_lane) in occupied_cells:
            continue
        key = (candidate.token, candidate.x_step, candidate.y_lane)
        if key in seen_interactives:
            continue
        if candidate.token not in CONTROL_TOKENS:
            occupied_cells.add((candidate.x_step, candidate.y_lane))
        seen_interactives.add(key)
        step_counts[candidate.x_step] = step_counts.get(candidate.x_step, 0) + 1
        polished.append(candidate)

    return sorted(polished, key=lambda value: (value.x_step, value.y_lane, value.sequence))


def normalize_runtime_object(item: RuntimeObject) -> RuntimeObject | None:
    if item.token in SOLID_TOKENS:
        if item.y_lane > MAX_SOLID_LANE:
            return None
        return replace(
            item,
            y=runtime_y(item.token, item.y_lane),
            width=max(1, min(item.width, GROUND_SEGMENT_WIDTH)),
        )
    if item.token in ORB_TOKENS | PAD_TOKENS | HAZARD_TOKENS:
        y_lane = max(1, min(item.y_lane, MAX_INTERACTIVE_LANE))
        return replace(item, y_lane=y_lane, y=runtime_y(item.token, y_lane))
    if item.token in CONTROL_TOKENS:
        y_lane = max(2, min(item.y_lane, MAX_INTERACTIVE_LANE))
        return replace(item, y_lane=y_lane, y=runtime_y(item.token, y_lane))
    return item


def control_is_allowed(item: RuntimeObject, last_control_step: dict[str, int]) -> bool:
    category = control_category(item.token)
    last_step = last_control_step.get(category)
    if last_step is not None and item.x_step - last_step < CONTROL_SPACING_STEPS:
        return False
    last_control_step[category] = item.x_step
    return True


def control_category(token: str) -> str:
    if token in PORTAL_TOKENS:
        return "portal"
    if token in GRAVITY_TOKENS:
        return "gravity"
    if token in SPEED_TOKENS:
        return "speed"
    return token


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def gd_object_string(object_id: int, x: int, y: int) -> str:
    return f"1,{object_id},2,{x},3,{y}"


def runtime_y(token: str, y_lane: int) -> int:
    return Y_ORIGIN + y_lane * Y_UNITS
