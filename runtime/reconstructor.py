from __future__ import annotations

from dataclasses import dataclass

from gd_scraper.gd_objects import SOLID_TOKENS
from gd_scraper.reconstruction import reconstruct_tokens

from .mappings import STEP_UNITS, TOKEN_TO_GD_ID, X_ORIGIN, Y_ORIGIN, Y_UNITS


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
                y=Y_ORIGIN + item.y_lane * Y_UNITS,
                width=item.width,
                sequence=item.sequence,
            )
        )

    object_strings = [raw for item in runtime_objects for raw in item.gd_object_strings()]
    level_string = "kS1,0;" + ";".join(object_strings)
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


def gd_object_string(object_id: int, x: int, y: int) -> str:
    return f"1,{object_id},2,{x},3,{y}"
