from __future__ import annotations

from dataclasses import dataclass

from .gd_objects import ORB_TOKENS, PORTAL_TOKENS, PRIMARY_MECHANIC_TOKENS, SOLID_TOKENS
from .reconstruction import reconstruct_tokens


GRAVITY_TOKENS = {"GRAVITY_UP", "GRAVITY_DOWN"}
SPEED_TOKENS = {"SPEED_SLOW", "SPEED_NORMAL", "SPEED_FAST"}
CONTROL_TOKENS = PORTAL_TOKENS | GRAVITY_TOKENS | SPEED_TOKENS


@dataclass(frozen=True)
class TokenQualityReport:
    score: float
    valid: bool
    errors: list[str]
    object_count: int
    step_count: int
    duplicate_positions: int
    max_objects_per_step: int
    control_spam: int
    path_gaps: int
    path_jumps: int
    path_obstructions: int
    diversity: float

    def to_json(self) -> dict[str, float | int | bool | list[str]]:
        return {
            "score": round(self.score, 6),
            "valid": self.valid,
            "errors": self.errors,
            "object_count": self.object_count,
            "step_count": self.step_count,
            "duplicate_positions": self.duplicate_positions,
            "max_objects_per_step": self.max_objects_per_step,
            "control_spam": self.control_spam,
            "path_gaps": self.path_gaps,
            "path_jumps": self.path_jumps,
            "path_obstructions": self.path_obstructions,
            "diversity": round(self.diversity, 6),
        }


def evaluate_token_sequence_quality(tokens: list[str]) -> TokenQualityReport:
    objects, errors = reconstruct_tokens(tokens)
    object_count = len(objects)
    step_count = tokens.count("STEP")
    diversity = len(set(tokens)) / max(len(tokens), 1)

    position_counts: dict[tuple[int, int], int] = {}
    object_steps: dict[int, int] = {}
    last_control_step: dict[str, int] = {}
    duplicate_positions = 0
    control_spam = 0
    path = estimate_player_path(tokens)
    path_gaps = 0
    path_jumps = 0
    path_obstructions = 0
    current_path_lane = 1

    for item in sorted(objects, key=lambda value: (value.x_step, value.sequence)):
        width = max(1, item.width) if item.token in SOLID_TOKENS else 1
        for offset in range(width):
            key = (item.x_step + offset, item.y_lane)
            previous = position_counts.get(key, 0)
            if previous:
                duplicate_positions += 1
            position_counts[key] = previous + 1

        object_steps[item.x_step] = object_steps.get(item.x_step, 0) + 1
        if item.token in CONTROL_TOKENS:
            category = control_category(item.token)
            previous_step = last_control_step.get(category)
            if previous_step is not None and item.x_step - previous_step < 8:
                control_spam += 1
            last_control_step[category] = item.x_step

        if item.token == "BLOCK" and item.y_lane > 0 and item.y_lane in {current_path_lane, current_path_lane + 1}:
            path_obstructions += 1
        current_path_lane = next_path_lane(current_path_lane, item)

    max_objects_per_step = max(object_steps.values(), default=0)
    for left_step, right_step in zip(sorted(path), sorted(path)[1:]):
        left_lane = path[left_step]
        right_lane = path[right_step]
        if right_step - left_step > 6:
            path_gaps += 1
        if abs(right_lane - left_lane) > 4:
            path_jumps += 1

    target_objects = max(8, min(160, step_count))
    object_balance = 1.0 - min(1.0, abs(object_count - target_objects) / max(target_objects, 1))
    step_density = step_count / max(len(tokens), 1)
    mechanics = sum(1 for token in tokens if token in PRIMARY_MECHANIC_TOKENS)
    mechanic_density = mechanics / max(len(tokens), 1)
    orb_count = sum(1 for token in tokens if token in ORB_TOKENS)
    portal_count = sum(1 for token in tokens if token in PORTAL_TOKENS)

    score = 100.0
    score -= len(errors) * 18.0
    score -= duplicate_positions * 5.0
    score -= max(0, max_objects_per_step - 3) * 6.0
    score -= control_spam * 8.0
    score -= path_gaps * 4.0
    score -= path_jumps * 7.0
    score -= path_obstructions * 9.0
    score += object_balance * 24.0
    score += min(diversity, 0.45) * 30.0
    score += min(mechanic_density, 0.45) * 18.0
    score += min(step_density, 0.45) * 10.0
    score += min(orb_count, 12) * 0.8
    score += min(portal_count, 4) * 0.8
    if object_count == 0:
        score -= 80.0
    if step_count < 12:
        score -= 30.0

    return TokenQualityReport(
        score=max(0.0, score),
        valid=not errors and object_count > 0,
        errors=errors[:20],
        object_count=object_count,
        step_count=step_count,
        duplicate_positions=duplicate_positions,
        max_objects_per_step=max_objects_per_step,
        control_spam=control_spam,
        path_gaps=path_gaps,
        path_jumps=path_jumps,
        path_obstructions=path_obstructions,
        diversity=diversity,
    )


def control_category(token: str) -> str:
    if token in PORTAL_TOKENS:
        return "portal"
    if token in GRAVITY_TOKENS:
        return "gravity"
    if token in SPEED_TOKENS:
        return "speed"
    return token


def estimate_player_path(tokens: list[str]) -> dict[int, int]:
    objects, _errors = reconstruct_tokens(tokens)
    path: dict[int, int] = {}
    current_lane = 1

    for item in sorted(objects, key=lambda value: (value.x_step, value.sequence)):
        current_lane = next_path_lane(current_lane, item)

        previous = path.get(item.x_step)
        if previous is None:
            path[item.x_step] = current_lane
        else:
            path[item.x_step] = round((previous + current_lane) / 2)

    if not path and tokens:
        path[0] = current_lane
    return path


def next_path_lane(current_lane: int, item: object) -> int:
    token = str(getattr(item, "token", ""))
    y_lane = int(getattr(item, "y_lane", current_lane))
    if token in SOLID_TOKENS:
        if y_lane <= current_lane and y_lane + 1 >= current_lane:
            return max(current_lane, y_lane + 1)
        if y_lane < current_lane:
            return max(1, y_lane + 1)
        return current_lane
    if token in ORB_TOKENS:
        return max(max(1, current_lane - 2), min(min(15, current_lane + 2), y_lane))
    if token in {"PAD_YELLOW", "PAD_BLUE"}:
        return max(1, min(15, y_lane + 1))
    if token in GRAVITY_TOKENS:
        return current_lane
    if token in PORTAL_TOKENS:
        return current_lane
    return current_lane
