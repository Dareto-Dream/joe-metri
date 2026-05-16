from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

from gd_scraper.gd_objects import ORB_TOKENS, PORTAL_TOKENS, PRIMARY_MECHANIC_TOKENS, SOLID_TOKENS
from gd_scraper.quality import evaluate_token_sequence_quality
from gd_scraper.reconstruction import reconstruct_tokens

from .conditioning import ConditioningProfile


HAZARD_TOKENS = {"SPIKE", "SAW"}
PAD_TOKENS = {"PAD_YELLOW", "PAD_BLUE"}
GRAVITY_TOKENS = {"GRAVITY_UP", "GRAVITY_DOWN"}
SPEED_TOKENS = {"SPEED_SLOW", "SPEED_NORMAL", "SPEED_FAST"}
CONTROL_TOKENS = PORTAL_TOKENS | GRAVITY_TOKENS | SPEED_TOKENS
FLOW_ORBS = ("ORB_YELLOW", "ORB_BLUE", "ORB_PINK", "ORB_BLACK")
FLOW_PORTALS = ("PORTAL_CUBE", "PORTAL_SHIP", "PORTAL_BALL", "PORTAL_UFO", "PORTAL_WAVE")
GROUND_WIDTH = 4
GROUND_LANE = 0
MAX_TERRAIN_TOKEN_SHARE = 0.34
MAX_EVENTS_PER_STEP = 2


@dataclass(frozen=True)
class FlowSyncReport:
    score: float
    sync_score: float
    flow_score: float
    synced_objects: int
    unsynced_objects: int
    beat_aligned_objects: int
    onset_aligned_objects: int
    max_lane_jump: int
    path_obstructions: int
    duplicate_positions: int
    max_objects_per_step: int
    object_count: int
    step_count: int
    source_token_count: int
    arranged_token_count: int

    def to_json(self) -> dict[str, float | int]:
        return {
            "score": round(self.score, 6),
            "sync_score": round(self.sync_score, 6),
            "flow_score": round(self.flow_score, 6),
            "synced_objects": self.synced_objects,
            "unsynced_objects": self.unsynced_objects,
            "beat_aligned_objects": self.beat_aligned_objects,
            "onset_aligned_objects": self.onset_aligned_objects,
            "max_lane_jump": self.max_lane_jump,
            "path_obstructions": self.path_obstructions,
            "duplicate_positions": self.duplicate_positions,
            "max_objects_per_step": self.max_objects_per_step,
            "object_count": self.object_count,
            "step_count": self.step_count,
            "source_token_count": self.source_token_count,
            "arranged_token_count": self.arranged_token_count,
        }


@dataclass
class _FlowState:
    lane: int = 1
    max_lane_jump: int = 0
    last_portal_step: int = -999
    last_speed_step: int = -999
    last_gravity_step: int = -999
    last_token: str = ""
    orb_run: int = 0


def arrange_flow_synced_tokens(
    source_tokens: list[str],
    conditioning: ConditioningProfile,
    *,
    seed: int,
) -> tuple[list[str], FlowSyncReport]:
    rng = random.Random(seed)
    steps = _planned_step_count(conditioning)
    object_count = _target_object_count(conditioning, steps)
    event_steps = _select_event_steps(conditioning, steps, object_count, rng)
    style = _style_weights(source_tokens, conditioning.alignments)

    tokens = _normalized_prefix(conditioning)
    state = _FlowState()
    events_by_step = _build_events(event_steps, conditioning, style, state, rng)
    _add_playable_motifs(events_by_step, event_steps, steps, conditioning, rng)
    _add_raised_scaffolds(events_by_step, steps, conditioning, rng)
    _add_support_terrain(events_by_step, steps, conditioning)

    for step in range(steps):
        for event in events_by_step.get(step, []):
            _append_event(tokens, event)
        tokens.append("STEP")
    tokens.append("END")

    report = score_flow_sync(tokens, conditioning, source_token_count=len(source_tokens))
    return tokens, report


def score_flow_sync(
    tokens: list[str],
    conditioning: ConditioningProfile,
    *,
    source_token_count: int | None = None,
) -> FlowSyncReport:
    objects, _errors = reconstruct_tokens(tokens)
    quality = evaluate_token_sequence_quality(tokens)
    if not objects:
        return FlowSyncReport(
            score=0.0,
            sync_score=0.0,
            flow_score=0.0,
            synced_objects=0,
            unsynced_objects=0,
            beat_aligned_objects=0,
            onset_aligned_objects=0,
            max_lane_jump=0,
            path_obstructions=quality.path_obstructions,
            duplicate_positions=quality.duplicate_positions,
            max_objects_per_step=quality.max_objects_per_step,
            object_count=0,
            step_count=tokens.count("STEP"),
            source_token_count=len(tokens) if source_token_count is None else source_token_count,
            arranged_token_count=len(tokens),
        )

    beat_steps = _expanded_sync_steps(conditioning.beat_steps, tokens.count("STEP"))
    onset_steps = conditioning.onset_steps
    beat_aligned = 0
    onset_aligned = 0
    synced = 0
    lanes: list[tuple[int, int]] = []
    current_lane = 1
    sync_objects = [item for item in objects if not _is_support_terrain(item.token, item.y_lane)]
    for item in sorted(sync_objects, key=lambda value: (value.x_step, value.sequence)):
        beat_distance = _nearest_distance(item.x_step, beat_steps)
        onset_distance = _nearest_distance(item.x_step, onset_steps)
        if beat_distance == 0:
            beat_aligned += 1
        if onset_distance == 0:
            onset_aligned += 1
        if min(beat_distance, onset_distance) <= 1:
            synced += 1

        next_lane = _next_path_lane(current_lane, item.token, item.y_lane)
        lanes.append((item.x_step, next_lane))
        current_lane = next_lane

    max_lane_jump = 0
    for (_left_step, left_lane), (_right_step, right_lane) in zip(lanes, lanes[1:]):
        max_lane_jump = max(max_lane_jump, abs(right_lane - left_lane))

    object_count = len(objects)
    scored_object_count = len(sync_objects)
    sync_score = 100.0 * synced / max(scored_object_count, 1)
    flow_score = 100.0
    flow_score -= max(0, max_lane_jump - 2) * 10.0
    flow_score -= quality.path_obstructions * 18.0
    flow_score -= quality.duplicate_positions * 8.0
    flow_score -= max(0, quality.max_objects_per_step - 2) * 10.0
    flow_score -= quality.control_spam * 10.0
    flow_score -= quality.path_jumps * 8.0
    flow_score = max(0.0, min(100.0, flow_score))

    combined = sync_score * 0.58 + flow_score * 0.42
    return FlowSyncReport(
        score=combined,
        sync_score=sync_score,
        flow_score=flow_score,
        synced_objects=synced,
        unsynced_objects=max(0, scored_object_count - synced),
        beat_aligned_objects=beat_aligned,
        onset_aligned_objects=onset_aligned,
        max_lane_jump=max_lane_jump,
        path_obstructions=quality.path_obstructions,
        duplicate_positions=quality.duplicate_positions,
        max_objects_per_step=quality.max_objects_per_step,
        object_count=object_count,
        step_count=tokens.count("STEP"),
        source_token_count=len(tokens) if source_token_count is None else source_token_count,
        arranged_token_count=len(tokens),
    )


def _planned_step_count(conditioning: ConditioningProfile) -> int:
    audio_steps = max(conditioning.energy_by_step, default=24)
    budget_steps = max(18, int(conditioning.target_tokens * 0.66))
    return max(18, min(audio_steps, budget_steps, 620))


def _target_object_count(conditioning: ConditioningProfile, steps: int) -> int:
    budget = max(1, (conditioning.target_tokens - len(conditioning.prefix) - steps - 1) // 2)
    alignment_rate = 0.0
    for alignment in conditioning.alignments:
        normalized = alignment.lower()
        if normalized == "dense":
            alignment_rate += 0.08
        elif normalized == "sync-heavy":
            alignment_rate += 0.05
        elif normalized == "technical":
            alignment_rate += 0.04
        elif normalized == "flow":
            alignment_rate -= 0.02
    rate = max(0.30, min(0.68, 0.30 + conditioning.density * 0.18 + alignment_rate))
    desired = max(6, int(round(steps * rate)))
    return max(1, min(desired, budget))


def _select_event_steps(
    conditioning: ConditioningProfile,
    steps: int,
    object_count: int,
    rng: random.Random,
) -> list[int]:
    priorities: dict[int, float] = {}
    beat_steps = _expanded_sync_steps(conditioning.beat_steps, steps)
    for step in beat_steps:
        if 2 <= step < steps - 2:
            priorities[step] = max(priorities.get(step, 0.0), 2.0 + conditioning.energy_for_step(step))
    for step in conditioning.onset_steps:
        if 2 <= step < steps - 2:
            priorities[step] = max(priorities.get(step, 0.0), 4.0 + conditioning.energy_for_step(step) * 1.4)

    if not priorities:
        for step in range(3, max(3, steps - 2), 4):
            priorities[step] = 1.0 + conditioning.energy_for_step(step)

    ranked = sorted(priorities, key=lambda step: (priorities[step], rng.random()), reverse=True)
    selected = _pick_spaced_steps(ranked, object_count, min_gap=2)
    if len(selected) < object_count:
        fill = sorted(
            (step for step in range(2, steps - 2) if step not in selected),
            key=lambda step: (conditioning.energy_for_step(step), rng.random()),
            reverse=True,
        )
        selected.extend(_pick_spaced_steps(fill, object_count - len(selected), min_gap=1, existing=set(selected)))
    return sorted(selected[:object_count])


def _pick_spaced_steps(
    candidates: list[int],
    count: int,
    *,
    min_gap: int,
    existing: set[int] | None = None,
) -> list[int]:
    selected = set(existing or set())
    picked: list[int] = []
    for step in candidates:
        if len(picked) >= count:
            break
        if any(abs(step - other) < min_gap for other in selected):
            continue
        selected.add(step)
        picked.append(step)
    return picked


def _style_weights(source_tokens: list[str], alignments: tuple[str, ...]) -> Counter[str]:
    weights: Counter[str] = Counter({token: 1 for token in PRIMARY_MECHANIC_TOKENS})
    source_counts = Counter(token for token in source_tokens if token in PRIMARY_MECHANIC_TOKENS)
    for token, count in source_counts.items():
        weights[token] += min(5, int(count**0.5))

    normalized = {alignment.lower() for alignment in alignments}
    if "flow" in normalized:
        weights.update({"ORB_YELLOW": 4, "ORB_BLUE": 3, "PAD_YELLOW": 2, "SPIKE": 2, "SAW": -1})
    if "sync-heavy" in normalized:
        weights.update({"SPIKE": 4, "ORB_YELLOW": 3, "ORB_BLUE": 3, "SPEED_FAST": 2})
    if "technical" in normalized:
        weights.update({"ORB_BLUE": 4, "ORB_BLACK": 3, "GRAVITY_UP": 2, "GRAVITY_DOWN": 2})
    if "jump-heavy" in normalized:
        weights.update({"SPIKE": 4, "PAD_YELLOW": 3, "ORB_YELLOW": 3})
    if "wave-heavy" in normalized:
        weights.update({"PORTAL_WAVE": 5, "SPIKE": 3, "ORB_BLUE": 2})
    if "ship-focused" in normalized:
        weights.update({"PORTAL_SHIP": 5, "PORTAL_CUBE": 3, "SPIKE": 2})
    return weights


def _normalized_prefix(conditioning: ConditioningProfile) -> list[str]:
    prefix = conditioning.prefix[:]
    if not prefix or prefix[0] != "START":
        prefix.insert(0, "START")
    if len(prefix) == 1:
        prefix.append("DIFF_HARD")
    if len(prefix) == 2:
        prefix.append("ALIGN_UNKNOWN")
    return prefix[:3]


def _build_events(
    event_steps: list[int],
    conditioning: ConditioningProfile,
    style: Counter[str],
    state: _FlowState,
    rng: random.Random,
) -> dict[int, list[tuple[str, int, int]]]:
    events: dict[int, list[tuple[str, int, int]]] = {}
    for index, step in enumerate(event_steps):
        token = _choose_token(step, index, conditioning, style, state, rng)
        lane = _choose_lane(token, state.lane, step, conditioning, rng)
        width = 1
        if token in SOLID_TOKENS:
            width = 1 if conditioning.energy_for_step(step) > 0.45 else 2
        events.setdefault(step, []).append((token, lane, width))
        next_lane = _next_path_lane(state.lane, token, lane)
        state.max_lane_jump = max(state.max_lane_jump, abs(next_lane - state.lane))
        state.lane = next_lane
        state.last_token = token
        state.orb_run = state.orb_run + 1 if token in ORB_TOKENS else 0
        if token in PORTAL_TOKENS:
            state.last_portal_step = step
        elif token in SPEED_TOKENS:
            state.last_speed_step = step
        elif token in GRAVITY_TOKENS:
            state.last_gravity_step = step
    return events


def _choose_token(
    step: int,
    index: int,
    conditioning: ConditioningProfile,
    style: Counter[str],
    state: _FlowState,
    rng: random.Random,
) -> str:
    energy = conditioning.energy_for_step(step)
    beat = _beat_proximity(conditioning, step)
    onset = conditioning.onset_proximity(step)
    normalized = {alignment.lower() for alignment in conditioning.alignments}

    if (
        index > 0
        and step - state.last_portal_step >= 28
        and (index % 13 == 0 or "wave-heavy" in normalized or ("ship-focused" in normalized and index % 9 == 0))
    ):
        return _weighted_choice(FLOW_PORTALS, style, rng)
    if index > 0 and step - state.last_speed_step >= 34 and index % 17 == 0:
        return "SPEED_FAST" if energy > 0.58 else "SPEED_NORMAL"
    if "technical" in normalized and step - state.last_gravity_step >= 44 and onset > 0.0 and index % 8 == 0:
        return rng.choice(("GRAVITY_UP", "GRAVITY_DOWN"))

    choices: Counter[str] = Counter()
    if onset >= 1.0:
        choices.update({"ORB_YELLOW": 5, "ORB_BLUE": 4, "SPIKE": 4, "PAD_YELLOW": 2})
        if energy > 0.7:
            choices.update({"ORB_PINK": 2, "SAW": 2})
    elif beat >= 1.0:
        choices.update({"SPIKE": 5, "ORB_YELLOW": 3, "PAD_YELLOW": 2, "ORB_BLUE": 2})
    elif energy > 0.62:
        choices.update({"ORB_BLUE": 4, "SPIKE": 3, "ORB_YELLOW": 3, "PAD_BLUE": 1, "SAW": 1})
    else:
        choices.update({"SPIKE": 3, "ORB_YELLOW": 2, "PAD_YELLOW": 1})

    if "flow" in normalized:
        choices.update({"ORB_YELLOW": 3, "ORB_BLUE": 3, "PAD_YELLOW": 1, "SPIKE": -1})
    if "sync-heavy" in normalized and beat > 0.0:
        choices.update({"SPIKE": 2, "ORB_YELLOW": 1})
    if "jump-heavy" in normalized:
        choices.update({"SPIKE": 2, "PAD_YELLOW": 2})
    if "technical" in normalized:
        choices.update({"ORB_BLUE": 2, "ORB_BLACK": 1})

    if state.orb_run >= 2:
        for token in ORB_TOKENS:
            choices[token] -= 3
    if state.last_token == "SPIKE":
        choices["SPIKE"] -= 3
    for token, weight in style.items():
        if token in choices:
            choices[token] += max(0, min(3, weight // 2))

    return _weighted_choice(tuple(token for token, weight in choices.items() if weight > 0), choices, rng)


def _choose_lane(
    token: str,
    current_lane: int,
    step: int,
    conditioning: ConditioningProfile,
    rng: random.Random,
) -> int:
    energy = conditioning.energy_for_step(step)
    if token == "SPIKE":
        return 1
    if token == "SAW":
        return _clamp_lane(current_lane + (3 if energy > 0.55 else 2), 4, 8)
    if token in PAD_TOKENS:
        return 1
    if token in ORB_TOKENS:
        drift = rng.choice((-1, 0, 1, 1, 2 if energy > 0.55 else 0))
        return _clamp_lane(current_lane + drift, 2, 7)
    if token in PORTAL_TOKENS | GRAVITY_TOKENS | SPEED_TOKENS:
        return _clamp_lane(current_lane + rng.choice((0, 1, 2)), 3, 7)
    if token in SOLID_TOKENS:
        return _clamp_lane(current_lane + 3, 4, 8)
    return _clamp_lane(current_lane, 1, 9)


def _append_event(tokens: list[str], event: tuple[str, int, int]) -> None:
    token, lane, width = event
    tokens.append(token)
    tokens.append(f"Y{_clamp_lane(lane, 0, 15)}")
    if token in SOLID_TOKENS:
        tokens.append(f"WIDTH_{max(1, min(16, width))}")


def _next_path_lane(current_lane: int, token: str, y_lane: int) -> int:
    if token in SOLID_TOKENS:
        if y_lane <= 0:
            return current_lane
        if y_lane <= current_lane and y_lane + 1 >= current_lane:
            return max(current_lane, y_lane + 1)
        if y_lane < current_lane:
            return max(1, y_lane + 1)
        return current_lane
    if token in ORB_TOKENS:
        return _clamp_lane(y_lane, max(1, current_lane - 2), min(15, current_lane + 2))
    if token in PAD_TOKENS:
        return _clamp_lane(y_lane + 1, 1, 15)
    if token in CONTROL_TOKENS:
        return current_lane
    if token in HAZARD_TOKENS:
        return _clamp_lane(current_lane - 2, 1, 15)
    return current_lane


def _nearest_distance(step: int, targets: set[int]) -> int:
    if not targets:
        return 999
    return min(abs(step - target) for target in targets)


def _beat_proximity(conditioning: ConditioningProfile, step: int) -> float:
    direct = conditioning.beat_proximity(step)
    if direct > 0.0:
        return direct
    expanded = _expanded_sync_steps(conditioning.beat_steps, step + 2)
    if step in expanded:
        return 1.0
    if step - 1 in expanded or step + 1 in expanded:
        return 0.45
    return 0.0


def _expanded_sync_steps(steps: set[int], limit: int) -> set[int]:
    if not steps:
        return set()
    expanded = {step for step in steps if 0 <= step <= limit}
    ordered = sorted(steps)
    intervals = [right - left for left, right in zip(ordered, ordered[1:]) if right > left]
    interval = int(round(sum(intervals) / len(intervals))) if intervals else 2
    interval = max(1, interval)
    step = ordered[0]
    while step <= limit:
        expanded.add(step)
        step += interval
    return expanded


def _weighted_choice(tokens: tuple[str, ...], weights: Counter[str], rng: random.Random) -> str:
    if not tokens:
        return "SPIKE"
    weighted = [(token, max(1, int(weights.get(token, 1)))) for token in tokens]
    total = sum(weight for _token, weight in weighted)
    cursor = rng.randint(1, total)
    running = 0
    for token, weight in weighted:
        running += weight
        if running >= cursor:
            return token
    return weighted[-1][0]


def _clamp_lane(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _add_support_terrain(
    events_by_step: dict[int, list[tuple[str, int, int]]],
    steps: int,
    conditioning: ConditioningProfile,
) -> None:
    terrain_token_budget = max(0, int(conditioning.target_tokens * MAX_TERRAIN_TOKEN_SHARE))
    max_segments = max(1, terrain_token_budget // 3)
    starts = list(range(0, steps, GROUND_WIDTH))[:max_segments]
    for start in starts:
        if _is_floor_gap(start, steps):
            _put_event(events_by_step, start - 2, "PAD_YELLOW", 1, steps=steps)
            _put_event(events_by_step, start + 2, "ORB_YELLOW", 3, steps=steps)
            continue
        width = max(1, min(GROUND_WIDTH, steps - start))
        events = events_by_step.setdefault(start, [])
        if len(events) < MAX_EVENTS_PER_STEP:
            events.insert(0, ("BLOCK", GROUND_LANE, width))


def _is_support_terrain(token: str, y_lane: int) -> bool:
    return token in SOLID_TOKENS and y_lane <= GROUND_LANE


def _add_playable_motifs(
    events_by_step: dict[int, list[tuple[str, int, int]]],
    event_steps: list[int],
    steps: int,
    conditioning: ConditioningProfile,
    rng: random.Random,
) -> None:
    if not event_steps:
        return

    for index, step in enumerate(event_steps):
        if step < 4 or step >= steps - 4:
            continue

        phase = index % 12
        if phase in {1, 7}:
            _replace_first_interactive(events_by_step, step, "SPIKE", 1)
            _put_event(events_by_step, step + 2, "SPIKE", 1, steps=steps)
        elif phase in {2, 8}:
            _replace_first_interactive(events_by_step, step, "PAD_YELLOW", 1)
            _put_event(events_by_step, step + 2, "ORB_YELLOW", 3, steps=steps)
            _put_event(events_by_step, step + 4, "SPIKE", 1, steps=steps)
        elif phase == 4:
            platform_lane = 2 + (index // 12) % 3
            _put_event(events_by_step, step + 1, "PLATFORM", platform_lane, width=3, steps=steps)
            _put_event(events_by_step, step + 3, rng.choice(FLOW_ORBS), platform_lane + 2, steps=steps)
        elif phase == 6:
            _replace_first_interactive(events_by_step, step, "PAD_BLUE", 1)
            _put_event(events_by_step, step + 2, "ORB_BLUE", 4, steps=steps)
        elif phase == 10:
            _replace_first_interactive(events_by_step, step, rng.choice(("GRAVITY_UP", "GRAVITY_DOWN")), 5)
            _put_event(events_by_step, step + 2, rng.choice(FLOW_PORTALS), 5, steps=steps)

    if "ship-focused" in {item.lower() for item in conditioning.alignments}:
        for step in event_steps[::10]:
            _put_event(events_by_step, step, "PORTAL_SHIP", 5, steps=steps)


def _add_raised_scaffolds(
    events_by_step: dict[int, list[tuple[str, int, int]]],
    steps: int,
    conditioning: ConditioningProfile,
    rng: random.Random,
) -> None:
    normalized = {item.lower() for item in conditioning.alignments}
    spacing = 10 if "dense" in normalized else 12
    for index, start in enumerate(range(10, max(10, steps - 8), spacing)):
        lane = 2 + index % 4
        width = 2 + (index % 2)
        _put_event(events_by_step, start + 1, "PLATFORM", lane, width=width, steps=steps)

        if index % 3 == 0:
            _put_event(events_by_step, start + 4, rng.choice(FLOW_ORBS), min(7, lane + 2), steps=steps)
        elif index % 3 == 1:
            _put_event(events_by_step, start + 4, "SPIKE", 1, steps=steps)
        else:
            _put_event(events_by_step, start + 4, "PAD_YELLOW", 1, steps=steps)


def _put_event(
    events_by_step: dict[int, list[tuple[str, int, int]]],
    step: int,
    token: str,
    lane: int,
    *,
    steps: int,
    width: int = 1,
) -> bool:
    if step < 1 or step >= steps - 1:
        return False

    events = events_by_step.setdefault(step, [])
    if any(existing_token == token and existing_lane == lane for existing_token, existing_lane, _width in events):
        return False
    if any(existing_lane == lane for _existing_token, existing_lane, _width in events):
        return False
    if token in SOLID_TOKENS and any(existing_token in SOLID_TOKENS for existing_token, _lane, _width in events):
        return False
    if len(events) >= MAX_EVENTS_PER_STEP:
        return False

    events.append((token, lane, width))
    return True


def _replace_first_interactive(
    events_by_step: dict[int, list[tuple[str, int, int]]],
    step: int,
    token: str,
    lane: int,
) -> None:
    events = events_by_step.get(step)
    if not events:
        events_by_step[step] = [(token, lane, 1)]
        return

    for index, (existing_token, _existing_lane, _width) in enumerate(events):
        if existing_token not in SOLID_TOKENS:
            events[index] = (token, lane, 1)
            events[:] = _dedupe_event_lanes(events)
            return
    if len(events) < MAX_EVENTS_PER_STEP:
        events.append((token, lane, 1))
        events[:] = _dedupe_event_lanes(events)


def _is_floor_gap(start: int, steps: int) -> bool:
    segment = start // GROUND_WIDTH
    if start < 16 or start > steps - 16:
        return False
    return segment % 11 == 6 or segment % 17 == 10


def _dedupe_event_lanes(events: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    occupied_lanes: set[int] = set()
    for token, lane, width in events:
        if lane in occupied_lanes:
            continue
        occupied_lanes.add(lane)
        result.append((token, lane, width))
    return result[:MAX_EVENTS_PER_STEP]
