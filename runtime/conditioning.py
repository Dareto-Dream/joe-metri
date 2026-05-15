from __future__ import annotations

from dataclasses import dataclass

from gd_scraper.gd_objects import DIFFICULTY_TOKENS

from .audio import AudioAnalysis, energy_at


DIFFICULTY_TO_TOKEN = {
    "easy": "DIFF_EASY",
    "medium": "DIFF_MEDIUM",
    "hard": "DIFF_HARD",
    "insane": "DIFF_INSANE",
    "extreme": "DIFF_EXTREME",
}

DIFFICULTY_DENSITY = {
    "easy": 0.58,
    "medium": 0.76,
    "hard": 0.96,
    "insane": 1.16,
    "extreme": 1.36,
}


@dataclass(frozen=True)
class GenerationControls:
    difficulty: str = "Hard"
    alignments: tuple[str, ...] = ("Flow",)
    temperature: float = 0.9
    top_k: int = 40
    max_tokens: int = 360
    seed: int = 7
    planning_iterations: int = 4


@dataclass(frozen=True)
class ConditioningProfile:
    prefix: list[str]
    difficulty: str
    alignments: tuple[str, ...]
    target_tokens: int
    min_tokens: int
    step_seconds: float
    beat_steps: set[int]
    onset_steps: set[int]
    energy_by_step: dict[int, float]
    density: float
    object_bias: dict[str, float]

    def energy_for_step(self, step: int) -> float:
        if not self.energy_by_step:
            return 0.35
        if step in self.energy_by_step:
            return self.energy_by_step[step]
        closest = min(self.energy_by_step, key=lambda value: abs(value - step))
        return self.energy_by_step[closest]

    def beat_proximity(self, step: int) -> float:
        if step in self.beat_steps:
            return 1.0
        if step - 1 in self.beat_steps or step + 1 in self.beat_steps:
            return 0.45
        return 0.0

    def onset_proximity(self, step: int) -> float:
        if step in self.onset_steps:
            return 1.0
        if step - 1 in self.onset_steps or step + 1 in self.onset_steps:
            return 0.5
        return 0.0


def build_conditioning(analysis: AudioAnalysis, controls: GenerationControls) -> ConditioningProfile:
    difficulty_key = controls.difficulty.strip().lower()
    difficulty_token = DIFFICULTY_TO_TOKEN.get(difficulty_key, "DIFF_HARD")
    if difficulty_token not in DIFFICULTY_TOKENS:
        difficulty_token = "DIFF_HARD"

    alignments = tuple(item.strip() for item in controls.alignments if item.strip()) or ("Flow",)
    density = DIFFICULTY_DENSITY.get(difficulty_key, DIFFICULTY_DENSITY["hard"])
    for alignment in alignments:
        normalized = alignment.lower()
        if normalized == "dense":
            density += 0.22
        elif normalized == "sync-heavy":
            density += 0.12
        elif normalized == "technical":
            density += 0.16
        elif normalized == "flow":
            density -= 0.04

    beat_seconds = 60.0 / max(analysis.bpm, 1)
    step_seconds = max(0.08, beat_seconds / 2.0)
    estimated_steps = max(24, int(analysis.duration_seconds / step_seconds))
    duration_token_target = int(estimated_steps * (1.65 + density * 0.45))
    target_tokens = max(90, min(controls.max_tokens, duration_token_target))
    min_tokens = max(36, min(target_tokens - 8, int(target_tokens * 0.38)))

    beat_steps = {max(0, int(round(beat / step_seconds))) for beat in analysis.beats}
    onset_steps = {max(0, int(round(onset / step_seconds))) for onset in analysis.onsets}
    max_step = max(estimated_steps, max(beat_steps, default=0), max(onset_steps, default=0))
    energy_by_step = {
        step: energy_at(analysis, step * step_seconds)
        for step in range(max_step + 2)
    }

    return ConditioningProfile(
        prefix=["START", difficulty_token, "ALIGN_UNKNOWN"],
        difficulty=controls.difficulty,
        alignments=alignments,
        target_tokens=target_tokens,
        min_tokens=min_tokens,
        step_seconds=step_seconds,
        beat_steps=beat_steps,
        onset_steps=onset_steps,
        energy_by_step=energy_by_step,
        density=max(0.35, min(1.75, density)),
        object_bias=_object_bias(alignments),
    )


def _object_bias(alignments: tuple[str, ...]) -> dict[str, float]:
    bias: dict[str, float] = {}
    for alignment in alignments:
        normalized = alignment.lower()
        if normalized == "jump-heavy":
            bias.update({"SPIKE": 1.35, "ORB_YELLOW": 1.35, "ORB_BLUE": 1.25, "PAD_YELLOW": 1.15})
        elif normalized == "technical":
            bias.update({"ORB_BLUE": 1.3, "ORB_BLACK": 1.25, "GRAVITY_UP": 1.18, "GRAVITY_DOWN": 1.18})
        elif normalized == "sync-heavy":
            bias.update({"ORB_YELLOW": 1.28, "ORB_BLUE": 1.22, "SPIKE": 1.15, "SPEED_FAST": 1.18})
        elif normalized == "wave-heavy":
            bias.update({"PORTAL_WAVE": 2.2, "SPIKE": 1.2, "BLOCK": 0.85, "PLATFORM": 0.9})
        elif normalized == "ship-focused":
            bias.update({"PORTAL_SHIP": 2.1, "PORTAL_CUBE": 1.2, "SPIKE": 1.15, "ORB_YELLOW": 0.85})
        elif normalized == "dense":
            bias.update({"BLOCK": 1.12, "SPIKE": 1.18, "ORB_YELLOW": 1.12})
        elif normalized == "flow":
            bias.update({"BLOCK": 1.08, "PLATFORM": 1.08, "SAW": 0.85})
    return bias

