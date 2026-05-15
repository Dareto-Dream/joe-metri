from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gd_scraper.quality import TokenQualityReport, evaluate_token_sequence_quality
from gd_scraper.train_mechanics import TinyNgramModel

from .conditioning import ConditioningProfile
from .sampler import sample_tokens


PlanningEventCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class GenerationCandidate:
    index: int
    seed: int
    tokens: list[str]
    quality: TokenQualityReport

    def to_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "token_count": len(self.tokens),
            "quality": self.quality.to_json(),
        }


@dataclass(frozen=True)
class GenerationPlan:
    best: GenerationCandidate
    candidates: list[GenerationCandidate]

    def to_json(self) -> dict[str, object]:
        return {
            "selected_candidate": self.best.index,
            "iterations": len(self.candidates),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


def plan_tokens(
    model: TinyNgramModel,
    *,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    conditioning: ConditioningProfile,
    seed: int,
    temperature: float,
    top_k: int,
    iterations: int,
    event_callback: PlanningEventCallback | None = None,
) -> GenerationPlan:
    candidates: list[GenerationCandidate] = []
    active_iterations = max(1, iterations)

    for index in range(active_iterations):
        candidate_seed = seed + index * 9_973
        if event_callback is not None:
            event_callback({"type": "candidate_start", "candidate": index, "seed": candidate_seed})

        tokens = sample_tokens(
            model,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            conditioning=conditioning,
            seed=candidate_seed,
            temperature=temperature,
            top_k=top_k,
            token_callback=_candidate_token_callback(index, event_callback),
        )
        quality = evaluate_token_sequence_quality(tokens)
        candidate = GenerationCandidate(index=index, seed=candidate_seed, tokens=tokens, quality=quality)
        candidates.append(candidate)

        if event_callback is not None:
            event_callback(
                {
                    "type": "candidate_done",
                    "candidate": index,
                    "seed": candidate_seed,
                    "quality": quality.to_json(),
                    "token_count": len(tokens),
                }
            )

    best = max(candidates, key=lambda candidate: (candidate.quality.score, candidate.quality.valid, -candidate.index))
    if event_callback is not None:
        event_callback({"type": "selected", "candidate": best.index, "quality": best.quality.to_json()})
    return GenerationPlan(best=best, candidates=candidates)


def _candidate_token_callback(
    candidate_index: int,
    event_callback: PlanningEventCallback | None,
) -> Callable[[str, list[str]], None] | None:
    if event_callback is None:
        return None

    def emit(token: str, tokens: list[str]) -> None:
        event_callback(
            {
                "type": "token",
                "candidate": candidate_index,
                "token": token,
                "token_count": len(tokens),
            }
        )

    return emit
