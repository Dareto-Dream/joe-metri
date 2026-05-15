from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from gd_scraper.quality import TokenQualityReport, evaluate_token_sequence_quality
from gd_scraper.train_mechanics import TinyNgramModel

from .conditioning import ConditioningProfile
from .flow import FlowSyncReport, arrange_flow_synced_tokens
from .sampler import sample_tokens


PlanningEventCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class GenerationCandidate:
    index: int
    seed: int
    tokens: list[str]
    quality: TokenQualityReport
    flow_sync: FlowSyncReport

    def to_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "token_count": len(self.tokens),
            "quality": self.quality.to_json(),
            "flow_sync": self.flow_sync.to_json(),
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

        sampled_tokens = sample_tokens(
            model,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            conditioning=conditioning,
            seed=candidate_seed,
            temperature=temperature,
            top_k=top_k,
        )
        tokens, flow_sync = arrange_flow_synced_tokens(sampled_tokens, conditioning, seed=candidate_seed)
        _emit_candidate_tokens(index, tokens, event_callback)
        quality = evaluate_token_sequence_quality(tokens)
        candidate = GenerationCandidate(
            index=index,
            seed=candidate_seed,
            tokens=tokens,
            quality=quality,
            flow_sync=flow_sync,
        )
        candidates.append(candidate)

        if event_callback is not None:
            event_callback(
                {
                    "type": "candidate_done",
                    "candidate": index,
                    "seed": candidate_seed,
                    "quality": quality.to_json(),
                    "flow_sync": flow_sync.to_json(),
                    "token_count": len(tokens),
                    "sampled_token_count": len(sampled_tokens),
                }
            )

    best = max(
        candidates,
        key=lambda candidate: (
            candidate.quality.valid,
            candidate.quality.score + candidate.flow_sync.score * 0.55,
            candidate.flow_sync.sync_score,
            -candidate.index,
        ),
    )
    if event_callback is not None:
        event_callback(
            {
                "type": "selected",
                "candidate": best.index,
                "quality": best.quality.to_json(),
                "flow_sync": best.flow_sync.to_json(),
            }
        )
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


def _emit_candidate_tokens(
    candidate_index: int,
    tokens: list[str],
    event_callback: PlanningEventCallback | None,
) -> None:
    callback = _candidate_token_callback(candidate_index, event_callback)
    if callback is None:
        return
    emitted: list[str] = []
    for token in tokens:
        emitted.append(token)
        callback(token, emitted[:])
