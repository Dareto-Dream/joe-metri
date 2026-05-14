from __future__ import annotations

import random

from gd_scraper.gd_objects import ORB_TOKENS, PORTAL_TOKENS, PRIMARY_MECHANIC_TOKENS, SOLID_TOKENS
from gd_scraper.train_mechanics import TinyNgramModel, allowed_next_tokens, close_sample

from .conditioning import ConditioningProfile


def sample_tokens(
    model: TinyNgramModel,
    *,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    conditioning: ConditioningProfile,
    seed: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    rng = random.Random(seed)
    pad_id = token_to_id["<PAD>"]
    unk_id = token_to_id["<UNK>"]
    tokens = conditioning.prefix[:]

    while len(tokens) < conditioning.target_tokens and tokens[-1] != "END":
        context_tokens = tokens[-model.context_size :]
        context_ids = [token_to_id.get(token, unk_id) for token in context_tokens]
        context_ids = [pad_id] * max(0, model.context_size - len(context_ids)) + context_ids
        probabilities = model.probabilities(context_ids)
        allowed = allowed_next_tokens(tokens, min_generated=conditioning.min_tokens, max_consecutive_steps=8)
        next_token = _sample_from_allowed(
            probabilities=probabilities,
            allowed_tokens=allowed,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            tokens=tokens,
            conditioning=conditioning,
            rng=rng,
            temperature=temperature,
            top_k=top_k,
        )
        tokens.append(next_token)

    if tokens[-1] != "END":
        close_sample(tokens)
    return tokens


def _sample_from_allowed(
    *,
    probabilities: list[float],
    allowed_tokens: list[str],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    tokens: list[str],
    conditioning: ConditioningProfile,
    rng: random.Random,
    temperature: float,
    top_k: int,
) -> str:
    current_step = tokens.count("STEP")
    same_step_objects = _same_step_object_count(tokens)
    energy = conditioning.energy_for_step(current_step)
    beat = conditioning.beat_proximity(current_step)
    onset = conditioning.onset_proximity(current_step)

    weighted: list[tuple[str, float]] = []
    for token in allowed_tokens:
        token_id = token_to_id.get(token)
        if token_id is None:
            continue
        probability = max(probabilities[token_id], 1e-12)
        weight = probability ** (1.0 / max(temperature, 1e-6))
        weight *= _audio_bias(token, energy=energy, beat=beat, onset=onset, same_step_objects=same_step_objects)
        weight *= _style_bias(token, conditioning)
        weight *= _length_bias(token, len(tokens), conditioning)
        weighted.append((id_to_token[token_id], max(weight, 1e-12)))

    if not weighted:
        return "END"

    weighted.sort(key=lambda item: item[1], reverse=True)
    if top_k > 0:
        weighted = weighted[:top_k]

    total = sum(weight for _token, weight in weighted)
    threshold = rng.random() * total
    running = 0.0
    for token, weight in weighted:
        running += weight
        if running >= threshold:
            return token
    return weighted[-1][0]


def _audio_bias(token: str, *, energy: float, beat: float, onset: float, same_step_objects: int) -> float:
    object_drive = 0.72 + energy * 1.35 + beat * 0.7 + onset * 0.9
    quiet_drive = 1.45 - energy * 0.55

    if token == "STEP":
        return max(0.35, quiet_drive - beat * 0.35 - onset * 0.45 + same_step_objects * 0.42)
    if token == "END":
        return 1.0
    if token in PRIMARY_MECHANIC_TOKENS:
        drive = object_drive
        if token in ORB_TOKENS:
            drive += beat * 0.55 + onset * 0.25
        if token in PORTAL_TOKENS:
            drive += max(0.0, energy - 0.55) * 0.75
        if token in SOLID_TOKENS:
            drive += max(0.0, 0.55 - energy) * 0.25
        if same_step_objects >= 4:
            drive *= 0.25
        elif same_step_objects >= 2:
            drive *= 0.55
        return max(0.1, drive)
    return 1.0


def _style_bias(token: str, conditioning: ConditioningProfile) -> float:
    return conditioning.object_bias.get(token, 1.0) * conditioning.density if token in PRIMARY_MECHANIC_TOKENS else 1.0


def _length_bias(token: str, token_count: int, conditioning: ConditioningProfile) -> float:
    progress = token_count / max(conditioning.target_tokens, 1)
    if token == "END":
        if progress < 0.72:
            return 0.03
        return 0.25 + progress
    if token == "STEP" and progress > 0.92:
        return 1.35
    return 1.0


def _same_step_object_count(tokens: list[str]) -> int:
    count = 0
    index = len(tokens) - 1
    while index >= 0 and tokens[index] != "STEP":
        if tokens[index] in PRIMARY_MECHANIC_TOKENS:
            count += 1
        index -= 1
    return count

