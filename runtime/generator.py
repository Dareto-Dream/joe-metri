from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any
import uuid

from gd_scraper.gd_objects import ORB_TOKENS, PORTAL_TOKENS
from gd_scraper.storage import loads_json
from gd_scraper.train_mechanics import TinyNgramModel, model_from_json

from .audio import AudioAnalysis, analyze_audio
from .conditioning import ConditioningProfile, GenerationControls, build_conditioning
from .reconstructor import RuntimeLayout, reconstruct_layout
from .sampler import sample_tokens
from .validator import ValidationResult, validate_generation


@dataclass(frozen=True)
class GenerationMetrics:
    generation_time_seconds: float
    tokens_per_second: float
    sequence_length: int
    step_density: float
    portal_count: int
    orb_count: int
    invalid_token_rate: float
    object_count: int

    def to_json(self) -> dict[str, float | int]:
        return {
            "generation_time_seconds": round(self.generation_time_seconds, 4),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "sequence_length": self.sequence_length,
            "step_density": round(self.step_density, 6),
            "portal_count": self.portal_count,
            "orb_count": self.orb_count,
            "invalid_token_rate": round(self.invalid_token_rate, 6),
            "object_count": self.object_count,
        }


@dataclass(frozen=True)
class GenerationResult:
    generation_id: str
    audio: AudioAnalysis
    controls: GenerationControls
    conditioning: ConditioningProfile
    tokens: list[str]
    validation: ValidationResult
    layout: RuntimeLayout
    metrics: GenerationMetrics
    model_path: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.generation_id,
            "audio": self.audio.to_json(),
            "controls": {
                "difficulty": self.controls.difficulty,
                "alignments": list(self.controls.alignments),
                "temperature": self.controls.temperature,
                "top_k": self.controls.top_k,
                "max_tokens": self.controls.max_tokens,
                "seed": self.controls.seed,
            },
            "conditioning": {
                "prefix": self.conditioning.prefix,
                "target_tokens": self.conditioning.target_tokens,
                "min_tokens": self.conditioning.min_tokens,
                "step_seconds": round(self.conditioning.step_seconds, 4),
                "density": round(self.conditioning.density, 4),
            },
            "tokens": self.tokens,
            "validation": self.validation.to_json(),
            "preview": self.layout.to_json(),
            "metrics": self.metrics.to_json(),
            "model_path": self.model_path,
        }


class MechanicsRuntime:
    def __init__(self, model_dir: Path = Path("models") / "mechanics_v1") -> None:
        self.model_dir = model_dir
        self.model_path = resolve_model_path(model_dir)
        payload = loads_json(self.model_path.read_bytes())
        self.model: TinyNgramModel = model_from_json(payload)
        vocab = payload.get("vocab") or {}
        self.token_to_id = {str(key): int(value) for key, value in vocab.get("token_to_id", {}).items()}
        self.id_to_token = [str(token) for token in vocab.get("id_to_token", [])]
        if not self.token_to_id or not self.id_to_token:
            raise ValueError(f"model_vocab_missing:{self.model_path}")
        self.known_tokens = set(self.id_to_token)

    def generate_from_audio(
        self,
        audio_path: Path,
        *,
        filename: str | None = None,
        controls: GenerationControls | None = None,
    ) -> GenerationResult:
        active_controls = controls or GenerationControls()
        started = time.perf_counter()
        audio = analyze_audio(audio_path, filename=filename)
        conditioning = build_conditioning(audio, active_controls)
        tokens = sample_tokens(
            self.model,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            conditioning=conditioning,
            seed=active_controls.seed,
            temperature=active_controls.temperature,
            top_k=active_controls.top_k,
        )
        validation = validate_generation(tokens, known_tokens=self.known_tokens)
        layout = reconstruct_layout(tokens)
        elapsed = max(time.perf_counter() - started, 1e-9)
        metrics = build_metrics(tokens, validation, layout, elapsed)
        return GenerationResult(
            generation_id=uuid.uuid4().hex,
            audio=audio,
            controls=active_controls,
            conditioning=conditioning,
            tokens=tokens,
            validation=validation,
            layout=layout,
            metrics=metrics,
            model_path=str(self.model_path),
        )


def build_metrics(
    tokens: list[str],
    validation: ValidationResult,
    layout: RuntimeLayout,
    elapsed_seconds: float,
) -> GenerationMetrics:
    sequence_length = len(tokens)
    portal_count = sum(1 for token in tokens if token in PORTAL_TOKENS)
    orb_count = sum(1 for token in tokens if token in ORB_TOKENS)
    return GenerationMetrics(
        generation_time_seconds=elapsed_seconds,
        tokens_per_second=sequence_length / max(elapsed_seconds, 1e-9),
        sequence_length=sequence_length,
        step_density=tokens.count("STEP") / max(sequence_length, 1),
        portal_count=portal_count,
        orb_count=orb_count,
        invalid_token_rate=validation.invalid_token_rate,
        object_count=len(layout.objects),
    )


def resolve_model_path(model_dir: Path) -> Path:
    live = model_dir / "model_live.json"
    if live.exists():
        return live

    checkpoints = model_dir / "checkpoints"
    if checkpoints.exists():
        candidates = sorted(
            checkpoints.glob("checkpoint_step_*.json"),
            key=lambda path: _checkpoint_step(path),
            reverse=True,
        )
        if candidates:
            return candidates[0]

    model = model_dir / "model.json"
    if model.exists():
        return model
    raise FileNotFoundError(f"No mechanics model found in {model_dir}")


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)\.json$", path.name)
    if not match:
        return -1
    return int(match.group(1))

