from __future__ import annotations

from dataclasses import dataclass

from gd_scraper.gd_objects import BASE_VOCAB_TOKENS
from gd_scraper.reconstruction import validate_token_grammar


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]
    invalid_token_count: int
    invalid_token_rate: float

    def to_json(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "invalid_token_count": self.invalid_token_count,
            "invalid_token_rate": round(self.invalid_token_rate, 6),
        }


def validate_generation(tokens: list[str], *, known_tokens: set[str] | None = None) -> ValidationResult:
    token_scope = known_tokens or set(BASE_VOCAB_TOKENS)
    unknown = [token for token in tokens if token not in token_scope]
    errors = validate_token_grammar(tokens)
    if unknown:
        errors.extend(f"unknown_token:{token}" for token in sorted(set(unknown)))
    invalid_count = len(unknown)
    return ValidationResult(
        valid=not errors,
        errors=errors,
        invalid_token_count=invalid_count,
        invalid_token_rate=invalid_count / max(len(tokens), 1),
    )

