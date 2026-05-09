from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiscoverySource:
    name: str
    params: dict[str, Any]


DEFAULT_SOURCES: dict[str, DiscoverySource] = {
    "featured": DiscoverySource("featured", {"type": 6}),
    "rated": DiscoverySource("rated", {"type": 11, "star": 1}),
    "trending": DiscoverySource("trending", {"type": 3, "star": 1}),
    "popular": DiscoverySource("popular", {"type": 1, "star": 1}),
    "most_liked": DiscoverySource("most_liked", {"type": 2, "star": 1}),
    "demons": DiscoverySource("demons", {"type": 11, "diff": -2}),
    "easy_demons": DiscoverySource("easy_demons", {"type": 11, "diff": -2, "demonFilter": 1}),
    "medium_demons": DiscoverySource("medium_demons", {"type": 11, "diff": -2, "demonFilter": 2}),
    "hard_demons": DiscoverySource("hard_demons", {"type": 11, "diff": -2, "demonFilter": 3}),
    "insane_demons": DiscoverySource("insane_demons", {"type": 11, "diff": -2, "demonFilter": 4}),
    "extreme_demons": DiscoverySource("extreme_demons", {"type": 11, "diff": -2, "demonFilter": 5}),
}


def resolve_sources(names: list[str]) -> list[DiscoverySource]:
    unknown = [name for name in names if name not in DEFAULT_SOURCES]
    if unknown:
        valid = ", ".join(sorted(DEFAULT_SOURCES))
        raise ValueError(f"unknown sources {unknown}; valid sources: {valid}")
    return [DEFAULT_SOURCES[name] for name in names]
