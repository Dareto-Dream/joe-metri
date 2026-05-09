from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TOKENIZER_VERSION = "0.1.0"
X_STEP_RESOLUTION = 30
Y_LANES = 16
MAX_WIDTH = 16


PRIMARY_MECHANIC_TOKENS = [
    "BLOCK",
    "PLATFORM",
    "SPIKE",
    "SAW",
    "ORB_YELLOW",
    "ORB_BLUE",
    "ORB_PINK",
    "ORB_BLACK",
    "PAD_YELLOW",
    "PAD_BLUE",
    "PORTAL_CUBE",
    "PORTAL_SHIP",
    "PORTAL_BALL",
    "PORTAL_UFO",
    "PORTAL_WAVE",
    "GRAVITY_UP",
    "GRAVITY_DOWN",
    "SPEED_SLOW",
    "SPEED_NORMAL",
    "SPEED_FAST",
]

SOLID_TOKENS = {"BLOCK", "PLATFORM"}
ORB_TOKENS = {"ORB_YELLOW", "ORB_BLUE", "ORB_PINK", "ORB_BLACK"}
PORTAL_TOKENS = {"PORTAL_CUBE", "PORTAL_SHIP", "PORTAL_BALL", "PORTAL_UFO", "PORTAL_WAVE"}
CONTROL_TOKENS = ["START", "END", "STEP", "ALIGN_UNKNOWN"]
DIFFICULTY_TOKENS = [
    "DIFF_NA",
    "DIFF_AUTO",
    "DIFF_EASY",
    "DIFF_NORMAL",
    "DIFF_MEDIUM",
    "DIFF_HARD",
    "DIFF_HARDER",
    "DIFF_INSANE",
    "DIFF_EXTREME",
]
Y_TOKENS = [f"Y{index}" for index in range(Y_LANES)]
WIDTH_TOKENS = [f"WIDTH_{index}" for index in range(1, MAX_WIDTH + 1)]

BASE_VOCAB_TOKENS = [
    "<PAD>",
    "<UNK>",
    *CONTROL_TOKENS,
    *DIFFICULTY_TOKENS,
    *PRIMARY_MECHANIC_TOKENS,
    *Y_TOKENS,
    *WIDTH_TOKENS,
]


@dataclass(frozen=True)
class ObjectMapping:
    object_id: int
    token: str
    source: str


BLOCK_IDS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
}

PLATFORM_IDS = {
    40,
    62,
    63,
    64,
    65,
    66,
    68,
}

SPIKE_IDS = {
    8,
    9,
    18,
    19,
    20,
    21,
    39,
}

SAW_IDS = {
    88,
    89,
    98,
}

OBJECT_TOKEN_BY_ID: dict[int, str] = {
    **{object_id: "BLOCK" for object_id in BLOCK_IDS},
    **{object_id: "PLATFORM" for object_id in PLATFORM_IDS},
    **{object_id: "SPIKE" for object_id in SPIKE_IDS},
    **{object_id: "SAW" for object_id in SAW_IDS},
    36: "ORB_YELLOW",
    84: "ORB_BLUE",
    141: "ORB_PINK",
    1333: "ORB_BLACK",
    35: "PAD_YELLOW",
    67: "PAD_BLUE",
    12: "PORTAL_CUBE",
    13: "PORTAL_SHIP",
    47: "PORTAL_BALL",
    111: "PORTAL_UFO",
    660: "PORTAL_WAVE",
    11: "GRAVITY_UP",
    10: "GRAVITY_DOWN",
    200: "SPEED_SLOW",
    201: "SPEED_NORMAL",
    202: "SPEED_FAST",
    203: "SPEED_FAST",
    1334: "SPEED_FAST",
}


# Known non-v1 gameplay or editor/control objects. They are intentionally skipped
# instead of counted as unknowns when calculating tokenizer diagnostics.
IGNORED_OBJECT_IDS = {
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
    34,
    45,
    46,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    85,
    86,
    87,
    97,
    99,
    101,
    286,
    287,
    745,
    747,
    899,
    901,
    914,
    1006,
    1007,
    1049,
    1268,
    1329,
    1330,
    1331,
    1332,
    1346,
    1347,
    1520,
    1585,
    1595,
    1611,
    1612,
    1613,
    1615,
    1616,
    1811,
    1812,
    1813,
    1814,
    1815,
    1816,
    1817,
    1818,
    1819,
    1829,
    1859,
}


def difficulty_token(difficulty: str) -> str:
    normalized = difficulty.strip().upper().replace(" ", "_")
    if normalized.endswith("_DEMON"):
        normalized = normalized[: -len("_DEMON")]
    if normalized == "EASY_DEMON":
        normalized = "EASY"
    if normalized == "MEDIUM_DEMON":
        normalized = "MEDIUM"
    if normalized == "HARD_DEMON":
        normalized = "HARD"
    token = f"DIFF_{normalized or 'NA'}"
    if token not in DIFFICULTY_TOKENS:
        return "DIFF_NA"
    return token


def map_object_id(object_id: int) -> str | None:
    return OBJECT_TOKEN_BY_ID.get(object_id)


def is_ignored_object_id(object_id: int) -> bool:
    return object_id in IGNORED_OBJECT_IDS


def y_token(lane: int) -> str:
    return f"Y{max(0, min(Y_LANES - 1, lane))}"


def width_token(width: int) -> str:
    return f"WIDTH_{max(1, min(MAX_WIDTH, width))}"


def ordered_unique(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result
