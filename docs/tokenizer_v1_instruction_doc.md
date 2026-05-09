# Tokenizer v1 Instruction Doc

## Goal

Build the first mechanics-token pipeline for Geometry Dash level data:

```text
data/raw/levels.jsonl
-> decode level_data
-> parse GD objects
-> filter gameplay objects
-> normalize x/y
-> emit mechanics tokens
-> train a tiny proof-of-life baseline
```

This is not the final generation model. It is a tokenizer and sanity-check baseline whose job is to prove that the data can produce legal, GD-like mechanics-token sequences.

## Output Paths

```text
data/processed/parsed_levels.jsonl
data/processed/gameplay_objects.jsonl
data/tokenized/mechanics_tokens.jsonl
data/tokenized/vocab.json
data/tokenized/tokenizer_stats.jsonl
data/tokenized/tokenizer_analytics.json
data/tokenized/reconstruction_validation.jsonl
data/tokenized/reconstruction_summary.json
models/mechanics_v1/
```

## Token Scope

Tokenizer v1 keeps:

```text
BLOCK
PLATFORM
SPIKE
SAW
ORB_YELLOW
ORB_BLUE
ORB_PINK
ORB_BLACK
PAD_YELLOW
PAD_BLUE
PORTAL_CUBE
PORTAL_SHIP
PORTAL_BALL
PORTAL_UFO
PORTAL_WAVE
GRAVITY_UP
GRAVITY_DOWN
SPEED_SLOW
SPEED_NORMAL
SPEED_FAST
STEP
START
END
ALIGN_UNKNOWN
DIFF_*
Y0-Y15
WIDTH_1-WIDTH_16
```

Tokenizer v1 skips decoration, color/move/alpha/rotation trigger detail, scale detail, dual mode, custom object groups, mirror/size portals, teleport portals, robot/spider portals, and other non-v1 controls.

## Level Object Parsing

Raw level records are read from `data/raw/levels.jsonl`. The `level_data` field is base64-url decoded and gzip/zlib decompressed with the existing scraper decoder.

Decoded level data is split on `;`. The first section is the level header. Later sections are GD object records encoded as comma-separated key/value pairs. Tokenizer v1 reads at least:

```text
1 = object id
2 = x
3 = y
6 = rotation
32 = scale
```

Rotation and scale are parsed for diagnostics only. Tokenizer v1 does not emit rotation or scale tokens.

## Gameplay Mapping

The v1 gameplay mapping table lives in `gd_scraper/gd_objects.py`.

Core mappings:

```text
35 -> PAD_YELLOW
36 -> ORB_YELLOW
67 -> PAD_BLUE
84 -> ORB_BLUE
141 -> ORB_PINK
1333 -> ORB_BLACK

12 -> PORTAL_CUBE
13 -> PORTAL_SHIP
47 -> PORTAL_BALL
111 -> PORTAL_UFO
660 -> PORTAL_WAVE

10 -> GRAVITY_DOWN
11 -> GRAVITY_UP

200 -> SPEED_SLOW
201 -> SPEED_NORMAL
202,203,1334 -> SPEED_FAST
```

Selected early solid block, platform, spike, and saw object IDs are also mapped. Unknown IDs are counted in tokenizer stats. Known skipped control/decor IDs are counted as ignored, not unknown.

## Normalization

Coordinates are normalized per level after gameplay filtering:

```text
x_origin = minimum mapped gameplay x
y_origin = 5th-percentile mapped gameplay y, used as a robust local floor
x_step = round((x - x_origin) / 30)
y_lane = clamp(round((y - y_origin) / 30), 0, 15)
```

`x_step_resolution` is stored on each tokenized record and defaults to `30`.

`parsed_levels.jsonl` stores compact parsed summaries by default: decoded length, header, coordinate bounds, raw object count, and object ID counts. Use `--write-parsed-objects` only when a local debugging run needs every parsed raw object serialized.

## Event Sequencing

Tokenizer v1 emits event-based mechanics sequences:

```text
START DIFF_EXTREME ALIGN_UNKNOWN
BLOCK Y1 WIDTH_4
STEP
SPIKE Y2
ORB_YELLOW Y5
STEP
PORTAL_SHIP Y3
STEP
END
```

Rules:

- Sort gameplay objects by `x_step`.
- Objects sharing one `x_step` are emitted together before the sequence advances to the next timestep.
- Every time advance is represented with one or more `STEP` tokens.
- Spacing is never implicit.
- `BLOCK` and `PLATFORM` runs on the same lane are merged into one event with `WIDTH_1` through `WIDTH_16`.
- Non-solid objects emit only the object token and a `Y*` lane token.
- Every tokenized record preserves the raw `level_id`.

## Tokenized Record Schema

```json
{
  "dataset_version": 1,
  "tokenizer_version": "0.1.0",
  "level_id": 123,
  "difficulty": "EXTREME",
  "song_id": 999,
  "source": "featured",
  "object_count_raw": 1234,
  "object_count_gameplay": 231,
  "x_step_resolution": 30,
  "y_lanes": 16,
  "tokens": ["START", "DIFF_EXTREME", "ALIGN_UNKNOWN", "BLOCK", "Y1", "WIDTH_4", "STEP", "SPIKE", "Y2", "END"]
}
```

## Validation

Reject a level before training if any condition is true:

- Decoded level object count is zero.
- Fewer than 20 gameplay objects.
- Fewer than 50 tokens.
- Missing `START` or `END`.
- Unknown object ratio is above the configured threshold.
- Token length is above the configured maximum.
- All mapped gameplay objects collapse into one timestep.

Rejected levels are not written to `mechanics_tokens.jsonl`, but their reason is written to `tokenizer_stats.jsonl`.

## Vocabulary And Analytics

`vocab.json` includes token-to-ID mappings, ID-to-token mappings, per-token frequency counts, unknown token count, unknown object count, and top unknown object IDs.

`tokenizer_analytics.json` is generated after tokenizer runs unless `--skip-analytics` is passed. It includes:

- `levels_processed`
- `levels_accepted`
- `levels_rejected`
- `avg_token_length`
- `max_token_length`
- `avg_step_density`
- `unknown_object_rate`
- `top_tokens`
- `rarest_tokens`
- `token_entropy`
- `object_frequency`
- `portal_frequency`
- `orb_frequency`

## Reconstruction Validation

`gd_scraper.reconstruction` parses token sequences back into symbolic gameplay objects and compares them with `gameplay_objects.jsonl`.

It validates grammar, object ordering, explicit spacing, portals, orb placement, stacking behavior, and object widths. The validator writes per-level results to `data/tokenized/reconstruction_validation.jsonl` and a summary to `data/tokenized/reconstruction_summary.json`.

## Commands

Tokenizer proof run on the first 100 raw levels:

```powershell
python -m gd_scraper.tokenizer --limit 100 --overwrite --inspect 10
```

Full local run:

```powershell
python -m gd_scraper.tokenizer --limit 0 --overwrite --inspect 10
```

Analytics:

```powershell
python -m gd_scraper.analytics
```

Reconstruction validation:

```powershell
python -m gd_scraper.reconstruction --overwrite
```

Tiny baseline proof-of-life training:

```powershell
python -m gd_scraper.train_mechanics --max-records 100 --epochs 8
```

The tiny baseline saves:

```text
models/mechanics_v1/model.json
models/mechanics_v1/training_stats.jsonl
models/mechanics_v1/sample_generation.json
```

Success criteria:

- Training loss decreases.
- Generated samples use legal token grammar.
- Generated samples use `STEP`.
- Samples do not repeat one token forever.
- Samples resemble GD mechanics sequences well enough to justify scaling tokenizer runs.
