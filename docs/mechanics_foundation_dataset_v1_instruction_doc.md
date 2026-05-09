# Geometry Dash Mechanics Foundation Dataset v1

## Purpose

Build the first large-scale, stable, trainable mechanics dataset for the Geometry Dash AI generation system.

This phase focuses on scaling tokenization, validating token fidelity, stabilizing the vocabulary, analyzing token distributions, and preparing for serious conditioned model training. It does not include decoration generation, visual styling, audio conditioning, RL/playability scoring, or final architecture optimization.

The target artifact is a mechanically coherent symbolic gameplay dataset.

## Pipeline

```text
Raw Geometry Dash levels
-> decoder/parser
-> gameplay object filter
-> coordinate normalization
-> mechanics tokenization
-> analytics
-> reconstruction validation
-> mechanics dataset v1
```

## Scale Targets

- Initial target: `10,000` tokenized levels.
- Preferred target: `50,000+` tokenized levels.
- Local runs are bounded by available `data/raw/levels.jsonl` records.

## Accepted Tokens

Accepted object categories:

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
```

Structural and attribute tokens:

```text
START
END
STEP
ALIGN_UNKNOWN
DIFF_*
Y0-Y15
WIDTH_1-WIDTH_16
```

Excluded categories remain ignored: decoration, color triggers, pulse triggers, move triggers, alpha triggers, visual effects, glow objects, and editor-only objects.

## Grammar

The core event grammar is:

```text
OBJECT ATTRIBUTES STEP
```

Constraints:

- Every sequence starts with `START` and ends with `END`.
- Optional conditioning tokens appear after `START`.
- Gameplay objects always appear before the `STEP` that advances time.
- Spacing is never implicit.
- Objects sharing a timestep may stack naturally.
- Attributes follow their object immediately.
- Solids require `Y* WIDTH_*`.
- Non-solids require only `Y*`.

## Validation

Reject levels before training when any condition is true:

- gameplay object count `< 20`
- token count `< 50`
- missing `START` or `END`
- invalid grammar ordering
- excessive unknown objects
- zero STEP progression
- absurd sequence length
- decode inconsistency

Rejected levels preserve reasons in `data/tokenized/tokenizer_stats.jsonl`.

## Required Outputs

```text
data/processed/parsed_levels.jsonl
data/processed/gameplay_objects.jsonl
data/tokenized/mechanics_tokens.jsonl
data/tokenized/vocab.json
data/tokenized/tokenizer_stats.jsonl
data/tokenized/tokenizer_analytics.json
data/tokenized/reconstruction_validation.jsonl
data/tokenized/reconstruction_summary.json
```

## Analytics

Generate analytics after each major tokenizer run.

Dataset metrics:

```text
levels_processed
levels_accepted
levels_rejected
avg_token_length
max_token_length
avg_step_density
unknown_object_rate
```

Vocabulary metrics:

```text
top_tokens
rarest_tokens
token_entropy
object_frequency
portal_frequency
orb_frequency
```

`vocab.json` must include token IDs, reverse IDs, token frequency counts, unknown token count, unknown object count, and top unknown object IDs.

## Reconstruction

The reconstruction validator parses mechanics tokens into reconstructed gameplay objects and compares them against `gameplay_objects.jsonl`.

It validates preservation of:

- object ordering
- spacing
- portals
- orb placement
- stacking behavior
- gameplay structure

Minor raw level formatting differences are acceptable because reconstruction validates the symbolic gameplay representation, not exact GD level-string byte equivalence.

## Commands

Tokenize all locally available raw records:

```powershell
python -m gd_scraper.tokenizer --limit 0 --overwrite --inspect 10
```

Generate analytics explicitly:

```powershell
python -m gd_scraper.analytics
```

Validate reconstruction:

```powershell
python -m gd_scraper.reconstruction --overwrite
```

Train the tiny proof-of-life baseline:

```powershell
python -m gd_scraper.train_mechanics --max-records 0 --epochs 8
```
