# Geometry Dash Dataset Scraper

Production-grade async ingestion for raw Geometry Dash level data. The scraper discovers levels, downloads raw server responses, validates level-data integrity, records metadata, writes JSONL datasets, and resumes automatically after interruption.

The scraper stage preserves raw data only. Tokenizer v1 is included as the first downstream stage for mechanics-token preprocessing and a tiny proof-of-life baseline model.

## Layout

```text
data/
  raw/
    levels.jsonl
    comments.jsonl
    songs.jsonl

  processed/
    parsed_levels.jsonl
    gameplay_objects.jsonl

  tokenized/
    mechanics_tokens.jsonl
    vocab.json
    tokenizer_stats.jsonl
    tokenizer_analytics.json
    reconstruction_validation.jsonl
    reconstruction_summary.json

models/
  mechanics_v1/

  checkpoints/
    discovered_levels.jsonl
    downloaded_level_ids.txt
    failed_requests.jsonl
    rejected_levels.jsonl
    state.json

  logs/
    scraper.log
    metrics.jsonl
```

The scraper writes only `raw/`, `checkpoints/`, and `logs/`. Tokenizer commands write `processed/`, `tokenized/`, and `models/mechanics_v1/`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Small test run:

```powershell
python -m gd_scraper --target-count 25 --pages-per-source 3
```

Large production run:

```powershell
python -m gd_scraper --target-count 500000 --pages-per-source 5000 --concurrency 20
```

Infinite discovery for the selected sources:

```powershell
python -m gd_scraper --target-count 0
```

Restricted sources:

```powershell
python -m gd_scraper --sources featured,rated,extreme_demons
```

Optional raw comment pages for saved levels:

```powershell
python -m gd_scraper --target-count 100 --include-comments --comments-pages 1
```

## Tokenizer v1

First-100 proof run:

```powershell
python -m gd_scraper.tokenizer --limit 100 --overwrite --inspect 10
```

Tiny mechanics baseline:

```powershell
python -m gd_scraper.train_mechanics --max-records 100 --epochs 8
```

Full local tokenization and validation:

```powershell
python -m gd_scraper.tokenizer --limit 0 --overwrite --inspect 10
python -m gd_scraper.analytics
python -m gd_scraper.reconstruction --overwrite
```

The tokenizer spec is in `docs/tokenizer_v1_instruction_doc.md`. The dataset-scale instruction doc is in `docs/mechanics_foundation_dataset_v1_instruction_doc.md`.

## Defaults

- discovery rate: `2` requests/sec
- download rate: `0.33` requests/sec
- comment rate: `0.33` requests/sec
- concurrency: `20` workers
- retries: `3`
- timeout: `20` seconds

The client uses `POST`, form-encoded payloads, an empty `User-Agent`, and the GD common read secret for `getGJLevels21.php`, `downloadGJLevel22.php`, and `getGJComments21.php`.

## Raw Level Schema

Each line in `data/raw/levels.jsonl` is one independent JSON object:

```json
{
  "dataset_version": 1,
  "scraper_version": "1.0.0",
  "level_id": 123,
  "name": "Example",
  "author": "Player",
  "difficulty": "EXTREME_DEMON",
  "downloads": 123456,
  "likes": 5432,
  "song_id": 999,
  "song_type": "custom",
  "source": "featured",
  "source_page": 12,
  "object_count": 1234,
  "level_hash": "...",
  "fetched_at": 1770000000,
  "level_data": "...",
  "raw": "..."
}
```

Additional metadata is included under `metadata`, including `candidate_sequence` for deterministic recovery and downstream sorting.

## Validation

A level is rejected if:

- response is empty
- response is `-1`
- `level_data` is missing
- base64 decode fails
- zlib decompress fails
- decoded level has zero objects

Rejected levels are written to `data/checkpoints/rejected_levels.jsonl`:

```json
{"level_id":123,"reason":"zlib_decode_failed","timestamp":1770000000}
```

The scraper decodes only to validate integrity and count objects. It does not store decoded level strings or transform gameplay data.

## Resume

Resume is automatic:

- saved IDs are loaded from `data/raw/levels.jsonl`
- completed IDs are also loaded from `data/checkpoints/downloaded_level_ids.txt`
- discovered candidates are reloaded from `data/checkpoints/discovered_levels.jsonl`
- source page progress resumes from `data/checkpoints/state.json`
- rejected IDs are skipped from `data/checkpoints/rejected_levels.jsonl`
- network/API failures are logged to `data/checkpoints/failed_requests.jsonl`

Use `--retry-rejected` to ignore prior validation rejections for a run.

Press `Ctrl+C` once to stop discovery, let any active download finish writing, flush final metrics, and exit with code `130`. A second `Ctrl+C` cancels immediately.

## Metrics

Snapshots are appended to `data/logs/metrics.jsonl` and include:

- `levels_discovered`
- `levels_saved`
- `levels_rejected`
- `levels_failed`
- `duplicates_skipped`
- `avg_response_time`
- `requests_per_minute`
- `levels_per_minute`

`data/logs/scraper.log` receives the structured run log.

## Sources

Supported discovery sources:

```text
featured
rated
trending
popular
most_liked
demons
easy_demons
medium_demons
hard_demons
insane_demons
extreme_demons
```

## Design Notes

- Raw responses are preserved in JSONL and never overwritten.
- Checkpoints are append-only where practical, with source pagination stored atomically in `state.json`.
- Output records carry `dataset_version`, `scraper_version`, and `candidate_sequence`.
- Later processing should read from `data/raw/levels.jsonl` and write new artifacts under `processed/` or `tokenized/`.
