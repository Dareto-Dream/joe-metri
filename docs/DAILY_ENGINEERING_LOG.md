# Daily Engineering Log

This log was reconstructed from the Git commit history and the current unstaged work in the repository.

## Date: May 8, 2026

Commit: `0c4a3fa` - Initial Commit

Today I worked on:
- Started the Geometry Dash dataset scraper project and set up the main Python package.
- Built the first version of the API client, parser, scraper, source definitions, storage helpers, and command-line entry point.

What changed in my project:
- Added the repo structure, README, requirements, data folders, scraper code, parser tests, and shutdown tests.

Challenges or bugs:
- The raw Geometry Dash server responses needed custom parsing and validation before they could safely be saved.
- The first version still needed better interruption handling for stopping a scrape safely.

Next steps:
- Improve Ctrl+C/SIGINT shutdown behavior so the scraper can stop without corrupting saved data.

## Date: May 8, 2026

Commit: `d33d5e1` - SIGINT fix

Today I worked on:
- Improved graceful shutdown behavior for the scraper and API client.
- Added tests around stopping the scraper while downloads are active.

What changed in my project:
- Updated the API, scraper, CLI, errors, and storage code.
- Added API shutdown tests and expanded scraper shutdown tests.

Challenges or bugs:
- The scraper had to stop new work while still allowing active downloads to finish cleanly.
- I needed to avoid partial writes when the program receives an interrupt.

Next steps:
- Start turning downloaded level data into useful gameplay tokens for training.

## Date: May 8, 2026

Commit: `66f0f1f` - training and tokenization

Today I worked on:
- Built the first tokenizer and tiny mechanics training pipeline.
- Added reconstruction and analytics tools to check whether tokenized levels can be rebuilt.

What changed in my project:
- Added tokenizer, object vocabulary, reconstruction validation, analytics, training script, instruction docs, and tests.

Challenges or bugs:
- Converting Geometry Dash object strings into compact tokens was tricky because the tokens still need to preserve gameplay order and layout meaning.
- Bad or unusable levels needed to be rejected instead of polluting the training data.

Next steps:
- Connect scraping, tokenization, and training into one continuous pipeline.

## Date: May 8, 2026

Commit: `d0d3db6` - orchestrator

Today I worked on:
- Added a continuous orchestrator to run scraping, tokenization, training, evaluation, and monitoring together.
- Made the pipeline treat JSONL files as queues so progress can resume across runs.

What changed in my project:
- Added `gd_scraper/orchestrator.py`, updated storage/tokenizer/training code, documented orchestrator commands, and added orchestrator tests.

Challenges or bugs:
- The hard part was coordinating async scraping with tokenizer and trainer work without losing state.
- Vocabulary IDs needed to stay stable so old checkpoints would still load.

Next steps:
- Run longer training and add better monitoring for model progress.

## Date: May 11, 2026

Commit: `9255b18` - trained model

Today I worked on:
- Ran a long mechanics model training session and saved checkpoints.
- Added screensaver-style telemetry pages for watching orchestrator progress.

What changed in my project:
- Added many mechanics model checkpoint files, screensaver code, screensaver tests, and README updates.

Challenges or bugs:
- The trained checkpoint output became very large, so tracking and monitoring model progress became more important.
- I needed a way to see the pipeline status without reading raw log files constantly.

Next steps:
- Build a usable runtime that can generate levels from the model and export them.

## Date: May 13, 2026

Commit: `df68c01` - funny push

Today I worked on:
- Built the first runtime and web interface for generating Geometry Dash layouts.
- Added audio analysis, conditioning, sampling, layout reconstruction, validation, exporting, and save-file helpers.

What changed in my project:
- Added `app.py`, the `runtime/` package, the `web/` frontend, runtime tests, requirements updates, and more model checkpoints.

Challenges or bugs:
- Exporting Geometry Dash data is sensitive because save codecs and local save data need to be preserved correctly.
- Generated layouts needed validation so broken level strings would not be exported blindly.

Next steps:
- Improve the export workflow, especially GMD/GDShare output and example generated levels.

## Date: May 15, 2026

Commit: `1386317` - exporting

Today I worked on:
- Improved the primary export path for generated levels.
- Updated the app and docs around exporting GMD files, level strings, JSON, and metrics.

What changed in my project:
- Updated `app.py`, `runtime/exporter.py`, runtime tests, README export instructions, and added more training checkpoints.

Challenges or bugs:
- The export files needed to be valid enough for external Geometry Dash tools, not just readable by my own code.
- The CLI export workflow needed clearer output and validation.

Next steps:
- Add example exports and improve the web controls around generated layouts.

## Date: May 15, 2026

Commit: `76f5817` - more exporting and examples

Today I worked on:
- Added an example generated layout and improved export behavior in the runtime and web app.
- Continued tuning how reconstructed layouts become Geometry Dash object strings.

What changed in my project:
- Added `examples/Femtanyl layout 2.gmd` and updated app, exporter, reconstructor, web UI, and runtime tests.

Challenges or bugs:
- The coordinate mapping and export format needed more checking so objects would appear where expected after import.

Next steps:
- Improve generation quality, token planning, and parser/tokenizer edge cases.

## Date: May 15, 2026

Commit: `a960a13` - improvements

Today I worked on:
- Improved scraper, parser, tokenizer, and runtime generation quality.
- Added planning and quality scoring so generated token sequences can be evaluated before export.

What changed in my project:
- Added `gd_scraper/quality.py` and `runtime/planner.py`.
- Updated scraper/API/parser/storage/tokenizer/runtime files and expanded parser, tokenizer, and runtime tests.

Challenges or bugs:
- Some generated token sequences could collapse or repeat too much, so they needed quality checks.
- Parser and tokenizer edge cases needed stronger tests before using more scraped data.

Next steps:
- Add flow/sync arranging so generated layouts are more playable and better matched to audio.

## Date: May 15, 2026

Commit: `417df71` - COMMIT

Today I worked on:
- Added flow syncing for generated gameplay and improved runtime planning output.
- Updated the app and web UI to work with the new runtime flow information.

What changed in my project:
- Added `runtime/flow.py`, updated generator/planner/conditioning/app/web files, added process logs, and expanded runtime tests.

Challenges or bugs:
- Generated samples needed to be rearranged into playable layouts while still keeping good sync and flow scores.
- It was important to stream useful planning events without breaking the existing runtime API.

Next steps:
- Fix remaining export coordinate issues and add support for a layout-only training corpus.

## Date: May 15, 2026

Current unstaged work

Today I worked on:
- Added support for scraping layout-only levels and fixed coordinate handling in runtime reconstruction.
- Added tests to make sure tokenizer output roundtrips back to expected Geometry Dash object coordinates.

What changed in my project:
- Added a `layout` discovery source and source name filtering so layout scraping does not collect unrelated search results.
- Updated the orchestrator default sources, README layout-only commands, runtime Y-coordinate mapping, reconstruction logic, and tokenizer/runtime tests.

Challenges or bugs:
- Searching for "layout" can return levels that are not actually layout levels, so the scraper needed a name filter.
- Floor blocks were being dropped or exported at the wrong Y coordinate, so the runtime needed a clearer coordinate origin and roundtrip tests.

Next steps:
- Run the focused scraper/tokenizer/runtime tests, then collect a layout-only corpus and train a separate layout model.
