from __future__ import annotations

import argparse
import asyncio
import contextlib
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import logging
import math
from pathlib import Path
import random
import signal
import time
from typing import Any, Iterable

import aiofiles
import orjson

from .api import GDClient, GDClientConfig
from .gd_objects import BASE_VOCAB_TOKENS, TOKENIZER_VERSION
from .models import DATASET_VERSION, epoch_seconds
from .quality import evaluate_token_sequence_quality
from .reconstruction import validate_token_grammar
from .scraper import GeometryDashScraper
from .sources import DEFAULT_SOURCES, resolve_sources
from .storage import (
    CheckpointStore,
    append_id_async,
    coerce_source_names,
    count_lines,
    dumps_jsonl,
    dumps_pretty,
    load_ids_from_jsonl,
    load_ids_from_text,
    loads_json,
)
from .tokenizer import TokenizerConfig, build_vocab, tokenized_record_from_level
from .train_mechanics import (
    TinyNgramModel,
    build_examples,
    evaluate_loss,
    generate_sample,
    model_from_json,
)


LOGGER = logging.getLogger(__name__)


def default_source_names() -> list[str]:
    return [name for name, source in DEFAULT_SOURCES.items() if not source.name_contains]


@dataclass
class ResourceAllocation:
    cpu_slots: int = 1
    gpu_slots: int = 0
    scraper_concurrency: int = 8
    tokenizer_workers: int = 2
    tokenizer_records_per_cycle: int = 100
    training_examples_per_cycle: int = 2_000
    training_interval_seconds: float = 30.0


@dataclass(frozen=True)
class OrchestratorConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models") / "mechanics_v1"
    mode: str = "assisted"
    sources: list[str] = field(default_factory=default_source_names)
    scraper_enabled: bool = True
    tokenizer_enabled: bool = True
    trainer_enabled: bool = True
    evaluator_enabled: bool = True
    scraper_pages_per_cycle: int = 25
    scraper_target_count: int = 0
    scraper_restart_seconds: float = 5.0
    timeout_seconds: float = 20.0
    retries: int = 3
    backoff_seconds: float = 1.0
    search_rate: float = 2.0
    download_rate: float = 0.33
    comment_rate: float = 0.33
    include_comments: bool = False
    download_audio: bool = False
    comments_pages: int = 0
    tokenizer_poll_seconds: float = 2.0
    metrics_interval_seconds: float = 10.0
    evaluation_interval_seconds: float = 60.0
    evaluation_interval_steps: int = 2_000
    max_runtime_seconds: float = 0.0
    min_gameplay_objects: int = 20
    min_tokens: int = 50
    max_unknown_ratio: float = 0.995
    max_token_length: int = 12_000
    train_min_records: int = 5
    train_window_records: int = 250
    train_max_examples: int = 30_000
    context_size: int = 4
    embedding_dim: int = 24
    learning_rate: float = 0.05
    seed: int = 7
    sample_prefix: str = "START DIFF_HARD ALIGN_UNKNOWN"
    sample_tokens: int = 120
    min_sample_tokens: int = 40
    temperature: float = 0.9
    checkpoint_interval_steps: int = 10_000
    raw_backlog_warning: int = 100
    token_starvation_threshold: int = 3
    unknown_object_warning_rate: float = 0.05
    min_token_entropy: float = 2.0
    collapse_diversity_threshold: float = 0.18
    collapse_max_repetition: int = 16
    min_sample_quality_score: float = 80.0
    max_sample_path_obstructions: int = 0
    max_sample_control_spam: int = 2
    collapse_pause_seconds: float = 120.0
    plateau_patience: int = 3
    policy_cooldown_seconds: float = 300.0
    shutdown_timeout_seconds: float = 30.0


class OrchestratorSignalHandler:
    def __init__(self, orchestrator: "ContinuousPipelineOrchestrator") -> None:
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.main_task: asyncio.Task[Any] | None = None
        self.exit_code = 0
        self._signals = [signal.SIGINT]
        if hasattr(signal, "SIGTERM"):
            self._signals.append(signal.SIGTERM)
        self._previous_handlers: dict[signal.Signals, Any] = {}
        self._loop_handlers: set[signal.Signals] = set()
        self._requests = 0

    def __enter__(self) -> "OrchestratorSignalHandler":
        self.loop = asyncio.get_running_loop()
        self.main_task = asyncio.current_task()
        for sig in self._signals:
            self._previous_handlers[sig] = signal.getsignal(sig)
            try:
                self.loop.add_signal_handler(sig, self._handle_signal, sig)
                self._loop_handlers.add(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                signal.signal(sig, self._sync_handle_signal)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for sig in self._signals:
            if sig in self._loop_handlers and self.loop is not None:
                self.loop.remove_signal_handler(sig)
            previous = self._previous_handlers.get(sig)
            if previous is not None:
                signal.signal(sig, previous)

    def _sync_handle_signal(self, signum: int, _frame: object | None) -> None:
        self._handle_signal(signal.Signals(signum))

    def _handle_signal(self, sig: signal.Signals) -> None:
        self._requests += 1
        self.exit_code = 128 + sig.value
        if self._requests == 1:
            self.logger.warning("received %s; stopping orchestrator gracefully", sig.name)
            self.orchestrator.request_shutdown(sig.name)
            return

        self.logger.warning("received %s again; forcing orchestrator shutdown", sig.name)
        self.orchestrator.request_shutdown(sig.name, force=True)
        if self.main_task is not None:
            self.main_task.cancel()


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return loads_json(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return default


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(dumps_pretty(value))
    tmp_path.replace(path)


async def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "ab") as handle:
        await handle.write(dumps_jsonl(value))
        await handle.flush()


def read_jsonl_at_offset(path: Path, offset: int, limit: int) -> tuple[list[tuple[dict[str, Any], int]], int]:
    if not path.exists():
        return [], 0
    size = path.stat().st_size
    if offset > size:
        offset = 0

    records: list[tuple[dict[str, Any], int]] = []
    current_offset = offset
    with path.open("rb") as handle:
        handle.seek(offset)
        while limit <= 0 or len(records) < limit:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            if not raw_line.endswith(b"\n"):
                current_offset = line_start
                break
            current_offset = handle.tell()
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = loads_json(raw_line)
            except orjson.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append((record, current_offset))
    return records, current_offset


def load_latest_jsonl(path: Path) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    if not path.exists():
        return latest
    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                value = loads_json(raw_line)
            except orjson.JSONDecodeError:
                continue
            if isinstance(value, dict):
                latest = value
    return latest


def load_recent_token_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: deque[dict[str, Any]] = deque(maxlen=max(limit, 1))
    if not path.exists():
        return []
    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = loads_json(raw_line)
            except orjson.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("tokens"), list):
                records.append(record)
    return list(records)


def load_tokenizer_processed_ids(store: CheckpointStore) -> set[int]:
    ids = load_ids_from_text(store.orchestrator_raw_processed_ids_path)
    ids.update(load_ids_from_jsonl(store.tokenizer_stats_path, "level_id"))
    return ids


def merge_vocab_append_only(
    existing_vocab: dict[str, Any] | None,
    token_records: Iterable[dict[str, Any]],
    stats_records: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    existing_vocab = existing_vocab or build_vocab(token_counts=Counter())
    id_to_token = [str(token) for token in existing_vocab.get("id_to_token", [])]
    if not id_to_token:
        token_to_id = {
            str(token): int(token_id)
            for token, token_id in dict(existing_vocab.get("token_to_id", {})).items()
        }
        id_to_token = [token for token, _token_id in sorted(token_to_id.items(), key=lambda item: item[1])]

    tokens: list[str] = []
    seen_tokens: set[str] = set()
    for token in [*id_to_token, *BASE_VOCAB_TOKENS]:
        token = str(token)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        tokens.append(token)

    token_to_id = {token: index for index, token in enumerate(tokens)}
    token_counts = {
        token: int(count)
        for token, count in dict(existing_vocab.get("token_counts", {})).items()
        if str(token) in token_to_id
    }
    token_counts = {str(token): count for token, count in token_counts.items()}
    added_tokens = False
    for record in token_records:
        for raw_token in record.get("tokens", []) or []:
            token = str(raw_token)
            if token not in token_to_id:
                token_to_id[token] = len(tokens)
                tokens.append(token)
                added_tokens = True
            token_counts[token] = token_counts.get(token, 0) + 1

    for token in tokens:
        token_counts.setdefault(token, 0)

    unknown_object_ids: Counter[int] = Counter()
    for item in existing_vocab.get("top_unknown_object_ids", []) or []:
        try:
            unknown_object_ids[int(item["object_id"])] += int(item["count"])
        except (KeyError, TypeError, ValueError):
            continue

    unknown_object_count = int(existing_vocab.get("unknown_object_count", 0) or 0)
    for stats in stats_records:
        unknown_object_count += int(stats.get("unknown_object_count", 0) or 0)
        for item in stats.get("top_unknown_object_ids", []) or []:
            try:
                unknown_object_ids[int(item["object_id"])] += int(item["count"])
            except (KeyError, TypeError, ValueError):
                continue

    vocab_version = int(existing_vocab.get("vocab_version", 1) or 1)
    if added_tokens:
        vocab_version += 1

    return {
        "dataset_version": DATASET_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
        "vocab_version": vocab_version,
        "frequency_revision": int(existing_vocab.get("frequency_revision", 0) or 0) + 1,
        "updated_at": epoch_seconds(),
        "vocab_size": len(tokens),
        "tokens": tokens,
        "token_to_id": {token: index for index, token in enumerate(tokens)},
        "id_to_token": tokens,
        "token_counts": {token: int(token_counts.get(token, 0)) for token in tokens},
        "unknown_token_count": int(token_counts.get("<UNK>", 0)),
        "unknown_object_count": int(unknown_object_count),
        "top_unknown_object_ids": [
            {"object_id": object_id, "count": count}
            for object_id, count in unknown_object_ids.most_common(50)
        ],
    }


def token_entropy(records: Iterable[dict[str, Any]]) -> float:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(str(token) for token in record.get("tokens", []) or [])
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def max_repetition(tokens: list[str]) -> int:
    longest = 0
    current = 0
    previous = ""
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous = token
            current = 1
        longest = max(longest, current)
    return longest


class ContinuousPipelineOrchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.store = CheckpointStore(config.data_dir)
        self.allocation = ResourceAllocation()
        self.allocation.scraper_concurrency = 8
        self.stop_event = asyncio.Event()
        self.started_at = time.monotonic()
        self.started_epoch = time.time()
        self.force_shutdown_requested = False
        self.state: dict[str, Any] = {}
        self.processed_raw_ids: set[int] = set()
        self.model: TinyNgramModel | None = None
        self.model_lock = asyncio.Lock()
        self.vocab_lock = asyncio.Lock()
        self.active_scraper: GeometryDashScraper | None = None
        self.policy_last_emitted: dict[str, float] = {}
        self.runtime_stats: dict[str, int] = {
            "levels_tokenized": 0,
            "tokens_tokenized": 0,
            "samples_generated": 0,
            "sample_tokens_generated": 0,
        }
        self.live_model_path = config.model_dir / "model_live.json"
        self.live_training_stats_path = config.model_dir / "training_stats_live.jsonl"
        self.live_evaluation_stats_path = config.model_dir / "evaluation_stats_live.jsonl"
        self.live_sample_path = config.model_dir / "sample_generation_live.json"
        self.live_checkpoint_dir = config.model_dir / "checkpoints"

    async def run(self) -> None:
        self._initialize()
        await self._event("ORCHESTRATOR_STARTED", config=self._config_json(), allocation=asdict(self.allocation))
        if self.config.max_runtime_seconds > 0:
            asyncio.create_task(self._stop_after_runtime())

        tasks: list[asyncio.Task[None]] = []
        if self.config.scraper_enabled:
            tasks.append(asyncio.create_task(self._scraper_loop(), name="scraper-loop"))
        if self.config.tokenizer_enabled:
            tasks.append(asyncio.create_task(self._tokenizer_loop(), name="tokenizer-loop"))
        if self.config.trainer_enabled:
            tasks.append(asyncio.create_task(self._trainer_loop(), name="trainer-loop"))
        if self.config.evaluator_enabled:
            tasks.append(asyncio.create_task(self._evaluation_loop(), name="evaluation-loop"))
        tasks.append(asyncio.create_task(self._monitor_loop(), name="monitor-loop"))

        cancelled = False
        try:
            await self.stop_event.wait()
        except asyncio.CancelledError:
            cancelled = True
            self.request_shutdown("cancelled", force=True)
            raise
        finally:
            await self._shutdown_tasks(tasks, force=cancelled or self.force_shutdown_requested)
            await self._save_state()
            await self._event("ORCHESTRATOR_STOPPED", forced=self.force_shutdown_requested)

    async def run_once(self) -> dict[str, Any]:
        self._initialize()
        if self.config.tokenizer_enabled:
            await self.tokenize_cycle()
        if self.config.trainer_enabled:
            await self.training_cycle()
        if self.config.evaluator_enabled:
            await self.evaluation_cycle(force=True)
        return await self.monitor_cycle()

    def request_shutdown(self, reason: str = "requested", *, force: bool = False) -> None:
        LOGGER.warning("orchestrator shutdown requested: %s", reason)
        if force:
            self.force_shutdown_requested = True
        if self.active_scraper is not None:
            self.active_scraper.request_shutdown(reason, force=force)
        self.stop_event.set()

    def _initialize(self) -> None:
        self.store.ensure()
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        self.live_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state = read_json_file(self.store.orchestrator_state_path, {})
        self.state.setdefault("raw_offset", 0)
        self.state.setdefault("trainer_steps", 0)
        self.state.setdefault("trainer_token_records_seen", 0)
        self.state.setdefault("training_paused_until", 0.0)
        self.state.setdefault("plateau_count", 0)
        self.processed_raw_ids = load_tokenizer_processed_ids(self.store)
        if not self.store.vocab_path.exists():
            write_json_atomic(self.store.vocab_path, build_vocab(token_counts=Counter()))

    async def _stop_after_runtime(self) -> None:
        await asyncio.sleep(self.config.max_runtime_seconds)
        self.request_shutdown("max_runtime_seconds")

    async def _shutdown_tasks(self, tasks: list[asyncio.Task[None]], *, force: bool) -> None:
        if self.active_scraper is not None:
            self.active_scraper.request_shutdown("orchestrator_stopping", force=force)
        if not force:
            pending = [task for task in tasks if not task.done()]
            if pending:
                done, still_pending = await asyncio.wait(
                    pending,
                    timeout=max(self.config.shutdown_timeout_seconds, 0.0),
                )
                for task in done:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                if not still_pending:
                    return
                await self._event(
                    "ORCHESTRATOR_FORCE_SHUTDOWN",
                    pending_tasks=[task.get_name() for task in still_pending],
                )
                if self.active_scraper is not None:
                    self.active_scraper.request_shutdown("orchestrator_shutdown_timeout", force=True)
                pending = list(still_pending)
            else:
                return
        else:
            pending = [task for task in tasks if not task.done()]

        self.force_shutdown_requested = True
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def _scraper_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self._event(
                    "SCRAPER_STARTED",
                    sources=self.config.sources,
                    concurrency=self.allocation.scraper_concurrency,
                    pages_per_cycle=self.config.scraper_pages_per_cycle,
                )
                client_config = GDClientConfig(
                    timeout_seconds=self.config.timeout_seconds,
                    retries=self.config.retries,
                    backoff_seconds=self.config.backoff_seconds,
                    search_rate_per_second=self.config.search_rate,
                    download_rate_per_second=self.config.download_rate,
                    comment_rate_per_second=self.config.comment_rate,
                )
                async with GDClient(client_config) as client:
                    scraper = GeometryDashScraper(
                        client=client,
                        data_dir=self.config.data_dir,
                        sources=resolve_sources(self.config.sources),
                        pages_per_source=max(self.config.scraper_pages_per_cycle, 1),
                        target_count=self.config.scraper_target_count,
                        concurrency=max(self.allocation.scraper_concurrency, 1),
                        include_comments=self.config.include_comments,
                        download_audio=self.config.download_audio,
                        comments_pages=self.config.comments_pages,
                    )
                    self.active_scraper = scraper
                    stats = await scraper.run()
                await self._event("SCRAPER_STOPPED", stats=stats.to_json())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await self._event("SCRAPER_FAILED", error=str(exc))
            finally:
                self.active_scraper = None
            await self._sleep(self.config.scraper_restart_seconds)

    async def _tokenizer_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self.tokenize_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await self._event("TOKENIZER_FAILED", error=str(exc))
            await self._sleep(self.config.tokenizer_poll_seconds)

    async def tokenize_cycle(self) -> dict[str, int]:
        config = TokenizerConfig(
            min_gameplay_objects=self.config.min_gameplay_objects,
            min_tokens=self.config.min_tokens,
            max_unknown_ratio=self.config.max_unknown_ratio,
            max_token_length=self.config.max_token_length,
        )
        batch_limit = max(1, self.allocation.tokenizer_records_per_cycle * max(self.allocation.tokenizer_workers, 1))
        records, new_offset = read_jsonl_at_offset(
            self.store.levels_path,
            int(self.state.get("raw_offset", 0) or 0),
            batch_limit,
        )
        accepted = 0
        rejected = 0
        token_records: list[dict[str, Any]] = []
        stats_records: list[dict[str, Any]] = []

        parsed_path = self.store.parsed_levels_path
        gameplay_path = self.store.gameplay_objects_path
        token_path = self.store.mechanics_tokens_path
        stats_path = self.store.tokenizer_stats_path
        for path in (parsed_path, gameplay_path, token_path, stats_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        async with (
            aiofiles.open(parsed_path, "ab") as parsed_handle,
            aiofiles.open(gameplay_path, "ab") as gameplay_handle,
            aiofiles.open(token_path, "ab") as token_handle,
            aiofiles.open(stats_path, "ab") as stats_handle,
        ):
            for raw_level, offset in records:
                level_id = self._level_id(raw_level)
                if level_id <= 0:
                    self.state["raw_offset"] = offset
                    continue
                if level_id in self.processed_raw_ids:
                    self.state["raw_offset"] = offset
                    continue

                artifacts = tokenized_record_from_level(raw_level, config)
                if artifacts.parsed_record is not None:
                    await parsed_handle.write(dumps_jsonl(artifacts.parsed_record))
                if artifacts.gameplay_record is not None:
                    await gameplay_handle.write(dumps_jsonl(artifacts.gameplay_record))
                if artifacts.token_record is not None:
                    await token_handle.write(dumps_jsonl(artifacts.token_record))
                    token_records.append(artifacts.token_record)
                    accepted += 1
                    self.runtime_stats["tokens_tokenized"] += len(artifacts.token_record["tokens"])
                else:
                    rejected += 1
                await stats_handle.write(dumps_jsonl(artifacts.stats_record))
                stats_records.append(artifacts.stats_record)
                self.processed_raw_ids.add(level_id)
                await append_id_async(self.store.orchestrator_raw_processed_ids_path, level_id)
                self.state["raw_offset"] = offset

            await parsed_handle.flush()
            await gameplay_handle.flush()
            await token_handle.flush()
            await stats_handle.flush()

        if not records:
            self.state["raw_offset"] = new_offset
        if token_records or stats_records:
            await self._merge_vocab(token_records, stats_records)
            self.runtime_stats["levels_tokenized"] += accepted
            await self._save_state()
            await self._event(
                "TOKENIZER_BATCH",
                levels_seen=accepted + rejected,
                levels_accepted=accepted,
                levels_rejected=rejected,
            )
        return {"seen": accepted + rejected, "accepted": accepted, "rejected": rejected}

    async def _merge_vocab(
        self,
        token_records: list[dict[str, Any]],
        stats_records: list[dict[str, Any]],
    ) -> None:
        async with self.vocab_lock:
            existing_vocab = read_json_file(self.store.vocab_path, None)
            merged = merge_vocab_append_only(existing_vocab, token_records, stats_records)
            write_json_atomic(self.store.vocab_path, merged)

    async def _trainer_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self.training_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await self._event("TRAINING_FAILED", error=str(exc))
            await self._sleep(self.allocation.training_interval_seconds)

    async def training_cycle(self) -> dict[str, Any] | None:
        if time.time() < float(self.state.get("training_paused_until", 0.0) or 0.0):
            await self._event("TRAINING_PAUSED", paused_until=self.state.get("training_paused_until"))
            return None

        records = load_recent_token_records(self.store.mechanics_tokens_path, self.config.train_window_records)
        if len(records) < self.config.train_min_records:
            await self._event(
                "DATASET_STARVATION",
                tokenized_records=len(records),
                required_records=self.config.train_min_records,
            )
            return None

        vocab = read_json_file(self.store.vocab_path, None)
        if not isinstance(vocab, dict):
            await self._event("TRAINING_WAITING_FOR_VOCAB")
            return None
        token_to_id = {str(key): int(value) for key, value in vocab["token_to_id"].items()}
        id_to_token = [str(token) for token in vocab["id_to_token"]]

        rng = random.Random(self.config.seed + int(self.state.get("trainer_steps", 0) or 0))
        shuffled_records = records[:]
        rng.shuffle(shuffled_records)
        split_index = max(1, int(len(shuffled_records) * 0.8))
        train_records = shuffled_records[:split_index]
        eval_records = shuffled_records[split_index:] or shuffled_records[:]
        train_examples = build_examples(
            train_records,
            token_to_id,
            self.config.context_size,
            self.config.train_max_examples,
        )
        eval_examples = build_examples(
            eval_records,
            token_to_id,
            self.config.context_size,
            max(1_000, self.config.train_max_examples // 5),
        )
        if not train_examples:
            await self._event("TRAINING_QUEUE_EMPTY")
            return None

        rng.shuffle(train_examples)
        train_examples = train_examples[: max(self.allocation.training_examples_per_cycle, 1)]

        async with self.model_lock:
            model = self._load_or_init_model(len(id_to_token))
            total_loss = 0.0
            for context, target_id in train_examples:
                total_loss += model.train_one(context, target_id, self.config.learning_rate)
            train_loss = total_loss / max(len(train_examples), 1)
            eval_loss = evaluate_loss(model, eval_examples)
            self.state["trainer_steps"] = int(self.state.get("trainer_steps", 0) or 0) + len(train_examples)
            self.state["trainer_token_records_seen"] = count_lines(self.store.mechanics_tokens_path)
            metric = {
                "dataset_version": DATASET_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
                "model_type": "tiny_embedding_ngram",
                "timestamp": epoch_seconds(),
                "orchestrator_mode": self.config.mode,
                "step": self.state["trainer_steps"],
                "records": len(records),
                "examples": len(train_examples),
                "train_loss": round(train_loss, 6),
                "eval_loss": round(eval_loss, 6),
                "vocab_size": len(id_to_token),
            }
            await append_jsonl(self.live_training_stats_path, metric)
            self._write_live_model(model, vocab)
            if self._should_checkpoint():
                self._write_live_model(
                    model,
                    vocab,
                    path=self.live_checkpoint_dir / f"checkpoint_step_{self.state['trainer_steps']}.json",
                )

        await self._detect_training_plateau(train_loss, eval_loss)
        await self._save_state()
        await self._event("TRAINING_STEP", **metric)
        if self.config.evaluator_enabled and self._should_evaluate_for_steps():
            await self.evaluation_cycle(force=True)
        return metric

    def _load_or_init_model(self, vocab_size: int) -> TinyNgramModel:
        if self.model is not None:
            self.model.expand_vocab(vocab_size, seed=self.config.seed)
            return self.model

        loaded = read_json_file(self.live_model_path, None)
        if isinstance(loaded, dict):
            try:
                model = model_from_json(loaded)
                if model.context_size == self.config.context_size and model.embedding_dim == self.config.embedding_dim:
                    model.expand_vocab(vocab_size, seed=self.config.seed)
                    self.model = model
                    return model
            except (KeyError, TypeError, ValueError):
                pass

        self.model = TinyNgramModel(
            vocab_size=vocab_size,
            context_size=self.config.context_size,
            embedding_dim=self.config.embedding_dim,
            seed=self.config.seed,
        )
        return self.model

    def _write_live_model(
        self,
        model: TinyNgramModel,
        vocab: dict[str, Any],
        *,
        path: Path | None = None,
    ) -> None:
        output_path = path or self.live_model_path
        model_json = {
            **model.to_json(),
            "vocab": vocab,
            "trained_at": epoch_seconds(),
            "training": {
                "orchestrator": True,
                "mode": self.config.mode,
                "steps": int(self.state.get("trainer_steps", 0) or 0),
                "learning_rate": self.config.learning_rate,
            },
        }
        write_json_atomic(output_path, model_json)

    def _should_checkpoint(self) -> bool:
        interval = max(self.config.checkpoint_interval_steps, 1)
        steps = int(self.state.get("trainer_steps", 0) or 0)
        previous = int(self.state.get("last_checkpoint_step", 0) or 0)
        if steps - previous < interval:
            return False
        self.state["last_checkpoint_step"] = steps
        return True

    def _should_evaluate_for_steps(self) -> bool:
        interval = max(self.config.evaluation_interval_steps, 1)
        steps = int(self.state.get("trainer_steps", 0) or 0)
        previous = int(self.state.get("last_step_evaluation_step", 0) or 0)
        if steps - previous < interval:
            return False
        self.state["last_step_evaluation_step"] = steps
        return True

    async def _detect_training_plateau(self, train_loss: float, eval_loss: float) -> None:
        previous_train = self.state.get("last_train_loss")
        previous_eval = self.state.get("last_eval_loss")
        plateau_count = int(self.state.get("plateau_count", 0) or 0)
        if previous_train is not None and previous_eval is not None:
            train_improved = train_loss < float(previous_train) - 1e-6
            eval_stagnated = eval_loss >= float(previous_eval) - 1e-4
            plateau_count = plateau_count + 1 if train_improved and eval_stagnated else 0
        self.state["plateau_count"] = plateau_count
        self.state["last_train_loss"] = train_loss
        self.state["last_eval_loss"] = eval_loss
        if plateau_count >= self.config.plateau_patience:
            await self._policy_event(
                "TRAINING_PLATEAU",
                train_loss=round(train_loss, 6),
                eval_loss=round(eval_loss, 6),
                recommendation="increase scraping priority and source diversity",
            )
            if self.config.mode == "autonomous":
                self.allocation.scraper_concurrency += 1

    async def _evaluation_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self.evaluation_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await self._event("EVALUATION_FAILED", error=str(exc))
            await self._sleep(self.config.evaluation_interval_seconds)

    async def evaluation_cycle(self, *, force: bool = False) -> dict[str, Any] | None:
        if not force:
            last_eval = float(self.state.get("last_sample_evaluation_at", 0.0) or 0.0)
            if time.time() - last_eval < self.config.evaluation_interval_seconds:
                return None

        vocab = read_json_file(self.store.vocab_path, None)
        if not isinstance(vocab, dict) or not self.live_model_path.exists():
            return None
        token_to_id = {str(key): int(value) for key, value in vocab["token_to_id"].items()}
        id_to_token = [str(token) for token in vocab["id_to_token"]]

        async with self.model_lock:
            model = self._load_or_init_model(len(id_to_token))
            sample = generate_sample(
                model,
                token_to_id=token_to_id,
                id_to_token=id_to_token,
                prefix=self.config.sample_prefix.split(),
                max_new_tokens=self.config.sample_tokens,
                min_generated=self.config.min_sample_tokens,
                seed=self.config.seed + int(self.state.get("trainer_steps", 0) or 0) + 1,
                temperature=self.config.temperature,
            )

        grammar_errors = validate_token_grammar(sample)
        quality = evaluate_token_sequence_quality(sample)
        diversity = len(set(sample)) / max(len(sample), 1)
        longest_repetition = max_repetition(sample)
        collapsed = (
            diversity < self.config.collapse_diversity_threshold
            or longest_repetition > self.config.collapse_max_repetition
            or quality.score < self.config.min_sample_quality_score
            or quality.path_obstructions > self.config.max_sample_path_obstructions
            or quality.control_spam > self.config.max_sample_control_spam
        )
        result = {
            "dataset_version": DATASET_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "timestamp": epoch_seconds(),
            "step": int(self.state.get("trainer_steps", 0) or 0),
            "grammar_valid": not grammar_errors,
            "grammar_errors": grammar_errors[:20],
            "token_count": len(sample),
            "unique_tokens": len(set(sample)),
            "diversity": round(diversity, 6),
            "max_repetition": longest_repetition,
            "quality": quality.to_json(),
            "collapsed": collapsed,
        }
        await append_jsonl(self.live_evaluation_stats_path, result)
        write_json_atomic(
            self.live_sample_path,
            {
                **result,
                "model_type": "tiny_embedding_ngram",
                "prefix": self.config.sample_prefix.split(),
                "tokens": sample,
            },
        )
        self.runtime_stats["samples_generated"] += 1
        self.runtime_stats["sample_tokens_generated"] += len(sample)
        self.state["last_sample_evaluation_at"] = time.time()
        await self._save_state()
        await self._event("EVALUATION_SAMPLE", **result)
        if collapsed:
            await self._policy_event(
                "TOKEN_COLLAPSE_WARNING",
                diversity=round(diversity, 6),
                max_repetition=longest_repetition,
                quality_score=round(quality.score, 6),
                path_obstructions=quality.path_obstructions,
                control_spam=quality.control_spam,
                recommendation="pause aggressive training and increase gameplay-path diversity",
            )
            if self.config.mode == "autonomous":
                self.state["training_paused_until"] = time.time() + self.config.collapse_pause_seconds
        return result

    async def _monitor_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                await self.monitor_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await self._event("MONITOR_FAILED", error=str(exc))
            await self._sleep(self.config.metrics_interval_seconds)

    async def monitor_cycle(self) -> dict[str, Any]:
        metrics = self._collect_metrics()
        await append_jsonl(self.store.orchestrator_metrics_path, metrics)
        await self._apply_policies(metrics)
        return metrics

    def _collect_metrics(self) -> dict[str, Any]:
        levels_scraped = count_lines(self.store.levels_path)
        levels_tokenized = count_lines(self.store.mechanics_tokens_path)
        processed_raw = len(self.processed_raw_ids)
        raw_queue_size = max(levels_scraped - processed_raw, 0)
        trainer_seen = int(self.state.get("trainer_token_records_seen", 0) or 0)
        token_queue_size = max(levels_tokenized - trainer_seen, 0)
        recent_tokens = load_recent_token_records(self.store.mechanics_tokens_path, 200)
        token_count_total = self._vocab_token_total()
        latest_training = load_latest_jsonl(self.live_training_stats_path) or {}
        latest_eval = load_latest_jsonl(self.live_evaluation_stats_path) or {}
        latest_scraper = load_latest_jsonl(self.store.metrics_path) or {}
        recent_unknown_rate = self._recent_unknown_object_rate()
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        tokens_per_second = self.runtime_stats["tokens_tokenized"] / elapsed
        eval_loss = latest_training.get("eval_loss")
        validation_quality = 0.0
        if isinstance(eval_loss, (int, float)):
            validation_quality = round(1.0 / (1.0 + float(eval_loss)), 6)
        return {
            "dataset_version": DATASET_VERSION,
            "timestamp": epoch_seconds(),
            "started_at": int(self.started_epoch),
            "uptime_seconds": round(time.time() - self.started_epoch, 6),
            "mode": self.config.mode,
            "dashboard": {
                "levels_scraped": levels_scraped,
                "levels_tokenized": levels_tokenized,
                "tokens_generated": token_count_total,
                "tokens_per_second": round(tokens_per_second, 6),
                "training_loss": latest_training.get("train_loss"),
                "validation_quality": validation_quality,
                "queue_sizes": {
                    "raw_queue": raw_queue_size,
                    "token_queue": token_queue_size,
                    "training_queue": int(latest_training.get("examples", 0) or 0),
                },
                "dataset_size": levels_tokenized,
            },
            "scraper": {
                "levels_per_minute": latest_scraper.get("levels_per_minute", 0),
                "request_failures": latest_scraper.get("levels_failed", 0),
                "duplicates_skipped": latest_scraper.get("duplicates_skipped", 0),
                "source_diversity": self._source_diversity(),
                "difficulty_distribution": self._difficulty_distribution(),
            },
            "tokenizer": {
                "token_throughput": round(tokens_per_second, 6),
                "unknown_object_rate": round(recent_unknown_rate, 6),
                "grammar_validity": round(self._grammar_validity(recent_tokens), 6),
                "token_entropy": round(token_entropy(recent_tokens), 6),
                "avg_step_density": round(self._avg_step_density(recent_tokens), 6),
            },
            "trainer": {
                "training_loss": latest_training.get("train_loss"),
                "validation_loss": latest_training.get("eval_loss"),
                "generation_diversity": latest_eval.get("diversity"),
                "collapse_indicator": latest_eval.get("collapsed", False),
                "grammar_legal": latest_eval.get("grammar_valid"),
                "steps": int(self.state.get("trainer_steps", 0) or 0),
            },
            "allocation": asdict(self.allocation),
        }

    async def _apply_policies(self, metrics: dict[str, Any]) -> None:
        dashboard = metrics["dashboard"]
        tokenizer_metrics = metrics["tokenizer"]
        trainer_metrics = metrics["trainer"]
        raw_queue = int(dashboard["queue_sizes"]["raw_queue"])
        token_queue = int(dashboard["queue_sizes"]["token_queue"])

        if raw_queue >= self.config.raw_backlog_warning:
            await self._policy_event(
                "TOKENIZER_BACKLOG",
                raw_queue=raw_queue,
                recommendation="increase tokenizer allocation",
            )
            if self.config.mode == "autonomous":
                self.allocation.tokenizer_records_per_cycle += 50

        if token_queue <= self.config.token_starvation_threshold and raw_queue > 0:
            await self._policy_event(
                "DATASET_STARVATION",
                raw_queue=raw_queue,
                token_queue=token_queue,
                recommendation="increase tokenizer allocation and scraper throughput",
            )
            if self.config.mode == "autonomous":
                self.allocation.scraper_concurrency += 1
                self.allocation.tokenizer_records_per_cycle += 25

        if float(tokenizer_metrics["unknown_object_rate"]) >= self.config.unknown_object_warning_rate:
            await self._policy_event(
                "UNKNOWN_OBJECTS_INCREASE",
                unknown_object_rate=tokenizer_metrics["unknown_object_rate"],
                recommendation="trigger mapping review warning",
            )

        if (
            int(dashboard["levels_tokenized"]) >= self.config.train_min_records
            and float(tokenizer_metrics["token_entropy"]) < self.config.min_token_entropy
        ):
            await self._policy_event(
                "DATASET_DIVERSITY_LOW",
                token_entropy=tokenizer_metrics["token_entropy"],
                recommendation="increase scrape diversity and emphasize underrepresented difficulties",
            )
            if self.config.mode == "autonomous":
                self.allocation.scraper_concurrency += 1

        if trainer_metrics.get("collapse_indicator"):
            await self._policy_event(
                "TOKEN_COLLAPSE_WARNING",
                recommendation="pause aggressive training and increase dataset diversity",
            )

    async def _policy_event(self, code: str, **details: Any) -> None:
        now = time.monotonic()
        last = self.policy_last_emitted.get(code, 0.0)
        if now - last < self.config.policy_cooldown_seconds:
            return
        self.policy_last_emitted[code] = now
        details.setdefault("mode", self.config.mode)
        if self.config.mode == "assisted":
            details.setdefault("action_applied", False)
        else:
            details.setdefault("action_applied", True)
        await self._event(code, **details)

    async def _event(self, code: str, **details: Any) -> None:
        record = {
            "dataset_version": DATASET_VERSION,
            "timestamp": epoch_seconds(),
            "event": code,
            **details,
        }
        LOGGER.info("%s %s", code, details)
        await append_jsonl(self.store.orchestrator_events_path, record)

    async def _save_state(self) -> None:
        write_json_atomic(self.store.orchestrator_state_path, self.state)

    async def _sleep(self, seconds: float) -> None:
        sleep_task = asyncio.create_task(asyncio.sleep(max(seconds, 0.0)))
        stop_task = asyncio.create_task(self.stop_event.wait())
        try:
            done, pending = await asyncio.wait(
                {sleep_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if stop_task in done:
                return
        finally:
            for task in (sleep_task, stop_task):
                if not task.done():
                    task.cancel()

    def _config_json(self) -> dict[str, Any]:
        value = asdict(self.config)
        value["data_dir"] = str(self.config.data_dir)
        value["model_dir"] = str(self.config.model_dir)
        return value

    def _level_id(self, record: dict[str, Any]) -> int:
        try:
            return int(record.get("level_id", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _vocab_token_total(self) -> int:
        vocab = read_json_file(self.store.vocab_path, {})
        counts = dict(vocab.get("token_counts", {})) if isinstance(vocab, dict) else {}
        total = 0
        for value in counts.values():
            try:
                total += int(value)
            except (TypeError, ValueError):
                continue
        return total

    def _recent_unknown_object_rate(self) -> float:
        raw_objects = 0
        unknown_objects = 0
        if not self.store.tokenizer_stats_path.exists():
            return 0.0
        recent: deque[dict[str, Any]] = deque(maxlen=200)
        with self.store.tokenizer_stats_path.open("rb") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = loads_json(raw_line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("record_type") == "level":
                    recent.append(record)
        for record in recent:
            raw_objects += int(record.get("object_count_raw", 0) or 0)
            unknown_objects += int(record.get("unknown_object_count", 0) or 0)
        return unknown_objects / max(raw_objects, 1)

    def _source_diversity(self) -> int:
        sources: set[str] = set()
        if not self.store.levels_path.exists():
            return 0
        with self.store.levels_path.open("rb") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = loads_json(raw_line)
                except orjson.JSONDecodeError:
                    continue
                sources.add(str(record.get("source", "")))
        sources.discard("")
        return len(sources)

    def _difficulty_distribution(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        if not self.store.levels_path.exists():
            return {}
        with self.store.levels_path.open("rb") as handle:
            for raw_line in handle:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = loads_json(raw_line)
                except orjson.JSONDecodeError:
                    continue
                counts[str(record.get("difficulty", "NA"))] += 1
        return dict(counts)

    def _avg_step_density(self, records: Iterable[dict[str, Any]]) -> float:
        total = 0.0
        count = 0
        for record in records:
            tokens = [str(token) for token in record.get("tokens", []) or []]
            if not tokens:
                continue
            total += tokens.count("STEP") / len(tokens)
            count += 1
        return total / max(count, 1)

    def _grammar_validity(self, records: Iterable[dict[str, Any]]) -> float:
        valid = 0
        total = 0
        for record in records:
            tokens = [str(token) for token in record.get("tokens", []) or []]
            if not tokens:
                continue
            total += 1
            if not validate_token_grammar(tokens):
                valid += 1
        return valid / max(total, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-orchestrate-mechanics",
        description="Run scraping, tokenization, training, evaluation, and monitoring as one continuous pipeline.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models") / "mechanics_v1")
    parser.add_argument("--mode", choices=["assisted", "autonomous"], default="assisted")
    parser.add_argument("--sources", nargs="*", default=default_source_names())
    parser.add_argument("--disable-scraper", action="store_true")
    parser.add_argument("--disable-tokenizer", action="store_true")
    parser.add_argument("--disable-trainer", action="store_true")
    parser.add_argument("--disable-evaluator", action="store_true")
    parser.add_argument("--scraper-pages-per-cycle", type=int, default=25)
    parser.add_argument("--scraper-target-count", type=int, default=0)
    parser.add_argument("--scraper-restart-seconds", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=1.0)
    parser.add_argument("--search-rate", type=float, default=2.0)
    parser.add_argument("--download-rate", type=float, default=0.33)
    parser.add_argument("--comment-rate", type=float, default=0.33)
    parser.add_argument("--include-comments", action="store_true")
    parser.add_argument("--download-audio", action="store_true")
    parser.add_argument("--comments-pages", type=int, default=0)
    parser.add_argument("--tokenizer-poll-seconds", type=float, default=2.0)
    parser.add_argument("--metrics-interval-seconds", type=float, default=10.0)
    parser.add_argument("--evaluation-interval-seconds", type=float, default=60.0)
    parser.add_argument("--evaluation-interval-steps", type=int, default=2_000)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0)
    parser.add_argument("--min-gameplay-objects", type=int, default=20)
    parser.add_argument("--min-tokens", type=int, default=50)
    parser.add_argument("--max-unknown-ratio", type=float, default=0.995)
    parser.add_argument("--max-token-length", type=int, default=12_000)
    parser.add_argument("--train-min-records", type=int, default=5)
    parser.add_argument("--train-window-records", type=int, default=250)
    parser.add_argument("--train-max-examples", type=int, default=30_000)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-prefix", default="START DIFF_HARD ALIGN_UNKNOWN")
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--min-sample-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=10_000)
    parser.add_argument("--min-sample-quality-score", type=float, default=80.0)
    parser.add_argument("--max-sample-path-obstructions", type=int, default=0)
    parser.add_argument("--max-sample-control-spam", type=int, default=2)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--once", action="store_true", help="Process currently queued local data once without starting scraper.")
    return parser


def config_from_args(args: argparse.Namespace) -> OrchestratorConfig:
    return OrchestratorConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        mode=args.mode,
        sources=coerce_source_names(args.sources),
        scraper_enabled=not args.disable_scraper and not args.once,
        tokenizer_enabled=not args.disable_tokenizer,
        trainer_enabled=not args.disable_trainer,
        evaluator_enabled=not args.disable_evaluator,
        scraper_pages_per_cycle=args.scraper_pages_per_cycle,
        scraper_target_count=args.scraper_target_count,
        scraper_restart_seconds=args.scraper_restart_seconds,
        timeout_seconds=args.timeout,
        retries=args.retries,
        backoff_seconds=args.backoff,
        search_rate=args.search_rate,
        download_rate=args.download_rate,
        comment_rate=args.comment_rate,
        include_comments=args.include_comments,
        download_audio=args.download_audio,
        comments_pages=args.comments_pages,
        tokenizer_poll_seconds=args.tokenizer_poll_seconds,
        metrics_interval_seconds=args.metrics_interval_seconds,
        evaluation_interval_seconds=args.evaluation_interval_seconds,
        evaluation_interval_steps=args.evaluation_interval_steps,
        max_runtime_seconds=args.max_runtime_seconds,
        min_gameplay_objects=args.min_gameplay_objects,
        min_tokens=args.min_tokens,
        max_unknown_ratio=args.max_unknown_ratio,
        max_token_length=args.max_token_length,
        train_min_records=args.train_min_records,
        train_window_records=args.train_window_records,
        train_max_examples=args.train_max_examples,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        sample_prefix=args.sample_prefix,
        sample_tokens=args.sample_tokens,
        min_sample_tokens=args.min_sample_tokens,
        temperature=args.temperature,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        min_sample_quality_score=args.min_sample_quality_score,
        max_sample_path_obstructions=args.max_sample_path_obstructions,
        max_sample_control_spam=args.max_sample_control_spam,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
    )


async def async_main(args: argparse.Namespace) -> int:
    log_dir = args.data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "orchestrator.log", encoding="utf-8"),
        ],
        force=True,
    )
    orchestrator = ContinuousPipelineOrchestrator(config_from_args(args))
    with OrchestratorSignalHandler(orchestrator) as shutdown:
        try:
            if args.once:
                await orchestrator.run_once()
            else:
                await orchestrator.run()
        except KeyboardInterrupt:
            orchestrator.request_shutdown("keyboard_interrupt")
            return 130
        except asyncio.CancelledError:
            if shutdown.exit_code:
                return shutdown.exit_code
            raise
        if shutdown.exit_code:
            return shutdown.exit_code
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
