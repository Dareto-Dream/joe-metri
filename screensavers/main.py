from __future__ import annotations

import argparse
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
SCREEN_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "mechanics_v1"
DEFAULT_PORT = 5000
DEFAULT_STALE_AFTER_SECONDS = 180


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def latest_jsonl(path: Path) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    latest = item
    except OSError:
        return None
    return latest


def tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    items: deque[dict[str, Any]] = deque(maxlen=max(limit, 1))
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    items.append(item)
    except OSError:
        return []
    return list(items)


def count_jsonl(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def file_size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 3)
    except OSError:
        return 0.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def get_system_stats() -> dict[str, float]:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return {
            "cpu": 0.0,
            "ram": 0.0,
            "ram_used": 0.0,
            "ram_total": 0.0,
            "gpu": 0.0,
            "vram_used": 0.0,
            "vram_total": 0.0,
        }

    memory = psutil.virtual_memory()
    gpu_load = 0.0
    vram_used = 0.0
    vram_total = 0.0
    try:
        import GPUtil  # type: ignore[import-not-found]

        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_load = float(gpu.load) * 100.0
            vram_used = float(gpu.memoryUsed)
            vram_total = float(gpu.memoryTotal)
    except Exception:
        pass

    return {
        "cpu": float(psutil.cpu_percent(interval=0.05)),
        "ram": float(memory.percent),
        "ram_used": float(memory.used),
        "ram_total": float(memory.total),
        "gpu": gpu_load,
        "vram_used": vram_used,
        "vram_total": vram_total,
    }


def latest_orchestrator_paths(data_dir: Path, model_dir: Path) -> dict[str, Path]:
    return {
        "metrics": data_dir / "logs" / "orchestrator_metrics.jsonl",
        "events": data_dir / "logs" / "orchestrator_events.jsonl",
        "tokens": data_dir / "tokenized" / "mechanics_tokens.jsonl",
        "vocab": data_dir / "tokenized" / "vocab.json",
        "model": model_dir / "model_live.json",
        "training": model_dir / "training_stats_live.jsonl",
        "evaluation": model_dir / "evaluation_stats_live.jsonl",
        "sample": model_dir / "sample_generation_live.json",
    }


def is_recent(timestamp: Any, *, stale_after_seconds: int) -> bool:
    try:
        return time.time() - float(timestamp) <= stale_after_seconds
    except (TypeError, ValueError):
        return False


def build_training_compat_payload(
    *,
    data_dir: Path,
    model_dir: Path,
    latest_metrics: dict[str, Any],
    latest_training: dict[str, Any] | None,
    latest_eval: dict[str, Any] | None,
    latest_sample: dict[str, Any],
    events: list[dict[str, Any]],
    active: bool,
) -> dict[str, Any]:
    paths = latest_orchestrator_paths(data_dir, model_dir)
    dashboard = dict(latest_metrics.get("dashboard") or {})
    trainer = dict(latest_metrics.get("trainer") or {})
    tokenizer = dict(latest_metrics.get("tokenizer") or {})
    allocation = dict(latest_metrics.get("allocation") or {})
    model = read_json(paths["model"], {})
    model_training = dict(model.get("training") or {}) if isinstance(model, dict) else {}

    steps = int((latest_training or {}).get("step") or trainer.get("steps") or 0)
    steps_per_epoch = max(1, int(model_training.get("checkpoint_interval_steps") or 10_000))
    epoch = max(1, steps // steps_per_epoch + 1)
    dataset_size = int(dashboard.get("dataset_size") or count_jsonl(paths["tokens"]))
    raw_queue = int((dashboard.get("queue_sizes") or {}).get("raw_queue") or 0)
    token_queue = int((dashboard.get("queue_sizes") or {}).get("token_queue") or 0)
    queue_total = max(dataset_size + raw_queue + token_queue, 1)
    stream_progress = clamp(dataset_size / queue_total * 100.0, 0.0, 100.0)
    train_loss = float((latest_training or {}).get("train_loss") or trainer.get("training_loss") or 0.0)
    val_loss = float((latest_training or {}).get("eval_loss") or trainer.get("validation_loss") or train_loss)
    sample_tokens = latest_sample.get("tokens") if isinstance(latest_sample.get("tokens"), list) else []
    event_name = str(events[-1].get("event", "orchestrator_idle")) if events else "orchestrator_idle"

    return {
        "live": active,
        "active": active,
        "spoofed": False,
        "source": "orchestrator",
        "run_name": "GD Mechanics Orchestrator",
        "model_name": "mechanics_v1 live ngram",
        "dataset_name": "mechanics_tokens.jsonl",
        "dataset_path": str(paths["tokens"]),
        "dataset_samples": dataset_size,
        "dataset_size_mb": file_size_mb(paths["tokens"]),
        "avg_instruction_chars": 0.0,
        "avg_response_chars": 0.0,
        "max_seq_length": int(model.get("context_size") or 0) if isinstance(model, dict) else 0,
        "per_device_batch_size": max(1, int(allocation.get("training_examples_per_cycle") or 1)),
        "gradient_accumulation_steps": 1,
        "effective_batch_size": max(1, int(allocation.get("training_examples_per_cycle") or 1)),
        "optimizer": "online_ngram_sgd",
        "configured_epochs": epoch,
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_dropout": 0.0,
        "lora_target_modules": [],
        "checkpoint_path": str(paths["model"]),
        "epoch": epoch,
        "epochs_total": epoch,
        "step": steps,
        "steps_per_epoch": steps_per_epoch,
        "progress_pct": stream_progress,
        "loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": float(model_training.get("learning_rate") or 0.0),
        "grad_norm": 0.0,
        "tokens_per_second": float(dashboard.get("tokens_per_second") or 0.0),
        "started_at": latest_metrics.get("timestamp"),
        "eta_seconds": None,
        "uptime_seconds": max(0.0, time.time() - float(latest_metrics.get("timestamp") or time.time())),
        "shutdown_allowed": not active,
        "shutdown_message": (
            "The Geometry Dash AI orchestrator is active. Do not shut down, sleep, or restart this machine."
            if active
            else "No recent orchestrator heartbeat was detected. Shutdown is allowed."
        ),
        "disclaimer": "" if active else "*orchestrator telemetry is stale or unavailable",
        "event": event_name,
        "raw_queue": raw_queue,
        "token_queue": token_queue,
        "training_queue": int((dashboard.get("queue_sizes") or {}).get("training_queue") or 0),
        "levels_scraped": int(dashboard.get("levels_scraped") or 0),
        "levels_tokenized": dataset_size,
        "tokens_generated": int(dashboard.get("tokens_generated") or 0),
        "token_entropy": float(tokenizer.get("token_entropy") or 0.0),
        "grammar_validity": float(tokenizer.get("grammar_validity") or 0.0),
        "generation_diversity": float((latest_eval or {}).get("diversity") or trainer.get("generation_diversity") or 0.0),
        "sample_preview": " ".join(str(token) for token in sample_tokens[:24]),
    }


def build_payload(
    *,
    data_dir: Path,
    model_dir: Path,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
) -> dict[str, Any]:
    paths = latest_orchestrator_paths(data_dir, model_dir)
    metrics = latest_jsonl(paths["metrics"]) or {}
    training = latest_jsonl(paths["training"])
    evaluation = latest_jsonl(paths["evaluation"])
    events = tail_jsonl(paths["events"], 16)
    sample = read_json(paths["sample"], {})
    vocab = read_json(paths["vocab"], {})

    active = is_recent(metrics.get("timestamp"), stale_after_seconds=stale_after_seconds)
    system_stats = get_system_stats()
    training_payload = build_training_compat_payload(
        data_dir=data_dir,
        model_dir=model_dir,
        latest_metrics=metrics,
        latest_training=training,
        latest_eval=evaluation,
        latest_sample=sample if isinstance(sample, dict) else {},
        events=events,
        active=active,
    )

    return {
        **system_stats,
        "training": training_payload,
        "training_live": training_payload["live"],
        "shutdown_allowed": training_payload["shutdown_allowed"],
        "generated_at": time.time(),
        "orchestrator": {
            "active": active,
            "mode": metrics.get("mode", "unknown"),
            "metrics": metrics,
            "latest_training": training or {},
            "latest_evaluation": evaluation or {},
            "events": events,
            "sample": sample if isinstance(sample, dict) else {},
            "vocab_size": int(vocab.get("vocab_size") or 0) if isinstance(vocab, dict) else 0,
            "paths": {key: str(value) for key, value in paths.items()},
        },
    }


class ScreensaverHandler(BaseHTTPRequestHandler):
    data_dir: Path = DEFAULT_DATA_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers("text/plain")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", ""}:
            self._redirect("/screensaver_1_terminal.html")
            return
        if parsed.path == "/stats":
            query = parse_qs(parsed.query)
            data_dir = Path(query.get("data_dir", [str(self.data_dir)])[0])
            model_dir = Path(query.get("model_dir", [str(self.model_dir)])[0])
            payload = build_payload(
                data_dir=data_dir,
                model_dir=model_dir,
                stale_after_seconds=self.stale_after_seconds,
            )
            self._send_json(payload)
            return

        requested = parsed.path.lstrip("/")
        path = (SCREEN_DIR / requested).resolve()
        if SCREEN_DIR not in path.parents and path != SCREEN_DIR:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.exists() or path.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(path)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._send_common_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._send_common_headers(content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, target: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self._send_common_headers("text/plain")
        self.send_header("Location", target)
        self.end_headers()

    def _send_common_headers(self, content_type: str) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-screensavers",
        description="Serve orchestrator-aware Geometry Dash AI screensaver telemetry.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    return parser


def run_server(args: argparse.Namespace) -> int:
    handler = ScreensaverHandler
    handler.data_dir = args.data_dir
    handler.model_dir = args.model_dir
    handler.stale_after_seconds = args.stale_after_seconds
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving screensavers at http://{args.host}:{args.port}/")
    print(f"terminal: http://{args.host}:{args.port}/screensaver_1_terminal.html")
    print(f"minimal:  http://{args.host}:{args.port}/screensaver_2_minimal.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


def main() -> int:
    return run_server(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
