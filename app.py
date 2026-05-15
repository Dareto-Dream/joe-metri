from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from runtime.audio import AudioAnalysis, EnergyPoint, EnergySection, SUPPORTED_AUDIO_EXTENSIONS, analyze_audio
from runtime.conditioning import GenerationControls, build_conditioning
from runtime.exporter import (
    ExportError,
    GmdMetadata,
    export_gmd,
    export_generation_json,
    export_k4,
    export_level_string,
    export_object_strings,
    inject_generation_into_local_save,
    inject_generation_into_save,
    write_export_metrics,
    write_layout_exports,
    write_generation_exports,
)
from runtime.generator import GenerationResult, MechanicsRuntime
from runtime.flow import arrange_flow_synced_tokens
from runtime.planner import plan_tokens
from runtime.reconstructor import reconstruct_layout
from runtime.save_codec import SaveCodecError, inject_level_string_into_local_save, inject_level_string_into_save
from runtime.validator import validate_generation
from gd_scraper.storage import loads_json


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data" / "runtime"
UPLOAD_DIR = DATA_DIR / "uploads"
SAVE_UPLOAD_DIR = DATA_DIR / "saves"
EXPORT_DIR = ROOT / "exports"

app = FastAPI(title="Geometry Dash AI Runtime v1")
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

runtime = MechanicsRuntime(ROOT / "models" / "mechanics_v1")
uploaded_audio: dict[str, Path] = {}
generations: dict[str, GenerationResult] = {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_path": str(runtime.model_path),
        "supported_audio": sorted(SUPPORTED_AUDIO_EXTENSIONS),
        "generations": len(generations),
    }


@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)) -> dict[str, object]:
    path = await _save_upload(audio)
    upload_id = path.stem
    uploaded_audio[upload_id] = path
    analysis = analyze_audio(path, filename=audio.filename)
    return {"upload_id": upload_id, "audio": analysis.to_json()}


@app.post("/generate")
async def generate_level(
    audio: UploadFile | None = File(default=None),
    upload_id: str | None = Form(default=None),
    difficulty: str = Form(default="Hard"),
    alignments: str = Form(default="Flow,Sync-heavy"),
    temperature: float = Form(default=0.9),
    top_k: int = Form(default=40),
    max_tokens: int = Form(default=360),
    seed: int = Form(default=7),
    planning_iterations: int = Form(default=4),
) -> JSONResponse:
    if audio is not None:
        path = await _save_upload(audio)
        filename = audio.filename
    elif upload_id:
        path = uploaded_audio.get(upload_id)
        filename = path.name if path is not None else None
        if path is None:
            raise HTTPException(status_code=404, detail="upload_not_found")
    else:
        raise HTTPException(status_code=400, detail="audio_required")

    controls = _generation_controls(difficulty, alignments, temperature, top_k, max_tokens, seed, planning_iterations)
    result = runtime.generate_from_audio(path, filename=filename, controls=controls)
    generations[result.generation_id] = result
    return JSONResponse(result.to_json())


@app.post("/generate-stream")
async def generate_level_stream(
    audio: UploadFile | None = File(default=None),
    upload_id: str | None = Form(default=None),
    difficulty: str = Form(default="Hard"),
    alignments: str = Form(default="Flow,Sync-heavy"),
    temperature: float = Form(default=0.9),
    top_k: int = Form(default=40),
    max_tokens: int = Form(default=360),
    seed: int = Form(default=7),
    planning_iterations: int = Form(default=4),
) -> StreamingResponse:
    if audio is not None:
        path = await _save_upload(audio)
        filename = audio.filename
    elif upload_id:
        path = uploaded_audio.get(upload_id)
        filename = path.name if path is not None else None
        if path is None:
            raise HTTPException(status_code=404, detail="upload_not_found")
    else:
        raise HTTPException(status_code=400, detail="audio_required")

    controls = _generation_controls(difficulty, alignments, temperature, top_k, max_tokens, seed, planning_iterations)

    def stream_events() -> object:
        events: queue.Queue[dict[str, object] | None] = queue.Queue()

        def emit(event: dict[str, object]) -> None:
            events.put(event)

        def run_generation() -> None:
            try:
                result = runtime.generate_from_audio(path, filename=filename, controls=controls, event_callback=emit)
                generations[result.generation_id] = result
                events.put({"type": "done", "generation": result.to_json()})
            except Exception as exc:  # noqa: BLE001
                events.put({"type": "error", "error": str(exc)})
            finally:
                events.put(None)

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@app.get("/preview/{generation_id}")
def get_preview(generation_id: str) -> dict[str, object]:
    result = _generation_or_404(generation_id)
    return result.to_json()


@app.get("/export/{generation_id}", response_model=None)
def export_generation(generation_id: str, format: str = "json") -> Response:
    result = _generation_or_404(generation_id)
    normalized = format.strip().lower()
    if normalized == "json":
        return JSONResponse(export_generation_json(result))
    if normalized in {"objects", "object_strings", "raw"}:
        return PlainTextResponse(export_object_strings(result), media_type="text/plain")
    if normalized in {"level", "level_string"}:
        return PlainTextResponse(export_level_string(result), media_type="text/plain")
    if normalized == "gmd":
        return Response(
            export_gmd(result),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="generated.gmd"'},
        )
    if normalized == "k4":
        return PlainTextResponse(export_k4(result), media_type="text/plain")
    raise HTTPException(status_code=400, detail="unknown_export_format")


@app.post("/inject/{generation_id}", response_model=None)
async def inject_save(
    generation_id: str,
    save: UploadFile = File(...),
    target_level_key: str | None = Form(default=None),
    target_level_name: str | None = Form(default=None),
) -> JSONResponse:
    result = _generation_or_404(generation_id)
    save_path = await _save_dat_upload(save)
    export_dir = EXPORT_DIR / generation_id
    try:
        injection, metrics = inject_generation_into_save(
            result,
            save_path,
            export_dir,
            target_level_key=target_level_key or None,
            target_level_name=target_level_name or None,
        )
    except (ExportError, SaveCodecError) as exc:
        raise HTTPException(status_code=400, detail=getattr(exc, "reason", str(exc))) from exc

    return JSONResponse(
        {
            "injection": injection.to_json(),
            "metrics": metrics.to_json(),
        }
    )


@app.post("/inject-local/{generation_id}", response_model=None)
async def inject_local_save(
    generation_id: str,
    save: UploadFile = File(...),
    target_level_name: str | None = Form(default=None),
    target_slot: int | None = Form(default=None),
) -> JSONResponse:
    result = _generation_or_404(generation_id)
    save_path = await _save_dat_upload(save)
    export_dir = EXPORT_DIR / generation_id
    try:
        injection, metrics = inject_generation_into_local_save(
            result,
            save_path,
            export_dir,
            target_level_name=target_level_name or None,
            target_slot=target_slot,
        )
    except (ExportError, SaveCodecError) as exc:
        raise HTTPException(status_code=400, detail=getattr(exc, "reason", str(exc))) from exc

    return JSONResponse(
        {
            "injection": injection.to_json(),
            "metrics": metrics.to_json(),
        }
    )


async def _save_upload(upload: UploadFile) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"unsupported_audio_type:{extension or 'none'}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    with path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return path


async def _save_dat_upload(upload: UploadFile) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension != ".dat":
        raise HTTPException(status_code=400, detail=f"unsupported_save_type:{extension or 'none'}")
    SAVE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"
    with path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return path


def _generation_or_404(generation_id: str) -> GenerationResult:
    result = generations.get(generation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="generation_not_found")
    return result


def _parse_alignments(value: str) -> tuple[str, ...]:
    alignments = tuple(item.strip() for item in value.split(",") if item.strip())
    return alignments or ("Flow", "Sync-heavy")


def _generation_controls(
    difficulty: str,
    alignments: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    seed: int,
    planning_iterations: int,
) -> GenerationControls:
    return GenerationControls(
        difficulty=difficulty,
        alignments=_parse_alignments(alignments),
        temperature=max(0.2, min(1.8, float(temperature))),
        top_k=max(1, min(80, int(top_k))),
        max_tokens=max(90, min(900, int(max_tokens))),
        seed=int(seed),
        planning_iterations=max(1, min(16, int(planning_iterations))),
    )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python app.py",
        description="Geometry Dash AI runtime server and export CLI.",
    )
    parser.add_argument("--export-gmd", action="store_true", help="Write exports/generated.gmd and companion artifacts.")
    parser.add_argument("--export-object-string", action="store_true", help="Write exports/level_string.txt.")
    parser.add_argument("--inject-save", action="store_true", help="Inject generated level data into a GD save.")
    parser.add_argument("--inject-local-level", action="store_true", help="Inject generated level data into CCLocalLevels.dat.")
    parser.add_argument("--open-exports", action="store_true", help="Open the export folder after writing artifacts.")
    parser.add_argument("--audio", type=Path, help="Optional audio file used for CLI generation.")
    parser.add_argument(
        "--tokens-file",
        type=Path,
        default=ROOT / "models" / "mechanics_v1" / "sample_generation_live.json",
        help="Token JSON file used when --audio is not provided.",
    )
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--target-level-key", default="")
    parser.add_argument("--target-level-name", default="")
    parser.add_argument("--target-slot", type=int)
    parser.add_argument("--level-name", default="AI Generated Level")
    parser.add_argument("--level-description", default="Generated by Geometry Dash AI runtime.")
    parser.add_argument("--creator", default="Geometry Dash AI")
    parser.add_argument("--official-song-id", type=int, default=0)
    parser.add_argument("--custom-song-id", type=int, default=0)
    parser.add_argument("--difficulty", default="Hard")
    parser.add_argument("--alignments", default="Flow,Sync-heavy")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planning-iterations", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def run_cli(args: argparse.Namespace) -> int:
    if args.inject_save and args.inject_local_level:
        raise ValueError("--inject-save and --inject-local-level are mutually exclusive")
    if args.official_song_id and args.custom_song_id:
        raise ValueError("--official-song-id and --custom-song-id are mutually exclusive")

    export_dir: Path = args.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    metadata = GmdMetadata(
        level_name=args.level_name,
        description=args.level_description,
        creator=args.creator,
        official_song_id=max(0, int(args.official_song_id)),
        custom_song_id=max(0, int(args.custom_song_id)),
    )

    if args.audio:
        controls = GenerationControls(
            difficulty=args.difficulty,
            alignments=_parse_alignments(args.alignments),
            temperature=max(0.2, min(1.8, float(args.temperature))),
            top_k=max(1, min(80, int(args.top_k))),
            max_tokens=max(90, min(900, int(args.max_tokens))),
            seed=int(args.seed),
            planning_iterations=max(1, min(16, int(args.planning_iterations))),
        )
        result = runtime.generate_from_audio(args.audio, filename=args.audio.name, controls=controls)
        metrics = write_generation_exports(result, export_dir, metadata=metadata)
        level_string = result.layout.level_string
        token_count = len(result.tokens)
    else:
        tokens = load_cli_tokens(args.tokens_file)
        controls = GenerationControls(
            difficulty=args.difficulty,
            alignments=_parse_alignments(args.alignments),
            temperature=max(0.2, min(1.8, float(args.temperature))),
            top_k=max(1, min(80, int(args.top_k))),
            max_tokens=max(90, min(900, int(args.max_tokens))),
            seed=int(args.seed),
            planning_iterations=max(1, min(16, int(args.planning_iterations))),
        )
        if tokens is None:
            tokens = generate_cli_tokens(controls)
            print(f"sample token file not found; generated tokens from {runtime.model_path}")
        else:
            conditioning = build_conditioning(default_cli_audio_analysis(), controls)
            tokens, _flow_sync = arrange_flow_synced_tokens(tokens, conditioning, seed=controls.seed)
        validation = validate_generation(tokens, known_tokens=runtime.known_tokens)
        layout = reconstruct_layout(tokens)
        metrics = write_layout_exports(tokens, validation, layout, export_dir, metadata=metadata)
        level_string = layout.level_string
        token_count = len(tokens)

    print(f"wrote {export_dir / 'generated_level.json'}")
    print(f"wrote {export_dir / 'level_string.txt'}")
    print(f"wrote {export_dir / 'generated.gmd'}")
    print(f"wrote {export_dir / 'export_metrics.json'}")
    print(f"tokens={token_count}")
    print(f"objects_generated={metrics.objects_generated}")

    if args.inject_save:
        injection = inject_level_string_into_save(
            args.save_path or default_save_path(),
            level_string,
            export_dir,
            target_level_key=args.target_level_key or None,
            target_level_name=args.target_level_name or None,
        )
        metrics = replace(metrics, detected_codec=injection.detected_codec)
        write_export_metrics(export_dir, metrics)
        print(f"wrote {injection.backup_path}")
        print(f"wrote {injection.decoded_xml_path}")
        print(f"wrote {injection.generated_save_path}")
        print(f"target_level_key={injection.target_level_key}")
        print(f"detected_codec={injection.detected_codec}")

    if args.inject_local_level:
        injection = inject_level_string_into_local_save(
            args.save_path or default_local_levels_path(),
            level_string,
            export_dir,
            target_level_name=args.target_level_name or None,
            target_slot=args.target_slot,
        )
        metrics = replace(metrics, detected_codec=injection.detected_codec)
        write_export_metrics(export_dir, metrics)
        print(f"wrote {injection.backup_path}")
        print(f"wrote {injection.decoded_xml_path}")
        print(f"wrote {injection.generated_save_path}")
        print(f"target_container_key={injection.target_container_key}")
        print(f"target_level_key={injection.target_level_key}")
        print(f"target_slot={injection.target_slot}")
        print(f"detected_codec={injection.detected_codec}")

    if args.open_exports:
        open_export_folder(export_dir)

    return 0


def load_cli_tokens(path: Path) -> list[str] | None:
    default_live = ROOT / "models" / "mechanics_v1" / "sample_generation_live.json"
    default_static = ROOT / "models" / "mechanics_v1" / "sample_generation.json"
    if not path.exists() and path.name == "sample_generation_live.json":
        path = default_static
    if not path.exists():
        if path in (default_live, default_static):
            return None
        raise FileNotFoundError(f"tokens file not found: {path}")
    payload = loads_json(path.read_bytes())
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError(f"tokens file does not contain a token list: {path}")
    return [str(token) for token in tokens]


def generate_cli_tokens(controls: GenerationControls) -> list[str]:
    conditioning = build_conditioning(default_cli_audio_analysis(), controls)
    plan = plan_tokens(
        runtime.model,
        token_to_id=runtime.token_to_id,
        id_to_token=runtime.id_to_token,
        conditioning=conditioning,
        seed=controls.seed,
        temperature=controls.temperature,
        top_k=controls.top_k,
        iterations=controls.planning_iterations,
    )
    return plan.best.tokens


def default_cli_audio_analysis() -> AudioAnalysis:
    duration_seconds = 60.0
    beat_interval = 0.5
    beats = [index * beat_interval for index in range(1, int(duration_seconds / beat_interval))]
    energy = [
        EnergyPoint(time=index * beat_interval, value=0.45 + (0.18 if index % 4 == 0 else 0.0))
        for index in range(int(duration_seconds / beat_interval) + 1)
    ]
    return AudioAnalysis(
        filename="cli-default",
        extension="generated",
        duration_seconds=duration_seconds,
        bpm=120,
        beats=beats,
        onsets=beats[::2],
        energy=energy,
        energy_sections=[
            EnergySection(start=0.0, end=20.0, energy=0.42, label="steady"),
            EnergySection(start=20.0, end=40.0, energy=0.58, label="active"),
            EnergySection(start=40.0, end=duration_seconds, energy=0.5, label="steady"),
        ],
        decoder="synthetic",
    )


def default_save_path() -> Path:
    local_app_data = Path(os.environ["LOCALAPPDATA"]) if "LOCALAPPDATA" in os.environ else Path.home()
    return local_app_data / "GeometryDash" / "CCGameManager.dat"


def default_local_levels_path() -> Path:
    local_app_data = Path(os.environ["LOCALAPPDATA"]) if "LOCALAPPDATA" in os.environ else Path.home()
    return local_app_data / "GeometryDash" / "CCLocalLevels.dat"


def open_export_folder(export_dir: Path) -> None:
    path = export_dir.resolve()
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    if args.export_gmd or args.export_object_string or args.inject_save or args.inject_local_level:
        return run_cli(args)

    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
