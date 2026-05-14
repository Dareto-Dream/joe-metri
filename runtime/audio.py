from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import subprocess
import wave


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg"}


@dataclass(frozen=True)
class EnergyPoint:
    time: float
    value: float

    def to_json(self) -> dict[str, float]:
        return {"time": round(self.time, 3), "value": round(self.value, 4)}


@dataclass(frozen=True)
class EnergySection:
    start: float
    end: float
    energy: float
    label: str

    def to_json(self) -> dict[str, float | str]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "energy": round(self.energy, 4),
            "label": self.label,
        }


@dataclass(frozen=True)
class AudioAnalysis:
    filename: str
    extension: str
    duration_seconds: float
    bpm: int
    beats: list[float]
    onsets: list[float]
    energy: list[EnergyPoint]
    energy_sections: list[EnergySection]
    decoder: str

    def to_json(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "extension": self.extension,
            "duration_seconds": round(self.duration_seconds, 3),
            "bpm": self.bpm,
            "beats": [round(value, 3) for value in self.beats],
            "onsets": [round(value, 3) for value in self.onsets],
            "energy": [item.to_json() for item in self.energy],
            "energy_sections": [item.to_json() for item in self.energy_sections],
            "decoder": self.decoder,
        }


def analyze_audio(path: Path, *, filename: str | None = None) -> AudioAnalysis:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"unsupported_audio_type:{extension or 'none'}; supported={supported}")

    name = filename or path.name
    samples: list[float]
    sample_rate: int
    decoder: str
    if extension == ".wav":
        try:
            samples, sample_rate = _decode_wav(path)
            decoder = "wave"
        except (wave.Error, OSError, ValueError):
            samples, sample_rate, decoder = _decode_with_ffmpeg(path)
    else:
        try:
            samples, sample_rate, decoder = _decode_with_ffmpeg(path)
        except (OSError, subprocess.SubprocessError, ValueError):
            return _analyze_bytes(path, filename=name, extension=extension)

    if not samples or sample_rate <= 0:
        return _analyze_bytes(path, filename=name, extension=extension)

    duration = len(samples) / sample_rate
    energy = _rms_energy(samples, sample_rate)
    normalized_energy = _normalize_energy(energy)
    onsets = _detect_onsets(normalized_energy)
    bpm = _estimate_bpm(onsets, normalized_energy)
    beats = _build_beats(duration, bpm, onsets)
    sections = _energy_sections(normalized_energy, duration)

    return AudioAnalysis(
        filename=name,
        extension=extension.lstrip("."),
        duration_seconds=duration,
        bpm=bpm,
        beats=beats,
        onsets=onsets,
        energy=normalized_energy,
        energy_sections=sections,
        decoder=decoder,
    )


def _decode_wav(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if channels <= 0 or sample_rate <= 0:
        raise ValueError("invalid_wav_shape")

    if sample_width == 1:
        values = [(byte - 128) / 128.0 for byte in raw]
    elif sample_width == 2:
        values = array("h")
        values.frombytes(raw)
        if sample_width != values.itemsize:
            raise ValueError("unsupported_wav_width")
        scale = float(2**15)
        values_list = [max(-1.0, min(1.0, value / scale)) for value in values]
        values = values_list
    elif sample_width == 4:
        ints = array("i")
        ints.frombytes(raw)
        scale = float(2**31)
        values = [max(-1.0, min(1.0, value / scale)) for value in ints]
    else:
        raise ValueError("unsupported_wav_width")

    if channels == 1:
        return list(values), sample_rate

    mono: list[float] = []
    usable = len(values) - (len(values) % channels)
    for index in range(0, usable, channels):
        mono.append(sum(float(values[index + offset]) for offset in range(channels)) / channels)
    return mono, sample_rate


def _decode_with_ffmpeg(path: Path) -> tuple[list[float], int, str]:
    sample_rate = 22_050
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=True, timeout=30)
    pcm = array("h")
    pcm.frombytes(completed.stdout)
    if not pcm:
        raise ValueError("ffmpeg_decoded_empty")
    samples = [max(-1.0, min(1.0, value / float(2**15))) for value in pcm]
    return samples, sample_rate, "ffmpeg"


def _rms_energy(samples: list[float], sample_rate: int, *, window_seconds: float = 0.1) -> list[EnergyPoint]:
    window_size = max(1, int(sample_rate * window_seconds))
    energy: list[EnergyPoint] = []
    for start in range(0, len(samples), window_size):
        chunk = samples[start : start + window_size]
        if not chunk:
            continue
        rms = math.sqrt(sum(value * value for value in chunk) / len(chunk))
        energy.append(EnergyPoint(time=start / sample_rate, value=rms))
    return energy


def _normalize_energy(energy: list[EnergyPoint]) -> list[EnergyPoint]:
    if not energy:
        return []
    values = [item.value for item in energy]
    floor = min(values)
    peak = max(values)
    span = max(peak - floor, 1e-9)
    normalized: list[EnergyPoint] = []
    for item in energy:
        normalized.append(EnergyPoint(time=item.time, value=(item.value - floor) / span))
    return _smooth_energy(normalized)


def _smooth_energy(energy: list[EnergyPoint]) -> list[EnergyPoint]:
    if len(energy) < 3:
        return energy
    result: list[EnergyPoint] = []
    for index, item in enumerate(energy):
        start = max(0, index - 1)
        end = min(len(energy), index + 2)
        result.append(EnergyPoint(time=item.time, value=sum(e.value for e in energy[start:end]) / (end - start)))
    return result


def _detect_onsets(energy: list[EnergyPoint]) -> list[float]:
    if len(energy) < 4:
        return []

    deltas = [max(0.0, energy[index].value - energy[index - 1].value) for index in range(1, len(energy))]
    mean = statistics.fmean(deltas) if deltas else 0.0
    deviation = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    threshold = max(0.08, mean + deviation * 1.15)
    onsets: list[float] = []
    last_onset = -1.0
    for index, delta in enumerate(deltas, start=1):
        if delta < threshold:
            continue
        timestamp = energy[index].time
        if timestamp - last_onset < 0.16:
            continue
        onsets.append(timestamp)
        last_onset = timestamp
    return onsets


def _estimate_bpm(onsets: list[float], energy: list[EnergyPoint]) -> int:
    intervals: list[float] = []
    for left, right in zip(onsets, onsets[1:]):
        interval = right - left
        if 0.24 <= interval <= 1.2:
            intervals.append(interval)

    if intervals:
        interval = statistics.median(intervals)
        bpm = 60.0 / max(interval, 1e-9)
        return _normalize_bpm(bpm)

    autocorr_bpm = _estimate_bpm_from_energy(energy)
    return _normalize_bpm(autocorr_bpm)


def _estimate_bpm_from_energy(energy: list[EnergyPoint]) -> float:
    if len(energy) < 20:
        return 120.0
    values = [item.value for item in energy]
    step = max(energy[1].time - energy[0].time, 0.1)
    best_score = -1.0
    best_lag = int(round(0.5 / step))
    for lag in range(max(2, int(0.25 / step)), max(3, int(1.2 / step))):
        score = 0.0
        count = 0
        for index in range(lag, len(values)):
            score += values[index] * values[index - lag]
            count += 1
        if count:
            score /= count
        if score > best_score:
            best_score = score
            best_lag = lag
    return 60.0 / max(best_lag * step, 1e-9)


def _normalize_bpm(value: float) -> int:
    bpm = float(value)
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 210.0:
        bpm /= 2.0
    return int(round(max(80.0, min(210.0, bpm))))


def _build_beats(duration: float, bpm: int, onsets: list[float]) -> list[float]:
    interval = 60.0 / max(bpm, 1)
    start = onsets[0] if onsets else 0.0
    while start > interval:
        start -= interval
    beats: list[float] = []
    timestamp = max(0.0, start)
    while timestamp <= duration + interval:
        beats.append(timestamp)
        timestamp += interval
    return beats


def _energy_sections(energy: list[EnergyPoint], duration: float, *, count: int = 12) -> list[EnergySection]:
    if not energy:
        return [EnergySection(start=0.0, end=max(duration, 1.0), energy=0.35, label="steady")]

    count = max(1, min(count, max(1, len(energy))))
    section_seconds = max(duration / count, 0.1)
    sections: list[EnergySection] = []
    for index in range(count):
        start = index * section_seconds
        end = duration if index == count - 1 else (index + 1) * section_seconds
        values = [item.value for item in energy if start <= item.time < end]
        average = statistics.fmean(values) if values else 0.0
        if average >= 0.72:
            label = "drop"
        elif average >= 0.48:
            label = "build"
        elif average <= 0.22:
            label = "quiet"
        else:
            label = "steady"
        sections.append(EnergySection(start=start, end=end, energy=average, label=label))
    return sections


def _analyze_bytes(path: Path, *, filename: str, extension: str) -> AudioAnalysis:
    raw = path.read_bytes()
    size = len(raw)
    if size == 0:
        raise ValueError("empty_audio_file")

    duration = max(20.0, min(180.0, size / 32_000.0))
    window_count = 240
    chunk_size = max(1, size // window_count)
    values: list[EnergyPoint] = []
    for index in range(0, size, chunk_size):
        chunk = raw[index : index + chunk_size]
        if not chunk:
            continue
        centered = [(byte - 128) / 128.0 for byte in chunk]
        rms = math.sqrt(sum(value * value for value in centered) / len(centered))
        timestamp = duration * (index / max(size, 1))
        values.append(EnergyPoint(time=timestamp, value=rms))

    energy = _normalize_energy(values)
    onsets = _detect_onsets(energy)
    bpm = _estimate_bpm(onsets, energy)
    beats = _build_beats(duration, bpm, onsets)
    return AudioAnalysis(
        filename=filename,
        extension=extension.lstrip("."),
        duration_seconds=duration,
        bpm=bpm,
        beats=beats,
        onsets=onsets,
        energy=energy,
        energy_sections=_energy_sections(energy, duration),
        decoder="byte-fallback",
    )


def energy_at(analysis: AudioAnalysis, timestamp: float) -> float:
    if not analysis.energy:
        return 0.35
    closest = min(analysis.energy, key=lambda item: abs(item.time - timestamp))
    return closest.value

