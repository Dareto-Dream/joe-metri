from __future__ import annotations

from pathlib import Path
import base64
import gzip
import json
import tempfile
import unittest
import wave
from array import array
import math

from runtime.audio import analyze_audio
from runtime.conditioning import GenerationControls, build_conditioning
from runtime.exporter import is_valid_gd_object_string, validate_layout_export, write_layout_exports
from runtime.generator import MechanicsRuntime
from runtime.reconstructor import reconstruct_layout
from runtime.save_codec import (
    CODEC_BASE64_GZIP,
    CODEC_PLAINTEXT_XML,
    CODEC_RAW_GZIP,
    CODEC_XOR_BASE64_GZIP,
    SaveCodecError,
    decode_save_bytes_with_codec,
    decode_level_string_k4,
    decode_save_file,
    encode_level_string_k4,
    encode_save_xml,
    find_injected_k4,
    find_local_level_k4,
    inject_level_string_into_local_save,
    inject_level_string_into_save,
)
from runtime.validator import validate_generation


class RuntimeTests(unittest.TestCase):
    def test_audio_analysis_extracts_wav_beats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse.wav"
            write_pulse_wav(path)
            analysis = analyze_audio(path)

        self.assertGreaterEqual(analysis.bpm, 80)
        self.assertGreater(len(analysis.beats), 2)
        self.assertGreater(len(analysis.energy), 2)

    def test_conditioning_builds_runtime_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse.wav"
            write_pulse_wav(path)
            analysis = analyze_audio(path)
        conditioning = build_conditioning(
            analysis,
            GenerationControls(difficulty="Extreme", alignments=("Sync-heavy",), max_tokens=120),
        )

        self.assertEqual(conditioning.prefix, ["START", "DIFF_EXTREME", "ALIGN_UNKNOWN"])
        self.assertGreaterEqual(conditioning.target_tokens, 90)
        self.assertTrue(conditioning.beat_steps)

    def test_reconstruct_layout_exports_raw_object_strings(self) -> None:
        layout = reconstruct_layout(
            [
                "START",
                "DIFF_HARD",
                "ALIGN_UNKNOWN",
                "BLOCK",
                "Y1",
                "WIDTH_2",
                "STEP",
                "SPIKE",
                "Y2",
                "STEP",
                "END",
            ]
        )

        self.assertEqual(layout.errors, [])
        self.assertEqual(len(layout.objects), 2)
        self.assertIn("1,1,2,15,3,30", layout.gd_object_strings)
        self.assertIn("1,1,2,45,3,30", layout.gd_object_strings)
        self.assertTrue(layout.level_string.startswith("kS1,0;"))
        self.assertTrue(is_valid_gd_object_string(layout.gd_object_strings[0]))

    def test_runtime_generates_valid_layout_from_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pulse.wav"
            write_pulse_wav(path)
            runtime = MechanicsRuntime(Path("models") / "mechanics_v1")
            result = runtime.generate_from_audio(
                path,
                controls=GenerationControls(difficulty="Hard", alignments=("Flow",), max_tokens=120, seed=3),
            )

        self.assertTrue(result.validation.valid, result.validation.errors)
        self.assertGreater(len(result.tokens), 20)
        self.assertGreater(len(result.layout.gd_object_strings), 0)
        self.assertGreater(result.metrics.tokens_per_second, 0)

    def test_k4_level_string_codec_roundtrips(self) -> None:
        level_string = "kS1,0;1,1,2,15,3,30;1,8,2,45,3,60"
        k4 = encode_level_string_k4(level_string)

        self.assertNotIn("=", k4)
        self.assertEqual(decode_level_string_k4(k4), level_string)

    def test_layout_export_writes_metrics_artifact(self) -> None:
        tokens = [
            "START",
            "DIFF_HARD",
            "ALIGN_UNKNOWN",
            "BLOCK",
            "Y1",
            "WIDTH_1",
            "STEP",
            "SPIKE",
            "Y2",
            "STEP",
            "END",
        ]
        layout = reconstruct_layout(tokens)
        validation = validate_generation(tokens)

        with tempfile.TemporaryDirectory() as directory:
            export_dir = Path(directory)
            metrics = write_layout_exports(tokens, validation, layout, export_dir)

            self.assertTrue((export_dir / "generated_level.json").exists())
            self.assertTrue((export_dir / "level_string.txt").exists())
            metrics_payload = json.loads((export_dir / "export_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(metrics.objects_generated, 2)
        self.assertEqual(metrics_payload["objects_generated"], 2)
        self.assertEqual(metrics_payload["STEP_count"], 2)

    def test_layout_export_records_invalid_token_warnings_without_crashing(self) -> None:
        tokens = [
            "START",
            "DIFF_HARD",
            "ALIGN_UNKNOWN",
            "NOT_A_TOKEN",
            "BLOCK",
            "Y1",
            "WIDTH_1",
            "STEP",
            "END",
        ]
        layout = reconstruct_layout(tokens)
        validation = validate_generation(tokens)

        with tempfile.TemporaryDirectory() as directory:
            metrics = write_layout_exports(tokens, validation, layout, Path(directory))

        self.assertEqual(metrics.objects_generated, 1)
        self.assertIn("unknown_token:NOT_A_TOKEN", metrics.serialization_warnings)

    def test_save_decoder_accepts_whitespace_wrapped_base64(self) -> None:
        xml = save_xml("AI Test", encode_level_string_k4("kS1,0;1,1,2,15,3,0"))
        encoded = base64.b64encode(gzip.compress(xml.encode("utf-8"))).decode("ascii")
        wrapped = "\n".join(encoded[index : index + 32] for index in range(0, len(encoded), 32))

        with tempfile.TemporaryDirectory() as directory:
            save_path = Path(directory) / "CCGameManager.dat"
            save_path.write_text(wrapped, encoding="ascii")
            decoded = decode_save_file(save_path)

        self.assertIn("GLM_03", decoded)
        self.assertIn("AI Test", decoded)

    def test_save_decoder_reports_supported_codecs(self) -> None:
        xml = local_save_xml(
            [
                ("AI Test", "kS1,0;1,1,2,15,3,0"),
            ],
            container_key="LLM_02",
        )

        for codec in (CODEC_XOR_BASE64_GZIP, CODEC_BASE64_GZIP, CODEC_RAW_GZIP, CODEC_PLAINTEXT_XML):
            with self.subTest(codec=codec):
                decoded = decode_save_bytes_with_codec(encode_save_xml(xml, codec=codec))

            self.assertEqual(decoded.detected_codec, codec)
            self.assertIn("LLM_02", decoded.xml_text)

    def test_save_injection_writes_backup_generated_save_and_preserves_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_path = root / "CCGameManager.dat"
            export_dir = root / "exports"
            original_level = "kS1,0;1,1,2,15,3,0"
            generated_level = "kS1,0;1,1,2,15,3,30;1,8,2,45,3,60"
            xml = save_xml("AI Test", encode_level_string_k4(original_level))
            save_path.write_bytes(encode_save_xml(xml))

            result = inject_level_string_into_save(save_path, generated_level, export_dir)

            self.assertTrue(result.backup_path.exists())
            self.assertTrue(result.generated_save_path.exists())
            self.assertTrue(result.decoded_xml_path.exists())
            self.assertEqual(result.target_level_key, "k_1")

            decoded = decode_save_file(result.generated_save_path)
            self.assertIn("untouched", decoded)
            k4 = find_injected_k4(decoded, target_level_key="k_1")
            self.assertEqual(decode_level_string_k4(k4), generated_level)
            self.assertEqual(decode_level_string_k4(find_injected_k4(xml, target_level_key="k_1")), original_level)
            self.assertEqual(result.detected_codec, CODEC_XOR_BASE64_GZIP)

    def test_local_level_injection_targets_name_and_preserves_codec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_path = root / "CCLocalLevels.dat"
            export_dir = root / "exports"
            keep_level = "kS1,0;1,1,2,15,3,0"
            original_level = "kS1,0;1,1,2,45,3,0"
            generated_level = "kS1,0;1,1,2,15,3,30;1,8,2,45,3,60"
            xml = local_save_xml(
                [
                    ("Keep", keep_level),
                    ("AI Test", original_level),
                ],
                container_key="LLM_02",
            )
            save_path.write_bytes(encode_save_xml(xml, codec=CODEC_RAW_GZIP))

            result = inject_level_string_into_local_save(
                save_path,
                generated_level,
                export_dir,
                target_level_name="AI Test",
            )

            self.assertEqual(result.backup_path.name, "CCLocalLevels.backup.dat")
            self.assertEqual(result.generated_save_path.name, "CCLocalLevels.generated.dat")
            self.assertEqual(result.detected_codec, CODEC_RAW_GZIP)
            self.assertEqual(result.target_container_key, "LLM_02")
            self.assertEqual(result.target_level_key, "k_1")
            self.assertEqual(result.target_slot, 1)

            decoded = decode_save_bytes_with_codec(result.generated_save_path.read_bytes())
            self.assertEqual(decoded.detected_codec, CODEC_RAW_GZIP)
            self.assertIn("untouched", decoded.xml_text)
            self.assertEqual(
                decode_level_string_k4(find_local_level_k4(decoded.xml_text, target_slot=1)),
                generated_level,
            )
            self.assertEqual(
                decode_level_string_k4(find_local_level_k4(decoded.xml_text, target_slot=0)),
                keep_level,
            )

    def test_local_level_injection_targets_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_path = root / "CCLocalLevels.dat"
            export_dir = root / "exports"
            generated_level = "kS1,0;1,8,2,45,3,60"
            xml = local_save_xml(
                [
                    ("First", "kS1,0;1,1,2,15,3,0"),
                    ("Second", "kS1,0;1,1,2,45,3,0"),
                ]
            )
            save_path.write_bytes(encode_save_xml(xml, codec=CODEC_BASE64_GZIP))

            result = inject_level_string_into_local_save(
                save_path,
                generated_level,
                export_dir,
                target_slot=0,
            )

            decoded = decode_save_file(result.generated_save_path)
            self.assertEqual(result.target_level_name, "First")
            self.assertEqual(decode_level_string_k4(find_local_level_k4(decoded, target_slot=0)), generated_level)

    def test_local_level_injection_rejects_downloaded_level_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_path = root / "CCGameManager.dat"
            export_dir = root / "exports"
            xml = save_xml("Online Level", encode_level_string_k4("kS1,0;1,1,2,15,3,0"))
            save_path.write_bytes(encode_save_xml(xml))

            with self.assertRaises(SaveCodecError) as caught:
                inject_level_string_into_local_save(
                    save_path,
                    "kS1,0;1,8,2,45,3,60",
                    export_dir,
                    target_slot=0,
                )

        self.assertEqual(caught.exception.reason, "local_level_container_not_found")

    def test_export_validation_rejects_malformed_object_strings(self) -> None:
        layout = reconstruct_layout(
            [
                "START",
                "DIFF_HARD",
                "ALIGN_UNKNOWN",
                "BLOCK",
                "Y1",
                "WIDTH_1",
                "STEP",
                "END",
            ]
        )

        validate_layout_export(["START", "DIFF_HARD", "ALIGN_UNKNOWN", "STEP", "END"], layout)
        self.assertFalse(is_valid_gd_object_string("1,1,2,15"))


def write_pulse_wav(path: Path) -> None:
    sample_rate = 8_000
    duration = 2.0
    frames = array("h")
    for index in range(int(sample_rate * duration)):
        t = index / sample_rate
        pulse = 1.0 if (t % 0.5) < 0.045 else 0.12
        value = int(24_000 * pulse * math.sin(2 * math.pi * 440 * t))
        frames.append(value)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames.tobytes())


def save_xml(level_name: str, k4: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
  <dict>
    <key>GLM_03</key>
    <dict>
      <key>k_1</key>
      <dict>
        <key>k2</key>
        <string>{level_name}</string>
        <key>k4</key>
        <string>{k4}</string>
        <key>k99</key>
        <string>untouched</string>
      </dict>
    </dict>
  </dict>
</plist>"""


def local_save_xml(levels: list[tuple[str, str]], *, container_key: str = "LLM_01") -> str:
    entries = []
    for index, (level_name, level_string) in enumerate(levels):
        entries.append(
            f"""
      <key>k_{index}</key>
      <dict>
        <key>k2</key>
        <string>{level_name}</string>
        <key>k4</key>
        <string>{encode_level_string_k4(level_string)}</string>
        <key>k99</key>
        <string>untouched</string>
      </dict>"""
        )
    level_entries = "".join(entries)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
  <dict>
    <key>{container_key}</key>
    <dict>{level_entries}
    </dict>
  </dict>
</plist>"""


if __name__ == "__main__":
    unittest.main()
