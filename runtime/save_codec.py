from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import gzip
from pathlib import Path
import shutil
import zlib
import xml.etree.ElementTree as ET


XOR_KEY = 11
GLM_LEVELS_KEY = "GLM_03"
LOCAL_LEVELS_PREFIX = "LLM"
LEVEL_DATA_KEY = "k4"
CODEC_XOR_BASE64_GZIP = "xor_base64_gzip"
CODEC_BASE64_GZIP = "base64_gzip"
CODEC_RAW_GZIP = "raw_gzip"
CODEC_PLAINTEXT_XML = "plaintext_xml"


class SaveCodecError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail or reason


@dataclass(frozen=True)
class DecodedSave:
    xml_text: str
    detected_codec: str

    def to_json(self) -> dict[str, str]:
        return {
            "detected_codec": self.detected_codec,
        }


@dataclass(frozen=True)
class LocalLevelEntry:
    container_key: str
    level_key: str
    slot: int
    level: ET.Element


@dataclass(frozen=True)
class SaveInjectionResult:
    backup_path: Path
    generated_save_path: Path
    decoded_xml_path: Path
    target_level_key: str
    k4_length: int
    roundtrip_valid: bool
    detected_codec: str = ""
    target_slot: int | None = None
    target_level_name: str = ""
    target_container_key: str = ""

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backup_path": str(self.backup_path),
            "generated_save_path": str(self.generated_save_path),
            "decoded_xml_path": str(self.decoded_xml_path),
            "target_level_key": self.target_level_key,
            "k4_length": self.k4_length,
            "roundtrip_valid": self.roundtrip_valid,
        }
        if self.detected_codec:
            payload["detected_codec"] = self.detected_codec
        if self.target_slot is not None:
            payload["target_slot"] = self.target_slot
        if self.target_level_name:
            payload["target_level_name"] = self.target_level_name
        if self.target_container_key:
            payload["target_container_key"] = self.target_container_key
        return payload


def xor_bytes(data: bytes, *, key: int = XOR_KEY) -> bytes:
    return bytes(byte ^ key for byte in data)


def decode_save_bytes(raw: bytes) -> str:
    return decode_save_bytes_with_codec(raw).xml_text


def decode_save_bytes_with_codec(raw: bytes) -> DecodedSave:
    errors: list[str] = []
    strategies = [
        (CODEC_XOR_BASE64_GZIP, lambda: _decode_base64_gzip_xml(xor_bytes(raw))),
        (CODEC_BASE64_GZIP, lambda: _decode_base64_gzip_xml(raw)),
        (CODEC_RAW_GZIP, lambda: _decode_raw_gzip_xml(raw)),
        (CODEC_PLAINTEXT_XML, lambda: _decode_plaintext_xml(raw)),
    ]
    for codec, decoder in strategies:
        try:
            return DecodedSave(xml_text=decoder(), detected_codec=codec)
        except SaveCodecError as exc:
            errors.append(f"{codec}:{exc.reason}")

    if errors:
        raise SaveCodecError("decode_strategy_unsuccessful", ";".join(errors))
    raise SaveCodecError("decode_strategy_unsuccessful")


def _decode_base64_gzip_xml(raw: bytes) -> str:
    decoded = _base64_decode(raw)
    return _decode_raw_gzip_xml(decoded)


def _decode_raw_gzip_xml(raw: bytes) -> str:
    try:
        decompressed = zlib.decompress(raw, 15 | 32)
    except zlib.error as exc:
        raise SaveCodecError("gzip_decompress_failed", str(exc)) from exc

    return _decode_plaintext_xml(decompressed)


def _decode_plaintext_xml(raw: bytes) -> str:
    try:
        xml_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SaveCodecError("xml_unicode_decode_failed", str(exc)) from exc
    return _validated_xml_text(xml_text)


def _validated_xml_text(xml_text: str) -> str:
    root = parse_save_xml(xml_text)
    if _local_name(root.tag) != "plist":
        raise SaveCodecError("decoded_xml_not_plist", _local_name(root.tag))
    return xml_text


def encode_save_xml(xml_text: str, *, codec: str = CODEC_XOR_BASE64_GZIP) -> bytes:
    if codec == CODEC_PLAINTEXT_XML:
        return xml_text.encode("utf-8")
    compressed = gzip.compress(xml_text.encode("utf-8"))
    if codec == CODEC_RAW_GZIP:
        return compressed
    encoded = base64.b64encode(compressed)
    if codec == CODEC_BASE64_GZIP:
        return encoded
    if codec == CODEC_XOR_BASE64_GZIP:
        return xor_bytes(encoded)
    raise SaveCodecError("unsupported_save_codec", codec)


def decode_save_file(path: Path) -> str:
    return decode_save_file_with_codec(path).xml_text


def decode_save_file_with_codec(path: Path) -> DecodedSave:
    try:
        return decode_save_bytes_with_codec(path.read_bytes())
    except OSError as exc:
        raise SaveCodecError("save_read_failed", str(exc)) from exc


def encode_level_string_k4(level_string: str) -> str:
    compressed = gzip.compress(level_string.encode("utf-8"))
    return base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")


def decode_level_string_k4(value: str) -> str:
    try:
        decoded = _base64_decode(value.encode("ascii"))
        decompressed = zlib.decompress(decoded, 15 | 32)
        return decompressed.decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, zlib.error, SaveCodecError) as exc:
        raise SaveCodecError("k4_decode_failed", str(exc)) from exc


def inject_k4_into_save_xml(
    xml_text: str,
    k4_value: str,
    *,
    target_level_key: str | None = None,
    target_level_name: str | None = None,
) -> tuple[str, str]:
    root = parse_save_xml(xml_text)
    glm = find_value_after_key(root, GLM_LEVELS_KEY)
    if glm is None:
        raise SaveCodecError("glm_03_not_found")
    if _local_name(glm.tag) not in {"dict", "d", "array"}:
        raise SaveCodecError("glm_03_not_level_container", _local_name(glm.tag))

    selected_key = ""
    selected_level: ET.Element | None = None
    for level_key, level in iter_level_entries(glm):
        if _local_name(level.tag) not in {"dict", "d"}:
            continue
        if target_level_key and level_key != target_level_key:
            continue
        if target_level_name and level_name(level) != target_level_name:
            continue
        if find_value_after_key(level, LEVEL_DATA_KEY) is None:
            continue
        selected_key = level_key
        selected_level = level
        break

    if selected_level is None:
        if target_level_key or target_level_name:
            raise SaveCodecError("target_level_not_found")
        raise SaveCodecError("no_level_with_k4_found")

    k4_element = find_value_after_key(selected_level, LEVEL_DATA_KEY)
    if k4_element is None:
        raise SaveCodecError("target_level_missing_k4")
    k4_element.text = k4_value
    return serialize_xml(root), selected_key


def inject_level_string_into_save(
    save_path: Path,
    level_string: str,
    export_dir: Path,
    *,
    target_level_key: str | None = None,
    target_level_name: str | None = None,
) -> SaveInjectionResult:
    if not save_path.exists():
        raise SaveCodecError("save_file_not_found", str(save_path))
    if not level_string or ";" not in level_string:
        raise SaveCodecError("malformed_level_string")

    export_dir.mkdir(parents=True, exist_ok=True)
    backup_path = export_dir / "CCGameManager.backup.dat"
    generated_path = export_dir / f"{save_path.stem}.generated{save_path.suffix}"
    decoded_xml_path = export_dir / "decoded_save.xml"

    shutil.copy2(save_path, backup_path)
    decoded_save = decode_save_file_with_codec(save_path)
    original_xml = decoded_save.xml_text
    k4_value = encode_level_string_k4(level_string)
    updated_xml, selected_key = inject_k4_into_save_xml(
        original_xml,
        k4_value,
        target_level_key=target_level_key,
        target_level_name=target_level_name,
    )
    decoded_xml_path.write_text(updated_xml, encoding="utf-8", newline="\n")

    encoded_save = encode_save_xml(updated_xml, codec=decoded_save.detected_codec)
    generated_path.write_bytes(encoded_save)

    try:
        roundtrip_xml = decode_save_file_with_codec(generated_path).xml_text
        parse_save_xml(roundtrip_xml)
        roundtrip_k4 = find_injected_k4(
            roundtrip_xml,
            target_level_key=selected_key,
        )
        if decode_level_string_k4(roundtrip_k4) != level_string:
            raise SaveCodecError("k4_roundtrip_mismatch")
    except Exception:
        generated_path.unlink(missing_ok=True)
        raise

    return SaveInjectionResult(
        backup_path=backup_path,
        generated_save_path=generated_path,
        decoded_xml_path=decoded_xml_path,
        target_level_key=selected_key,
        k4_length=len(k4_value),
        roundtrip_valid=True,
        detected_codec=decoded_save.detected_codec,
    )


def inject_k4_into_local_save_xml(
    xml_text: str,
    k4_value: str,
    *,
    target_level_name: str | None = None,
    target_slot: int | None = None,
) -> tuple[str, LocalLevelEntry]:
    root = parse_save_xml(xml_text)
    selected = select_local_level_entry(
        discover_local_level_entries(root),
        target_level_name=target_level_name,
        target_slot=target_slot,
    )
    k4_element = find_value_after_key(selected.level, LEVEL_DATA_KEY)
    if k4_element is None:
        raise SaveCodecError("target_level_missing_k4")
    k4_element.text = k4_value
    return serialize_xml(root), selected


def inject_level_string_into_local_save(
    save_path: Path,
    level_string: str,
    export_dir: Path,
    *,
    target_level_name: str | None = None,
    target_slot: int | None = None,
) -> SaveInjectionResult:
    if not save_path.exists():
        raise SaveCodecError("save_file_not_found", str(save_path))
    if not level_string or ";" not in level_string:
        raise SaveCodecError("malformed_level_string")

    export_dir.mkdir(parents=True, exist_ok=True)
    backup_path = export_dir / "CCLocalLevels.backup.dat"
    generated_path = export_dir / "CCLocalLevels.generated.dat"
    decoded_xml_path = export_dir / "decoded_save.xml"

    shutil.copy2(save_path, backup_path)
    decoded_save = decode_save_file_with_codec(save_path)
    k4_value = encode_level_string_k4(level_string)
    updated_xml, selected = inject_k4_into_local_save_xml(
        decoded_save.xml_text,
        k4_value,
        target_level_name=target_level_name,
        target_slot=target_slot,
    )
    decoded_xml_path.write_text(updated_xml, encoding="utf-8", newline="\n")

    generated_path.write_bytes(encode_save_xml(updated_xml, codec=decoded_save.detected_codec))

    try:
        roundtrip = decode_save_file_with_codec(generated_path)
        parse_save_xml(roundtrip.xml_text)
        roundtrip_k4 = find_local_level_k4(roundtrip.xml_text, target_slot=selected.slot)
        if decode_level_string_k4(roundtrip_k4) != level_string:
            raise SaveCodecError("k4_roundtrip_mismatch")
    except Exception:
        generated_path.unlink(missing_ok=True)
        raise

    return SaveInjectionResult(
        backup_path=backup_path,
        generated_save_path=generated_path,
        decoded_xml_path=decoded_xml_path,
        target_level_key=selected.level_key,
        k4_length=len(k4_value),
        roundtrip_valid=True,
        detected_codec=decoded_save.detected_codec,
        target_slot=selected.slot,
        target_level_name=level_name(selected.level),
        target_container_key=selected.container_key,
    )


def find_injected_k4(xml_text: str, *, target_level_key: str) -> str:
    root = parse_save_xml(xml_text)
    glm = find_value_after_key(root, GLM_LEVELS_KEY)
    if glm is None:
        raise SaveCodecError("glm_03_not_found")
    for level_key, level in iter_level_entries(glm):
        if level_key != target_level_key:
            continue
        k4_element = find_value_after_key(level, LEVEL_DATA_KEY)
        if k4_element is None or not k4_element.text:
            raise SaveCodecError("target_level_missing_k4")
        return k4_element.text
    raise SaveCodecError("target_level_not_found")


def find_local_level_k4(
    xml_text: str,
    *,
    target_level_name: str | None = None,
    target_slot: int | None = None,
    target_level_key: str | None = None,
    target_container_key: str | None = None,
) -> str:
    root = parse_save_xml(xml_text)
    selected = select_local_level_entry(
        discover_local_level_entries(root),
        target_level_name=target_level_name,
        target_slot=target_slot,
        target_level_key=target_level_key,
        target_container_key=target_container_key,
    )
    k4_element = find_value_after_key(selected.level, LEVEL_DATA_KEY)
    if k4_element is None or not k4_element.text:
        raise SaveCodecError("target_level_missing_k4")
    return k4_element.text


def discover_local_level_entries(root: ET.Element) -> list[LocalLevelEntry]:
    containers = discover_local_level_containers(root)
    entries: list[LocalLevelEntry] = []
    slot = 0
    for container_key, container in containers:
        for level_key, level in iter_level_entries(container):
            if find_value_after_key(level, LEVEL_DATA_KEY) is None:
                continue
            entries.append(
                LocalLevelEntry(
                    container_key=container_key,
                    level_key=level_key,
                    slot=slot,
                    level=level,
                )
            )
            slot += 1
    if not entries:
        raise SaveCodecError("no_local_level_with_k4_found")
    return entries


def discover_local_level_containers(root: ET.Element) -> list[tuple[str, ET.Element]]:
    candidates: list[tuple[int, int, str, ET.Element]] = []
    for order, (key_name, value) in enumerate(iter_key_value_nodes(root)):
        if _local_name(value.tag) not in {"dict", "d", "array"}:
            continue
        entries = [
            level
            for _level_key, level in iter_level_entries(value)
            if find_value_after_key(level, LEVEL_DATA_KEY) is not None
        ]
        if not entries:
            continue
        priority = local_level_container_priority(key_name)
        if priority is None:
            continue
        candidates.append((priority, order, key_name, value))

    if not candidates:
        raise SaveCodecError("local_level_container_not_found")

    best_priority = min(priority for priority, _order, _key, _value in candidates)
    return [
        (key, value)
        for priority, _order, key, value in sorted(candidates, key=lambda item: (item[0], item[1]))
        if priority == best_priority
    ]


def local_level_container_priority(key_name: str) -> int | None:
    normalized_key = key_name.upper()
    if normalized_key.startswith(LOCAL_LEVELS_PREFIX):
        return 0
    if "LOCAL" in normalized_key and "LEVEL" in normalized_key:
        return 1
    return None


def select_local_level_entry(
    entries: list[LocalLevelEntry],
    *,
    target_level_name: str | None = None,
    target_slot: int | None = None,
    target_level_key: str | None = None,
    target_container_key: str | None = None,
) -> LocalLevelEntry:
    if target_slot is not None and target_slot < 0:
        raise SaveCodecError("target_slot_invalid", str(target_slot))

    for entry in entries:
        if target_slot is not None and entry.slot != target_slot:
            continue
        if target_level_key and entry.level_key != target_level_key:
            continue
        if target_container_key and entry.container_key != target_container_key:
            continue
        if target_level_name and level_name(entry.level) != target_level_name:
            continue
        return entry

    if target_slot is not None:
        raise SaveCodecError("target_slot_not_found", str(target_slot))
    if target_level_name:
        raise SaveCodecError("target_level_not_found", target_level_name)
    if target_level_key:
        raise SaveCodecError("target_level_not_found", target_level_key)
    raise SaveCodecError("no_local_level_with_k4_found")


def parse_save_xml(xml_text: str) -> ET.Element:
    try:
        return ET.fromstring(xml_text.encode("utf-8"))
    except ET.ParseError as exc:
        raise SaveCodecError("xml_parse_failed", str(exc)) from exc


def serialize_xml(root: ET.Element) -> str:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")


def find_value_after_key(parent: ET.Element, key: str) -> ET.Element | None:
    for _index, key_name, value in iter_key_values(parent):
        if key_name == key:
            return value
    for child in list(parent):
        result = find_value_after_key(child, key)
        if result is not None:
            return result
    return None


def iter_level_entries(container: ET.Element) -> list[tuple[str, ET.Element]]:
    tag = _local_name(container.tag)
    if tag in {"dict", "d"}:
        return [
            (key_name, value)
            for _index, key_name, value in iter_key_values(container)
            if _local_name(value.tag) in {"dict", "d"}
        ]
    if tag == "array":
        return [
            (f"index_{index}", child)
            for index, child in enumerate(list(container))
            if _local_name(child.tag) in {"dict", "d"}
        ]
    return []


def iter_key_values(parent: ET.Element) -> list[tuple[int, str, ET.Element]]:
    children = list(parent)
    pairs: list[tuple[int, str, ET.Element]] = []
    for index, child in enumerate(children[:-1]):
        if _local_name(child.tag) not in {"key", "k"}:
            continue
        key = (child.text or "").strip()
        if not key:
            continue
        pairs.append((index, key, children[index + 1]))
    return pairs


def iter_key_value_nodes(parent: ET.Element) -> list[tuple[str, ET.Element]]:
    pairs: list[tuple[str, ET.Element]] = []
    for _index, key_name, value in iter_key_values(parent):
        pairs.append((key_name, value))
    for child in list(parent):
        pairs.extend(iter_key_value_nodes(child))
    return pairs


def level_name(level: ET.Element) -> str:
    for key in ("k2", "k1", "name"):
        value = find_value_after_key(level, key)
        if value is not None and value.text:
            return value.text
    return ""


def _base64_decode(value: bytes) -> bytes:
    normalized = b"".join(value.split()).replace(b"-", b"+").replace(b"_", b"/")
    normalized += b"=" * (-len(normalized) % 4)
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SaveCodecError("base64_decode_failed", str(exc)) from exc


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag
